"""Explicit derivative contracts and installed native-header discovery."""

from pathlib import Path

import numpy as np
import pytest
from orsia_numerics import get_include
from orsia_numerics import tangents as autodiff
from orsia_numerics.tangents import directional_derivative, jacobian


@pytest.mark.parametrize("block_size", [None, 1, 2, 20])
def test_jacobian_blocks_and_ownership(block_size):
    x = np.array([2.0, 3.0, 4.0])
    calls = []
    matrix = np.arange(12.0).reshape(2, 2, 3)

    def evaluate(state, seeds):
        assert not state.flags.writeable
        assert not np.shares_memory(state, x)
        calls.append(seeds.copy())
        return matrix @ state, matrix @ seeds

    value, derivatives = jacobian(evaluate, x, block_size=block_size)
    np.testing.assert_array_equal(value, matrix @ x)
    np.testing.assert_array_equal(derivatives, matrix)
    np.testing.assert_array_equal(np.concatenate(calls, axis=1), np.eye(3))
    for output in (value, derivatives):
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        assert not np.shares_memory(output, matrix)


def test_empty_state_scalar_and_zero_output():
    calls = []

    def empty(state, seeds):
        calls.append(seeds.shape)
        return np.array(3.0), np.zeros(0)

    value, derivatives = jacobian(empty, np.empty(0))
    assert value.shape == () and derivatives.shape == (0,) and calls == [(0, 0)]
    value, derivatives = jacobian(
        lambda x, v: (np.empty(0), np.empty((0, v.shape[1]))), np.ones(2), block_size=1
    )
    assert value.shape == (0,) and derivatives.shape == (0, 2)


@pytest.mark.parametrize("x", [np.ones((1, 1)), np.ones(1, dtype=np.float32), np.array([np.inf])])
@pytest.mark.parametrize("api", [jacobian, directional_derivative])
def test_invalid_state(x, api):
    args = (np.ones(1),) if api is directional_derivative else ()
    with pytest.raises(ValueError, match="x must"):
        api(lambda x, v: (x, v), x, *args)


@pytest.mark.parametrize("block_size", [0, -1, True, 1.5])
def test_invalid_block(block_size):
    with pytest.raises(ValueError, match="block_size"):
        jacobian(lambda x, v: (x, v), np.ones(1), block_size=block_size)


@pytest.mark.parametrize(
    "outputs",
    [
        (np.ones(2, dtype=np.float32), np.ones((2, 1))),
        (np.ones(2), np.ones((2, 1), dtype=np.float32)),
        (np.ones(2), np.ones((1, 2))),
        (np.array([np.nan]), np.ones((1, 1))),
        (np.ones(1), np.array([[np.inf]])),
    ],
)
@pytest.mark.parametrize("api", [jacobian, directional_derivative])
def test_invalid_callback(outputs, api):
    args = (np.ones(1),) if api is directional_derivative else ()
    with pytest.raises(ValueError, match="callback"):
        api(lambda x, v: outputs, np.ones(1), *args)


def test_changing_value_and_callback_exception():
    with pytest.raises(ValueError, match="changed"):
        jacobian(lambda x, v: (v[:, 0], v), np.ones(2), block_size=1)

    def failure(x, v):
        raise RuntimeError("caller model")

    with pytest.raises(RuntimeError, match="caller model"):
        jacobian(failure, np.ones(1))


@pytest.mark.parametrize("block_size", [None, 1, 4])
def test_callback_keywords(block_size):
    def square(x, seeds):
        return x * x, 2 * x[:, None] * seeds

    x = np.array([2.0, 3.0])
    value, derivatives = jacobian(function=square, x=x, block_size=block_size)
    np.testing.assert_array_equal(value, [4.0, 9.0])
    np.testing.assert_array_equal(derivatives, np.diag([4.0, 6.0]))
    value, derivative = directional_derivative(function=square, x=x, direction=np.ones(2))
    np.testing.assert_array_equal(value, [4.0, 9.0])
    np.testing.assert_array_equal(derivative, [4.0, 6.0])


def test_removed_dense_jacobian_api():
    assert not hasattr(autodiff, "dense_jacobian")
    assert "dense_jacobian" not in autodiff.__all__


@pytest.mark.parametrize("output_shape", [(), (4,), (2, 2), (0,), (2, 0)])
@pytest.mark.parametrize("n", [0, 3])
@pytest.mark.parametrize("k", [None, 0, 1, 2])
def test_direction_shapes_products_and_ownership(output_shape, n, k):
    x = np.arange(float(n))
    matrix = np.arange(float(np.prod(output_shape) * n)).reshape(output_shape + (n,))
    direction = np.arange(float(n if k is None else n * k)).reshape((n,) if k is None else (n, k))
    calls = []
    callback_outputs = []

    def evaluate(state, seeds):
        assert state.flags.owndata and not state.flags.writeable
        assert seeds.flags.owndata and seeds.flags.writeable and seeds.flags.c_contiguous
        assert not np.shares_memory(state, x)
        assert not np.shares_memory(seeds, direction)
        calls.append(seeds.shape)
        result = (np.asarray(matrix @ state), matrix @ seeds)
        callback_outputs.extend(result)
        return result

    value, derivative = directional_derivative(evaluate, x, direction)
    assert calls == [(n, 1 if k is None else k)]
    assert value.shape == output_shape
    assert derivative.shape == output_shape + (() if k is None else (k,))
    np.testing.assert_array_equal(value, matrix @ x)
    np.testing.assert_array_equal(derivative, matrix @ direction)
    for output in (value, derivative):
        assert output.dtype == np.float64
        assert output.flags.owndata and output.flags.writeable and output.flags.c_contiguous
        for source in (x, direction, *callback_outputs):
            assert not np.shares_memory(output, source)


@pytest.mark.parametrize("vector", [False, True])
def test_direction_layouts_and_callback_aliases(vector):
    x = np.arange(6.0)[::2]
    storage = np.arange(24.0).reshape(6, 4)
    direction = storage[::2, 0] if vector else storage[::2, ::-2]
    direction.flags.writeable = False
    original = direction.copy()
    retained = []

    def identity(state, seeds):
        retained.extend((state, seeds))
        return state, seeds

    value, derivative = directional_derivative(identity, x, direction)
    np.testing.assert_array_equal(value, x)
    np.testing.assert_array_equal(derivative, original)
    retained[1][:] = -10
    x[:] = -20
    np.testing.assert_array_equal(value, [0, 2, 4])
    np.testing.assert_array_equal(derivative, original)
    np.testing.assert_array_equal(direction, original)


@pytest.mark.parametrize(
    "direction",
    [
        np.array(1.0),
        np.ones((2, 1, 1)),
        np.ones(3),
        np.ones((3, 0)),
        np.ones(2, dtype=np.float32),
        np.ones(2, dtype=np.int64),
        np.ones(2, dtype=np.dtype("float64").newbyteorder("S")),
        np.array([np.nan, 1.0]),
        np.array([[1.0], [np.inf]]),
    ],
)
def test_invalid_direction(direction):
    def unused(x, seeds):
        pytest.fail("invalid directions must be rejected before evaluation")

    with pytest.raises(ValueError, match="direction must"):
        directional_derivative(unused, np.ones(2), direction)


def test_direction_callback_exception():
    def failure(x, seeds):
        raise RuntimeError("caller model")

    with pytest.raises(RuntimeError, match="caller model"):
        directional_derivative(failure, np.ones(2), np.empty((2, 0)))


def test_header_discovery():
    path = Path(get_include())
    assert path.is_absolute()
    assert (path / "orsia_numerics" / "tangent.h").is_file()
