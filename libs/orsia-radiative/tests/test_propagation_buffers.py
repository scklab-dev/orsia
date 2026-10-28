"""Native buffer ABI: no conversion, exports held through calls, exact rejection."""

import _interpreters
import gc
import sys
from array import array
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from orsia_radiative.transfer.native import propagation_native


@pytest.mark.parametrize("format", ["f", "d"])
@pytest.mark.parametrize("solar", [False, True])
@pytest.mark.parametrize("centres", [False, True])
def test_non_numpy_buffers(format, solar, centres):
    depth = array(format, [0, 0.4, 1])
    mu = array(format, [-0.8, 0.8])
    matrices = [array(format, [0] * 6) for _ in range(6)]
    views = [memoryview(a).cast("B").cast(format, shape=[3, 2]) for a in matrices]
    assert (
        propagation_native.prepare(
            memoryview(depth).toreadonly(),
            memoryview(mu).toreadonly(),
            0.8,
            views[0],
            views[1],
            views[2],
            views[3],
            views[4] if solar else None,
            views[5] if solar else None,
        )
        is None
    )
    dtype = np.float32 if format == "f" else np.float64
    expected = [np.empty((3, 2), dtype=dtype) for _ in range(6)]
    propagation_native.prepare(
        np.array(depth, dtype=dtype),
        np.array(mu, dtype=dtype),
        0.8,
        expected[0],
        expected[1],
        expected[2],
        expected[3],
        expected[4] if solar else None,
        expected[5] if solar else None,
    )
    count = 6 if solar else 4
    for view, reference in zip(views[:count], expected[:count], strict=True):
        np.testing.assert_array_equal(view, reference)
    source = memoryview(array(format, [0.2] * 6)).cast("B").cast(format, shape=[3, 2])
    field = memoryview(array(format, [0] * 6)).cast("B").cast(format, shape=[3, 2])
    exits = array(format, [0] * 2)
    assert (
        propagation_native.apply(
            mu,
            source.toreadonly(),
            views[0].toreadonly(),
            views[4 if solar else 2].toreadonly(),
            views[1].toreadonly() if centres else None,
            views[5 if solar else 3].toreadonly() if centres else None,
            0.2,
            field if centres else None,
            exits,
        )
        is None
    )
    expected_field, expected_exits = np.empty((3, 2), dtype=dtype), np.empty(2, dtype=dtype)
    propagation_native.apply(
        np.array(mu, dtype=dtype),
        np.array(source),
        expected[0],
        expected[4 if solar else 2],
        expected[1] if centres else None,
        expected[5 if solar else 3] if centres else None,
        0.2,
        expected_field if centres else None,
        expected_exits,
    )
    if centres:
        np.testing.assert_array_equal(field, expected_field)
    np.testing.assert_array_equal(exits, expected_exits)
    # No exports survive the call: resizes would fail while arrays are exported.
    depth.append(1)
    mu.append(1)
    exits.append(1)


class Exporter:
    """Python buffer exporter retaining storage and recording export lifetimes."""

    def __init__(self, view):
        self.view = view
        self.exports = 0
        self.releases = 0

    def __buffer__(self, flags):
        self.exports += 1
        return self.view

    def __release_buffer__(self, view):
        self.releases += 1


@pytest.mark.parametrize("fail", [None, "rank", "precision", "acquisition"])
def test_export_release_on_success_and_error(fail):
    depth, mu = Exporter(memoryview(array("d", [0.4]))), Exporter(memoryview(array("d", [0.8])))
    outputs = [
        Exporter(memoryview(array("d", [0])).cast("B").cast("d", shape=[1, 1])) for _ in range(4)
    ]
    if fail == "acquisition":
        # A later acquisition failure must release all earlier successful exports.
        with pytest.raises(TypeError):
            propagation_native.prepare(
                depth,
                mu,
                0,
                outputs[0],
                outputs[1],
                outputs[2],
                object(),  # type: ignore[bad-argument-type]
                None,
                None,
            )
        outputs.pop()  # this unused exporter was never acquired
    elif fail:
        if fail == "precision":
            outputs[-1] = Exporter(memoryview(array("f", [0])).cast("B").cast("f", shape=[1, 1]))
        else:
            outputs[-1] = Exporter(memoryview(array("d", [0])))  # wrong rank, after acquisition
        with pytest.raises(ValueError, match="matching native" if fail == "precision" else "grid"):
            propagation_native.prepare(
                depth, mu, 0, outputs[0], outputs[1], outputs[2], outputs[3], None, None
            )
    else:
        propagation_native.prepare(
            depth, mu, 0, outputs[0], outputs[1], outputs[2], outputs[3], None, None
        )
        exits = Exporter(memoryview(array("d", [0])))
        # Exit-only deliberately ignores the two centre coefficients.
        propagation_native.apply(
            mu,
            outputs[0],
            outputs[0],
            outputs[2],
            object(),  # type: ignore[bad-argument-type]
            object(),  # type: ignore[bad-argument-type]
            0.2,
            None,
            exits,
        )
        assert exits.exports == exits.releases == 1
    gc.collect()
    for exporter in (depth, mu, *outputs):
        assert exporter.exports > 0 and exporter.exports == exporter.releases


@pytest.mark.parametrize("format", ["f", "d"])
def test_shared_readonly_buffers_concurrent(format):
    mu = memoryview(array(format, [-0.8, 0.8])).toreadonly()
    source = memoryview(array(format, [0.2] * 6)).cast("B").cast(format, shape=[3, 2]).toreadonly()
    transmission = (
        memoryview(array(format, [0.5] * 6)).cast("B").cast(format, shape=[3, 2]).toreadonly()
    )

    def run(_):
        exits = array(format, [0, 0])
        propagation_native.apply(mu, source, transmission, source, None, None, 0.2, None, exits)
        return exits

    expected = run(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(40)))
    assert all(result == expected for result in results)
    assert all(result is not expected for result in results)
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("which", ["field", "exits"])
@pytest.mark.parametrize("numpy", [False, True])
def test_readonly_outputs(which, numpy):
    mu = np.array([-0.8, 0.8])
    source = np.ones((3, 2))
    field, exits = np.empty_like(source), np.empty_like(mu)
    if which == "field":
        field.flags.writeable = False
    else:
        exits.flags.writeable = False
    with pytest.raises(ValueError if numpy else BufferError):
        if numpy:
            propagation_native.apply(mu, source, source, source, source, source, 0.2, field, exits)
        else:
            propagation_native.apply(
                memoryview(mu),
                memoryview(source),
                memoryview(source),
                memoryview(source),
                memoryview(source),
                memoryview(source),
                0.2,
                memoryview(field),
                memoryview(exits),
            )


@pytest.mark.parametrize("numpy", [False, True])
def test_noncontiguous_rejected_without_copy(numpy):
    mu = np.ones(4)[::2]
    source = np.ones((3, 2))
    exits = np.full(2, -99.0)
    with pytest.raises(ValueError if numpy else BufferError):
        propagation_native.apply(
            mu if numpy else memoryview(mu), source, source, source, None, None, 0, None, exits
        )
    np.testing.assert_array_equal(exits, [-99, -99])
    with pytest.raises(ValueError):
        propagation_native.apply(
            np.ones(2), source, np.asfortranarray(source), source, None, None, 0, None, exits
        )


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.dtype(">f8")])
def test_non_native_float_rejected(dtype):
    mu = np.ones(2, dtype=dtype)
    source = np.ones((3, 2), dtype=dtype)
    with pytest.raises(ValueError, match="matching native"):
        propagation_native.apply(mu, source, source, source, None, None, 0, None, mu.copy())


def test_native_positional_only_and_no_conversion():
    with pytest.raises(TypeError):
        propagation_native.prepare(depth=np.ones(1))  # type: ignore[unexpected-keyword, bad-argument-count]
    with pytest.raises(TypeError):
        propagation_native.apply(mu=np.ones(1))  # type: ignore[unexpected-keyword, bad-argument-count]
    with pytest.raises(TypeError):
        matrix = np.ones((1, 1))
        propagation_native.prepare(
            [1],  # type: ignore[bad-argument-type]
            np.ones(1),
            0,
            matrix,
            matrix,
            matrix,
            matrix,
            None,
            None,
        )


def test_native_in_subinterpreter():
    # Load only the extension: NumPy/SciPy are not qualified in subinterpreters.
    interpreter = _interpreters.create()
    try:
        _interpreters.exec(
            interpreter,
            f"""
import importlib.util
import sys
from array import array
spec = importlib.util.spec_from_file_location("propagation_native", {propagation_native.__file__!r})
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
mu = array("d", [0.8])
source = memoryview(array("d", [0.4])).cast("B").cast("d", shape=[1, 1])
exits = array("d", [0])
native.apply(mu, source, source, source, None, None, 0.2, None, exits)
assert exits[0] == 0.2 * 0.4 + 0.4 * 0.4
assert not sys._is_gil_enabled()
""",
        )
    finally:
        _interpreters.destroy(interpreter)


@pytest.mark.parametrize("operation", ["prepare", "apply"])
@pytest.mark.parametrize("bad_scalar", [10**1000, "0.8", None])
def test_scalar_conversion_errors(operation, bad_scalar):
    mu, matrix = np.array([0.8]), np.ones((1, 1))
    error = OverflowError if isinstance(bad_scalar, int) else TypeError
    with pytest.raises(error):
        if operation == "prepare":
            propagation_native.prepare(
                mu, mu, bad_scalar, matrix, matrix, matrix, matrix, None, None
            )
        else:
            propagation_native.apply(mu, matrix, matrix, matrix, None, None, bad_scalar, None, mu)


def test_scalar_user_exception_preserved():
    class Scalar:
        def __float__(self):
            raise RuntimeError("scalar sentinel")

    mu, matrix = np.array([0.8]), np.ones((1, 1))
    with pytest.raises(RuntimeError, match="scalar sentinel"):
        propagation_native.prepare(
            mu,
            mu,
            Scalar(),  # type: ignore[bad-argument-type]
            matrix,
            matrix,
            matrix,
            matrix,
            None,
            None,
        )
    with pytest.raises(RuntimeError, match="scalar sentinel"):
        propagation_native.apply(
            mu,
            matrix,
            matrix,
            matrix,
            None,
            None,
            Scalar(),  # type: ignore[bad-argument-type]
            None,
            mu,
        )
