"""Exact serial recurrence contracts for range dispatch and raw buffer aliases."""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from orsia_radiative.transfer.native import propagation_native


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("centres", [False, True])
@pytest.mark.parametrize("cells", [0, 1, 7])
@pytest.mark.parametrize(
    "layout", ["ordered", "reverse", "mixed", "up", "down", "zero", "nan", "empty"]
)
def test_signed_ranges(dtype, centres, cells, layout):
    mu = np.array([-1, -0.8, -1e-5, 1e-5, 0.8, 1], dtype=dtype)
    if layout == "reverse":
        mu = mu[::-1].copy()
    elif layout == "mixed":
        mu = mu[[0, 3, 1, 4, 2, 5]]
    elif layout == "up":
        mu = mu[3:].copy()
    elif layout == "down":
        mu = mu[:3].copy()
    elif layout == "zero":
        mu[2:4] = [0, -0.0]
    elif layout == "nan":
        mu[2] = np.nan
    elif layout == "empty":
        mu = mu[:0]
    args = _arguments(mu, cells, centres)
    expected = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    _original(*expected)
    propagation_native.apply(*args)
    _equal(args, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("centres", [False, True])
@pytest.mark.parametrize(
    "alias",
    [
        "field_source",
        "field_t",
        "field_w",
        "field_ht",
        "field_hw",
        "exit_mu",
        "exit_source",
        "exit_t",
        "exit_w",
        "exit_ht",
        "exit_hw",
        "outputs",
        "partial_field",
        "partial_exit",
        "inputs",
    ],
)
def test_raw_alias_fallback(dtype, centres, alias):
    def make():
        args = _arguments(np.array([-0.8, -0.2, 0.2, 0.8], dtype=dtype), 3, centres)
        index = {"source": 1, "t": 2, "w": 3, "ht": 4, "hw": 5}
        if alias.startswith("field_") and centres:
            args[7] = args[index[alias[6:]]]
        elif alias == "exit_mu":
            args[8] = args[0]
        elif alias.startswith("exit_"):
            matrix = args[index[alias[5:]]]
            if matrix is not None:
                args[8] = matrix[0]
        elif alias == "outputs" and centres:
            args[8] = args[7][1]
        elif alias == "partial_field" and centres:
            backing = np.arange(13, dtype=dtype) / dtype(20)
            args[1] = backing[:12].reshape(3, 4)
            args[7] = backing[1:].reshape(3, 4)
        elif alias == "partial_exit":
            args[8] = args[1].ravel()[1:5]
        elif alias == "inputs":
            args[2] = args[3] = args[1]
            if centres:
                args[4] = args[5] = args[1]
        return args

    actual, expected = make(), make()
    _original(*expected)
    propagation_native.apply(*actual)
    _equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("centres", [False, True])
def test_ieee_source_values(dtype, centres):
    args = _arguments(np.array([-0.8, -0.2, 0.2, 0.8], dtype=dtype), 7, centres)
    args[1][:] = np.array([np.inf, -np.inf, np.nan, -0.0], dtype=dtype)
    expected = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    with np.errstate(invalid="ignore"):
        _original(*expected)
    propagation_native.apply(*args)
    for a, b in zip(args, expected, strict=True):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
            finite = np.isfinite(b)
            assert a[finite].tobytes() == b[finite].tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("centres", [False, True])
def test_concurrent_native_prepare_apply(dtype, centres):
    # One immutable input set, independent writable buffers per invocation.
    depth = np.array([0, 1e-20, 1e-6, 0.4, 1000], dtype=dtype)
    mu = np.repeat(np.array([-0.8, 0.8], dtype=dtype), 129)
    source = np.linspace(-0.3, 0.7, depth.size * mu.size, dtype=dtype).reshape(depth.size, mu.size)
    for a in (depth, mu, source):
        a.flags.writeable = False

    def run(_):
        matrices = [np.empty_like(source) for _ in range(6)]
        propagation_native.prepare(depth, mu, 0.8, *matrices)
        field = np.empty_like(source) if centres else None
        exits = np.empty_like(mu)
        propagation_native.apply(
            mu,
            source,
            matrices[0],
            matrices[4],
            matrices[1] if centres else None,
            matrices[5] if centres else None,
            0.2,
            field,
            exits,
        )
        return matrices + [field, exits]

    expected = run(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for actual in pool.map(run, range(40)):
            _equal(actual, expected)
            for a, b in zip(actual, expected, strict=True):
                if a is not None:
                    assert not np.shares_memory(a, b)
    assert not sys._is_gil_enabled()


def _arguments(mu, cells, centres):
    shape = (cells, mu.size)
    depth = np.resize(np.array([0, 1e-20, 1e-6, 0.4, 1000], dtype=mu.dtype), cells)
    matrices = [np.empty(shape, dtype=mu.dtype) for _ in range(6)]
    # Raw apply also accepts zero/NaN directions; give those deterministic finite
    # coefficients to isolate the original sign-filtering behavior.
    safe_mu = mu.copy()
    safe_mu[~np.isfinite(safe_mu) | (safe_mu == 0)] = 0.8
    propagation_native.prepare(depth, safe_mu, 0.8, *matrices)
    source = np.linspace(-0.3, 0.7, cells * mu.size, dtype=mu.dtype).reshape(shape)
    return [
        mu,
        source,
        matrices[0],
        matrices[4],
        matrices[1] if centres else None,
        matrices[5] if centres else None,
        0.2,
        np.zeros_like(source) if centres else None,
        np.empty_like(mu),
    ]


def _original(mu, source, transmission, weight, ht, hw, bottom, field, exits):
    # Scalar transcription of the original native loop, including observable
    # alias mutations, initialization order and raw zero/NaN sign comparisons.
    real = source.dtype.type
    for d in range(mu.size):
        exits[d] = real(bottom) if mu[d] > 0 else real(0)
    for upward in (False, True):
        for step in range(source.shape[0]):
            c = step if upward else source.shape[0] - 1 - step
            for d in range(mu.size):
                if (mu[d] <= 0) if upward else (mu[d] >= 0):
                    continue
                incoming = exits[d]
                if field is not None:
                    field[c, d] = incoming * ht[c, d] + source[c, d] * hw[c, d]
                exits[d] = incoming * transmission[c, d] + source[c, d] * weight[c, d]


def _equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b
