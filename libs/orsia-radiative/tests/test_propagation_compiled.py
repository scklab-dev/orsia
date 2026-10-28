"""Packaged recurrence equivalence, ownership and free-threaded execution."""

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path

import formal_reference
import numpy as np
import pytest
from orsia_radiative.transfer import formal
from orsia_radiative.transfer.native import propagation_native


def test_native_package_layout():
    assert propagation_native.__name__ == "orsia_radiative.transfer.native.propagation_native"
    assert Path(propagation_native.__file__).parent.name == "native"
    assert find_spec("orsia_radiative.transfer.propagation_native") is None
    assert formal._native_apply is propagation_native.apply
    assert formal._native_prepare is propagation_native.prepare


@pytest.mark.parametrize("solar_mu", [None, 0.8, 1e-5])
@pytest.mark.parametrize("cells", [0, 1, 19])
@pytest.mark.parametrize("layout", ["strided", "repeated", "wide", "empty"])
def test_preparation_equivalence(monkeypatch, solar_mu, cells, layout):
    depths = np.repeat(np.resize(np.array([0.0, 1e-20, 1e-16, 1e-10, 0.3, 1000.0]), cells), 2)[::-2]
    mu = np.array([-1.0, -0.8, -np.nextafter(0.8, 0), -1e-5, 1e-5, 0.8, 1.0])[::-1]
    if layout == "repeated":
        mu = np.tile(mu, 17)
    elif layout == "wide":
        mu = np.linspace(1e-5, 1.0, 1025)
    elif layout == "empty":
        mu = mu[:0]
    snapshots = depths.copy(), mu.copy()
    assert formal._native_prepare is propagation_native.prepare
    actual = formal.propagation_coefficients(depths, mu, solar_mu=solar_mu)
    monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    expected = formal.propagation_coefficients(depths, mu, solar_mu=solar_mu)
    for a, e in zip(vars(actual).values(), vars(expected).values(), strict=True):
        if e is None:
            assert a is None
        else:
            np.testing.assert_allclose(a, e, rtol=8e-15, atol=0.0)
            assert np.isfinite(a).all() and not a.flags.writeable
            assert a.flags.owndata
            assert not np.shares_memory(a, depths) and not np.shares_memory(a, mu)
    source = np.linspace(-0.1, 0.7, cells * mu.size).reshape(cells, mu.size)
    options = dict(solar_source=solar_mu is not None, bottom_radiance=0.2)
    for a, e in zip(
        formal.apply_propagation(source, actual, **options),
        formal.apply_propagation(source, expected, **options),
        strict=True,
    ):
        np.testing.assert_allclose(a, e, rtol=2e-14, atol=1e-15)
    np.testing.assert_array_equal(depths, snapshots[0])
    np.testing.assert_array_equal(mu, snapshots[1])


def test_preparation_near_zero_exprel(monkeypatch):
    # Includes SciPy's epsilon branch boundary and the exact equal solar angle.
    depth = np.r_[0.0, np.logspace(-22, -2, 101), np.finfo(float).eps]
    mu = np.array([-0.8, -np.nextafter(0.8, 0), 0.8])
    actual = formal.propagation_coefficients(depth, mu, solar_mu=0.8)
    monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    expected = formal.propagation_coefficients(depth, mu, solar_mu=0.8)
    for a, e in zip(vars(actual).values(), vars(expected).values(), strict=True):
        np.testing.assert_allclose(a, e, rtol=8e-16, atol=0)


@pytest.mark.parametrize("size", [32, 1024, 1025])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raw_preparation_unique_cosines(monkeypatch, size, dtype):
    # Exercise the raw scalar loop and lookup cap even when automatic dispatch
    # conservatively selects NumPy for these unique-angle matrices.
    depth = np.array([0.0, 1e-20, 0.4, 1000.0], dtype=dtype)
    mu = np.linspace(-1.0, -1e-5, size, dtype=dtype)
    outputs = [np.empty((depth.size, mu.size), dtype=dtype) for _ in range(6)]
    propagation_native.prepare(depth, mu, 0.8, *outputs)
    monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    expected = formal.propagation_coefficients(depth, mu, solar_mu=0.8)
    for a, e in zip(outputs, list(vars(expected).values())[1:], strict=True):
        assert a.dtype == e.dtype == dtype
        np.testing.assert_allclose(
            a,
            e,
            rtol=4e-6 if dtype == np.float32 else 8e-15,
            atol=3e-45 if dtype == np.float32 else 0,
        )


def test_preparation_fallback(monkeypatch):
    def forbidden(*args):
        pytest.fail("large unique grids must not reach the native ABI")

    monkeypatch.setattr(formal, "_native_prepare", forbidden)
    # Native scalar transcendental functions lose to vector NumPy on wide
    # unique grids; dispatch intentionally preserves the reference path.
    for size in (257, 1025):
        formal.propagation_coefficients(np.ones(3), np.linspace(0.1, 1, size), solar_mu=0.8)


def test_preparation_buffer_validation():
    depth, mu = np.ones(3), np.ones(2)
    outputs = [np.empty((3, 2)) for _ in range(6)]
    with pytest.raises(ValueError, match="doubles"):
        propagation_native.prepare(depth.astype(np.float32), mu, 0.8, *outputs)  # type: ignore[bad-argument-type]
    with pytest.raises(ValueError, match="vector"):
        propagation_native.prepare(depth.reshape(1, 3), mu, 0.8, *outputs)
    with pytest.raises(ValueError, match="grid"):
        propagation_native.prepare(depth[:1], mu, 0.8, *outputs)
    with pytest.raises(ValueError, match="solar"):
        propagation_native.prepare(depth, mu, 0.0, *outputs)
    with pytest.raises(ValueError, match="solar"):
        propagation_native.prepare(
            depth, mu, 0.8, outputs[0], outputs[1], outputs[2], outputs[3], None, outputs[5]
        )
    outputs[0].flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        propagation_native.prepare(depth, mu, 0.8, *outputs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_concurrent_preparation(dtype):
    depth = np.array([0.0, 1e-12, 0.2, 1000.0], dtype=dtype)
    mu = np.tile(np.array([-0.8, 0.8], dtype=dtype), 17)
    expected = formal.propagation_coefficients(depth, mu, solar_mu=0.8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: formal.propagation_coefficients(depth, mu, solar_mu=0.8), range(40))
        )
    for result in results:
        for a, e in zip(vars(result).values(), vars(expected).values(), strict=True):
            np.testing.assert_array_equal(a, e)
            assert not np.shares_memory(a, e)
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("depth", [0.0, 1e-16, 1e-6, 0.4, 1000.0])
@pytest.mark.parametrize("solar", [False, True])
@pytest.mark.parametrize("centres", [False, True])
@pytest.mark.parametrize("layout", ["ordered", "interleaved", "up", "down", "empty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_equivalence(monkeypatch, depth, solar, centres, layout, dtype):
    mu = np.array([-1.0, -0.8, -0.01, 0.01, 0.8, 1.0], dtype=dtype)
    if layout == "interleaved":
        mu = mu[[0, 3, 1, 4, 2, 5]]
    elif layout == "up":
        mu = mu[3:]
    elif layout == "down":
        mu = mu[:3]
    elif layout == "empty":
        mu = mu[:0]
    source = np.linspace(-0.1, 0.7, 7 * mu.size * 2, dtype=dtype).reshape(7, mu.size * 2)[:, ::-2]
    coefficients = formal.propagation_coefficients(
        np.full(7, depth, dtype=dtype), mu, solar_mu=0.8 if solar else None
    )
    inputs = [source] + [a for a in vars(coefficients).values() if a is not None]
    snapshots = [a.copy() for a in inputs]
    options = dict(solar_source=solar, return_centres=centres, bottom_radiance=2.0)
    assert formal._native_apply is propagation_native.apply
    actual = formal.apply_propagation(source, coefficients, **options)
    monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    expected = formal.apply_propagation(source, coefficients, **options)
    for a, e in zip(actual, expected, strict=True):
        if e is None:
            assert a is None
        else:
            np.testing.assert_array_equal(a, e)
            assert a.dtype == dtype
            assert a.flags.writeable
            assert all(not np.shares_memory(a, original) for original in inputs)
    for original, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("cells", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cell_edges(monkeypatch, cells, dtype):
    mu = np.array([0.3, -0.3], dtype=dtype)
    source = np.ones((cells, 2), dtype=dtype)
    coefficients = formal.propagation_coefficients(np.zeros(cells, dtype=dtype), mu)
    actual = formal.apply_propagation(source, coefficients, bottom_radiance=3.0)
    monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    expected = formal.apply_propagation(source, coefficients, bottom_radiance=3.0)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def test_errors_and_fallback(monkeypatch):
    coefficients = formal.propagation_coefficients(np.ones(3), np.array([0.8, -0.8]))
    source = np.ones((3, 2))
    with pytest.raises(ValueError, match="solar forcing"):
        formal.apply_propagation(source, coefficients, solar_source=True)
    with pytest.raises(ValueError, match="grid"):
        formal.apply_propagation(source[:1], coefficients)

    # Mixed precision promotes both output fields, using the NumPy fallback.
    def forbidden(*args):
        pytest.fail("mixed precision must not reach the native ABI")

    monkeypatch.setattr(formal, "_native_apply", forbidden)
    centres, exits = formal.apply_propagation(source.astype(np.float32), coefficients)
    assert centres.dtype == np.float64
    assert exits.dtype == np.float64


def test_native_buffer_validation():
    mu = np.ones(2)
    source = np.ones((3, 2))
    with pytest.raises(ValueError, match="doubles"):
        propagation_native.apply(
            mu,
            source.astype(np.float32),  # type: ignore[bad-argument-type]
            source,
            source,
            None,
            None,
            0.0,
            None,
            mu.copy(),
        )
    with pytest.raises(ValueError, match="grid"):
        propagation_native.apply(mu, source, source[:1], source, None, None, 0.0, None, mu.copy())
    with pytest.raises(ValueError, match="grid"):
        propagation_native.apply(mu, source, source, source, None, None, 0.0, None, np.ones(1))
    with pytest.raises(ValueError, match="vector"):
        propagation_native.apply(
            mu.reshape(1, 2), source, source, source, None, None, 0.0, None, mu.copy()
        )


def test_shared_coefficients_free_threaded():
    assert sys.version_info[:2] == (3, 14)
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    assert not sys._is_gil_enabled()
    coefficients = formal.propagation_coefficients(
        np.array([0.0, 1e-12, 1000.0]), np.array([0.8, -0.8])
    )
    source = np.ones((3, 2))
    expected = formal.apply_propagation(source, coefficients)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: formal.apply_propagation(source, coefficients), range(40))
        )
    for result in results:
        for actual, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
    assert not sys._is_gil_enabled()
