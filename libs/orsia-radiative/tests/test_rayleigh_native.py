"""Rayleigh arithmetic parity, callback independence and native buffer safety."""

import builtins
import gc
import json
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import numpy as np
import orders_reference as orders
import pytest
from orsia_radiative.transfer import formal, solar
from orsia_radiative.transfer import orders as production_orders
from orsia_radiative.transfer.native import propagation_native
from test_solar_accumulation import _inputs
from test_solar_boundary import _sweep_reference


@pytest.fixture(params=["portable", "blas"], autouse=True)
def native_backend(request, monkeypatch):
    if request.param == "blas":
        monkeypatch.setattr(
            propagation_native,
            "rayleigh_orders",
            propagation_native.bind_rayleigh_blas(import_module("scipy.linalg.cython_blas")),
        )


def test_blas_owner_lifetime_and_one_time_resolution() -> None:
    cython_blas = import_module("scipy.linalg.cython_blas")
    owner = ModuleType("retained_blas_owner")
    owner.__dict__["__pyx_capi__"] = dict(cython_blas.__pyx_capi__)
    reference = weakref.ref(owner)
    bound = propagation_native.bind_rayleigh_blas(owner)
    args = _native_args(_point(2, False, 4))
    expected = bound(*args)
    owner.__dict__["__pyx_capi__"] = None
    del owner
    gc.collect()
    assert reference() is not None
    with patch.object(builtins, "__import__", side_effect=RuntimeError("import forbidden")):
        actual = bound(*args)
    assert actual.tobytes() == expected.tobytes()
    del bound
    gc.collect()
    assert reference() is None


def test_production_binds_scipy_blas_at_import() -> None:
    script = """
import importlib
import sys
from unittest.mock import patch
import numpy as np
from scipy.linalg import cython_blas
from orsia_radiative.transfer import orders
from orsia_radiative.transfer.native import propagation_native
sys.path.insert(0, sys.argv[1])
from test_solar_accumulation import _inputs
scene, source, optics = _inputs('positive', 0.3)
expected = orders.scattering_orders(scene, source=source, optics=optics)
with patch.object(propagation_native, 'bind_spectral_blas',
                  wraps=propagation_native.bind_spectral_blas) as bind:
    importlib.reload(orders)
    bind.assert_called_once_with(cython_blas)
    for _ in range(2):
        actual = orders.scattering_orders(scene, source=source, optics=optics)
        np.testing.assert_array_equal(actual, expected)
    bind.assert_called_once_with(cython_blas)
assert not sys._is_gil_enabled()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).parent)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("max_order", [0, 1, 4, 16])
@pytest.mark.parametrize("reflectance", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("case", ["positive", "zero_layer", "zero_point", "dark", "opaque"])
def test_public_parity(case: str, reflectance: float, max_order: int) -> None:
    scene, source, optics = _inputs(case, reflectance)
    controls = dict(max_order=max_order, angular_order=3, layer_subdivisions=2)
    with patch.object(orders, "_accumulate_orders", _sweep_reference):
        expected = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    with patch.object(
        production_orders,
        "_native_spectral_orders",
        wraps=production_orders._native_spectral_orders,
    ) as native:
        actual = production_orders.scattering_orders(
            scene, source=source, optics=optics, **controls
        )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)
    assert native.call_count == 1
    assert actual.flags.owndata and actual.flags.writeable
    total = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    assert total.values.tobytes() == actual.sum(axis=0).tobytes()


@pytest.mark.parametrize("subdivisions", [1, 3, 8])
def test_public_dispatch_prepares_native_arguments(subdivisions: int) -> None:
    scene, source, optics = _inputs("positive", 0.3)
    assert optics.rayleigh is not None and optics.absorption is not None
    with patch.object(
        production_orders,
        "_native_spectral_orders",
        wraps=production_orders._native_spectral_orders,
    ) as native:
        production_orders.scattering_orders(
            scene,
            source=source,
            optics=optics,
            max_order=4,
            angular_order=3,
            layer_subdivisions=subdivisions,
        )
    assert native.call_count == 1
    optical, irradiance, reflectance, directions, weights, layers, sub, maximum, moments, *_ = (
        native.call_args.args
    )
    assert optical[0] is optics.absorption.optical_depth
    assert optical[1] is optics.rayleigh.optical_depth
    assert irradiance is source.irradiance.values
    assert reflectance is scene.surface.reflectance
    assert directions.shape == (36, 3) and weights.shape == (36,)
    assert (layers, sub, maximum, moments) == (3, subdivisions, 4, 0)


@pytest.mark.parametrize("angular_order", [1, 2, 5, 8, 12, 16])
@pytest.mark.parametrize("permutation", [False, True])
@pytest.mark.parametrize("max_order", [0, 1, 6])
def test_point_parity_and_callback_independence(
    angular_order: int, permutation: bool, max_order: int
) -> None:
    args = _point(angular_order, permutation, max_order)
    expected = _point_reference(angular_order, permutation, max_order)
    with (
        patch.object(np, "matmul", wraps=np.matmul) as matmul,
        patch.object(np, "sum", wraps=np.sum) as reduction,
    ):
        actual = propagation_native.rayleigh_orders(*_native_args(args))
    assert matmul.call_count == 0
    assert reduction.call_count == 0
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0)


def test_concurrent_owned_scratch_and_readonly_inputs() -> None:
    args = _native_args(_point(5, True, 8))
    arrays = [args[0], *args[1], args[2], *args[3], args[4], args[5], args[6]]
    snapshots = [a.tobytes() for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    expected = propagation_native.rayleigh_orders(*args)

    def calculate(_: int):
        return propagation_native.rayleigh_orders(*args)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(16)))
    assert not sys._is_gil_enabled()
    for result in results:
        assert result.tobytes() == expected.tobytes()
        assert result.flags.writeable and result.flags.owndata
        assert not np.shares_memory(result, expected)
    results[0][:] = -1
    assert results[1].tobytes() == expected.tobytes()
    assert [a.tobytes() for a in arrays] == snapshots


@pytest.mark.parametrize(
    "fault",
    [
        "float32",
        "unaligned",
        "strided",
        "depth_shape",
        "beam_shape",
        "factor_shape",
        "weight_shape",
        "coefficient_shape",
        "coefficient_tuple",
        "factor_tuple",
        "negative_orders",
        "zero_mu",
        "nan_mu",
        "missing_solar",
    ],
)
def test_native_rejects_invalid_buffers(fault: str) -> None:
    args: list[Any] = list(_native_args(_point(2, False, 2)))
    if fault == "float32":
        args[0] = args[0].astype(np.float32)
    elif fault == "unaligned":
        args[0] = np.ndarray(
            args[0].shape, buffer=bytearray(args[0].nbytes + 1), offset=1, dtype=np.float64
        )
    elif fault == "strided":
        args[0] = np.repeat(args[0], 2)[::2]
    elif fault == "depth_shape":
        args[0] = args[0][:, None]
    elif fault == "beam_shape":
        args[2] = args[2][:-1]
    elif fault == "factor_shape":
        args[1] = (args[1][0][:-1], args[1][1])
    elif fault == "weight_shape":
        args[6] = args[6][:-1]
    elif fault == "coefficient_shape":
        args[3] = (args[3][0], args[3][1][:-1], *args[3][2:])
    elif fault == "coefficient_tuple":
        args[3] = args[3][:-1]
    elif fault == "factor_tuple":
        args[1] = args[1][:1]
    elif fault == "negative_orders":
        args[-1] = -1
    elif fault in ("zero_mu", "nan_mu"):
        mu = args[3][0].copy()
        mu[0] = 0 if fault == "zero_mu" else np.nan
        args[3] = (mu, *args[3][1:])
    elif fault == "missing_solar":
        args[3] = (*args[3][:-1], None)
    with pytest.raises((ValueError, TypeError)):
        propagation_native.rayleigh_orders(*args)


def test_numpy_callbacks_can_raise_without_affecting_core() -> None:
    args = _native_args(_point(2, False, 2))
    expected = propagation_native.rayleigh_orders(*args)
    with (
        patch.object(np, "matmul", side_effect=RuntimeError("callback forbidden")),
        patch.object(np, "sum", side_effect=RuntimeError("callback forbidden")),
    ):
        actual = propagation_native.rayleigh_orders(*args)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("cells", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_empty_cells_and_single_hemisphere(cells: int, sign: int) -> None:
    args: list[Any] = list(_native_args(_point(2, False, 3)))
    mu = sign * np.abs(args[3][0])
    coefficients = formal.propagation_coefficients(np.zeros(cells), mu, solar_mu=0.7)
    args[0] = np.zeros(cells)
    args[2] = np.zeros((cells, mu.size))
    args[3] = (
        coefficients.mu,
        coefficients.transmission,
        coefficients.centre_transmission,
        coefficients.source_weight,
        coefficients.centre_source_weight,
        coefficients.solar_source_weight,
        coefficients.solar_centre_source_weight,
    )
    args[4] = np.zeros_like(args[2])
    args[5] = np.zeros(mu.size)
    np.testing.assert_array_equal(propagation_native.rayleigh_orders(*args), np.zeros(3))


@pytest.mark.parametrize("angular_directions", [0, 1, 7, 8, 9, 127, 128, 129, 513])
@pytest.mark.parametrize("signed", [False, True])
def test_seven_factor_arithmetic_against_extended_precision(
    angular_directions: int, signed: bool
) -> None:
    """Isolate both products with a transparent one-cell formal response.

    Signed inputs deliberately exercise cancellation in the private numerical ABI.
    The absolute bound scales with product mass, preserving near-zero comparisons.
    """
    rng = np.random.default_rng(27)
    n = angular_directions + 1
    field = rng.uniform(0.1, 1.0, (1, n))
    incoming = rng.uniform(0.1, 1.0, (angular_directions, 7))
    outgoing = rng.uniform(0.1, 1.0, (7, n))
    if signed:
        incoming[::2] *= -1
        outgoing[::2] *= -1
    zeros, ones = np.zeros((1, n)), np.ones((1, n))
    args = (
        np.ones(1),
        (incoming, outgoing),
        zeros,
        (np.ones(n), zeros, zeros, ones, ones, ones, ones),
        field,
        np.zeros(n),
        np.ones(angular_directions),
        1.0,
        0.0,
        1,
    )
    actual = propagation_native.rayleigh_orders(*args)[0]
    f, a, b = (x.astype(np.longdouble) for x in (field[0, :-1], incoming, outgoing[:, -1]))
    expected = (f @ a) @ b
    mass = (np.abs(f) @ np.abs(a)) @ np.abs(b)
    bound = 8 * np.finfo(np.float64).eps * (angular_directions + 7) * mass
    assert abs(np.longdouble(actual) - expected) <= bound
    assert np.isfinite(actual)


@pytest.mark.parametrize("invalid", [False, True])
def test_general_buffer_exports_release_once(invalid: bool) -> None:
    class Exporter:
        def __init__(self, array):
            self.array = array
            self.acquired = 0
            self.released = 0

        def __buffer__(self, flags):
            self.acquired += 1
            return memoryview(self.array)

        def __release_buffer__(self, view):
            self.released += 1

    args: list[Any] = list(_native_args(_point(2, False, 3)))
    expected = propagation_native.rayleigh_orders(*args)
    arrays = [args[0], *args[1], args[2], *args[3], args[4], args[5], args[6]]
    wrappers = [Exporter(a) for a in arrays]
    args[:7] = [
        wrappers[0],
        tuple(wrappers[1:3]),
        wrappers[3],
        tuple(wrappers[4:11]),
        *wrappers[11:],
    ]
    if invalid:
        wrappers[8].array = arrays[8][:-1]
        with pytest.raises(ValueError, match="grid mismatch"):
            propagation_native.rayleigh_orders(*args)
    else:
        assert propagation_native.rayleigh_orders(*args).tobytes() == expected.tobytes()
    assert all(w.acquired == 1 and w.released == 1 for w in wrappers)


def _point_reference(angular_order: int, permutation: bool, max_order: int):
    """Frozen pre-removal Python outputs; provenance accompanies the small fixture."""
    fixture = json.loads(Path(__file__).with_name("rayleigh_orders_reference.json").read_text())
    return np.array(fixture["values"][f"{angular_order}:{int(permutation)}"][:max_order])


def _point(angular_order: int, permutation: bool, max_order: int):
    rng = np.random.default_rng(34)
    directions, weights = orders.angular_grid(angular_order)
    if permutation:
        index = rng.permutation(weights.size)
        directions, weights = directions[index], weights[index]
    outgoing = np.vstack((directions, [0.6, 0.0, 0.8]))
    factors = orders._prepare_rayleigh_factors(directions, outgoing, weights)
    layer_depth = np.array([0.001, 0.03, 0.12])
    transport = formal.propagation_coefficients(
        np.repeat(layer_depth / 3, 3), outgoing[:, 2], solar_mu=0.7
    )
    beam = rng.uniform(0, 0.1, transport.transmission.shape)
    centres, exits = formal.apply_propagation(np.zeros_like(beam), transport, bottom_radiance=1.0)
    return (0.6, layer_depth, 3, factors, beam, transport, centres, exits, weights, 0.3, max_order)


def _native_args(args):
    (
        direct,
        layer_depth,
        subdivisions,
        factors,
        beam,
        t,
        centres,
        exits,
        weights,
        reflectance,
        max_order,
    ) = args
    return (
        np.repeat(layer_depth / subdivisions, subdivisions),
        factors,
        beam,
        (
            t.mu,
            t.transmission,
            t.centre_transmission,
            t.source_weight,
            t.centre_source_weight,
            t.solar_source_weight,
            t.solar_centre_source_weight,
        ),
        centres,
        exits,
        weights,
        direct,
        reflectance,
        max_order,
    )
