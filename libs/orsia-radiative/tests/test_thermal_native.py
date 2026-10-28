"""Native thermal regression against the pre-port NumPy/SciPy calculation."""

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, localcontext
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import thermal_reference as reference
from orsia_radiative import AerosolComponent, AtmosphericOptics, wavenumber_grid
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.transfer import thermal
from orsia_radiative.transfer.native import propagation_native
from scipy import constants
from scipy.special import roots_legendre
from test_thermal import case


@pytest.mark.parametrize("method", ["quadrature", "expn"])
@pytest.mark.parametrize("shape", [(1, 1), (3, 11), (41, 1), (41, 127)])
@pytest.mark.parametrize("view", [0.0, 53.0, 89.9])
@pytest.mark.parametrize("roles", ["absorption", "aerosol", "mixed"])
def test_complete_calls_against_pre_port_reference(method, shape, view, roles):
    rng = np.random.default_rng(5901)
    depth = 10.0 ** rng.uniform(-25, 3, shape)
    depth[rng.random(shape) < 0.15] = 0.0
    scene, optics = case(depth, rng.uniform(180, 330, shape[0]), emissivity=0.43, view=view)
    emissivity = rng.uniform(0, 1, shape[1])
    emissivity[::3] = 1
    scene = replace(scene, surface=replace(scene.surface, emissivity=emissivity))
    if roles != "absorption":
        aerosol = AerosolComponent(optics.grid, depth, np.zeros_like(depth), np.zeros_like(depth))
        absorption = optics.absorption
        if roles == "aerosol":
            absorption = None
        else:
            assert absorption is not None
            absorption = replace(absorption, optical_depth=depth * 0.25)
            aerosol = replace(aerosol, optical_depth=depth * 0.75)
        optics = replace(optics, absorption=absorption, aerosol=aerosol)
    expected = reference.schwarzschild_terms(
        scene, optics=optics, angular_order=12, hemispheric_method=method
    )
    actual = thermal.schwarzschild_terms(
        scene, optics=optics, angular_order=12, hemispheric_method=method
    )
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=4e-13, atol=1e-315)
    radiance = thermal.schwarzschild_radiance(
        scene, optics=optics, angular_order=12, hemispheric_method=method
    )
    np.testing.assert_array_equal(radiance.values, sum(actual.values()))
    assert radiance.grid is optics.grid and radiance.geometry is scene.geometry


@pytest.mark.parametrize("temperature", [1e-100, 3.0, 300.0, 1e7, 1e100])
def test_native_planck_limits_against_radiometry(temperature):
    grid = wavenumber_grid(np.array([1e-100, 1e-8, 1.0, 1000.0, 1e8, 1e100]))
    scene, _ = case(np.zeros((1, 6)))
    scene = replace(
        scene,
        atmosphere=replace(scene.atmosphere, temperature_k=np.array([temperature])),
        surface=replace(scene.surface, grid=grid, temperature_k=temperature),
    )
    optics = AtmosphericOptics(grid, None, None, None)
    actual = thermal.schwarzschild_terms(scene, optics=optics)
    expected = reference.schwarzschild_terms(scene, optics=optics)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=5e-13, atol=5e-324)


def test_native_planck_overflow_and_mutated_physical_inputs():
    scene, optics = case(np.ones((2, 3)))
    assert optics.absorption is not None
    for bad in (-1.0, np.inf, np.nan):
        scene.atmosphere.temperature_k[0] = bad
        with pytest.raises(ValueError, match="temperature"):
            thermal.schwarzschild_terms(scene, optics=optics)
    scene.atmosphere.temperature_k[:] = 280
    for bad in (-1.0, np.inf, np.nan):
        optics.absorption.optical_depth[0, 0] = bad
        with pytest.raises(ValueError, match="finite and nonnegative"):
            thermal.schwarzschild_terms(scene, optics=optics)
    grid = wavenumber_grid(np.array([1e100, 1e101, 1e102]))
    hot = replace(scene, surface=replace(scene.surface, grid=grid, temperature_k=1e200))
    with pytest.raises(ValueError, match="Planck radiance"):
        thermal.schwarzschild_terms(hot, optics=AtmosphericOptics(grid, None, None, None))


@pytest.mark.parametrize("depth,emissivity", [(0.0, 0.4), (1.0, 1.0)])
def test_inactive_reflection_preserves_lazy_quadrature(monkeypatch, depth, emissivity):
    scene, optics = case(np.full((2, 3), depth), emissivity=emissivity)

    def unexpected_nodes(*args):
        raise AssertionError("inactive reflection reached angular node construction")

    monkeypatch.setattr(thermal, "roots_legendre", unexpected_nodes)
    result = thermal.schwarzschild_terms(scene, optics=optics, angular_order=1000000)
    np.testing.assert_array_equal(result["reflected_atmospheric_emission"], 0.0)


@pytest.mark.parametrize("method", ["quadrature", "expn"])
def test_strided_readonly_inputs_owned_outputs_and_concurrent_calls(method):
    scene, optics = case(np.array([[0, 1e-20, 0.1, 0.9], [0.1, 0.4, 0.3, 1e308]]), emissivity=0.4)
    assert optics.absorption is not None
    original = optics.absorption.optical_depth.copy()
    backing = np.empty((2, 8))
    backing[:, ::2] = original
    strided = backing[:, ::2]
    strided.flags.writeable = False
    object.__setattr__(optics.absorption, "optical_depth", strided)
    scene.atmosphere.temperature_k.flags.writeable = False
    assert scene.surface.emissivity is not None
    scene.surface.emissivity.flags.writeable = False

    def evaluate():
        return thermal.schwarzschild_terms(scene, optics=optics, hemispheric_method=method)

    expected = evaluate()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: evaluate(), range(9)))
    for result in results:
        for key, values in result.items():
            np.testing.assert_array_equal(values, expected[key])
            assert values.flags.owndata and values.flags.writeable
            assert values.dtype == np.float64 and values.shape == (4,)
            for other in (*expected.values(), strided):
                assert not np.shares_memory(values, other)
        assert not np.shares_memory(result["surface_emission"], result["atmospheric_emission"])
    np.testing.assert_array_equal(strided, original)
    assert not sys._is_gil_enabled()


def native_inputs() -> list[Any]:
    nodes, weights = roots_legendre(4)
    return [
        np.array([500.0, 1000.0]),
        np.array([280.0, 230.0]),
        np.array([0.3, 1.0]),
        np.ones((2, 2)) * 0.1,
        None,
        (nodes + 1) / 2,
        weights,
        300.0,
        0.8,
        False,
    ]


@pytest.mark.parametrize("layout", ["big_endian", "unaligned"])
def test_wrapper_normalizes_binary64_storage(layout):
    scene, optics = case(np.full((2, 3), 0.2), emissivity=0.4)
    assert optics.absorption is not None
    for owner, field in (
        (scene.atmosphere, "temperature_k"),
        (scene.surface, "emissivity"),
        (optics.absorption, "optical_depth"),
    ):
        values = getattr(owner, field)
        if layout == "big_endian":
            converted = values.astype(">f8")
        else:
            converted = np.ndarray(
                values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1
            )
            converted[:] = values
        object.__setattr__(owner, field, converted)
    actual = thermal.schwarzschild_terms(scene, optics=optics, hemispheric_method="expn")
    expected = reference.schwarzschild_terms(scene, optics=optics, hemispheric_method="expn")
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=4e-13)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("bad", ["rank", "dtype", "strides", "alignment", "support"])
def test_private_binding_rejects_incompatible_buffers(index, bad):
    values = native_inputs()
    array = values[index]
    if array is None:
        array = np.ones((2, 2))
    if bad == "rank":
        array = array.reshape(1, *array.shape)
    elif bad == "dtype":
        array = array.astype(np.float32)
    elif bad == "strides":
        array = array[::-1]
    elif bad == "alignment":
        unaligned = np.ndarray(
            array.shape, dtype=np.float64, buffer=bytearray(array.nbytes + 1), offset=1
        )
        unaligned[:] = array
        array = unaligned
    else:
        array = np.ones((1, 3)) if array.ndim == 2 else np.empty(0)
    values[index] = array
    with pytest.raises((ValueError, BufferError)):
        thermal._native_terms(*values)


@pytest.mark.parametrize("index,bad", [(7, 0.0), (7, np.inf), (8, 0.0), (8, 1.1), (8, np.nan)])
def test_private_binding_scalar_support(index, bad):
    values = native_inputs()
    values[index] = bad
    with pytest.raises(ValueError, match="physical support"):
        thermal._native_terms(*values)


def test_private_binding_angular_support_and_interval_shapes():
    for index, bad in ((5, 0.0), (5, np.inf), (6, -1.0), (6, np.nan)):
        values = native_inputs()
        values[index][0] = bad
        with pytest.raises(ValueError, match="angular grid"):
            thermal._native_terms(*values)
    with pytest.raises(ValueError, match="interval support"):
        thermal._native_intervals(np.ones(2), np.ones(3))
    assert thermal._native_intervals(np.empty(0), np.empty(0)).shape == (0,)


def test_expn_capsule_signature_and_owner_lifetime():
    provider = import_module("scipy.special.cython_special")
    owner = ModuleType("thermal_provider")
    owner.__dict__["__pyx_capi__"] = provider.__pyx_capi__
    retained = weakref.ref(owner)
    second_radiation = 100.0 * constants.h * constants.c / constants.k
    rayleigh_jeans = 2.0 * constants.k * constants.c * 1e6
    terms, intervals = propagation_native.bind_thermal_expn(owner, second_radiation, rayleigh_jeans)
    del owner
    gc.collect()
    assert retained() is not None
    assert intervals(np.array([0.0]), np.array([1e-20]))[0] > 0
    assert len(terms(*native_inputs())) == 3
    del terms, intervals
    gc.collect()
    assert retained() is None
    invalid = ModuleType("incompatible_provider")
    invalid.__dict__["__pyx_capi__"] = {
        "__pyx_fuse_0expn": import_module("scipy.linalg.cython_blas").__pyx_capi__["dgemm"]
    }
    with pytest.raises(ValueError, match="incorrect name"):
        propagation_native.bind_thermal_expn(invalid, second_radiation, rayleigh_jeans)


@pytest.mark.parametrize("analytic", [False, True])
def test_binding_captures_supplied_planck_coefficients(analytic):
    provider = import_module("scipy.special.cython_special")
    second_radiation = 100.0 * constants.h * constants.c / constants.k
    rayleigh_jeans = 2.0 * constants.k * constants.c * 1e6
    terms, _ = propagation_native.bind_thermal_expn(
        provider, 2.0 * second_radiation, 3.0 * rayleigh_jeans
    )
    values = native_inputs()
    values[-1] = analytic
    original = thermal._native_terms(*values)
    actual = terms(*values)
    values[1] = values[1] / 2.0
    values[7] /= 2.0
    expected = thermal._native_terms(*values)
    for result, baseline in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, 6.0 * baseline, rtol=4e-14)
    values[1] *= 2.0
    values[7] *= 2.0
    for result, baseline in zip(thermal._native_terms(*values), original, strict=True):
        np.testing.assert_array_equal(result, baseline)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize("index", [0, 1])
def test_binding_rejects_invalid_planck_coefficients(index, bad):
    coefficients = [
        100.0 * constants.h * constants.c / constants.k,
        2.0 * constants.k * constants.c * 1e6,
    ]
    coefficients[index] = bad
    with pytest.raises(ValueError, match="Planck coefficients"):
        propagation_native.bind_thermal_expn(
            import_module("scipy.special.cython_special"), *coefficients
        )


def test_free_threaded_runtime_after_thermal_import():
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("temperature", [1e-100, 3.0, 300.0, 1e7, 1e100])
def test_emitting_layers_at_planck_fast_path_boundaries(temperature):
    second_radiation = 100.0 * constants.h * constants.c / constants.k
    ratios = np.array(
        [1e-8, 0.999999e-4, 1e-4, 1.000001e-4, 1.0, 499.9999, 500.0, 500.0001, 700.0, 745.0, 1000.0]
    )
    grid = ratios * (temperature / second_radiation)
    values = native_inputs()
    values[:7] = [
        grid,
        np.array([temperature]),
        np.ones(grid.size),
        np.full((1, grid.size), 1000.0),
        None,
        np.empty(0),
        np.empty(0),
    ]
    values[7] = temperature
    values[-1] = True
    actual = thermal._native_terms(*values)[1]
    expected = planck_wavenumber(grid, temperature)
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize(
    "second_radiation,rayleigh_jeans,nu,temperature",
    [
        (1e300, 1e-300, 1e-10, 1e290),
        (1e-300, 1e300, 1e10, 1e-290),
        (1e100, 1e-200, 1e-55, 1e45),
    ],
)
def test_planck_numerator_intermediate_extremes(second_radiation, rayleigh_jeans, nu, temperature):
    terms, _ = propagation_native.bind_thermal_expn(
        import_module("scipy.special.cython_special"), second_radiation, rayleigh_jeans
    )
    actual = terms(
        np.array([nu]),
        np.array([temperature]),
        np.ones(1),
        np.full((1, 1), 1000.0),
        None,
        np.empty(0),
        np.empty(0),
        temperature,
        1.0,
        True,
    )[1][0]
    # Decimal products retain extreme intermediate magnitudes and evaluate the
    # algebraic law independently of the binary64 logarithmic fallback.
    with localcontext() as context:
        context.prec = 70
        c2, rj, wave, temp = map(
            Decimal.from_float, (second_radiation, rayleigh_jeans, nu, temperature)
        )
        expected = float(rj * c2 * wave**3 / (((wave / temp) * c2).exp() - 1))
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
@pytest.mark.parametrize("role", [3, 4])
def test_native_rejects_each_depth_component(bad, role):
    values = native_inputs()
    values[3] = np.ones((2, 2)) * 2
    values[4] = np.ones((2, 2)) * 2
    values[role][0, 0] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        thermal._native_terms(*values)


def test_native_rejects_assembled_depth_overflow():
    values = native_inputs()
    values[3] = np.full((2, 2), 1e308)
    values[4] = np.full((2, 2), 1e308)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        thermal._native_terms(*values)


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_native_rejects_invalid_output_terms(bad):
    values = native_inputs()
    values[2][0] = bad
    with pytest.raises(ValueError, match="thermal radiance terms"):
        thermal._native_terms(*values)


def test_native_rejects_reflected_overflow():
    values = native_inputs()
    values[1][:] = 1e6
    values[6][:] = 1e308
    with pytest.raises(ValueError, match="thermal radiance terms"):
        thermal._native_terms(*values)


def test_public_radiance_retains_assembled_overflow_check(monkeypatch):
    scene, optics = case(np.ones((1, 1)))
    monkeypatch.setattr(thermal, "_native_terms", lambda *args: (np.array([1e308]),) * 3)
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="radiance values must be finite"),
    ):
        thermal.schwarzschild_radiance(scene, optics=optics)


def test_native_layout_fast_path_borrows_stable_arrays():
    values = np.ones((2, 3))
    values.flags.writeable = False
    assert thermal._thermal_buffer(values) is values
    strided = values[:, ::2]
    normalized = thermal._thermal_buffer(strided)
    assert normalized.flags.c_contiguous and normalized.flags.aligned
    assert not np.shares_memory(strided, normalized)
    np.testing.assert_array_equal(strided, normalized)
