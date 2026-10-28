"""Complete solar calls: current inputs, independent results and native scratch."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import orders_reference as reference
import pytest
from orsia_radiative import AtmosphericOptics, wavenumber_grid
from orsia_radiative.transfer import orders, solar
from orsia_radiative.transfer.native import propagation_native
from test_spectral_native import mixed_inputs
from threadpoolctl import threadpool_info


@pytest.mark.parametrize("maximum", [0, 1, 4])
@pytest.mark.parametrize("moments", [None, 4, 6])
@pytest.mark.parametrize("layers", [2, 5])
@pytest.mark.parametrize("roles", ["direct", "rayleigh", "aerosol", "mixed"])
def test_complete_calls_match_reference(maximum, moments, layers, roles):
    scene, source, optics = mixed_inputs(layers)
    if roles in ("direct", "aerosol"):
        optics = replace(optics, rayleigh=None)
    if roles in ("direct", "rayleigh"):
        optics = replace(optics, aerosol=None)
    controls = dict(
        max_order=maximum, angular_order=3, layer_subdivisions=2, delta_m_moments=moments
    )
    expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    actual = orders.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    for _ in range(2):
        total = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
        np.testing.assert_array_equal(total.values, actual.sum(axis=0))
        assert total.geometry == scene.geometry and total.grid is source.irradiance.grid
        assert total.values.flags.owndata and total.values.flags.writeable
        assert not np.shares_memory(total.values, actual)


@pytest.mark.parametrize("moments", [None, 4])
def test_a_b_a_changes_roles_extinction_and_phase(moments):
    scene, source, a = mixed_inputs()
    assert a.absorption is not None and a.aerosol is not None
    b = replace(
        a,
        absorption=replace(a.absorption, optical_depth=a.absorption.optical_depth * 3),
        rayleigh=None,
        aerosol=replace(
            a.aerosol,
            asymmetry=np.full_like(a.aerosol.asymmetry, -0.3),
            optical_depth=a.aerosol.optical_depth * 2,
            scattering_depth=a.aerosol.scattering_depth * 0.5,
        ),
    )
    empty = AtmosphericOptics(a.grid, None, None, None)
    controls = dict(max_order=4, angular_order=4, layer_subdivisions=3, delta_m_moments=moments)
    old = []
    for optics in (a, b, a, empty, b, a):
        expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
        actual = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
        np.testing.assert_allclose(actual.values, expected.sum(axis=0), rtol=3e-13, atol=0)
        old.append((actual.values, actual.values.copy()))
    for actual, saved in old:
        np.testing.assert_array_equal(actual, saved)
    old[0][0][:] = -10
    np.testing.assert_array_equal(old[2][0], old[2][1])
    saved = a.aerosol.asymmetry.copy()
    a.aerosol.asymmetry[:] = 0.7
    changed = solar.reflected_radiance(scene, source=source, optics=a, **controls)
    expected = reference.scattering_orders(scene, source=source, optics=a, **controls)
    np.testing.assert_allclose(changed.values, expected.sum(axis=0), rtol=3e-13, atol=0)
    a.aerosol.asymmetry[:] = saved
    np.testing.assert_array_equal(
        solar.reflected_radiance(scene, source=source, optics=a, **controls).values, old[2][1]
    )


def test_current_illumination_surface_geometry_and_controls():
    scene, source, optics = mixed_inputs()
    assert scene.surface.reflectance is not None
    controls = dict(max_order=3, angular_order=3, layer_subdivisions=2)
    saved = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    original = saved.values.copy()
    source.irradiance.values[:] *= 2
    doubled = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    np.testing.assert_array_equal(doubled.values, 2 * original)
    scene.surface.reflectance[:] = 0
    scene = replace(scene, geometry=replace(scene.geometry, view_zenith_deg=60.0))
    source = replace(source, zenith_deg=45.0, relative_azimuth_deg=80.0)
    controls.update(max_order=4, angular_order=4, layer_subdivisions=3)
    actual = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    expected = reference.scattering_orders(scene, source=source, optics=optics, **controls)
    np.testing.assert_allclose(actual.values, expected.sum(axis=0), rtol=3e-13, atol=0)
    assert actual.geometry == scene.geometry
    np.testing.assert_array_equal(saved.values, original)


def test_concurrent_complete_calls_and_runtime():
    scene, source, optics = mixed_inputs()

    def calculate(_):
        return solar.reflected_radiance(
            scene, source=source, optics=optics, angular_order=4, max_order=6, delta_m_moments=6
        )

    expected = calculate(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(calculate, range(12)))
    for result in results:
        np.testing.assert_array_equal(result.values, expected.values)
        assert not np.shares_memory(result.values, expected.values)
    assert not sys._is_gil_enabled()
    assert set(os.sched_getaffinity(0)) <= set(range(8))
    assert all(pool["num_threads"] == 1 for pool in threadpool_info())


@pytest.mark.parametrize("maximum", [0, 4])
@pytest.mark.parametrize("scattering", [False, True])
def test_complete_call_uses_one_native_boundary(maximum, scattering):
    scene, source, optics = mixed_inputs()
    if not scattering:
        optics = replace(optics, rayleigh=None, aerosol=None)
    with (
        patch.object(orders, "angular_grid", wraps=orders.angular_grid) as angular,
        patch.object(
            orders, "_native_spectral_orders", wraps=orders._native_spectral_orders
        ) as native,
    ):
        solar.reflected_radiance(
            scene, source=source, optics=optics, max_order=maximum, angular_order=3
        )
    assert native.call_count == 1
    assert angular.call_count == int(bool(maximum) and scattering)


@pytest.mark.parametrize("maximum", [0, 4])
def test_support_errors_precede_native_computation(maximum):
    scene, source, optics = mixed_inputs()
    _, _, wrong_layers = mixed_inputs(3)
    wrong_grid = AtmosphericOptics(wavenumber_grid(optics.grid.points + 1), None, None, None)
    with patch.object(orders, "_native_spectral_orders") as native:
        with pytest.raises(ValueError, match="source and optical spectral axes"):
            solar.reflected_radiance(scene, source=source, optics=wrong_grid, max_order=maximum)
        with pytest.raises(ValueError, match="optical layer order"):
            solar.reflected_radiance(scene, source=source, optics=wrong_layers, max_order=maximum)
        native.assert_not_called()


@pytest.mark.parametrize("maximum", [0, 4])
@pytest.mark.parametrize(
    ("role", "field", "value"),
    [
        ("absorption", "optical_depth", -1),
        ("rayleigh", "optical_depth", np.nan),
        ("aerosol", "optical_depth", -1),
        ("aerosol", "scattering_depth", 1),
        ("aerosol", "asymmetry", np.nan),
    ],
)
def test_physical_validation_and_recovery(maximum, role, field, value):
    scene, source, optics = mixed_inputs()
    controls = dict(max_order=maximum, angular_order=3)
    expected = solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    array = getattr(getattr(optics, role), field)
    saved = array.copy()
    array[0, 0] = value
    with pytest.raises(ValueError):
        solar.reflected_radiance(scene, source=source, optics=optics, **controls)
    array[:] = saved
    np.testing.assert_array_equal(
        solar.reflected_radiance(scene, source=source, optics=optics, **controls).values,
        expected.values,
    )


def test_retained_resource_surface_is_removed():
    for module, names in (
        (
            solar,
            (
                "SolarScatteringWorkspace",
                "prepare_series_of_scatterers",
                "evaluate_series_of_scatterers",
            ),
        ),
        (
            orders,
            (
                "PreparedScattering",
                "prepare_scattering_orders",
                "allocate_scattering_buffers",
                "evaluate_scattering_orders",
            ),
        ),
        (
            propagation_native,
            (
                "ScatteringGeometry",
                "ScatteringBuffers",
                "prepare_scattering",
                "evaluate_scattering",
                "bind_scattering_blas",
            ),
        ),
    ):
        assert all(not hasattr(module, name) for name in names)
    assert orders.__all__ == ["angular_grid", "phase_function", "scattering_orders"]
