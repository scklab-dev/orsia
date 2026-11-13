"""Analytical native tangents, value agreement and fixed-support contracts."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import cos, pi, radians
from typing import Any

import numpy as np
import pytest
from orsia_radiative import (
    Absorption,
    AerosolComponent,
    ColumnScene,
    LayerAtmosphere,
    ObservationGeometry,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    spectral_band,
    wavenumber_grid,
)
from orsia_radiative.optics.cross_sections import cross_section_optical_depth_tangent
from orsia_radiative.transfer.native import transfer_native
from orsia_radiative.transfer.solar import reflected_radiance, reflected_radiance_tangent


def inputs(depth, rho=0.4, incident=8.0, zenith=30.0, view=20.0):
    layers, points = depth.shape
    grid = wavenumber_grid(np.linspace(12000.0, 12002.0, points))
    atmosphere = LayerAtmosphere(
        np.arange(layers + 1, dtype=np.float64) * 1000,
        np.linspace(1e5, 2e4, layers + 1),
        (np.linspace(1e5, 2e4, layers + 1)[:-1] + np.linspace(1e5, 2e4, layers + 1)[1:]) / 2,
        np.full(layers, 280.0),
        np.full(layers, 2e29),
        np.full(layers, 1.9e29),
        np.full(layers, 1e28),
    )
    scene = ColumnScene(
        atmosphere,
        SurfaceState(grid, reflectance=np.full(points, rho)),
        ObservationGeometry(view),
    )
    source = solar_source(
        SpectralIrradiance(grid, np.full(points, incident)), direction=(zenith, 20.0)
    )
    optics = compose_atmospheric_optics(atmosphere, source, absorption=Absorption(grid, depth))
    return scene, source, optics


def solar(depth, dtau, drho, **kwargs):
    scene, source, optics = inputs(depth, **kwargs)
    return reflected_radiance_tangent(
        scene,
        source=source,
        optics=optics,
        optical_depth_tangent=dtau,
        reflectance_tangent=drho,
    )


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_column_product_and_directional_reference(k):
    sigma = np.array([[0.0, 1e-25, 4e-25], [2e-25, 3e-25, 5e-25]])
    columns = np.array([0.0, 2e24])
    directions = np.arange(2 * k, dtype=np.float64).reshape(2, k) * 1e23 - 1e23
    tau, dtau = cross_section_optical_depth_tangent(sigma, columns, directions)
    np.testing.assert_array_equal(tau, sigma * columns[:, None])
    np.testing.assert_array_equal(dtau, sigma[:, :, None] * directions[:, None, :])
    assert dtau.shape == (2, 3, k)
    for array in (tau, dtau):
        assert array.dtype == np.float64 and array.flags.owndata and array.flags.c_contiguous
        assert array.flags.writeable
        assert all(not np.shares_memory(array, a) for a in (sigma, columns, directions))


@pytest.mark.parametrize("k", [0, 1, 2, 7])
@pytest.mark.parametrize(
    "rho,incident,tau", [(0.4, 8.0, 0.1), (0.0, 8.0, 0.0), (1.0, 0.0, 0.1), (1.0, 8.0, 1e4)]
)
def test_solar_analytic_and_existing_value_reference(k, rho, incident, tau):
    depth = np.full((3, 4), tau)
    dtau = np.arange(12 * k, dtype=np.float64).reshape(3, 4, k) / 30 - 0.2
    drho = np.arange(4 * k, dtype=np.float64).reshape(4, k) / 30 - 0.3
    scene, source, optics = inputs(depth, rho=rho, incident=incident)
    value, tangent = reflected_radiance_tangent(
        scene,
        source=source,
        optics=optics,
        optical_depth_tangent=dtau,
        reflectance_tangent=drho,
    )
    mu0, mu = cos(radians(30)), cos(radians(20))
    airmass = 1 / mu0 + 1 / mu
    gain = incident * mu0 / pi * np.exp(-airmass * depth.sum(axis=0))
    np.testing.assert_allclose(value.values, rho * gain, rtol=3e-15, atol=0)
    np.testing.assert_allclose(
        tangent,
        gain[:, None] * (drho - rho * airmass * dtau.sum(axis=0)),
        rtol=2e-14,
        atol=2e-15,
    )
    original = reflected_radiance(scene, source=source, optics=optics, max_order=0)
    np.testing.assert_allclose(value.values, original.values, rtol=3e-15, atol=0)
    assert tangent.shape == (4, k)
    assert tangent.flags.owndata and tangent.flags.writeable and tangent.flags.c_contiguous
    assert value.values.flags.owndata and not np.shares_memory(value.values, tangent)


def test_composed_columns_and_reflectance_finite_difference_step_sweep():
    sigma = np.array([[1e-25, 3e-25, 2e-25], [3e-25, 2e-25, 1e-25]])
    columns = np.array([1e24, 2e24])
    dcolumns = np.array([[1e23, 0, -2e23], [2e23, 0, 1e23]])
    drho = np.broadcast_to(np.array([0.0, 1.0, 0.2]), (3, 3)).copy()
    depth, dtau = cross_section_optical_depth_tangent(sigma, columns, dcolumns)
    _, analytical = solar(depth, dtau, drho)
    errors = []
    for h in (1e-2, 1e-3, 1e-4):
        numerical = np.empty_like(analytical)
        for d in range(3):

            def prediction(sign, h=h, d=d):
                trial = sigma * (columns + sign * h * dcolumns[:, d])[:, None]
                scene, source, optics = inputs(trial)
                surface = replace(
                    scene.surface, reflectance=scene.surface.reflectance + sign * h * drho[:, d]
                )
                return reflected_radiance(
                    replace(scene, surface=surface), source=source, optics=optics, max_order=0
                ).values

            numerical[:, d] = (prediction(1) - prediction(-1)) / (2 * h)
        errors.append(float(np.max(np.abs(numerical - analytical))))
    assert errors[1] < errors[0] / 50
    assert errors[2] < errors[1] / 20
    np.testing.assert_allclose(numerical, analytical, rtol=2e-9, atol=3e-12)


@pytest.mark.parametrize(
    "zenith,view", [(0.0, 0.0), (89.99999999999989, 20.0), (30.0, float(np.nextafter(90.0, 0.0)))]
)
def test_grazing_geometry_and_zero_directions(zenith, view):
    depth = np.full((2, 3), 1e-16)
    value, tangent = solar(depth, np.zeros((2, 3, 2)), np.zeros((3, 2)), zenith=zenith, view=view)
    scene, source, optics = inputs(depth, zenith=zenith, view=view)
    np.testing.assert_allclose(
        value.values,
        reflected_radiance(scene, source=source, optics=optics, max_order=0).values,
        rtol=3e-15,
    )
    np.testing.assert_array_equal(tangent, 0)


def test_layer_splitting_preserves_values_and_directions():
    depth = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    dtau = np.arange(12, dtype=np.float64).reshape(2, 3, 2) / 100
    drho = np.ones((3, 2))
    reference = solar(depth, dtau, drho)
    for subdivisions in (2, 5, 16):
        actual = solar(
            np.repeat(depth / subdivisions, subdivisions, axis=0),
            np.repeat(dtau / subdivisions, subdivisions, axis=0),
            drho,
        )
        np.testing.assert_allclose(actual[0].values, reference[0].values, rtol=3e-15)
        np.testing.assert_allclose(actual[1], reference[1], rtol=4e-15, atol=1e-16)


def test_spectral_refinement_and_fixed_projection():
    # Smooth synthetic spectrum: trapezoidal band projection converges to the
    # independently integrated exponential and its total-depth-scale derivative.
    beta = 0.8
    mu0, mu = cos(radians(30)), cos(radians(20))
    airmass = 1 / mu0 + 1 / mu
    c = airmass * beta
    factor = 8.0 * mu0 / pi
    expected = factor * 0.4 * (1 - np.exp(-c)) / c
    expected_d = factor * 0.4 * ((c + 1) * np.exp(-c) - 1) / c
    errors = []
    for n in (33, 65, 129):
        x = np.linspace(0, 1, n)
        depth = (beta * x)[None, :]
        scene, source, optics = inputs(depth)
        radiance, tangent = reflected_radiance_tangent(
            scene,
            source=source,
            optics=optics,
            optical_depth_tangent=depth[:, :, None],
            reflectance_tangent=np.zeros((n, 1)),
        )
        band = spectral_band("smooth", source.irradiance.grid, np.ones(n))
        projected = np.array([band.weights @ radiance.values, (band.weights @ tangent)[0]])
        errors.append(np.abs(projected - [expected, expected_d]))
    assert np.all(errors[1] < errors[0] / 3.9)
    assert np.all(errors[2] < errors[1] / 3.9)
    np.testing.assert_allclose(projected, [expected, expected_d], rtol=1e-4)


def test_absent_absorption_has_transparent_value_and_nonzero_depth_direction():
    scene, source, optics = inputs(np.zeros((2, 3)))
    args = dict(
        source=source,
        optical_depth_tangent=np.ones((2, 3, 1)),
        reflectance_tangent=np.zeros((3, 1)),
    )
    explicit = reflected_radiance_tangent(scene, optics=optics, **args)
    absent = reflected_radiance_tangent(scene, optics=replace(optics, absorption=None), **args)
    np.testing.assert_array_equal(absent[0].values, explicit[0].values)
    np.testing.assert_array_equal(absent[1], explicit[1])
    assert (absent[1] < 0).all()


def layout_copy(array, layout):
    if layout == "F":
        result = np.asfortranarray(array)
    elif layout == "unaligned":
        result = np.ndarray(
            array.shape, dtype=np.float64, buffer=bytearray(array.nbytes + 1), offset=1
        )
        result[:] = array
    else:
        storage = np.zeros(tuple(2 * n for n in array.shape))
        result = storage[tuple(slice(None, None, -2) for _ in array.shape)]
        result[:] = array
    result.flags.writeable = False
    return result


@pytest.mark.parametrize("layout", ["F", "reversed", "unaligned"])
def test_layouts_ownership_and_input_preservation(layout):
    sigma, columns, dc = np.full((2, 3), 1e-25), np.full(2, 1e24), np.ones((2, 4)) * 1e23
    expected = cross_section_optical_depth_tangent(sigma, columns, dc)
    arrays = [layout_copy(a, layout) for a in (sigma, columns, dc)]
    snapshots = [a.tobytes() for a in arrays]
    actual = cross_section_optical_depth_tangent(*arrays)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)
    assert [a.tobytes() for a in arrays] == snapshots
    scene, source, optics = inputs(expected[0])
    for owner, name in (
        (optics.absorption, "optical_depth"),
        (source.irradiance, "values"),
        (scene.surface, "reflectance"),
    ):
        a = layout_copy(getattr(owner, name), layout)
        object.__setattr__(owner, name, a)
        arrays.append(a)
    dtau, drho = layout_copy(expected[1], layout), layout_copy(np.ones((3, 4)), layout)
    arrays += [dtau, drho]
    snapshots = [a.tobytes() for a in arrays]
    actual_solar = reflected_radiance_tangent(
        scene,
        source=source,
        optics=optics,
        optical_depth_tangent=dtau,
        reflectance_tangent=drho,
    )
    reference = solar(expected[0], expected[1], np.ones((3, 4)))
    for a, e in zip(
        (actual_solar[0].values, actual_solar[1]), (reference[0].values, reference[1]), strict=True
    ):
        np.testing.assert_array_equal(a, e)
        assert a.flags.owndata and a.flags.writeable
        assert all(not np.shares_memory(a, other) for other in arrays)
    assert [a.tobytes() for a in arrays] == snapshots


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, np.zeros((0, 3))),
        (0, np.zeros((2, 0))),
        (0, np.zeros(2)),
        (0, np.zeros((2, 3), dtype=np.float32)),
        (0, np.full((2, 3), -1.0)),
        (0, np.full((2, 3), np.nan)),
        (1, np.zeros(3)),
        (1, np.ones((2, 1))),
        (1, np.array([np.inf, 1.0])),
        (1, np.array([-1.0, 1.0])),
        (2, np.zeros((3, 2))),
        (2, np.ones(2)),
        (2, np.full((2, 2), np.inf)),
        (2, np.ones((2, 2), dtype=">f8")),
    ],
)
def test_absorption_invalid_inputs(index, bad):
    arrays = [np.ones((2, 3)), np.ones(2), np.ones((2, 2))]
    arrays[index] = bad
    with pytest.raises(ValueError):
        cross_section_optical_depth_tangent(*arrays)


@pytest.mark.parametrize("which", ["value", "tangent"])
def test_absorption_overflow(which):
    columns = np.full(2, 2.0 if which == "value" else 0.0)
    with pytest.raises(ValueError, match="overflow"):
        cross_section_optical_depth_tangent(np.full((2, 3), 1e308), columns, np.full((2, 1), 2.0))


@pytest.mark.parametrize("role", ["rayleigh", "aerosol"])
def test_zero_scattering_roles_preserve_default_direct_call(role):
    scene, source, optics = inputs(np.zeros((2, 3)))
    zero = np.zeros((2, 3))
    explicit = (
        replace(optics, rayleigh=RayleighComponent(optics.grid, zero))
        if role == "rayleigh"
        else replace(optics, aerosol=AerosolComponent(optics.grid, zero, zero, zero))
    )
    args = dict(
        source=source,
        optical_depth_tangent=np.zeros((2, 3, 1)),
        reflectance_tangent=np.zeros((3, 1)),
    )
    value, tangent = reflected_radiance_tangent(scene, optics=explicit, **args)
    expected, expected_tangent = reflected_radiance_tangent(scene, optics=optics, **args)
    np.testing.assert_array_equal(value.values, expected.values)
    np.testing.assert_array_equal(tangent, expected_tangent)


def test_missing_reflectance_and_mismatched_scene_grid():
    scene, source, optics = inputs(np.zeros((2, 3)))
    args = dict(
        source=source,
        optics=optics,
        optical_depth_tangent=np.zeros((2, 3, 1)),
        reflectance_tangent=np.zeros((3, 1)),
    )
    thermal_surface = SurfaceState(optics.grid, temperature_k=280.0, emissivity=np.ones(3))
    with pytest.raises(ValueError, match="reflectance"):
        reflected_radiance_tangent(replace(scene, surface=thermal_surface), **args)
    other = SurfaceState(wavenumber_grid(np.array([1.0, 2.0, 3.0])), reflectance=np.ones(3))
    with pytest.raises(ValueError, match="spectral"):
        reflected_radiance_tangent(replace(scene, surface=other), **args)


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, np.zeros(2)),
        (0, np.zeros((0, 3))),
        (0, np.zeros((2, 0))),
        (0, np.full((2, 3), -1.0)),
        (0, np.full((2, 3), np.nan)),
        (1, np.zeros((2, 3))),
        (1, np.zeros((1, 3, 2))),
        (1, np.zeros((2, 4, 2))),
        (1, np.zeros((2, 3, 2), dtype=np.float32)),
        (1, np.full((2, 3, 2), np.nan)),
        (2, np.zeros(4)),
        (2, np.array([0.0, -1.0, 1.0])),
        (2, np.full(3, np.inf)),
        (3, np.zeros((3, 1))),
        (3, np.zeros(2)),
        (3, np.full(3, 1.1)),
        (4, np.zeros((3, 1))),
        (4, np.full((3, 2), np.inf)),
        (5, 90.0),
        (5, -1.0),
        (5, float("nan")),
        (6, 90.0),
    ],
)
def test_native_solar_invalid_inputs(index, bad):
    # Deliberately heterogeneous malformed native arguments.
    args: list[Any] = [
        np.ones((2, 3)),
        np.ones((2, 3, 2)),
        np.ones(3),
        np.ones(3),
        np.ones((3, 2)),
        30.0,
        20.0,
    ]
    args[index] = bad
    with pytest.raises(ValueError):
        transfer_native.reflected_radiance_tangent(*args)


@pytest.mark.parametrize("which", ["depth", "depth_tangent", "radiance_tangent"])
def test_solar_overflow(which):
    depth = np.full((2, 3), 1e308 if which == "depth" else 0.0)
    dtau = np.full((2, 3, 1), 1e308 if which == "depth_tangent" else 0.0)
    drho = np.full((3, 1), 1e308 if which == "radiance_tangent" else 0.0)
    with pytest.raises(ValueError, match="overflow"):
        solar(depth, dtau, drho, incident=100.0)


def test_concurrent_calls_and_native_import_gil():
    sigma, columns, dc = np.full((2, 3), 1e-25), np.full(2, 1e24), np.ones((2, 3)) * 1e23
    scene, source, optics = inputs(sigma * columns[:, None])
    drho = np.ones((3, 3))

    def calculate(_):
        tau, dtau = cross_section_optical_depth_tangent(sigma, columns, dc)
        assert optics.absorption is not None
        np.testing.assert_array_equal(tau, optics.absorption.optical_depth)
        return reflected_radiance_tangent(
            scene,
            source=source,
            optics=optics,
            optical_depth_tangent=dtau,
            reflectance_tangent=drho,
        )

    reference = calculate(0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(calculate, range(12)))
    for value, tangent in results:
        np.testing.assert_array_equal(value.values, reference[0].values)
        np.testing.assert_array_equal(tangent, reference[1])
        assert not np.shares_memory(tangent, reference[1])
    code = """
import os, sys
assert os.environ.get('PYTHON_GIL') != '0'
assert sys._xoptions.get('gil') != '0'
from orsia_radiative.optics import absorption_native
assert not sys._is_gil_enabled()
from orsia_radiative.transfer.native import transfer_native
assert not sys._is_gil_enabled()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy()
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
