"""Independent conservation, ragged reduction and direct-path derivative checks."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_radiative import (
    Absorber,
    LayerAtmosphere,
    clip_atmosphere_height,
    integrate_path_optical_depth,
    integrate_path_optical_depth_tangent,
    retained_layer_fractions,
)
from orsia_radiative.transfer.paths import direct_surface_radiance, direct_surface_radiance_tangent


def atmosphere():
    return LayerAtmosphere(
        np.array([0.0, 100.0, 300.0]),
        np.array([1000.0, 800.0, 400.0]),
        np.array([900.0, 600.0]),
        np.array([280.0, 270.0]),
        np.array([110.0, 220.0]),
        np.array([100.0, 200.0]),
        np.array([10.0, 20.0]),
        (
            Absorber("X", np.array([2.0, 4.0]), "dry_air"),
            Absorber("H2O", np.array([10.0, 20.0]), "total_air"),
        ),
    )


@pytest.mark.parametrize(
    "height,expected",
    [(0.0, [1.0, 1.0]), (25.0, [0.75, 1.0]), (100.0, [0.0, 1.0]), (250.0, [0.0, 0.25])],
)
def test_clip_conserves_actual_partial_support(height, expected):
    source = atmosphere()
    result = clip_atmosphere_height(source, height)
    fraction = retained_layer_fractions(source.height_boundaries_m, height)
    assert_array_equal(fraction, expected)
    keep = fraction > 0
    for field in (
        "total_air_columns_molecules_m2",
        "dry_air_columns_molecules_m2",
        "water_vapour_columns_molecules_m2",
    ):
        assert_allclose(getattr(result, field), (getattr(source, field) * fraction)[keep])
    assert_allclose(
        result.total_air_columns_molecules_m2,
        result.dry_air_columns_molecules_m2 + result.water_vapour_columns_molecules_m2,
    )
    for a, b in zip(source.absorbers, result.absorbers, strict=True):
        assert_allclose(b.columns_molecules_m2, (a.columns_molecules_m2 * fraction)[keep])
    assert result.height_boundaries_m[0] == height
    assert result.pressure_boundaries_pa[0] == pytest.approx(1000 - 2 * height)
    assert_allclose(
        result.pressure_pa,
        (result.pressure_boundaries_pa[:-1] + result.pressure_boundaries_pa[1:]) / 2,
    )
    result.temperature_k[:] = 1
    assert_array_equal(source.temperature_k, [280.0, 270.0])


@pytest.mark.parametrize("height", [-1.0, 300.0, 301.0, np.nan])
def test_clip_rejects_uncovered_height(height):
    with pytest.raises(ValueError):
        clip_atmosphere_height(atmosphere(), height)
    with pytest.raises(ValueError):
        retained_layer_fractions(np.array([0.0, 100.0, 300.0]), height)


def test_clip_rejects_bases_and_invalid_edges():
    from orsia_radiative import LayerProfileBasis

    source = atmosphere()
    basis = LayerProfileBasis(
        domain="layer",
        baseline=source.temperature_k,
        directions=np.ones((1, 2)),
        names=("offset",),
        unit="K",
    )
    with pytest.raises(ValueError, match="bases"):
        clip_atmosphere_height(replace(source, temperature_basis=basis), 25)
    for edges in (np.array([0.0]), np.array([1.0, 0.0]), np.array([0.0, np.nan])):
        with pytest.raises(ValueError):
            retained_layer_fractions(edges, 0.0)


def test_path_reduction_and_tangent_species_scattering_empty_paths():
    columns = np.array([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    sigma = np.arange(1.0, 13.0).reshape(3, 2, 2) * 0.01
    offsets = np.array([0, 2, 2, 3], dtype=np.int64)
    scattering = np.full((3, 2), 0.2)
    seed = np.stack((columns, -columns / 2), axis=-1)
    depth, tangent = integrate_path_optical_depth_tangent(
        columns, sigma, seed, path_offsets=offsets, segment_scattering_optical_depth=scattering
    )
    segment = np.sum(columns[:, :, None] * sigma, axis=1)
    assert_allclose(depth, np.stack((segment[:2].sum(axis=0) + 0.4, np.zeros(2), segment[2] + 0.2)))
    assert_allclose(tangent[:, :, 0], depth - np.array([[0.4, 0.4], [0.0, 0.0], [0.2, 0.2]]))
    assert_allclose(tangent[:, :, 1], -tangent[:, :, 0] / 2)
    for step in [1e-3, 1e-5]:
        plus = integrate_path_optical_depth(
            columns + step * seed[:, :, 1], sigma, path_offsets=offsets
        )
        minus = integrate_path_optical_depth(
            columns - step * seed[:, :, 1], sigma, path_offsets=offsets
        )
        assert_allclose((plus - minus) / (2 * step), tangent[:, :, 1], rtol=2e-10)
    empty = integrate_path_optical_depth_tangent(
        np.empty((0, 1)), np.empty((0, 1, 2)), np.empty((0, 1, 0)), path_offsets=np.array([0, 0])
    )
    assert_array_equal(empty[0], np.zeros((1, 2)))
    assert empty[1].shape == (1, 2, 0)


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([]),
        np.array([1, 3]),
        np.array([0, 2]),
        np.array([0, 3, 2, 3]),
        np.array([0.0, 3.0]),
    ],
)
def test_path_offsets_rejected(offsets):
    with pytest.raises(ValueError):
        integrate_path_optical_depth(np.ones((3, 1)), np.ones((3, 1, 2)), path_offsets=offsets)


def test_invalid_optical_arrays():
    columns, sigma, offsets = np.ones((2, 1)), np.ones((2, 1, 3)), np.array([0, 2])
    for bad in (np.array([1.0, 2.0]), -columns, columns * np.nan):
        with pytest.raises(ValueError):
            integrate_path_optical_depth(bad, sigma, path_offsets=offsets)
    with pytest.raises(ValueError):
        integrate_path_optical_depth(
            columns, sigma, path_offsets=offsets, segment_scattering_optical_depth=-np.ones((2, 3))
        )
    with pytest.raises(ValueError):
        integrate_path_optical_depth_tangent(columns, sigma, np.ones((2, 1)), path_offsets=offsets)


def test_direct_formula_shadow_zero_albedo_and_tangents():
    sun = np.array([[0.2, 0.4], [0.3, 0.8]])
    view = sun * 2
    rho = np.array([[0.0, 0.3], [0.4, 0.5]])
    beam = np.full((2, 2), 2.0)
    mu = np.array([0.8, 0.0])
    ds = np.ones((2, 2, 2)) * 0.1
    dv = ds * -0.3
    dr = ds * 2
    value, tangent = direct_surface_radiance_tangent(
        sun,
        view,
        rho,
        beam,
        mu,
        solar_optical_depth_tangent=ds,
        viewing_optical_depth_tangent=dv,
        reflectance_tangent=dr,
    )
    expected = rho * beam * mu[:, None] / np.pi * np.exp(-sun - view)
    assert_allclose(value, expected, rtol=1e-15)
    assert_allclose(
        tangent,
        (beam * mu[:, None] / np.pi * np.exp(-sun - view))[..., None] * dr
        - expected[..., None] * (ds + dv),
        rtol=1e-15,
    )
    assert tangent[0, 0, 0] > 0
    assert_array_equal(tangent[1], 0)
    # Centred FD away from the reflectance boundary; the zero-albedo boundary has
    # an analytic one-sided derivative, checked above.
    rho[:] = 0.3
    _, tangent = direct_surface_radiance_tangent(
        sun,
        view,
        rho,
        beam,
        mu,
        solar_optical_depth_tangent=ds,
        viewing_optical_depth_tangent=dv,
        reflectance_tangent=dr,
    )
    for step in [1e-3, 1e-5, 1e-6]:
        plus = direct_surface_radiance(
            sun + step * ds[..., 0], view + step * dv[..., 0], rho + step * dr[..., 0], beam, mu
        )
        minus = direct_surface_radiance(
            sun - step * ds[..., 0], view - step * dv[..., 0], rho - step * dr[..., 0], beam, mu
        )
        assert_allclose((plus - minus) / (2 * step), tangent[..., 0], rtol=2e-8)


def test_direct_errors_and_extreme_depth():
    valid = np.ones((1, 2)) * 0.1
    for index, bad in ((0, -valid), (1, valid * np.nan), (2, valid * 20), (3, np.ones((2, 2)))):
        args = [valid, valid, valid, valid, np.ones(1)]
        args[index] = bad
        with pytest.raises(ValueError):
            direct_surface_radiance(*args)
    with pytest.raises(ValueError):
        direct_surface_radiance(valid, valid, valid, valid, np.array([2.0]))
    with pytest.raises(ValueError):
        direct_surface_radiance_tangent(
            valid,
            valid,
            valid,
            valid,
            np.ones(1),
            solar_optical_depth_tangent=np.ones((1, 2, 1)),
            viewing_optical_depth_tangent=np.ones((1, 2, 2)),
            reflectance_tangent=np.ones((1, 2, 1)),
        )
    assert_array_equal(
        direct_surface_radiance(valid * 1e308, valid * 1e308, valid, valid, np.ones(1)), 0
    )


@pytest.mark.parametrize(
    "sun,view,rho,beam,mu,ds,dv,dr",
    [
        (1000, 0, 1, np.pi, 1, 1e300, 0, 0),
        (1000, 0, 1, np.pi, 1, -1e300, 0, 0),
        (600, 400, 1, np.pi, 1, 0, -1e300, 0),
        (1, 2, 1, np.pi, 1, 1e308, 1e308, 0),
        (1, 1, 0.5, np.pi, 1, 1e308, 1e308, 0),
        (1000, 0, 1, 1e308, 1, 1e308, 1e308, 0),
        (0, 0, 0, np.pi, 1, 1e308, 1e308, 1),
        (1000, 0, 1, np.pi, 0, 1e308, 1e308, 1e308),
        (1000, 0, 0, np.pi, 1, 0, 0, 1e300),
        (1000, 0, 0, np.pi, 1, 0, 0, -1e300),
        (1000, 0, 1, np.pi, 1, 1e300, -1e300, 1e200),
        (0, 0, 0.5, np.pi, 1, 1e308, 1e308, 1e308),
        (0, 0, 1, 1e308, 1, 1e308, 0, 1e308),
        (500, 500, 1, 1e300, 1, 0, 0, 0),
        (1000, 0, 1, 1e300, 1, 0, 0, 0),
        (0, 0, 0.5, np.nextafter(0.0, 1.0), 1, 0, 0, 1e308),
        (0, 0, 1, 1e300, 1, 1e308, -1e308, 1e-300),
    ],
)
def test_direct_scaled_attenuation_against_decimal(sun, view, rho, beam, mu, ds, dv, dr):
    # Decimal evaluates the physical formula independently, including its products
    # and cancellation, before a single conversion to the output precision.
    with localcontext() as context:
        context.prec = 1000
        s, v, r, b, m, x, y, z, pi = map(
            Decimal.from_float, map(float, (sun, view, rho, beam, mu, ds, dv, dr, np.pi))
        )
        light = b * m / pi * (-s - v).exp()
        expected_value = float(r * light)
        expected_tangent = float(light * (z - r * (x + y)))
    solar, viewing, albedo, irradiance = (
        np.full((2, 3), item, dtype=np.float64) for item in (sun, view, rho, beam)
    )
    args = (solar, viewing, albedo, irradiance, np.full(2, mu, dtype=np.float64))
    seeds = [np.full((2, 3, 4), item, dtype=np.float64) for item in (ds, dv, dr)]
    value, tangent = direct_surface_radiance_tangent(
        *args,
        solar_optical_depth_tangent=seeds[0],
        viewing_optical_depth_tangent=seeds[1],
        reflectance_tangent=seeds[2],
    )
    assert value.shape == (2, 3) and tangent.shape == (2, 3, 4)
    assert_allclose(value, expected_value, rtol=3e-14, atol=0)
    assert_allclose(tangent, expected_tangent, rtol=3e-14, atol=0)
    assert_array_equal(direct_surface_radiance(*args), value)


def test_direct_zero_direction_count():
    physical = np.ones((2, 3))
    seeds = np.empty((2, 3, 0))
    value, tangent = direct_surface_radiance_tangent(
        physical,
        physical,
        physical,
        physical,
        np.ones(2),
        solar_optical_depth_tangent=seeds,
        viewing_optical_depth_tangent=seeds,
        reflectance_tangent=seeds,
    )
    assert value.shape == (2, 3) and tangent.shape == (2, 3, 0)
    assert_allclose(value, np.exp(-2) / np.pi, rtol=1e-15)


def test_direct_fully_opaque_and_unrepresentable_tangents():
    ones = np.ones((1, 2))
    seeds = np.full((1, 2, 1), 1e308)
    value, tangent = direct_surface_radiance_tangent(
        ones * 1e308,
        ones * 1e308,
        ones,
        ones * 1e308,
        np.ones(1),
        solar_optical_depth_tangent=seeds,
        viewing_optical_depth_tangent=seeds,
        reflectance_tangent=seeds,
    )
    assert_array_equal(value, 0)
    assert_array_equal(tangent, 0)
    with pytest.raises(ValueError, match="overflow"):
        direct_surface_radiance_tangent(
            ones,
            ones,
            ones,
            ones * 1e308,
            np.ones(1),
            solar_optical_depth_tangent=seeds,
            viewing_optical_depth_tangent=seeds,
            reflectance_tangent=np.zeros_like(seeds),
        )


def test_retained_column_reuses_existing_scattering_and_thermal_contracts():
    from orsia_radiative import (
        Absorption,
        ColumnScene,
        ObservationGeometry,
        RayleighComponent,
        SolarSource,
        SpectralIrradiance,
        SurfaceState,
        atmospheric_optics,
        wavenumber_grid,
    )
    from orsia_radiative.transfer import solar, thermal
    from scipy.constants import c, h, k

    original = replace(atmosphere(), temperature_k=np.full(2, 280.0))
    column = clip_atmosphere_height(original, 25.0)
    fractions = retained_layer_fractions(original.height_boundaries_m, 25.0)
    grid = wavenumber_grid(np.array([900.0, 1000.0]))
    absorption = Absorption(grid, np.broadcast_to((0.1 * fractions)[:, None], (2, 2)))
    scattering = RayleighComponent(grid, np.broadcast_to((0.01 * fractions)[:, None], (2, 2)))
    scene = ColumnScene(
        column, SurfaceState(grid, reflectance=np.full(2, 0.3)), ObservationGeometry(20.0)
    )
    source = SolarSource(SpectralIrradiance(grid, np.ones(2)), 30.0, 40.0)
    optics = atmospheric_optics(column, grid, absorption=absorption, rayleigh=scattering)
    direct = solar.reflected_radiance(
        scene, source=source, optics=optics, max_order=0, delta_m_moments=None
    )
    scattered = solar.reflected_radiance(
        scene, source=source, optics=optics, max_order=1, angular_order=4, layer_subdivisions=4
    )
    assert np.all(scattered.values > direct.values)
    # Isothermal absorbing layers over a black surface recover Planck radiance.
    thermal_scene = replace(
        scene, surface=SurfaceState(grid, emissivity=np.ones(2), temperature_k=280.0)
    )
    value = thermal.thermal_radiance(
        thermal_scene, optics=atmospheric_optics(column, grid, absorption=absorption)
    )
    wave = 100 * grid.points
    expected = 100 * 2 * h * c**2 * wave**3 / np.expm1(h * c * wave / (k * 280))
    assert_allclose(value.values, expected, rtol=2e-14)
