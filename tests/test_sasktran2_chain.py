"""Analytic slab-to-photoelectron/inverse closure; no external qualification."""

import sys

import numpy as np
from numpy.typing import NDArray
from orsia_instruments import (
    DetectorMaps,
    exposure_sequence,
    filter_imager,
    regular_pixel_bounds,
    simulate_images,
    uniform_radiance_field,
)
from orsia_inverse import solve_map
from orsia_radiative import (
    Absorption,
    Atmosphere,
    ObservationGeometry,
    Radiance,
    RadiativeScene,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    transfer,
    wavenumber_grid,
)


def test_charge_linear_scaling():
    points = np.linspace(12000.0, 12001.0, 65)
    radiance = np.linspace(0.02, 0.03, 65)
    baseline = _charge(points, radiance, 9)
    assert baseline.shape == (1,) and baseline[0] > 0
    np.testing.assert_allclose(_charge(points, 2 * radiance, 9), 2 * baseline, rtol=1e-14)


def test_analytic_radiance_instrument_inverse_closure():
    x = np.linspace(0.0, 1.0, 65)
    points = 12000.0 + x
    grid = wavenumber_grid(points)
    depth = 0.2 + 0.15 * x
    irradiance = 1.0 + 0.1 * np.sin(2 * np.pi * x)
    mu0, muv, albedo = 0.8, 0.7, 0.3
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([290.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
    )
    source = solar_source(
        SpectralIrradiance(grid, irradiance), direction=(float(np.rad2deg(np.arccos(mu0))), 60.0)
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(grid, reflectance=np.full(65, albedo)),
        ObservationGeometry(float(np.rad2deg(np.arccos(muv)))),
    )
    optics = compose_atmospheric_optics(
        atmosphere, source, absorption=Absorption(grid, depth[None])
    )
    expected = irradiance * albedo * mu0 / np.pi * np.exp(-depth * (1 / mu0 + 1 / muv))
    actual = transfer.solar.reflected_radiance(
        scene,
        source=source,
        optics=optics,
        max_order=0,
        angular_order=2,
        layer_subdivisions=1,
        delta_m_moments=None,
    ).values
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    observation = _charge(points, expected, 17)
    model = _charge(points, actual, 17)
    np.testing.assert_allclose(_charge(points, actual, 9), model, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(model, observation, rtol=1e-8, atol=1e-10)
    for initial in (0.5, 1.5):
        estimate = solve_map(
            lambda state: model * state[0],
            lambda state: model[:, None],
            observation,
            np.array([initial]),
            observation_covariance=np.ones(observation.size),
        )
        assert estimate.success
        np.testing.assert_allclose(estimate.state, [1.0], rtol=0, atol=1e-7)
    assert not sys._is_gil_enabled()


def _charge(
    points: NDArray[np.float64], values: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """One noiseless physical read in expected electrons, with no DN conversion."""
    field = uniform_radiance_field(
        Radiance(wavenumber_grid(points), values, ObservationGeometry(0.0)),
        time_bounds_s=(0.0, 0.01),
        u_bounds=(-0.004, 0.004),
        v_bounds=(-0.004, 0.004),
        spectral_bounds_cm1=(float(points[0]), float(points[-1])),
    )
    return simulate_images(
        field,
        pixel_bounds_m=regular_pixel_bounds((1, 1), pitch_m=30e-6, active_size_m=28e-6),
        optics=filter_imager(
            focal_length_m=0.02,
            collecting_area_m2=1e-8,
            psf_sigma_m=30e-6,
            filter=lambda wavelength: np.ones_like(wavelength),
            throughput=0.8,
        ),
        exposures=exposure_sequence(
            start_s=[0.0],
            duration_s=0.01,
            direction_bounds_uv=(-0.004, 0.004, -0.004, 0.004),
        ),
        detector=DetectorMaps(quantum_efficiency=0.7),
        angular_order=order,
    ).photoelectrons.ravel()
