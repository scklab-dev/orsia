"""Frozen direct-solar replay and a separately integrated thin-scattering check."""

import json
from math import cos, exp, pi, radians, sin
from pathlib import Path

import numpy as np
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    Atmosphere,
    Line,
    LineSet,
    ObservationGeometry,
    RadiativeScene,
    ScatteringProperties,
    SpectralIrradiance,
    SurfaceState,
    compose_absorption,
    compose_atmospheric_optics,
    line_absorption,
    rayleigh_component,
    solar_source,
    wavenumber_grid,
)


def test_frozen_original_synthetic_direct_solar_replay() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/synthetic_direct_solar_radiance.json").read_text()
    )
    spectral = np.array(fixture["spectral_axis"]["coordinates"], dtype=np.float64)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([290.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
        (Absorber("X", np.array([1e25]), "dry_air"),),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(
            wavenumber_grid(spectral),
            reflectance=np.asarray(fixture["surface_reflectance"], dtype=np.float64),
        ),
        ObservationGeometry(fixture["view_zenith_deg"]),
    )
    irradiance = SpectralIrradiance(
        wavenumber_grid(spectral),
        np.asarray(fixture["irradiance"], dtype=np.float64),
    )
    absorption = AbsorptionComponent(
        wavenumber_grid(spectral),
        "X",
        "synthetic",
        "lorentz_lines",
        np.asarray(fixture["absorption_optical_depth"], dtype=np.float64),
    )
    source = solar_source(
        irradiance, direction=(fixture["solar_zenith_deg"], fixture["relative_azimuth_deg"])
    )
    result = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            source,
            absorption=compose_absorption(scene.atmosphere, source, (absorption,)),
        ),
    )
    assert result.values == pytest.approx(fixture["radiance"], rel=1e-13)
    # The frozen physical reference retains its historical model identity.
    assert fixture["model_id"] == "direct_solar_lambertian_absorption_only_v1"


def test_two_layer_black_surface_matches_independent_midpoint_quadrature() -> None:
    spectral = np.array([1000.0, 1001.0], dtype=np.float64)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([290.0, 280.0]),
        np.array([2e29, 1.8e29]),
        np.array([1.9e29, 1.7e29]),
        np.array([1e28, 1e28]),
        (Absorber("X", np.array([1e25, 1e25]), "dry_air"),),
    )
    material = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(spectral), reflectance=np.zeros(2)),
        ObservationGeometry(35.0),
    )
    lines = LineSet(
        "X",
        "synthetic",
        (Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7),),
        np.ones(2),
        (270.0, 300.0),
        (80000.0, 100000.0),
    )
    absorption = line_absorption(atmosphere, spectral, lines)
    scattering = ScatteringProperties(
        wavenumber_grid(spectral),
        np.array([[0.018, 0.015], [0.007, 0.012]]),
        np.tile([1.0, 0.0, 0.1], (2, 2, 1)),
        "source_untruncated",
    )
    solar = SpectralIrradiance(wavenumber_grid(spectral), np.array([10.0, 20.0]))
    result = terms_support.order_one(
        material,
        source=solar_source(solar, direction=(25.0, 50.0)),
        optics=compose_atmospheric_optics(
            material.atmosphere,
            solar_source(solar, direction=(25.0, 50.0)),
            absorption=compose_absorption(
                material.atmosphere, solar_source(solar, direction=(25.0, 50.0)), (absorption,)
            ),
            rayleigh=rayleigh_component(scattering),
        ),
    )
    mu0, mu = cos(radians(25.0)), cos(radians(35.0))
    theta = -mu0 * mu + sin(radians(25.0)) * sin(radians(35.0)) * cos(radians(50.0))
    phase = 0.75 * (1 + theta**2)
    cells = 2000
    for point, incident in enumerate(solar.values):
        extinction = [
            absorption.optical_depth[layer][point] + scattering.optical_depth[layer][point]
            for layer in range(2)
        ]
        # Independent discretised line-of-sight integral: each cell emits once,
        # then both incident and emergent rays pass through overlying extinction.
        integrated = 0.0
        for layer in range(2):
            overlying = sum(extinction[layer + 1 :])
            for cell in range(cells):
                overhead = overlying + extinction[layer] * (cell + 0.5) / cells
                integrated += (
                    scattering.optical_depth[layer][point]
                    / cells
                    * exp(-overhead / mu0 - overhead / mu)
                )
        expected = incident * phase / (4 * pi * mu) * integrated
        assert result.components[0].values[point] == 0
        assert result.values[point] == pytest.approx(expected, rel=1e-7)

    np.testing.assert_array_equal(result.wavenumbers_cm1, spectral)
