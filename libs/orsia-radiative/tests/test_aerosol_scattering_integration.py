"""Offline aerosol-scattering replay with independent numerical integrals on a synthetic case."""

import json
from dataclasses import replace
from math import cos, exp, pi, radians, sin
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import radiance_terms_support as terms_support
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    AerosolOptics,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    ScatteringProperties,
    SpectralIrradiance,
    SurfaceState,
    aerosol_component,
    compose_absorption,
    compose_atmospheric_optics,
    rayleigh_component,
    solar_source,
    wavenumber_grid,
)
from scipy.integrate import quad


def _case() -> tuple[
    dict[str, Any],
    RadiativeScene,
    SpectralIrradiance,
    AbsorptionComponent,
    ScatteringProperties,
    AerosolOptics,
]:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/aerosol_scattering_synthetic_case.json").read_text()
    )
    axis = fixture["axis"]
    assert axis["kind"] == "wavenumber" and axis["unit"] == "cm-1"
    spectral = np.array(axis["coordinates"], dtype=np.float64)
    material = RadiativeScene(
        Atmosphere(
            np.array([0.0, 1000.0, 2000.0]),
            np.array([100000.0, 90000.0, 80000.0]),
            np.array([95000.0, 85000.0]),
            np.array([290.0, 280.0]),
            np.array([2e29, 1.8e29]),
            np.array([1.9e29, 1.7e29]),
            np.array([1e28, 1e28]),
            (Absorber("X", np.array([1e25, 1e25]), "dry_air"),),
        ),
        SurfaceState(
            wavenumber_grid(spectral),
            reflectance=np.asarray(fixture["reflectance"], dtype=np.float64),
        ),
        ObservationGeometry(fixture["view_zenith_deg"]),
    )
    solar = SpectralIrradiance(
        wavenumber_grid(spectral),
        np.asarray(fixture["solar"], dtype=np.float64),
    )
    gas = AbsorptionComponent(
        wavenumber_grid(spectral),
        "X",
        "synthetic",
        "lorentz_lines",
        np.asarray(fixture["gas_absorption_depth"], dtype=np.float64),
    )
    rayleigh = ScatteringProperties(
        wavenumber_grid(spectral),
        np.asarray(fixture["rayleigh_depth"], dtype=np.float64),
        np.tile([1.0, 0.0, 0.1], (2, 1, 1)),
        "source_untruncated",
    )
    aerosol = AerosolOptics(
        wavenumber_grid(spectral),
        np.asarray(fixture["aerosol_extinction_depth"], dtype=np.float64),
        np.asarray(fixture["aerosol_single_scattering_albedo"], dtype=np.float64),
        np.asarray(fixture["aerosol_asymmetry"], dtype=np.float64),
    )
    return fixture, material, solar, gas, rayleigh, aerosol


def test_first_order_contains_frozen_single_ground_paths() -> None:
    fixture, scene, solar, gas, rayleigh, aerosol = _case()
    source = solar_source(
        solar, direction=(fixture["solar_zenith_deg"], fixture["relative_azimuth_deg"])
    )
    result = terms_support.order_one(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            source,
            absorption=compose_absorption(scene.atmosphere, source, (gas,)),
            rayleigh=rayleigh_component(rayleigh),
            aerosol=aerosol_component(aerosol),
        ),
    )
    assert fixture["model"] == "direct_solar_single_aerosol_hg_v1"
    solar_zenith = fixture["solar_zenith_deg"]
    azimuth = fixture["relative_azimuth_deg"]
    reflectance_values = scene.surface.reflectance
    assert solar_zenith is not None and azimuth is not None and reflectance_values is not None
    mu0 = cos(radians(solar_zenith))
    mu = cos(radians(scene.geometry.view_zenith_deg))
    theta = -mu0 * mu + sin(radians(solar_zenith)) * sin(
        radians(scene.geometry.view_zenith_deg)
    ) * cos(radians(azimuth))
    phase_r = 0.75 * (1 + theta**2)
    depths = [
        gas.optical_depth[i][0] + rayleigh.optical_depth[i][0] + aerosol.optical_depth[i][0]
        for i in range(2)
    ]
    total = sum(depths)
    irradiance = solar.values[0]
    reflectance = reflectance_values[0]
    expected_direct = irradiance * reflectance * mu0 / pi * exp(-total / mu0 - total / mu)
    expected_r = 0.0
    expected_a = 0.0
    expected_down = 0.0
    expected_up = 0.0
    # Integrate the original physical-ray attenuation independently, over layer
    # fraction t (bottom to top) and hemisphere cosine u (zero to one).
    for layer, depth in enumerate(depths):
        above, below = sum(depths[layer + 1 :]), sum(depths[:layer])
        scatter = aerosol.optical_depth[layer][0] * aerosol.single_scattering_albedo[layer][0]

        def rays(
            t: float, *, above: float = above, below: float = below, depth: float = depth
        ) -> tuple[float, float]:
            return above + (1 - t) * depth, below + t * depth

        line = quad(lambda t: exp(-rays(t)[0] * (1 / mu0 + 1 / mu)), 0, 1)[0]
        expected_r += rayleigh.optical_depth[layer][0] * line
        expected_a += scatter * line

        def down_at(u: float) -> float:
            u = max(u, 1e-12)
            return quad(lambda t: exp(-rays(t)[0] / mu0 - rays(t)[1] / u), 0, 1)[0]

        def up_at(u: float) -> float:
            u = max(u, 1e-12)
            return quad(lambda t: exp(-rays(t)[1] / u - rays(t)[0] / mu), 0, 1)[0]

        expected_down += scatter * quad(down_at, 0, 1, epsabs=1e-11)[0]
        expected_up += scatter * quad(up_at, 0, 1, epsabs=1e-11)[0]
    expected = (
        expected_direct,
        irradiance * phase_r / (4 * pi * mu) * expected_r,
        irradiance / (4 * pi * mu) * expected_a,  # HG(g=0)=1
        irradiance * reflectance / (2 * pi) * exp(-total / mu) * expected_down,
        irradiance * reflectance * mu0 / (2 * pi * mu) * exp(-total / mu0) * expected_up,
    )
    for value, frozen in zip(expected, fixture["component_radiance"], strict=True):
        assert value == pytest.approx(frozen, rel=1e-11)
    assert result.components[0].values[0] == pytest.approx(expected_direct, rel=1e-14)
    # The frozen selected-path values retain their historical physical meaning.
    # SOS includes additional Rayleigh/ground and repeated-ground contributions.
    assert result.components[1].values[0] > sum(expected[1:])


def test_aerosol_scattering_rejects_mismatched_axis_and_layers() -> None:
    fixture, scene, solar, gas, rayleigh, aerosol = _case()
    source = solar_source(
        solar, direction=(fixture["solar_zenith_deg"], fixture["relative_azimuth_deg"])
    )
    other = np.array([1001.0], dtype=np.float64)
    with pytest.raises(ValueError, match="scattering axis"):
        terms_support.order_one(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                source,
                absorption=compose_absorption(scene.atmosphere, source, (gas,)),
                rayleigh=rayleigh_component(rayleigh),
                aerosol=aerosol_component(replace(aerosol, grid=wavenumber_grid(other))),
            ),
        )
    with pytest.raises(ValueError, match="scattering axis"):
        terms_support.order_one(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere,
                source,
                absorption=compose_absorption(scene.atmosphere, source, (gas,)),
                rayleigh=rayleigh_component(rayleigh),
                aerosol=aerosol_component(
                    replace(
                        aerosol,
                        optical_depth=aerosol.optical_depth[:1],
                        single_scattering_albedo=aerosol.single_scattering_albedo[:1],
                        asymmetry=aerosol.asymmetry[:1],
                    )
                ),
            ),
        )


def test_aerosol_scattering_homogeneous_layer_split_does_not_change_radiance() -> None:
    fixture, scene, solar, gas, rayleigh, aerosol = _case()
    source = solar_source(
        solar, direction=(fixture["solar_zenith_deg"], fixture["relative_azimuth_deg"])
    )
    one_layer = replace(
        scene,
        atmosphere=Atmosphere(
            np.array([0.0, 2000.0]),
            np.array([100000.0, 80000.0]),
            np.array([90000.0]),
            np.array([285.0]),
            np.array([3.8e29]),
            np.array([3.6e29]),
            np.array([2e28]),
            (Absorber("X", np.array([2e25]), "dry_air"),),
        ),
    )
    molecular = replace(
        rayleigh, optical_depth=np.array([[0.03]]), phase_moments=np.array([[[1.0, 0.0, 0.1]]])
    )
    particulate = replace(
        aerosol,
        optical_depth=np.array([[0.07]]),
        single_scattering_albedo=np.array([[0.7]]),
        asymmetry=np.array([[0.2]]),
    )
    one = terms_support.order_one(
        one_layer,
        layer_subdivisions=16,
        source=source,
        optics=compose_atmospheric_optics(
            one_layer.atmosphere,
            source,
            absorption=compose_absorption(
                one_layer.atmosphere,
                source,
                (replace(gas, optical_depth=np.array([[0.05]])),),
            ),
            rayleigh=rayleigh_component(molecular),
            aerosol=aerosol_component(particulate),
        ),
    )
    split = terms_support.order_one(
        scene,
        layer_subdivisions=8,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere,
            source,
            absorption=compose_absorption(
                scene.atmosphere,
                source,
                (replace(gas, optical_depth=np.array([[0.025], [0.025]])),),
            ),
            rayleigh=rayleigh_component(
                replace(rayleigh, optical_depth=np.array([[0.015], [0.015]]))
            ),
            aerosol=aerosol_component(
                replace(
                    aerosol,
                    optical_depth=np.array([[0.035], [0.035]]),
                    single_scattering_albedo=np.array([[0.7], [0.7]]),
                    asymmetry=np.array([[0.2], [0.2]]),
                )
            ),
        ),
    )
    for original, divided in zip(one.components, split.components, strict=True):
        assert original.values == pytest.approx(divided.values, rel=1e-12, abs=1e-14)
