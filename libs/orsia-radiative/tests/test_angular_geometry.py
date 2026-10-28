"""Independent outward-vector oracles for the retained solar angular contract.

No oracle copies the transfer's scalar scattering-cosine expression. The SOS-2
reference shares angular nodes and one-cell collocation, but evaluates physical
ray integrals without production transport or phase helpers. It is a geometry
and assembly check, not a continuous-solver or convergence qualification.
"""

from dataclasses import replace
from math import acos, atan2, degrees, exp, expm1, pi
from unittest.mock import patch

import numpy as np
import orders_reference as reference
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SolarSource,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.transfer import orders
from scipy.integrate import quad

# Outward Cartesian vectors, not a scattering-angle formula. The azimuths and
# cosines below are independently known special cases, including signed angles.
_CASES = (
    pytest.param((1, 0, 1), (1, 0, 1), 180.0, -1.0, id="same-side-backscatter"),
    pytest.param((1, 0, 1), (-1, 0, 1), 0.0, 0.0, id="opposite-side"),
    pytest.param((1, 0, 1), (0, 1, 1), 270.0, -0.5, id="outward-plus-90"),
    pytest.param((1, 0, 1), (0, -1, 1), 90.0, -0.5, id="outward-minus-90"),
    pytest.param((0, 0, 1), (1, 0, 1), None, -1 / np.sqrt(2), id="solar-zenith"),
    pytest.param((1, 0, 1), (0, 0, 1), None, -1 / np.sqrt(2), id="view-zenith"),
    pytest.param((0, 0, 1), (0, 0, 1), None, -1.0, id="both-zenith"),
    pytest.param((2, -3, 4), (-1, 2, 3), None, None, id="oblique"),
    pytest.param((30, 1, 1), (-20, 3, 1), None, None, id="near-grazing"),
    pytest.param((1, 0, 1), (-1, 1e-6, 1), None, None, id="azimuth-near-360"),
    pytest.param((1, 0, 1), (-1, -1e-6, 1), None, None, id="azimuth-near-zero"),
)


@pytest.mark.parametrize("sun,view,azimuth,cosine", _CASES)
@pytest.mark.parametrize("rotation", [0.0, 0.731])
def test_public_first_event_from_physical_vectors(
    sun: tuple[float, float, float],
    view: tuple[float, float, float],
    azimuth: float | None,
    cosine: float | None,
    rotation: float,
) -> None:
    s, v = _vectors(sun, view, rotation)
    scene, source, molecular, aerosol = _inputs(s, v)
    if azimuth is not None:
        assert source.relative_azimuth_deg == pytest.approx(azimuth, abs=1e-12)
    if cosine is not None:
        assert -s @ v == pytest.approx(cosine, abs=1e-14)

    total_depth = molecular.optical_depth[0][0] + aerosol.optical_depth[0][0]
    ray_integral = quad(lambda t: exp(-t / s[2] - t / v[2]), 0, total_depth)[0]
    # HG signs distinguish forward and backward angles; Rayleigh alone cannot.
    for g in (-0.6, 0.6):
        hg = replace(aerosol, asymmetry=np.array([[g]]))
        sos = terms_support.reflected_radiance(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere, source, rayleigh=molecular, aerosol=hg
            ),
            max_order=1,
            angular_order=3,
            layer_subdivisions=1,
        )
        expected_total = 0.0
        for component, phase_g in ((molecular, None), (hg, g)):
            expected = (
                source.irradiance.values[0]
                / (4 * pi * v[2])
                * (
                    component.optical_depth
                    if isinstance(component, RayleighComponent)
                    else component.scattering_depth
                )[0][0]
                / total_depth
                * _phase(float(-s @ v), phase_g)
                * ray_integral
            )
            expected_total += expected
        assert sos.values == pytest.approx(expected_total, rel=3e-13, abs=1e-15)
        assert sos.components[0].values == (0.0,)


@pytest.mark.parametrize("sun,view,azimuth,cosine", _CASES)
def test_sos_beam_and_intermediate_cosines_are_physical_dot_products(
    sun: tuple[float, float, float],
    view: tuple[float, float, float],
    azimuth: float | None,
    cosine: float | None,
) -> None:
    s, v = _vectors(sun, view, 0.731)
    scene, source, molecular, aerosol = _inputs(s, v, reflectance=0.2)
    local, _ = orders.angular_grid(3)
    physical = _physical_directions(local, s)
    outgoing = np.vstack((physical, v))
    expected_beam = outgoing @ -s
    expected_intermediate = physical @ outgoing.T
    phase_function = orders.phase_function
    seen: set[int] = set()

    def check(cosines: NDArray[np.float64], g: float | None) -> NDArray[np.float64]:
        seen.add(cosines.ndim)
        np.testing.assert_allclose(
            cosines,
            expected_beam if cosines.ndim == 1 else expected_intermediate,
            rtol=0,
            atol=2e-14,
        )
        return phase_function(cosines, g)

    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=molecular, aerosol=aerosol
    )
    with patch.object(reference, "phase_function", side_effect=check):
        expected = reference.scattering_orders(
            scene,
            source=source,
            optics=optics,
            max_order=2,
            angular_order=3,
            layer_subdivisions=1,
        )
    assert seen == {1, 2}  # Do not pass solely through the shared analytic path.
    actual = terms_support.reflected_radiance(
        scene, source=source, optics=optics, max_order=2, angular_order=3, layer_subdivisions=1
    )
    np.testing.assert_allclose(actual.values, expected.sum(axis=0), rtol=3e-13, atol=0)
    for component, row in zip(actual.components, expected, strict=True):
        np.testing.assert_allclose(component.values, row, rtol=3e-13, atol=0)


@pytest.mark.parametrize("sun,view,azimuth,cosine", _CASES)
@pytest.mark.parametrize("g", [-0.6, 0.6])
def test_sos_order_two_volume_pairs_against_independent_one_cell_rays(
    sun: tuple[float, float, float],
    view: tuple[float, float, float],
    azimuth: float | None,
    cosine: float | None,
    g: float,
) -> None:
    s, v = _vectors(sun, view, 0.731)
    scene, source, molecular, aerosol = _inputs(s, v)
    aerosol = replace(aerosol, asymmetry=np.array([[g]]))
    constituents = (molecular, aerosol)
    phases = (None, g)
    local, weights = orders.angular_grid(3)
    physical = _physical_directions(local, s)
    total_depth = sum(c.optical_depth[0][0] for c in constituents)
    result = terms_support.reflected_radiance(
        scene,
        source=source,
        optics=compose_atmospheric_optics(
            scene.atmosphere, source, rayleigh=molecular, aerosol=aerosol
        ),
        max_order=2,
        angular_order=3,
        layer_subdivisions=1,
    )
    expected_order_two = 0.0
    for first, first_g in zip(constituents, phases, strict=True):
        # Trace backward from the slab centre to the upstream boundary. q is
        # optical distance along that ray, t is vertical depth from slab top.
        centres = []
        for direction in physical:
            mu = float(direction[2])
            integral = quad(
                lambda q, mu=mu: exp(-q - (total_depth / 2 + mu * q) / s[2]),
                0,
                total_depth / (2 * abs(mu)),
                epsabs=1e-13,
                epsrel=1e-13,
            )[0]
            centres.append(
                source.irradiance.values[0]
                / (4 * pi)
                * (
                    first.optical_depth
                    if isinstance(first, RayleighComponent)
                    else first.scattering_depth
                )[0][0]
                / total_depth
                * _phase(float(-s @ direction), first_g)
                * integral
            )
        for second, second_g in zip(constituents, phases, strict=True):
            scatter_in = sum(
                weight * centre * _phase(float(direction @ v), second_g)
                for weight, centre, direction in zip(weights, centres, physical, strict=True)
            )
            expected = (
                (
                    second.optical_depth
                    if isinstance(second, RayleighComponent)
                    else second.scattering_depth
                )[0][0]
                / total_depth
                * scatter_in
                * -expm1(-total_depth / v[2])
            )
            expected_order_two += expected
    assert _value(result, "atmospheric_order_2") == pytest.approx(
        expected_order_two, rel=3e-13, abs=1e-15
    )


@pytest.mark.parametrize("sun,view", [((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (0, 0, 1))])
def test_zenith_azimuth_is_arbitrary_for_every_public_component(
    sun: tuple[float, float, float], view: tuple[float, float, float]
) -> None:
    s, v = _vectors(sun, view, 0.0)
    scene, source, molecular, aerosol = _inputs(s, v, reflectance=0.2)
    for method in (
        terms_support.order_zero,
        terms_support.order_one,
        terms_support.reflected_radiance,
    ):
        reference = method(
            scene,
            source=source,
            optics=compose_atmospheric_optics(
                scene.atmosphere, source, rayleigh=molecular, aerosol=aerosol
            ),
        )
        # SOS's 16-point azimuth grid is invariant under quarter turns, but
        # arbitrary rotations at solar zenith have a finite-quadrature residual.
        azimuths = (0.0, 90.0, 180.0, 270.0)
        if method is terms_support.order_zero or sun != (0, 0, 1):
            azimuths += (359.0,)
        for azimuth in azimuths:
            result = method(
                scene,
                source=replace(source, relative_azimuth_deg=azimuth),
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    replace(source, relative_azimuth_deg=azimuth),
                    rayleigh=molecular,
                    aerosol=aerosol,
                ),
            )
            for actual, expected in zip(result.components, reference.components, strict=True):
                assert actual.values == pytest.approx(expected.values, rel=3e-13, abs=1e-15)


def test_solar_zenith_arbitrary_azimuth_residual_decreases_with_quadrature() -> None:
    s, v = _vectors((0, 0, 1), (1, 0, 1), 0.0)
    scene, source, molecular, aerosol = _inputs(s, v, reflectance=0.2)
    residuals = []
    for angular_order in (4, 8, 20):
        results = tuple(
            terms_support.reflected_radiance(
                scene,
                source=replace(source, relative_azimuth_deg=azimuth),
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    replace(source, relative_azimuth_deg=azimuth),
                    rayleigh=molecular,
                    aerosol=aerosol,
                ),
                angular_order=angular_order,
                layer_subdivisions=1,
            )
            for azimuth in (0.0, 359.0)
        )
        reference, rotated = (
            np.array([c.values[0] for c in result.components]) for result in results
        )
        residuals.append(float(np.max(np.abs(rotated - reference))))
    # Bounded synthetic check, not a universal accuracy promise. Do not impose
    # roundoff azimuth-invariance on a quadrature that does not possess it.
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[2] < 1e-12


def _vectors(
    sun: tuple[float, float, float], view: tuple[float, float, float], rotation: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    c, s = np.cos(rotation), np.sin(rotation)
    matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    solar, outgoing = (np.asarray(vector, dtype=np.float64) for vector in (sun, view))
    return matrix @ (solar / np.linalg.norm(solar)), matrix @ (outgoing / np.linalg.norm(outgoing))


def _inputs(
    sun: NDArray[np.float64], view: NDArray[np.float64], *, reflectance: float = 0.0
) -> tuple[RadiativeScene, SolarSource, RayleighComponent, AerosolComponent]:
    grid = np.array([12000.0], dtype=np.float64)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([285.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
        (),
    )
    scene = RadiativeScene(
        atmosphere,
        SurfaceState(wavenumber_grid(grid), reflectance=np.array([reflectance])),
        ObservationGeometry(degrees(acos(float(view[2])))),
    )
    incoming = -sun
    # atan2 of separately supplied Cartesian directions, not the production
    # scalar phase expression. atan2(0, 0) chooses an arbitrary zenith azimuth.
    azimuth = (
        degrees(
            atan2(float(view[1]), float(view[0])) - atan2(float(incoming[1]), float(incoming[0]))
        )
        % 360
    )
    source = solar_source(
        SpectralIrradiance(wavenumber_grid(grid), np.array([7.0])),
        direction=(degrees(acos(float(sun[2]))), azimuth),
    )
    return (
        scene,
        source,
        RayleighComponent(wavenumber_grid(grid), np.array([[0.05]])),
        AerosolComponent(
            wavenumber_grid(grid), np.array([[0.15]]), np.array([[0.1]]), np.array([[0.6]])
        ),
    )


def _physical_directions(
    local: NDArray[np.float64], sun: NDArray[np.float64]
) -> NDArray[np.float64]:
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([-sun[0], -sun[1], 0.0])
    if np.linalg.norm(x) == 0:
        # Match the arbitrary projection chosen by atan2 at solar zenith.
        phi = atan2(float(-sun[1]), float(-sun[0]))
        x = np.array([np.cos(phi), np.sin(phi), 0.0])
    else:
        x /= np.linalg.norm(x)
    frame = np.column_stack((x, np.cross(z, x), z))
    return local @ frame.T


def _phase(cosine: float, g: float | None) -> float:
    # Independently written, 4pi-mean-one laws: do not invoke optical evaluators.
    return 0.75 * (1 + cosine**2) if g is None else (1 - g**2) / (1 + g**2 - 2 * g * cosine) ** 1.5


def _value(
    result: terms_support.RadianceDecomposition,
    name: str,
    role: str | None = None,
    *,
    pair: tuple[str, str] | None = None,
) -> float:
    return next(c.values[0] for c in result.components if c.name == name and c.role == role)
