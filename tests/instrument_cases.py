"""Shared synthetic instrument fixtures; no files/providers or internal background."""

import numpy as np
from numpy.typing import NDArray
from orsia_instruments import (
    Acquisition,
    ExpectedElectrons,
    OpticalResponse,
    RadianceField,
    cavity_transmission,
    dispersive_response,
    expected_electrons,
    gaussian_filter,
    gaussian_pixel_fraction,
    lvf_centre,
    modulated_pixel_response,
    rectangular_acquisition,
    two_beam_ports,
)
from orsia_radiative import (
    Atmosphere,
    ObservationGeometry,
    Radiance,
    RadiativeScene,
    SpectralDomain,
    SpectralGrid,
    SpectralIrradiance,
    SurfaceState,
    atmospheric_optics,
    solar_source,
    transfer,
)

FAMILIES = ("dispersive", "filter", "lvf", "two-beam", "cavity")


def make_case(
    family: str,
    *,
    spectral_points: int = 257,
    time_order: int = 3,
    angular_order: int = 5,
    pixel_order: int = 8,
    repeats: int = 1,
    shift: float = 0,
    pointing_shift: float = 0,
    time_shift: float = 0,
) -> tuple[RadianceField, Acquisition, OpticalResponse]:
    """A fixed synthetic scene with finite pixels, sequential settings and moving pointing.

    shift changes a family parameter: wavelength centre [m] for filter/LVF,
    image x [m] for dispersive, OPD [m] for two-beam, spacing [m] for cavity.
    Scene axes are independent of quadrature order so refinement compares one
    interpolated field, not a changing scene. Local plane maps are illustrative.
    """
    if family not in FAMILIES:
        raise ValueError("unknown example response family")
    points = np.linspace(4900.0, 5100.0, spectral_points)
    grid = SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),))
    )
    times = np.array([-0.1, 0.0, 0.05, 0.1, 0.2])
    u = v = np.array([-0.004, 0.0, 0.004])
    spectrum = 1e-3 * (1 - 0.3 * np.exp(-(((points - 4995) / 7) ** 2)))
    spatial = 1 + 40 * u[:, None] - 20 * v[None, :] + 4000 * u[:, None] * v[None, :]
    values = (1 + 2 * times[:, None, None, None]) * spatial[None, :, :, None] * spectrum
    field = RadianceField(grid, times, u, v, values)
    identities = [
        (repeat, setting, pixel, port)
        for repeat in range(repeats)
        for setting in range(2)
        for pixel in range(2)
        for port in (("+", "-") if family == "two-beam" else ("direct",))
    ]
    acquisition = rectangular_acquisition(
        sample_ids=tuple(f"r{r}-s{s}-p{p}-{port}" for r, s, p, port in identities),
        pixel_ids=tuple(f"p{p}" for _, _, p, _ in identities),
        setting_ids=tuple(f"s{s}" for _, s, _, _ in identities),
        port_ids=tuple(port for _, _, _, port in identities),
        exposure_bounds_s=np.array(
            [[s * 0.05 + time_shift, s * 0.05 + 0.02 + time_shift] for _, s, _, _ in identities]
        ),
        pixel_bounds_m=np.array(
            [[(-1 + p) * 8e-5, p * 8e-5, -5e-5, 5e-5] for _, _, p, _ in identities]
        ),
        direction_bounds_uv=(-0.004, 0.004, -0.004, 0.004),
        pointing=lambda t: np.column_stack((0.006 * t + pointing_shift, -0.003 * t)),
        focal_length_m=0.02,
        time_order=time_order,
        angular_order=angular_order,
    )

    def response(
        i: int, wavelength: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        setting = int(acquisition.setting_ids[i][1:])
        image = acquisition.image_xy_m[i]
        bounds = acquisition.pixel_bounds_m[i]
        if family == "dispersive":
            shifted_image = image + np.array([shift, 0.0])
            return dispersive_response(
                wavelength,
                bounds,
                shifted_image,
                sigma_m=3e-5,
                reference_m=2e-6,
                dispersion=700,
                smile=0.1,
                keystone=30,
                transmission=0.8,
            )
        if family == "filter":
            fraction = gaussian_pixel_fraction(bounds, image, 3e-5)
            band = gaussian_filter(wavelength, 2e-6 + (setting - 0.5) * 2e-8 + shift, 1.5e-8, 0.85)
            return fraction[:, None] * band

        times = acquisition.times_s[i, :, None]
        angle = np.arcsin(acquisition.directions_uv[i, :, :1])

        def local(x: float, y: float, wave: NDArray[np.float64]) -> NDArray[np.float64]:
            if family == "lvf":
                centre = lvf_centre(
                    x,
                    reference_m=2e-6 + shift,
                    gradient=3e-4,
                    temperature_delta_k=times * 2,
                    thermal_shift_m_per_k=1e-9,
                    angle_rad=angle,
                )
                return gaussian_filter(wave, centre, 1.5e-8, 0.85)
            if family == "two-beam":
                opd = (setting * 8e-6 + 2e-5 * times + 0.02 * x + shift) * np.cos(angle)
                ports = two_beam_ports(wave, opd, visibility=0.9, transmission=0.8)
                return ports[0 if acquisition.port_ids[i] == "+" else 1]
            spacing = 1e-5 + setting * 1e-8 + 2e-4 * x + shift
            return cavity_transmission(
                wave,
                spacing,
                internal_angle_rad=angle,
                reflectivity=0.65,
                order_transmission=gaussian_filter(wave, 2e-6, 3e-8),
            )

        return modulated_pixel_response(
            wavelength, bounds, image, sigma_m=3e-5, transmission=local, pixel_order=pixel_order
        )

    return field, acquisition, response


def simulate(family: str, *, shift: float = 0) -> ExpectedElectrons:
    """Run the bounded synthetic native expectation; all internal backgrounds excluded."""
    field, acquisition, response = make_case(family, shift=shift)
    return expected_electrons(
        field,
        acquisition,
        response,
        collecting_area_m2=1e-8,
        throughput=0.8,
        quantum_efficiency=0.7,
    )


def radiative_inputs() -> tuple[Radiance, Radiance]:
    """Public solar and LTE thermal solver predictions for a transparent synthetic column."""
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.linspace(4900.0, 5100.0, 257),
        SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),)),
    )
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([280.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
        (),
    )
    surface = SurfaceState(
        grid, reflectance=np.full(257, 0.2), emissivity=np.full(257, 0.8), temperature_k=300.0
    )
    scene = RadiativeScene(atmosphere, surface, ObservationGeometry(0.0))
    optics = atmospheric_optics(atmosphere, grid)
    source = solar_source(SpectralIrradiance(grid, np.full(257, 0.1)), direction=(30.0, 0.0))
    solar = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    thermal = transfer.thermal.schwarzschild_radiance(scene, optics=optics)
    return solar, thermal
