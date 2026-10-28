"""Synthetic rolling LVF scan regressions; no physical qualification implied."""

import numpy as np
from numpy.typing import NDArray
from orsia_instruments import (
    Acquisition,
    DetectorMaps,
    ImagingOptics,
    RadianceField,
    exposure_sequence,
    gaussian_filter,
    lvf_centre,
    modulated_pixel_response,
    simulate_images,
)
from orsia_radiative import SpectralDomain, SpectralGrid


def test_lvf_scan_dense_image_preserves_read_geometry_and_electrons():
    image = _scan()
    assert image["reads"] == 24
    assert image["distinct_dn"] > 1
    assert image["electrons"] > 0


def _scan(*, rows: int = 3, cols: int = 4) -> dict:
    """Scan a heterogeneous moving scene through one complete acquisition."""
    if rows < 1 or cols < 1:
        raise ValueError("positive rows and columns required")
    points = np.linspace(4900.0, 5100.0, 129)
    grid = SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),))
    )
    width_x, width_y = 1.6e-4 / cols, 1.2e-4 / rows
    pixels = np.array(
        [
            [
                [
                    (col - cols / 2) * width_x,
                    (col + 1 - cols / 2) * width_x,
                    (row - rows / 2) * width_y,
                    (row + 1 - rows / 2) * width_y,
                ]
                for col in range(cols)
            ]
            for row in range(rows)
        ]
    )

    def pointing(t: NDArray[np.float64]) -> NDArray[np.float64]:
        # Supplied scan and jitter; not an inferred trajectory.
        return np.column_stack((0.002 * t + 1e-4 * np.sin(90 * t), -0.001 * t))

    def field_for_acquisition(acquisition: Acquisition) -> RadianceField:
        # Caller-owned scene covers the complete acquisition's exposure support.
        t = np.array([acquisition.exposure_bounds_s.min(), acquisition.exposure_bounds_s.max()])
        u = v = np.array([-0.003, 0.0, 0.003])
        spectrum = 1e-3 * (1 - 0.2 * np.exp(-(((points - 5000) / 12) ** 2)))
        values = (
            1
            + 2 * t[:, None, None, None]
            + 35 * u[None, :, None, None]
            - 20 * v[None, None, :, None]
        ) * spectrum
        return RadianceField(grid, t, u, v, values)

    def response(
        i: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        def local(x: float, y: float, wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
            centre = lvf_centre(
                x,
                reference_m=2e-6,
                gradient=4e-4,
                temperature_delta_k=acquisition.times_s[i, :, None],
                thermal_shift_m_per_k=2e-9,
            )
            return gaussian_filter(wavelength, centre, 1.5e-8, 0.8)

        return modulated_pixel_response(
            wave,
            acquisition.pixel_bounds_m[i],
            acquisition.image_xy_m[i],
            sigma_m=3e-5,
            transmission=local,
            pixel_order=5,
        )

    exposures = exposure_sequence(
        start_s=[0.0, 0.05],
        duration_s=0.02,
        row_delay_s=0.006 / (rows - 1),
        direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
        pointing_uv=pointing,
    )
    detector = DetectorMaps(
        quantum_efficiency=0.7,
        dark_electrons_per_s=2,
        read_noise_electrons=1,
        full_well_electrons=1e6,
        electrons_per_dn=2,
        bias_dn=100,
    )
    optics = ImagingOptics(
        0.02, 1e-8, lambda setting, read, wave, acq: response(read, wave, acq), {}, throughput=0.8
    )
    images = simulate_images(
        field_for_acquisition,
        pixel_bounds_m=pixels,
        exposures=exposures,
        optics=optics,
        detector=detector,
        rng=np.random.default_rng(17),
    )
    assert images.dn is not None and images.full_well_saturated is not None
    assert images.adc_clipped is not None
    assert not images.full_well_saturated.any() and not images.adc_clipped.any()
    assert np.all(images.photoelectrons > 0)
    return {
        "reads": images.photoelectrons.size,
        "electrons": float(images.photoelectrons.sum()),
        "distinct_dn": len(np.unique(images.dn)),
    }
