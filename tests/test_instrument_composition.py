"""Synthetic source-path composition and external-charge readout regression."""

import numpy as np
from orsia_instruments import (
    DetectorMaps,
    combine_image_source_paths,
    exposure_sequence,
    filter_imager,
    image_charge,
    image_photoelectrons,
    image_readout,
    regular_pixel_bounds,
    uniform_radiance_field,
)
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid


def test_source_paths_and_detector_only() -> None:
    pixels = regular_pixel_bounds((2, 3), pitch_m=30e-6)
    exposures = exposure_sequence(
        start_s=[0.0, 0.05],
        duration_s=0.01,
        row_delay_s=0.001,
        direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
    )
    detector = DetectorMaps(
        quantum_efficiency=0,  # Replaced by the explicit spectral QE, never multiplied.
        bias_dn=100,
        read_noise_electrons=1,
        dark_electrons_per_s=2,
    )

    def qe(indices, wavelengths_m):
        if np.any((wavelengths_m < 1.9e-6) | (wavelengths_m > 2.1e-6)):
            raise ValueError("wavelength outside supplied QE support")
        return (0.6 + 0.02 * indices[1][:, None]) * wavelengths_m[None] / 2e-6

    paths = []
    for name, points, radiance_value, transmission in (
        ("scene", np.linspace(4900.0, 5100.0, 65), 1e-3, 0.8),
        ("internal_path", np.linspace(4850.0, 5150.0, 49), 2e-4, 0.4),
    ):
        # Synthetic supplied radiance and remaining transport, not a thermal model.
        radiance = Radiance(
            wavenumber_grid(points), np.full(len(points), radiance_value), ObservationGeometry(0.0)
        )
        field = uniform_radiance_field(
            radiance,
            time_bounds_s=(0.0, 0.1),
            u_bounds=(-0.003, 0.003),
            v_bounds=(-0.003, 0.003),
            spectral_bounds_cm1=(float(points[0]), float(points[-1])),
        )
        optics = filter_imager(
            focal_length_m=0.02,
            collecting_area_m2=1e-8,
            psf_sigma_m=30e-6,
            filter=lambda w, transmission=transmission: np.full_like(w, transmission),
        )
        paths.append(
            image_photoelectrons(
                field,
                pixel_bounds_m=pixels,
                optics=optics,
                exposures=exposures,
                detector=detector,
                quantum_efficiency=qe,
                source_name=name,
            )
        )

    def external():
        # Already integrated means; detector QE and exposure remain optical-path properties.
        return image_charge(
            {"external": np.full((2, 2, 3), 4.0)},
            exposure_bounds_s=exposures.bounds_s,
            row_delay_s=exposures.row_delay_s,
        )

    images = image_readout(
        combine_image_source_paths(*paths, external()),
        detector=detector,
        rng=np.random.default_rng(71),
        realisations=2,
        row_noise_electrons=2,
    )
    assert images.dn is not None and images.dn.shape == (2, 2, 2, 3)
    assert images.contributions["scene"].min() > 0
    np.testing.assert_allclose(images.photoelectrons, sum(images.contributions.values()))
    np.testing.assert_array_equal(images.contributions["external"], 4.0)
    detector_only = image_readout(external(), detector=detector)
    np.testing.assert_array_equal(detector_only.photoelectrons, 4.0)
