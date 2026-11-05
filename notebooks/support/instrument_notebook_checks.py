"""Independent bounded integration checks for the instrument notebook's scenario.

These checks use public forward APIs. They never reconstruct the principal products.
"""

from time import perf_counter

import numpy as np
from orsia_instruments import (
    camera_acquisition_grid,
    cavity_imager,
    dispersive_spectrometer,
    focal_plane_imager,
    frame_response,
    interferometric_imager,
    lvf_imager,
    select_raster_reads,
    simulate_frames,
    spectrometer_acquisition_grid,
)
from orsia_radiative import wavenumber_grid
from scipy.special import erfc


def check_strip_boundaries(detector, law, values, *, base=0.0, order=6):
    """Check each active rectangle and the one-sided limits of fixed strip steps.

    Args:
        detector: Nominal 100-row detector with along-track columns.
        law: Pure detector-x/y offset callback in metres.
        values: Expected absolute strip OPD or cavity spacing, metres.
        base: Scalar base added to the offset, metres.
        order: Interior Gauss-node count per active x interval.

    Returns:
        Counts of checked strips, interior nodes and one-sided boundaries.

    Raises:
        AssertionError: A strip interior or one-sided value differs from its declaration.
    """
    nodes = np.polynomial.legendre.leggauss(order)[0]
    bounds = detector.pixel_bounds_m[49]
    for index, (left, right, bottom, top) in enumerate(bounds):
        samples = (left + right) / 2 + (right - left) * nodes / 2
        np.testing.assert_allclose(
            [base + law(x, (bottom + top) / 2) for x in samples],
            values[index],
            rtol=1e-14,
            atol=1e-18,
        )
    # Finite epsilon avoids floating-point ambiguity at exactly shared boundaries.
    for index, edge in enumerate(bounds[:-1, 1]):
        np.testing.assert_allclose(
            base + law(edge - 1e-12, 0), values[index], rtol=1e-14, atol=1e-18
        )
        np.testing.assert_allclose(
            base + law(edge + 1e-12, 0), values[index + 1], rtol=1e-14, atol=1e-18
        )
    return dict(
        strips=len(values), interior_order=order, one_sided_boundary_checks=2 * (len(values) - 1)
    )


def check_refinements(
    *,
    spectral,
    scene_radiance,
    scene_coverage,
    shared_optics,
    shared_geometry,
    aligned_instrument,
    row_strip_instrument,
    native_cases,
    instruments,
    two_beam_instruments,
    two_beam_grids,
    two_beam_frames,
    fixed_opd,
    opds,
    fixed_cavity,
    gaps,
    entrance_transmission,
    pixel_order,
    angular_order,
    slit_order,
):
    """Measure local quadrature sensitivity, static-strip closure and slit truncation.

    Args:
        spectral: Actual instrument wavenumber grid in cm^-1.
        scene_radiance: Shared heterogeneous source callback.
        scene_coverage: Shared continuous-support attestation.
        shared_optics: Notebook focal length, collecting area, PSF and throughput.
        shared_geometry: Notebook epoch, motion, surface and support bounds.
        aligned_instrument: Instrument attitude for detector-x spectral strips.
        row_strip_instrument: Quarter-turn attitude for detector-row filters.
        native_cases: Completed native measurement/product tuples from the notebook.
        instruments: Named native-band instruments used for the principal products.
        two_beam_instruments: Physical plus/minus instruments in stacking order.
        two_beam_grids: Matching true selected acquisition grids.
        two_beam_frames: Matching primary measurements.
        fixed_opd: Detector-plane OPD offset callback, metres.
        opds: Absolute OPD strip array, metres.
        fixed_cavity: Detector-plane cavity offset callback, metres.
        gaps: Absolute cavity strip array, metres.
        entrance_transmission: Declared wavelength-metre order-blocking law.
        pixel_order: Principal finite-pixel order.
        angular_order: Principal camera angular order.
        slit_order: Principal slit quadrature order.

    Returns:
        Representative-read sensitivities, boundary counts, tail bound and elapsed seconds.

    Raises:
        AssertionError: A boundary fails or relative sensitivity exceeds 0.003.
        ValueError: Public acquisition or source-support validation fails.
    """
    started = perf_counter()
    report = {
        "relative_criterion": 0.003,
        "scope": "representative actual reads; interpolated atmospheric spectrum",
    }
    report["opd_boundaries"] = check_strip_boundaries(
        two_beam_instruments[0].detector, fixed_opd, opds, order=pixel_order
    )
    report["cavity_boundaries"] = check_strip_boundaries(
        instruments["cavity"].detector, fixed_cavity, gaps, base=gaps[0], order=pixel_order
    )
    pairs = (
        ("uniform", "Uniform filter"),
        ("mosaic", "Filter mosaic"),
        ("lvf", "LVF"),
        ("slit", "Slit dispersive"),
        ("cavity", "Fabry-Perot"),
    )
    fine_geometry = dict(shared_geometry, time_order=5)
    for key, name in pairs:
        frames, _, _, _, _, grid, _ = native_cases[name]
        instrument = instruments[key]
        selection = np.array([len(grid.schedule.native_indices) // 2], dtype=np.int64)
        schedule = select_raster_reads(grid.schedule, selection)
        if key == "slit":
            fine_grid = spectrometer_acquisition_grid(
                instrument,
                schedule,
                **fine_geometry,
                instrument_attitude=aligned_instrument,
                slit_order=64,
            )
        else:
            fine_grid = camera_acquisition_grid(
                instrument,
                schedule,
                **fine_geometry,
                instrument_attitude=row_strip_instrument
                if key in ("uniform", "mosaic")
                else aligned_instrument,
                angular_order=32,
            )
        fine = simulate_frames(
            instrument, fine_grid, spectral, scene_radiance, scene_coverage, read_batch_size=1
        )
        coarse = frames.images.photoelectrons.ravel()[selection]
        error = float(
            np.max(abs(fine.images.photoelectrons - coarse) / abs(fine.images.photoelectrons))
        )
        report[key + "_angular_time_relative"] = error
        assert error < 0.003, (key, error)
        if key == "lvf":
            refined = focal_plane_imager(
                instrument.detector,
                lvf_imager(
                    **shared_optics,
                    reference_m=1650e-9,
                    gradient=10e-9 / (16 * 20e-6),
                    sigma_m=0.35e-9,
                    peak=0.9,
                    pixel_order=12,
                ),
                support_radius_m=28e-6,
            )
            refined_grid = fine_grid
            pixel_fine = simulate_frames(
                refined, refined_grid, spectral, scene_radiance, scene_coverage, read_batch_size=1
            )
            pixel_error = float(
                np.max(
                    abs(pixel_fine.images.photoelectrons - fine.images.photoelectrons)
                    / abs(pixel_fine.images.photoelectrons)
                )
            )
            report["lvf_pixel_relative"] = pixel_error
            assert pixel_error < 0.003
        if key == "slit":
            long_slit = dispersive_spectrometer(
                instrument.detector,
                **shared_optics,
                slit_bounds_m=np.array([-10e-6, 10e-6, -80e-6, 80e-6]),
                reference_m=1650e-9,
                dispersion=32000.0,
                transmission=lambda wave: 0.9 * entrance_transmission(wave),
            )
            long_grid = spectrometer_acquisition_grid(
                long_slit,
                schedule,
                **fine_geometry,
                instrument_attitude=aligned_instrument,
                slit_order=80,
            )
            extended = simulate_frames(
                long_slit, long_grid, spectral, scene_radiance, scene_coverage, read_batch_size=1
            )
            report["slit_longer_entrance_relative"] = float(
                np.max(
                    abs(extended.images.photoelectrons - fine.images.photoelectrons)
                    / abs(extended.images.photoelectrons)
                )
            )
            report["slit_gaussian_two_tail_mass_bound"] = float(erfc(10 / np.sqrt(2)))
            report["slit_cross_guard_m"] = 40e-6
            assert report["slit_longer_entrance_relative"] < 0.003
        if key == "cavity":
            column = int(schedule.native_indices[0, 2])
            scalar = focal_plane_imager(
                instrument.detector,
                cavity_imager(
                    **shared_optics,
                    spacing_m=gaps[column],
                    refractive_index=1.0,
                    reflectivity=0.85,
                    order_transmission=entrance_transmission,
                ),
                support_radius_m=28e-6,
            )
            scalar_response = frame_response(scalar, fine_grid, spectral, read_batch_size=1)
            report["cavity_fixed_vs_scalar_response_relative"] = _relative_response(
                fine.response, scalar_response
            )
            assert report["cavity_fixed_vs_scalar_response_relative"] < 0.003
    # Check first, middle and last fixed OPD columns in both physical ports.
    for port_index, (instrument, grid, frames) in enumerate(
        zip(two_beam_instruments, two_beam_grids, two_beam_frames, strict=True)
    ):
        errors, scalar_errors = [], []
        for column in (0, 15, 31):
            index = int(np.flatnonzero(grid.native_indices[:, 2] == column)[0])
            schedule = select_raster_reads(grid.schedule, np.array([index], dtype=np.int64))
            fine_grid = camera_acquisition_grid(
                instrument,
                schedule,
                **fine_geometry,
                instrument_attitude=aligned_instrument,
                angular_order=32,
            )
            fine = simulate_frames(
                instrument, fine_grid, spectral, scene_radiance, scene_coverage, read_batch_size=1
            )
            errors.append(
                float(
                    abs(fine.images.photoelectrons[0] - frames.images.photoelectrons[index])
                    / fine.images.photoelectrons[0]
                )
            )
            scalar = focal_plane_imager(
                instrument.detector,
                interferometric_imager(
                    **shared_optics,
                    opd_m=opds[column],
                    port=instrument.optics.port,
                    visibility=0.9,
                    transmission=lambda wave: 0.9 * entrance_transmission(wave),
                ),
                support_radius_m=28e-6,
            )
            scalar_response = frame_response(scalar, fine_grid, spectral, read_batch_size=1)
            scalar_errors.append(_relative_response(fine.response, scalar_response))
        report[f"port_{port_index}_angular_time_relative"] = max(errors)
        report[f"port_{port_index}_fixed_vs_scalar_response_relative"] = max(scalar_errors)
        assert max(errors) < 0.003 and max(scalar_errors) < 0.003
    # Same interpolated source, nested instrument nodes; atmospheric knots are unchanged.
    fine_spectral = wavenumber_grid(
        np.linspace(spectral.points[0], spectral.points[-1], 2 * len(spectral.points) - 1),
        domain=spectral.domain,
    )
    grid = two_beam_grids[0]
    selected = select_raster_reads(grid.schedule, np.array([0], dtype=np.int64))
    selected_grid = camera_acquisition_grid(
        two_beam_instruments[0],
        selected,
        **shared_geometry,
        instrument_attitude=aligned_instrument,
        angular_order=angular_order,
    )
    spectral_fine = simulate_frames(
        two_beam_instruments[0],
        selected_grid,
        fine_spectral,
        scene_radiance,
        scene_coverage,
        read_batch_size=1,
    )
    error = float(
        abs(spectral_fine.images.photoelectrons[0] - two_beam_frames[0].images.photoelectrons[0])
        / spectral_fine.images.photoelectrons[0]
    )
    report["nested_instrument_spectral_relative"] = error
    assert error < 0.003
    report["wall_s"] = perf_counter() - started
    report["orders"] = dict(
        pixel=pixel_order,
        angular=angular_order,
        time=shared_geometry["time_order"],
        slit=slit_order,
    )
    return report


def _relative_response(first, second):
    return float(
        np.max(
            np.sum(abs(first.spectral_coefficients - second.spectral_coefficients), axis=1)
            / second.spectral_coefficients.sum(axis=1)
        )
    )
