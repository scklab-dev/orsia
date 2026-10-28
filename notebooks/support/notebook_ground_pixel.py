"""Illustrative nadir-scan acquisition and linear scene-pixel reconstruction.

Arrays use float64. Ground axes are (along, cross), spectra are per cm^-1,
physical kernels have axes (read, along_offset, cross_offset, wavenumber),
and signed reconstruction weights have axes (channel, read). Inputs are
borrowed; returned arrays own storage. The separable Gaussian spatial calculation
uses public Orsia optical responses and explicit paraxial dOmega=dx dy / h^2.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray
from orsia_instruments import (
    DetectorElectronics,
    DetectorResponse,
    ExpectedElectrons,
    LinearTrajectory,
    RadianceField,
    ReadSchedule,
    along_track_acquisition,
    along_track_read_schedule,
    cavity_imager,
    correct_digital_readings,
    detector_response,
    digitize,
    dispersive_imager,
    electron_covariance,
    electron_moments,
    expected_electrons,
    filter_imager,
    gaussian_cross_track_response,
    gaussian_filter,
    interferometric_imager,
    lvf_imager,
    response_electrons,
    sample_readout,
    spatial_filter_imager,
)
from orsia_radiative import Radiance, SpectralDomain, SpectralGrid
from orsia_uncertainty import propagate_covariance
from scipy.special import ndtr

FAMILIES = (
    "Uniform filter",
    "Filter mosaic",
    "LVF",
    "Dispersive imager",
    "Two-beam interferometer",
    "Fabry–Pérot",
)
OFFSETS = np.arange(-2, 3, dtype=np.float64)


@dataclass(frozen=True)
class ScanParameters:
    """Illustrative scan geometry, detector optics and integration controls in SI units.

    Arrays supplied by the caller retain their own ownership; freezing this record
    protects field assignment, rather than the contents of an array field.
    """

    height_m: float
    speed_m_s: float
    sampling_m: float
    pitch_m: float
    focal_length_m: float
    diameter_m: float
    psf_sigma_m: float
    throughput: float
    qe: float
    exposure_s: float
    line_period_s: float
    entrance_nm: tuple[float, float]
    flat_radiance: float
    halo_pixels: int
    pixel_order: int
    time_order: int
    angular_order: int


@dataclass(frozen=True)
class DetectorLayout:
    """Detector stage/port assignment and ground mapping for one optical family.

    architecture() creates the arrays; consumers borrow them. A dataclass
    replacement can share array buffers and this frozen record leaves their
    contents mutable. Read order follows stage, with both ports of
    the two-beam family occupying successive blocks.
    """

    family: str
    stages: int
    stage: NDArray[np.int64]
    ports: tuple[str, ...]
    detector_x_m: NDArray[np.float64]
    target_offsets_at_exposure_centre_m: NDArray[np.float64]
    channel_nm: NDArray[np.float64]

    @property
    def read_count(self) -> int:
        return len(self.stage)

    @property
    def active_pixels(self) -> int:
        return 100 * self.read_count


@dataclass(frozen=True)
class AcquisitionWindow:
    """First/last exposure and per-pixel read timing over the 100-row image."""

    first_exposure_s: float
    last_exposure_s: float
    window_s: float
    per_pixel_span_s: float
    target_rows: int
    complete_edge_channels: int
    exposure_bounds_s: NDArray[np.float64]  # (along, read, start/end), borrowed by consumers


@dataclass(frozen=True)
class SceneResponseEvaluation:
    """One family's positive scene response, native reads and calibrated products.

    kernel: (read, along_offset, cross_offset, wavenumber) in electrons per
    spectral-radiance node; native: (read, along, cross) electrons; weights:
    (channel, read) signed inverse-electron calibration; effective:
    (channel, wavenumber); channels and digitized: reconstructed radiance;
    dn: detector counts. Arrays are caller-owned, not copied by this record.
    """

    kernel: NDArray[np.float64]
    native: NDArray[np.float64]
    weights: NDArray[np.float64]
    effective: NDArray[np.float64]
    diagnostics: dict
    channels: NDArray[np.float64]
    dn: NDArray[np.int64]
    digitized: NDArray[np.float64]
    detector: dict


def comparison_settings(scenario: dict) -> ScanParameters:
    """Nominal local constant-height/speed geometry; seconds relative to passage."""
    height = scenario["altitude_above_wgs84_ellipsoid_km"] * 1000
    speed = scenario["nadir_ground_speed_km_s"] * 1000
    return ScanParameters(
        height_m=height,
        speed_m_s=speed,
        sampling_m=50.0,
        pitch_m=20e-6,
        focal_length_m=height * 20e-6 / 50,
        diameter_m=0.04,
        psf_sigma_m=4e-6,
        throughput=0.70,
        qe=0.65,
        exposure_s=25 / speed,
        line_period_s=50 / speed,
        entrance_nm=(1645.0, 1655.0),
        flat_radiance=1e-3,
        halo_pixels=2,
        pixel_order=12,
        time_order=3,
        angular_order=10,
    )


def architecture(family: str, settings: ScanParameters) -> DetectorLayout:
    """Static strip/slit layouts, physical read settings and output channel labels.

    x is along-track in the detector plane, y cross-track. Positive x strips
    observe each ground location earlier. Both ports occupy separate relays.
    Mosaic stripes span the full cross-track field; LVF varies continuously in x.
    """
    count = {
        "Uniform filter": 1,
        "Filter mosaic": 8,
        "LVF": 16,
        "Dispersive imager": 16,
        "Two-beam interferometer": 32,
        "Fabry–Pérot": 8,
    }[family]
    x = (np.arange(count, dtype=np.float64) - (count - 1) / 2) * settings.pitch_m
    wavelength = 1645 + (np.arange(count, dtype=np.float64) + 0.5) * 10 / count
    stage = np.arange(count)
    ports = ["direct"] * count
    if family == "Two-beam interferometer":
        stage = np.tile(stage, 2)
        ports = ["+"] * count + ["-"] * count
        wavelength = 1645 + (np.arange(8, dtype=np.float64) + 0.5) * 10 / 8
    if family == "Uniform filter":
        wavelength = np.array([1650.0])
    ground_offsets = x[stage] * settings.height_m / settings.focal_length_m
    if family == "Dispersive imager":
        ground_offsets = np.zeros(count, dtype=np.float64)
    return DetectorLayout(
        family=family,
        stages=count,
        stage=stage,
        ports=tuple(ports),
        detector_x_m=x[stage],
        target_offsets_at_exposure_centre_m=ground_offsets,
        channel_nm=wavelength,
    )


def acquisition_window(layout: DetectorLayout, settings: ScanParameters) -> AcquisitionWindow:
    """Every target centre and required channel/port is covered, including edges."""
    reads, _ = native_reads(layout, settings, (layout.read_count, 100, 1))
    bounds = reads.exposure_bounds_s.reshape(layout.read_count, 100, 2).transpose(1, 0, 2)
    centres = bounds.mean(axis=-1)
    return AcquisitionWindow(
        first_exposure_s=float(bounds.min()),
        last_exposure_s=float(bounds.max()),
        window_s=float(bounds.max() - bounds.min()),
        per_pixel_span_s=float(np.ptp(centres[0]) + settings.exposure_s),
        target_rows=100,
        complete_edge_channels=layout.read_count,
        exposure_bounds_s=bounds,
    )


def scan_geometry(
    layout: DetectorLayout, read: int, settings: ScanParameters
) -> tuple[ReadSchedule[np.float64], NDArray[np.float64]]:
    """Return one physical strip read and independent detector bounds in metres.

    The caller supplies scan height/speed, integration orders and scene support
    to along_track_acquisition. Returned arrays own their storage.
    """
    x_detector, pitch = layout.detector_x_m[read], settings.pitch_m
    reads = along_track_read_schedule(
        sample_ids=(f"read-{read}",),
        pixel_ids=(f"strip-{layout.stage[read]}",),
        setting_ids=(str(layout.stage[read]),),
        port_ids=(layout.ports[read],),
        target_along_m=np.zeros(1),
        ground_along_offsets_m=np.array([layout.target_offsets_at_exposure_centre_m[read]]),
        exposure_s=settings.exposure_s,
        trajectory=LinearTrajectory(height_m=settings.height_m, speed_m_s=settings.speed_m_s),
    )
    return reads, np.array(
        [[x_detector - pitch / 2, x_detector + pitch / 2, -pitch / 2, pitch / 2]]
    )


def physical_kernel(
    layout: DetectorLayout,
    radiance: Radiance,
    settings: ScanParameters,
    *,
    angular_order: int | None = None,
    time_order: int | None = None,
    pixel_order: int | None = None,
) -> NDArray[np.float64]:
    """Nonnegative e- per spectral-radiance node and bilinear ground-grid value.

    A 5x5 ground stencil spans +/-100m. Its tent functions represent the supplied
    50m-sampled, bilinearly interpolated scene. Spatial support is integrated on
    +/-100m, with Gaussian tails beyond it retained as a truncation limitation.
    Dispersion uses a physical 50m-wide entrance slit before spectral mapping.
    Spectral trapezoid weights, photons, exposure, area, throughput and QE enter
    exactly once. An ideal order/entrance filter accepts 1645–1655nm.
    """
    wavelength_edges = 1e7 / radiance.grid.points[[-1, 0]]
    if not np.allclose(wavelength_edges, settings.entrance_nm, rtol=0, atol=1e-9):
        raise ValueError("kernel grid must sample both ideal entrance-band endpoints")
    order = angular_order or settings.angular_order
    result = np.empty((layout.read_count, 5, 5, len(radiance.grid.points)))
    grid = _supported_grid(radiance)
    for read in range(layout.read_count):
        acquisition, ground_x, ground_y, _ = _nodes(
            layout, read, settings, order, time_order or settings.time_order, full=True
        )
        # These example optics are Gaussian and separable cross-track. Evaluate
        # their expensive spectral law only at y=0, then lift the exact Gaussian
        # acceptance to the full public quadrature. General optics can supply
        # their response directly to detector_response.
        line, _, _, _ = _nodes(
            layout, read, settings, order, time_order or settings.time_order, full=False
        )
        optics = _optics(layout, read, settings, pixel_order or settings.pixel_order)
        factory = gaussian_cross_track_response(line, optics, sigma_m=settings.psf_sigma_m)
        accepted = factory(acquisition)

        result[read] = detector_response(
            grid,
            acquisition,
            accepted,
            ground_xy_m=np.stack((ground_x, ground_y), axis=-1)[None],
            along_m=OFFSETS * 50.0,
            cross_m=OFFSETS * 50.0,
            support_bounds_m=np.array([[-100.0, 100.0, -100.0, 100.0]]),
            collecting_area_m2=optics.collecting_area_m2,
            throughput=optics.throughput,
            quantum_efficiency=settings.qe,
        ).coefficients[0]
    return result


def reconstruction(
    layout: DetectorLayout, kernel: NDArray[np.float64], radiance: Radiance
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict]:
    """Calibrated channel means; signed interferometer inverse is explicit.

    Direct families return weighted mean spectral radiance through their actual
    band/LSF. The interferometer uses balanced ports and a sampled piecewise-
    constant eight-bin prior basis with unregularized least squares (rcond=1e-10).
    W@H is evaluated on every high-resolution source node. Out-of-basis spectra
    therefore retain visible sidelobes, cross-talk and nullspace ambiguity.
    """
    physical = kernel.sum(axis=(1, 2))
    if layout.family != "Two-beam interferometer":
        weights = np.diag(1 / physical.sum(axis=1))
        diagnostics: dict = dict(method="per-read flat-radiance calibration")
    else:
        wavelength = 1e7 / radiance.grid.points
        bins = np.minimum(((wavelength - 1645) / 1.25).astype(int), 7)
        basis = np.eye(8)[bins]
        difference = np.concatenate((np.eye(32), -np.eye(32)), axis=1)
        matrix = difference @ physical @ basis
        weights = np.linalg.pinv(matrix, rcond=1e-10) @ difference
        diagnostics = dict(
            method="balanced-port eight-bin least squares",
            bins_nm=np.linspace(1645, 1655, 9).tolist(),
            basis_rank=int(np.linalg.matrix_rank(matrix)),
            condition_number=float(np.linalg.cond(matrix)),
            rcond=1e-10,
            apodization="unity",
            basis_closure_max=float(np.max(np.abs(weights @ physical @ basis - np.eye(8)))),
        )
    effective = weights @ physical
    diagnostics.update(
        physical_min=float(physical.min()),
        signed_weight_min=float(weights.min()),
        effective_min=float(effective.min()),
        row_sum=effective.sum(axis=1).tolist(),
    )
    return weights, effective, diagnostics


def synthetic_scene(radiance: Radiance) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Illustrative spatial reflectance/absorption variation, with a two-pixel halo.

    Three nonnegative spatial maps multiply per-cm^-1 spectral radiances. The
    localized spectral change is a synthetic absorption-shaped surface template.
    Axes are (component, along=104, cross=104) and (component,wavenumber).
    """
    x, y = np.meshgrid((np.arange(104) - 51.5) * 50, (np.arange(104) - 51.5) * 50, indexing="ij")
    patch = 0.35 * np.exp(-((x - 450) ** 2 + (y + 650) ** 2) / (2 * 500**2))
    step = 0.18 * ((x > -650) & (y > 100))
    base = 0.5 + 0.12 * np.sin(x / 700) * np.cos(y / 900)
    maps = np.stack((base, patch, step))
    wave = 1e7 / radiance.grid.points
    spectra = np.stack(
        (
            radiance.values,
            radiance.values * (1 - 0.7 * np.exp(-0.5 * ((wave - 1649) / 0.45) ** 2)),
            radiance.values * (0.7 + 0.3 * (wave - 1645) / 10),
        )
    )
    return maps, spectra


def image_channels(
    kernel: NDArray[np.float64],
    weights: NDArray[np.float64],
    maps: NDArray[np.float64],
    spectra: NDArray[np.float64],
    *,
    layout: DetectorLayout,
    radiance: Radiance,
    settings: ScanParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read images and reconstructed channels, cropped to the 100x100 scene.

    All edge pixels use the explicit synthetic two-pixel scene halo. Correlation
    outside the larger array is discarded; the returned crop uses valid support.
    """
    shape = (len(kernel), maps.shape[1] - 4, maps.shape[2] - 4)
    templates, _ = native_reads(layout, settings, (len(kernel), 1, 1))
    response = DetectorResponse(
        templates,
        _supported_grid(radiance),
        OFFSETS * 50.0,
        OFFSETS * 50.0,
        np.tile([-100.0, 100.0, -100.0, 100.0], (len(kernel), 1)),
        kernel,
    )
    reads, _ = native_reads(layout, settings, shape)
    native = response_electrons(
        response,
        maps,
        spectra=spectra,
        scene_along_m=(np.arange(maps.shape[1]) - (maps.shape[1] - 1) / 2) * 50,
        scene_cross_m=(np.arange(maps.shape[2]) - (maps.shape[2] - 1) / 2) * 50,
        target_indices=(
            np.arange(2, maps.shape[1] - 2, dtype=np.int64),
            np.arange(2, maps.shape[2] - 2, dtype=np.int64),
        ),
        reads=reads,
        method="correlation",
    ).total.reshape(shape)
    return native, np.einsum("cr,rij->cij", weights, native)


def detector_channels(
    layout: DetectorLayout,
    settings: ScanParameters,
    native: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    electronics: DetectorElectronics | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], dict]:
    """Digitize each read, subtract its pedestal, then combine calibrated electrons.

    Default electronics specify 2e-/DN, 100DN bias, 2e-/s dark,
    3e- read-noise sigma for variance propagation, 1e6e- well and 20-bit ADC.
    An explicit electronics value supplies the same transfer/correction quantities.
    Shot/dark/read variance is propagated through W; quantization is bounded
    separately by 0.5 DN * gain * sum(abs(W)). Noise draws are outside this demo.
    One deterministic realization contains every physical read, ordered
    (template,along,cross), with native pixel/setting/port and exposure identities.
    native_reads() also supplies physical row/column/exposure noise groups for
    sample_readout; the optional stochastic path uses repeated draws only.
    """
    if electronics is None:
        electronics = DetectorElectronics(
            dark_electrons_per_s=2,
            read_noise_electrons=3,
            full_well_electrons=1e6,
            electrons_per_dn=2,
            bias_dn=100,
            adc_bits=20,
        )
    reads, groups = native_reads(layout, settings, native.shape)
    signal = ExpectedElectrons(reads, {"scene": native.ravel()})
    mean, variance = electron_moments(signal, electronics)
    reading = digitize(reads, mean[None], np.zeros((1, mean.size)), electronics)
    corrected = correct_digital_readings(reading, electronics).reshape(native.shape)
    dn = reading.dn[0].reshape(native.shape)
    reconstructed = np.einsum("cr,rij->cij", weights, corrected)
    variance = np.einsum("cr,rij->cij", weights**2, variance.reshape(native.shape))
    centre = ((native.shape[1] - 1) // 2, (native.shape[2] - 1) // 2)
    indices = np.arange(native.size).reshape(native.shape)[:, centre[0], centre[1]]
    centre_reads = ReadSchedule(
        tuple(reads.sample_ids[i] for i in indices),
        tuple(reads.pixel_ids[i] for i in indices),
        tuple(reads.setting_ids[i] for i in indices),
        tuple(reads.port_ids[i] for i in indices),
        reads.exposure_bounds_s[indices],
    )
    covariance = electron_covariance(
        ExpectedElectrons(centre_reads, {"scene": native.ravel()[indices]}),
        electronics,
        row_ids=tuple(groups["row_ids"][i] for i in indices),
        column_ids=tuple(groups["column_ids"][i] for i in indices),
        exposure_ids=tuple(groups["exposure_ids"][i] for i in indices),
    )
    reconstructed_covariance = propagate_covariance(covariance, weights)
    return (
        dn,
        reconstructed,
        dict(
            max_native_electrons=float(native.max()),
            zero_light_analogue_dn=electronics.bias_dn
            + electronics.dark_electrons_per_s * settings.exposure_s / electronics.electrons_per_dn,
            quantization_bound=(
                electronics.electrons_per_dn * 0.5 * np.abs(weights).sum(axis=1)
            ).tolist(),
            centre_noise_sigma=np.sqrt(variance[:, centre[0], centre[1]]).tolist(),
            centre_radiance_covariance=reconstructed_covariance.tolist(),
            unsaturated=True,
            physical_reads=len(reads.sample_ids),
            exposure_groups=len(set(groups["exposure_ids"])),
            realisations=1,
            full_well_saturated=int(reading.full_well_saturated.sum()),
            adc_clipped=int(reading.adc_clipped.sum()),
        ),
    )


def native_reads(layout: DetectorLayout, settings: ScanParameters, shape: tuple[int, int, int]):
    """Physical reads/groups in C-order (template,ground along,ground cross).

    Detector rows are strips in distinct port relays; cross columns are distinct
    physical pixels. Exposures share noise exactly when their actual passage
    positions coincide. Reconstructed ground pixels may involve different times.
    """
    count, rows, columns = shape
    if count != layout.read_count:
        raise ValueError("native template count disagrees with layout")
    index = list(np.ndindex(shape))
    along = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * settings.sampling_m
    passage = along[None, :] - layout.target_offsets_at_exposure_centre_m[:, None]
    port_row = [f"{layout.ports[r]}-strip-{layout.stage[r]}" for r, i, j in index]
    column_ids = tuple(f"{layout.ports[r]}-column-{j}" for r, i, j in index)
    reads = along_track_read_schedule(
        sample_ids=tuple(f"read-{r}-ground-{i}-{j}" for r, i, j in index),
        pixel_ids=tuple(
            f"{row}-column-{j}" for row, (_, _, j) in zip(port_row, index, strict=True)
        ),
        setting_ids=tuple(str(layout.stage[r]) for r, i, j in index),
        port_ids=tuple(layout.ports[r] for r, i, j in index),
        target_along_m=np.broadcast_to(along[None, :, None], shape).ravel(),
        ground_along_offsets_m=np.broadcast_to(
            layout.target_offsets_at_exposure_centre_m[:, None, None], shape
        ).ravel(),
        exposure_s=settings.exposure_s,
        trajectory=LinearTrajectory(height_m=settings.height_m, speed_m_s=settings.speed_m_s),
    )
    return reads, dict(
        row_ids=tuple(port_row),
        column_ids=column_ids,
        exposure_ids=tuple(f"passage-{float(passage[r, i]).hex()}" for r, i, j in index),
    )


def sample_detector_channels(
    layout, settings, native, weights, rng, *, realisations=1, exposure_noise_electrons=0
):
    """Explicit repeated detector draws before signed reconstruction W.

    Return DetectorReadings (flags/identities retained), reconstructed radiances
    [realisation,channel,along,cross], and physical group IDs. RNG is caller-owned.
    """
    reads, groups = native_reads(layout, settings, native.shape)
    electronics = DetectorElectronics(
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        full_well_electrons=1e6,
        electrons_per_dn=2,
        bias_dn=100,
        adc_bits=20,
    )
    reading = sample_readout(
        ExpectedElectrons(reads, {"scene": native.ravel()}),
        electronics,
        rng,
        realisations=realisations,
        **groups,
        exposure_noise_electrons=exposure_noise_electrons,
    )
    corrected = correct_digital_readings(reading, electronics).reshape(
        (realisations, *native.shape)
    )
    return reading, np.einsum("cr,brij->bcij", weights, corrected), groups


def direct_scene_read(
    layout: DetectorLayout,
    read: int,
    radiance: Radiance,
    settings: ScanParameters,
    scene_values: NDArray[np.float64],
    *,
    angular_order: int | None = None,
) -> NDArray[np.float64]:
    """Independent public expected_electrons on a time-dependent RadianceField.

    scene_values is (5,5,wavenumber), bilinear at +/-100m scene knots.
    Field time knots include all acquisition times; direction knots include every
    moving ground knot. Thus field interpolation and stencil integration describe
    the same scene exactly, including motion and edge-halo support.
    """
    order = angular_order or settings.angular_order
    acquisition, _, _, time_nodes = _nodes(
        layout, read, settings, order, settings.time_order, full=True
    )
    height, speed = settings.height_m, settings.speed_m_s
    time_grid = np.unique(np.r_[acquisition.exposure_bounds_s[0], time_nodes])
    u = np.unique(((OFFSETS[:, None] * 50 - speed * time_grid[None, :]) / height).ravel())
    u = np.unique(np.r_[u, acquisition.direction_bounds_uv[0, :2]])
    v = OFFSETS * 50 / height
    values = np.empty((len(time_grid), len(u), len(v), len(radiance.grid.points)))
    # Every query lies inside the supplied bilinear scene except unused field
    # corners; constant corner extension completes the rectangular field support.
    for index, time in enumerate(time_grid):
        x = u * height + speed * time
        for cross in range(5):
            for spectral in range(len(radiance.grid.points)):
                values[index, :, cross, spectral] = np.interp(
                    x, OFFSETS * 50, scene_values[:, cross, spectral]
                )
    grid = SpectralGrid(
        "wavenumber_cm1",
        radiance.grid.points,
        SpectralDomain("wavenumber_cm1", ((radiance.grid.points[0], radiance.grid.points[-1]),)),
    )
    field = RadianceField(grid, time_grid, u, v, values)
    optics = _optics(layout, read, settings, settings.pixel_order)
    result = expected_electrons(
        field,
        acquisition,
        lambda current_read, wave, current: optics.response(0, current_read, wave, current),
        collecting_area_m2=optics.collecting_area_m2,
        throughput=optics.throughput,
        quantum_efficiency=settings.qe,
    )
    return result.total


def _supported_grid(radiance):
    points = radiance.grid.points
    return SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((points[0], points[-1]),))
    )


def _pixel_fraction(centroid, settings):
    sigma, half = settings.psf_sigma_m, settings.pitch_m / 2
    return ndtr((half - centroid) / sigma) - ndtr((-half - centroid) / sigma)


def _nodes(layout, read, settings, angular, time_order, *, full):
    height, speed, focal = settings.height_m, settings.speed_m_s, settings.focal_length_m
    dispersive = layout.family == "Dispersive imager"
    reads, pixels = scan_geometry(layout, read, settings)
    acquisition = along_track_acquisition(
        reads,
        LinearTrajectory(height_m=height, speed_m_s=speed),
        target_xy_m=np.zeros((1, 2)),
        pixel_bounds_m=pixels,
        along_edges_m=np.array([-25.0, 0.0, 25.0]) if dispersive else OFFSETS * 50.0,
        cross_edges_m=OFFSETS * 50.0 if full else np.array([-0.5, 0.5]),
        support_frame="platform" if dispersive else "ground",
        time_order=time_order,
        angular_order=angular,
        cross_order=angular if full else 1,
        focal_length_m=focal,
        pointing_law=lambda read_index, times_s: np.zeros((len(times_s), 2)),
    )
    time = acquisition.times_s[0]
    ground_x = acquisition.directions_uv[0, :, 0] * height + speed * time
    ground_y = acquisition.directions_uv[0, :, 1] * height
    return acquisition, ground_x, ground_y, np.unique(time)


def _optics(layout, read, settings, pixel_order):
    common = dict(
        focal_length_m=settings.focal_length_m,
        collecting_area_m2=np.pi * (settings.diameter_m / 2) ** 2,
        psf_sigma_m=settings.psf_sigma_m,
        throughput=settings.throughput,
        pixel_order=pixel_order,
    )
    family, stage = layout.family, layout.stage[read]
    centres = 1645 + (np.arange(8) + 0.5) * 1.25
    if family == "Uniform filter":
        return filter_imager(
            **common, filter=partial(gaussian_filter, centre_m=1650e-9, width_m=2.5e-9, peak=0.9)
        )
    if family == "Filter mosaic":
        # Full-width along-track stripes; each active pixel lies within one tile.
        def transmission(x, y, wave):
            column = int(np.clip(np.floor(x / settings.pitch_m + 4), 0, 7))
            return gaussian_filter(wave, centres[column] * 1e-9, 0.45e-9, peak=0.9)

        return spatial_filter_imager(**common, filter=transmission)
    if family == "LVF":
        return lvf_imager(
            **common,
            reference_m=1650e-9,
            gradient=10e-9 / (16 * settings.pitch_m),
            sigma_m=0.35e-9,
            peak=0.9,
        )
    if family == "Dispersive imager":
        return dispersive_imager(
            **common, reference_m=1650e-9, dispersion=32000.0, transmission=0.9
        )
    if family == "Two-beam interferometer":
        return interferometric_imager(
            **common,
            opd_m=np.linspace(0, 1.8e-3, 32)[stage],
            port=layout.ports[read],
            visibility=0.9,
            transmission=0.9,
        )
    return cavity_imager(
        **common,
        spacing_m=((1645 + (np.arange(8) + 0.5) * 1.25) * 1e-9 * 40)[stage],
        refractive_index=1.0,
        reflectivity=0.85,
        order_transmission=1.0,
    )
