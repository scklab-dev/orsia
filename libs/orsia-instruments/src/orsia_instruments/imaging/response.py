"""Positive bilinear scene-to-photoelectron responses and explicit valid correlation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid
from scipy.ndimage import correlate

from ..acquisition import (
    Acquisition,
    LinearTrajectory,
    ReadSchedule,
    along_track_acquisition,
    read_schedule,
)
from ..charge import ExpectedElectrons
from ..integration import Efficiency, QuantumEfficiency, photoelectron_response
from ..optics.pixels import gaussian_pixel_fraction
from ..optics.response import ImagingOptics, OpticalResponse


@dataclass(frozen=True, eq=False)
class GroundGrid:
    """Locate a ground-fixed radiance lattice for detector response evaluation.

    Attributes:
        along_m: Owned strictly increasing along-track knots in metres.
        cross_m: Owned strictly increasing cross-track knots in metres.
    """

    along_m: NDArray[np.float64]
    cross_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("along_m", "cross_m"):
            axis = np.array(getattr(self, name), dtype=np.float64, copy=True)
            _axis(axis)
            object.__setattr__(self, name, axis)


@dataclass(frozen=True, eq=False)
class SceneRadiance:
    """Ground-fixed per-cm-1 radiance (along,cross,spectral), in W m-2 sr-1.

    The spectral grid has the exact wavenumber support of the response. Arrays
    own writable float64 storage; keep them unchanged during response calls.

    Attributes:
        ground_grid: Along/cross metre support for the sampled radiance.
        spectral_grid: Wavenumber point support in cm-1.
        values: Owned (along,cross,spectral) radiance in W m-2 sr-1 per cm-1.
    """

    ground_grid: GroundGrid
    spectral_grid: SpectralGrid
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.shape != (
            len(self.ground_grid.along_m),
            len(self.ground_grid.cross_m),
            len(self.spectral_grid.points),
        ):
            raise ValueError("scene radiance must be (along,cross,spectral) on its grids")
        _positive(values, "scene radiance")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, eq=False)
class DetectorResponse:
    """Per-read spatio-spectral detector response from scene radiance to photoelectrons.

    coefficients[read,along,cross,wavenumber] multiply W m-2 sr-1 per cm-1
    radiance to give photoelectrons. Spatial axes and support bounds are metres
    in the caller's ground frame; bounds[read]=(xmin,xmax,ymin,ymax) enclose the
    complete integrated support. Reads retain native identities and exposure
    intervals. Arrays are independent writable float64 copies; grid is immutable.
    Keep buffers unchanged during reuse or concurrent calls. This positive K
    precedes detector/calibration; signed reconstruction belongs to its consumer.

    Attributes:
        reads: Native read identities and timing support.
        grid: Wavenumber sampling in cm-1 for the response.
        along_m: Increasing along-track ground knots in metres.
        cross_m: Increasing cross-track ground knots in metres.
        support_bounds_m: (reads,4) integrated ground support in metres.
        coefficients: (reads,along,cross,spectral) photoelectron response weights.
    """

    reads: ReadSchedule
    grid: SpectralGrid
    along_m: NDArray[np.float64]
    cross_m: NDArray[np.float64]
    support_bounds_m: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    def __post_init__(self) -> None:
        _snapshot_response(self)


@dataclass(frozen=True, eq=False)
class InstrumentResponse:
    """Along-track radiance-to-photoelectron coefficients on a ground-fixed lattice.

    Shares DetectorResponse's fields, physical units, validation and independent
    writable float64 array ownership. It identifies construction from a local
    along-track read layout, trajectory and quadrature rather than a supplied
    acquisition and ground projection. Keep arrays stable during reuse.

    Attributes:
        reads: Native read identities and timing support.
        grid: Wavenumber sampling in cm-1 for the response.
        along_m: Increasing along-track ground knots in metres.
        cross_m: Increasing cross-track ground knots in metres.
        support_bounds_m: (reads,4) integrated ground support in metres.
        coefficients: (reads,along,cross,spectral) photoelectron response weights.
    """

    reads: ReadSchedule
    grid: SpectralGrid
    along_m: NDArray[np.float64]
    cross_m: NDArray[np.float64]
    support_bounds_m: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    def __post_init__(self) -> None:
        _snapshot_response(self)


def detector_response(
    spectral_grid: SpectralGrid,
    acquisition: Acquisition,
    optical_response: OpticalResponse,
    *,
    ground_xy_m: NDArray[np.float64],
    along_m: NDArray[np.float64],
    cross_m: NDArray[np.float64],
    support_bounds_m: NDArray[np.float64],
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
) -> DetectorResponse:
    """Project optical acquisition and photon conversion onto positive bilinear hats.

    ground_xy_m[read,node,2] supplies ground coordinates corresponding exactly to
    Acquisition nodes. Explicit support_bounds_m must enclose the entire moving
    integration support, including exposure endpoints, and fit within the supplied
    ground axes. This declaration is caller-owned; sampled nodes are also checked.
    The spectral grid has one fully sampled declared cm-1 interval. No tail extension,
    extrapolation, renormalization or optical separability is assumed. Numerical
     work and retained K are dense float64 (reads,along,cross,k).
    The optical callback supplies accepted fractions; the returned detector K
    also includes collecting area, throughput, QE, photon energy, exposure and
    quadrature. This is a physical response before detector noise and digitization.
    Inputs are borrowed, outputs independent. Time-dependent radiance requires
    direct expected_electrons or separate responses for its actual time slices.

    Args:
        spectral_grid: Fully sampled connected wavenumber grid in cm-1.
        acquisition: Physical reads and time/direction quadrature nodes.
        optical_response: Callback supplying accepted optical fractions per node.
        ground_xy_m: (read,node,2) corresponding ground positions in metres.
        along_m: Increasing along-track ground-lattice coordinates in metres.
        cross_m: Increasing cross-track ground-lattice coordinates in metres.
        support_bounds_m: (reads,4) enclosing x/y spatial support in metres.
        collecting_area_m2: Projected collecting area in square metres.
        throughput: Fractions broadcastable across reads and spectral samples.
        quantum_efficiency: Fractions or read/wavelength QE callback.

    Returns:
        Physical detector response with owned (read,along,cross,wavenumber) weights.

    Raises:
        ValueError: Grid, support, response fractions or geometry is invalid.
    """
    kernel = _project_coefficients(
        spectral_grid,
        acquisition,
        optical_response,
        ground_xy_m=ground_xy_m,
        along_m=along_m,
        cross_m=cross_m,
        support_bounds_m=support_bounds_m,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    return DetectorResponse(
        read_schedule(acquisition), spectral_grid, along_m, cross_m, support_bounds_m, kernel
    )


def instrument_response(
    spectral_grid: SpectralGrid,
    reads: ReadSchedule[np.float64],
    trajectory: LinearTrajectory,
    optical_factory: Callable[[Acquisition], OpticalResponse],
    *,
    target_xy_m: NDArray[np.float64],
    pixel_bounds_m: NDArray[np.float64],
    along_edges_m: NDArray[np.float64],
    cross_edges_m: NDArray[np.float64],
    ground_grid: GroundGrid,
    focal_length_m: float,
    collecting_area_m2: float,
    pointing_law: Callable[[int, NDArray[np.float64]], NDArray[np.float64]],
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
    support_frame: str = "ground",
    cross_offset_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    angular_order: int = 5,
    cross_order: int | None = None,
) -> InstrumentResponse:
    """Construct a local scan's positive ground response on explicit support.

    The factory receives the full Acquisition, allowing genuinely joint optics or
    an explicitly selected separable optimization. Ground coordinates and the
    complete swept support are derived from that same scan (metres). The factory
    is borrowed and must be pure and stable.

    Args:
        spectral_grid: Fully sampled wavenumber support in cm-1.
        reads: Ordered physical read identities and exposure intervals in seconds.
        trajectory: Local constant-height/speed scan and time origin.
        optical_factory: Pure acquisition-to-optical-response callback.
        target_xy_m: (reads,2) ground target coordinates in metres.
        pixel_bounds_m: (reads,4) detector bounds in focal-plane metres.
        along_edges_m: Piecewise along-track ground support edges in metres.
        cross_edges_m: Piecewise target-relative cross-track support edges in metres.
        ground_grid: Along/cross response lattice in ground metres.
        focal_length_m: Image-plane focal length in metres.
        collecting_area_m2: Projected collecting area in square metres.
        pointing_law: Read-index/time-node callback returning (nodes,2) local slopes.
        throughput: Per-read/spectral physical optical transmission fractions.
        quantum_efficiency: Per-read/spectral fraction or QE callback.
        support_frame: 'ground' target-fixed or 'platform' moving along support.
        cross_offset_m: Optional (reads,) cross ground offset from column origins in metres.
        time_order: Exposure-time Gauss order.
        angular_order: Along-track spatial Gauss order.
        cross_order: Optional cross-track spatial Gauss order.

    Returns:
        Local scan response with copied physical reads and dense positive coefficients.

    Raises:
        ValueError: Scan geometry, spectral/ground support or optical fractions fail.
    """
    acquisition = along_track_acquisition(
        reads,
        trajectory,
        target_xy_m=target_xy_m,
        pixel_bounds_m=pixel_bounds_m,
        along_edges_m=along_edges_m,
        cross_edges_m=cross_edges_m,
        focal_length_m=focal_length_m,
        support_frame=support_frame,
        cross_offset_m=cross_offset_m,
        pointing_law=pointing_law,
        time_order=time_order,
        angular_order=angular_order,
        cross_order=cross_order,
    )
    xy = acquisition.directions_uv * trajectory.height_m
    xy[..., 0] += trajectory.speed_m_s * (acquisition.times_s - trajectory.time_origin_s)
    cross = np.zeros(len(reads.sample_ids)) if cross_offset_m is None else cross_offset_m
    xy[..., 1] += (target_xy_m[:, 1] - cross)[:, None]
    x0, x1 = along_edges_m[[0, -1]]
    y0, y1 = cross_edges_m[[0, -1]]
    if support_frame == "ground":
        along_low = target_xy_m[:, 0] + x0
        along_high = target_xy_m[:, 0] + x1
    else:
        centre = acquisition.exposure_bounds_s.mean(axis=1)
        along_offset = target_xy_m[:, 0] - trajectory.speed_m_s * (
            centre - trajectory.time_origin_s
        )
        along_low = (
            trajectory.speed_m_s * (acquisition.exposure_bounds_s[:, 0] - trajectory.time_origin_s)
            + along_offset
            + x0
        )
        along_high = (
            trajectory.speed_m_s * (acquisition.exposure_bounds_s[:, 1] - trajectory.time_origin_s)
            + along_offset
            + x1
        )
    support_bounds = np.column_stack(
        (
            along_low,
            along_high,
            target_xy_m[:, 1] + y0,
            target_xy_m[:, 1] + y1,
        )
    )
    kernel = _project_coefficients(
        spectral_grid,
        acquisition,
        optical_factory(acquisition),
        ground_xy_m=xy,
        along_m=ground_grid.along_m,
        cross_m=ground_grid.cross_m,
        support_bounds_m=support_bounds,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    return InstrumentResponse(
        read_schedule(acquisition),
        spectral_grid,
        ground_grid.along_m,
        ground_grid.cross_m,
        support_bounds,
        kernel,
    )


def gaussian_cross_track_response(
    line: Acquisition, optics: ImagingOptics, *, sigma_m: float
) -> Callable[[Acquisition], OpticalResponse]:
    """Lift a single cross-track-centred optical line onto a Gaussian scan.

    The caller asserts that the optical law and any spectral mapping have no
    cross-track dependence beyond a separable, wavelength-independent Gaussian
    PSF integrated over the same rectangular pixel. ``line`` has one read with
    reference-cross centroids; the full acquisition has identical time/x nodes with each
    line node repeated for every cross-track node. Both are borrowed and must
    remain unchanged while the returned factory/callback is used. The result is
    an accepted fraction (full nodes, requested wavelengths), excluding area, QE,
    throughput and quadrature; it retains the finite pixel's lost Gaussian tails.
    General joint optics should pass their response directly to the scan composer.

    Args:
        line: Single-read, cross-track-centred acquisition for reference optics.
        optics: Borrowed imaging response evaluated on the line's nodes.
        sigma_m: Positive cross-track Gaussian PSF width in detector metres.

    Returns:
        Factory accepting a matching full acquisition and yielding an optical
        accepted-fraction callback for its cross-track nodes and wavelengths.

    Raises:
        ValueError: Line or sigma is invalid, or a later acquisition differs.
    """
    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("Gaussian scan sigma must be finite and positive")
    if line.times_s.shape[0] != 1:
        raise ValueError("optical line must have one read")

    def on_acquisition(acquisition: Acquisition) -> OpticalResponse:
        count = line.times_s.shape[1]
        nodes = acquisition.times_s.shape[1]
        if (
            acquisition.times_s.shape[0] != 1
            or nodes % count != 0
            or not np.array_equal(acquisition.pixel_bounds_m, line.pixel_bounds_m)
            or not np.array_equal(acquisition.exposure_bounds_s, line.exposure_bounds_s)
            or not np.all(acquisition.times_s[0].reshape(count, -1) == line.times_s[0, :, None])
            or not np.all(
                acquisition.image_xy_m[0, :, 0].reshape(count, -1) == line.image_xy_m[0, :, 0, None]
            )
        ):
            raise ValueError("scan nodes must repeat the optical line in time and along track")
        cross = nodes // count
        bounds = line.pixel_bounds_m[0]
        y = acquisition.image_xy_m[0, :, 1]
        x_centre = (bounds[0] + bounds[1]) / 2
        centres = np.column_stack((np.full(nodes, x_centre), y))
        reference = np.column_stack((np.full(count, x_centre), line.image_xy_m[0, :, 1]))
        reference_fraction = gaussian_pixel_fraction(bounds, reference, sigma_m)
        if not np.isfinite(reference_fraction).all() or np.any(reference_fraction <= 0):
            raise ValueError("Gaussian reference acceptance must be finite and positive")
        factor = gaussian_pixel_fraction(bounds, centres, sigma_m) / np.repeat(
            reference_fraction, cross
        )

        def accepted(
            read: int, wave: NDArray[np.float64], current: Acquisition
        ) -> NDArray[np.float64]:
            if read != 0 or current is not acquisition:
                raise ValueError("Gaussian scan response belongs to its supplied acquisition")
            return np.repeat(optics.response(0, 0, wave, line), cross, axis=0) * factor[:, None]

        return accepted

    return on_acquisition


def gaussian_along_track_response(
    spectral_grid: SpectralGrid,
    reads: ReadSchedule[np.float64],
    trajectory: LinearTrajectory,
    optics_for_read: Callable[[int], ImagingOptics],
    *,
    pixel_bounds_m: NDArray[np.float64],
    along_edges_m: NDArray[np.float64],
    cross_edges_m: NDArray[np.float64],
    ground_grid: GroundGrid,
    focal_length_m: float,
    sigma_m: float,
    pointing_law: Callable[[int, NDArray[np.float64]], NDArray[np.float64]],
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
    support_frame: str = "ground",
    cross_offset_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    angular_order: int = 5,
    cross_order: int | None = None,
) -> InstrumentResponse:
    """Stack single-read Gaussian scan stencils into one positive physical response.

    Reads define ordered template identities and exposure intervals, with detector
    bounds supplied separately; each target is centred at ground (0,0). Both spatial
    axes of ground_grid are relative to that centre in metres. Each optical law
    supplied by optics_for_read(index) must satisfy the cross-track Gaussian
    separability contract of gaussian_cross_track_response, including
    wavelength-independent sigma_m in detector metres. The per-read optical area
    and throughput enter once; QE enters once here. Cross-track line integration
    uses one symmetric interval around zero and one Gauss node. The supplied
    cross-track quadrature supplies the full response support and projection.
    Returned arrays and physical read times are independent writable snapshots;
    patterns, optics, grid and callbacks are borrowed and remain stable during use.
    Stacking retains a dense (template,along,cross,spectral) float64 K.

    Args:
        spectral_grid: Fully sampled wavenumber grid in cm-1.
        reads: Ordered strip read identities and timed exposures.
        trajectory: Local constant-height/speed scan and time origin.
        optics_for_read: Callback selecting an imaging law for each template index.
        pixel_bounds_m: (reads,4) detector rectangles in metres.
        along_edges_m: Piecewise along edges in ground metres.
        cross_edges_m: Target-relative cross support edges in ground metres.
        ground_grid: Along/cross response lattice in local ground metres.
        focal_length_m: Image-plane focal length in metres.
        sigma_m: Wavelength-independent cross-track PSF width in detector metres.
        pointing_law: Global read-index/time-node local-slope callback.
        quantum_efficiency: Physical QE fractions or read/wavelength callback.
        support_frame: 'ground' fixed or 'platform' moving along spatial support.
        cross_offset_m: Optional (reads,) cross ground offset from column origin in metres.
        time_order: Exposure Gauss order.
        angular_order: Along spatial Gauss order.
        cross_order: Optional cross spatial Gauss order.

    Returns:
        Stacked physical scan response with independent read times and coefficients.

    Raises:
        ValueError: Templates, Gaussian separation or spatial support is invalid.
    """
    count = len(reads.sample_ids)
    if pixel_bounds_m.shape != (count, 4):
        raise ValueError("Gaussian templates require (reads,4) pixel bounds")
    if cross_offset_m is not None and cross_offset_m.shape != (count,):
        raise ValueError("Gaussian templates require (reads,) cross offsets")
    coefficients = []
    supports = []
    bounds = []
    for read in range(count):
        local_reads = ReadSchedule(
            (reads.sample_ids[read],),
            (reads.pixel_ids[read],),
            (reads.setting_ids[read],),
            (reads.port_ids[read],),
            reads.exposure_bounds_s[read : read + 1],
        )

        def local_pointing(
            _read: int, times_s: NDArray[np.float64], read_index: int = read
        ) -> NDArray[np.float64]:
            return pointing_law(read_index, times_s)

        cross = None if cross_offset_m is None else cross_offset_m[read : read + 1]
        line = along_track_acquisition(
            local_reads,
            trajectory,
            target_xy_m=np.zeros((1, 2)),
            pixel_bounds_m=pixel_bounds_m[read : read + 1],
            along_edges_m=along_edges_m,
            cross_edges_m=np.array([-0.5, 0.5]),
            support_frame=support_frame,
            cross_offset_m=cross,
            pointing_law=local_pointing,
            time_order=time_order,
            angular_order=angular_order,
            cross_order=1,
            focal_length_m=focal_length_m,
        )
        optics = optics_for_read(read)
        physical = instrument_response(
            spectral_grid,
            local_reads,
            trajectory,
            gaussian_cross_track_response(line, optics, sigma_m=sigma_m),
            target_xy_m=np.zeros((1, 2)),
            pixel_bounds_m=pixel_bounds_m[read : read + 1],
            along_edges_m=along_edges_m,
            cross_edges_m=cross_edges_m,
            support_frame=support_frame,
            cross_offset_m=cross,
            pointing_law=local_pointing,
            time_order=time_order,
            angular_order=angular_order,
            cross_order=cross_order,
            ground_grid=ground_grid,
            focal_length_m=focal_length_m,
            collecting_area_m2=optics.collecting_area_m2,
            throughput=optics.throughput,
            quantum_efficiency=quantum_efficiency,
        )
        coefficients.append(physical.coefficients[0])
        supports.append(physical.support_bounds_m[0])
        bounds.append(physical.reads.exposure_bounds_s[0])
    reads = ReadSchedule(
        reads.sample_ids,
        reads.pixel_ids,
        reads.setting_ids,
        reads.port_ids,
        np.stack(bounds),
    )
    return InstrumentResponse(
        reads,
        spectral_grid,
        ground_grid.along_m,
        ground_grid.cross_m,
        np.stack(supports),
        np.stack(coefficients),
    )


def response_electrons(
    response: DetectorResponse | InstrumentResponse,
    scene: NDArray[np.float64],
    *,
    spectra: NDArray[np.float64] | None = None,
    scene_along_m: NDArray[np.float64] | None = None,
    scene_cross_m: NDArray[np.float64] | None = None,
    target_indices: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
    reads: ReadSchedule | None = None,
    method: str = "direct",
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Integrate scene radiance to expected electrons using a constructed response K.

    Apply K directly, or reuse a caller-declared translation-invariant stencil.
    Both response types represent the same positive K, despite their distinct
    construction paths. Default scene axes are (along,cross,k) on its exact lattice/grid.
    Alternatively scene[factor,along,cross] and spectra[factor,k] describe
    a nonnegative mathematical factorization on the response's spectral grid.

    Supplying target_indices=(row_indices,column_indices) declares stationary
    scene, uniform translation-invariant acquisition/optics and a stencil centred
    at (0,0). Explicit scene axes must be uniform with the stencil spacing. Every
    requested target's entire stencil needs a supplied halo; padding is rejected.
    method='correlation' reuses each stencil; method='direct' sums each patch as
    an independent reference. Output read order is (template,target_row,target_col)
    flattened C-order. Caller-supplied reads are mandatory and carry real physical
    timing/pixels; setting/port IDs and durations must match their template.
    This mathematical reuse leaves temporal stationarity/optical invariance and
    spectral-grid agreement as explicit caller assertions. Input arrays are
    borrowed and must remain stable during use; returned ExpectedElectrons owns
    signal storage. No interpolation or signed mixing occurs.

    Args:
        response: Physical detector or instrument response with positive weights.
        scene: Nonnegative (along,cross,spectral) radiance, or (factor,along,cross)
            maps when spectra are supplied.
        spectra: Optional nonnegative (factor,spectral) component spectra.
        scene_along_m: Explicit along-track scene axis for translated stencils.
        scene_cross_m: Explicit cross-track scene axis for translated stencils.
        target_indices: Optional row/column index vectors for complete stencil halos.
        reads: Explicit native reads with matched timing for translation mode.
        method: Direct summation or correlation reuse for translated stencils.
        source_name: Label of the resulting expected-electron contribution.

    Returns:
        Source-labelled expected electrons in physical-read order with owned counts.

    Raises:
        ValueError: Shapes, support, read timing, halo or application mode is invalid.
    """
    if method not in ("direct", "correlation"):
        raise ValueError("direct/correlation method required")
    _positive(scene, "scene radiance or factor maps")
    k = len(response.grid.points)
    if spectra is None:
        if scene.ndim != 3 or scene.shape[-1] != k:
            raise ValueError("scene must be (along,cross,spectral)")
        maps = np.moveaxis(scene, -1, 0)
    else:
        _positive(spectra, "component spectra")
        if scene.ndim != 3 or spectra.shape != (len(scene), k):
            raise ValueError("maps/spectra must be (factor,along,cross)/(factor,spectral)")
        maps = scene
    if target_indices is None:
        if (
            method != "direct"
            or reads is not None
            or scene_along_m is not None
            or scene_cross_m is not None
        ):
            raise ValueError("translation options require explicit target indices")
        if maps.shape[1:] != (len(response.along_m), len(response.cross_m)):
            raise ValueError("scene ground lattice disagrees with response")
        if spectra is None:
            counts = np.einsum("rijk,ijk->r", response.coefficients, scene)
        else:
            counts = np.einsum("rijk,cij,ck->r", response.coefficients, maps, spectra)
        return ExpectedElectrons(response.reads, {source_name: counts})
    if scene_along_m is None or scene_cross_m is None or reads is None:
        raise ValueError("translation requires explicit scene axes and native reads")
    rows, columns = target_indices
    half_x = _stencil(response.along_m, scene_along_m)
    half_y = _stencil(response.cross_m, scene_cross_m)
    if maps.shape[1:] != (len(scene_along_m), len(scene_cross_m)):
        raise ValueError("scene axes disagree with maps")
    for indices, half, size in ((rows, half_x, maps.shape[1]), (columns, half_y, maps.shape[2])):
        if (
            indices.ndim != 1
            or not len(indices)
            or np.any(indices < half)
            or np.any(indices >= size - half)
        ):
            raise ValueError("every target requires complete explicit scene halo")
    shape = (len(response.reads.sample_ids), len(rows), len(columns))
    if len(reads.sample_ids) != np.prod(shape):
        raise ValueError("native reads must match (template,target_row,target_col)")
    repeat = len(rows) * len(columns)
    for name in ("setting_ids", "port_ids"):
        expected = tuple(value for value in getattr(response.reads, name) for _ in range(repeat))
        if getattr(reads, name) != expected:
            raise ValueError("native settings/ports disagree with response templates")
    durations = np.repeat(np.diff(response.reads.exposure_bounds_s, axis=1).ravel(), repeat)
    if not np.allclose(
        np.diff(reads.exposure_bounds_s, axis=1).ravel(), durations, rtol=1e-11, atol=1e-14
    ):
        raise ValueError("native exposure durations disagree with response templates")
    if spectra is None:
        stencils = np.moveaxis(response.coefficients, -1, 1)
        local_maps = maps
    else:
        stencils = np.einsum("rijk,ck->rcij", response.coefficients, spectra)
        local_maps = maps
    result = np.empty(shape)
    if method == "correlation":
        for read, stencil_set in enumerate(stencils):
            image = np.zeros(maps.shape[1:])
            for local_image, stencil in zip(local_maps, stencil_set, strict=True):
                image += correlate(local_image, stencil, mode="constant")
            result[read] = image[np.ix_(rows, columns)]
    else:
        # Sliding windows expose all target patches for a single dense contraction.
        from numpy.lib.stride_tricks import sliding_window_view

        patches = sliding_window_view(local_maps, (2 * half_x + 1, 2 * half_y + 1), axis=(1, 2))[
            :, rows[:, None] - half_x, columns[None, :] - half_y
        ]
        result[:] = np.einsum("cijxy,rcxy->rij", patches, stencils)
    return ExpectedElectrons(reads, {source_name: result.ravel()})


def _snapshot_response(response: DetectorResponse | InstrumentResponse) -> None:
    object.__setattr__(
        response,
        "reads",
        ReadSchedule(
            response.reads.sample_ids,
            response.reads.pixel_ids,
            response.reads.setting_ids,
            response.reads.port_ids,
            np.asarray(response.reads.exposure_bounds_s, dtype=np.float64),
        ),
    )
    if (
        response.grid.coordinate != "wavenumber_cm1"
        or response.grid.domain is None
        or len(response.grid.points) < 2
        or response.grid.domain.intervals != ((response.grid.points[0], response.grid.points[-1]),)
    ):
        raise ValueError("response grid needs one fully sampled declared cm-1 interval")
    for name in ("along_m", "cross_m", "support_bounds_m", "coefficients"):
        object.__setattr__(
            response, name, np.array(getattr(response, name), dtype=np.float64, copy=True)
        )
    _axis(response.along_m)
    _axis(response.cross_m)
    _support(
        response.support_bounds_m,
        response.along_m,
        response.cross_m,
        len(response.reads.sample_ids),
    )
    shape = (
        len(response.reads.sample_ids),
        len(response.along_m),
        len(response.cross_m),
        len(response.grid.points),
    )
    if response.coefficients.shape != shape:
        raise ValueError("coefficients must be (reads,along,cross,spectral)")
    _positive(response.coefficients, "physical coefficients")


def _project_coefficients(
    spectral_grid: SpectralGrid,
    acquisition: Acquisition,
    optical_response: OpticalResponse,
    *,
    ground_xy_m: NDArray[np.float64],
    along_m: NDArray[np.float64],
    cross_m: NDArray[np.float64],
    support_bounds_m: NDArray[np.float64],
    collecting_area_m2: float,
    throughput: Efficiency,
    quantum_efficiency: Efficiency | QuantumEfficiency,
) -> NDArray[np.float64]:
    """Project acquired photoelectron weights onto the declared ground grid.

    Returns:
        Fresh dense K with axes (read, along, cross, spectral), owned by the caller.

    Raises:
        ValueError: Ground nodes or declared support are invalid or lack grid coverage.
    """
    _axis(along_m)
    _axis(cross_m)
    n, nodes = acquisition.times_s.shape
    _support(support_bounds_m, along_m, cross_m, n)
    if ground_xy_m.shape != (n, nodes, 2) or not np.isfinite(ground_xy_m).all():
        raise ValueError("ground coordinates must be finite (reads,nodes,2)")
    if np.any(ground_xy_m < support_bounds_m[:, None, [0, 2]]) or np.any(
        ground_xy_m > support_bounds_m[:, None, [1, 3]]
    ):
        raise ValueError("ground nodes lie outside declared support")
    ix, ax = _brackets(along_m, ground_xy_m[..., 0])
    iy, ay = _brackets(cross_m, ground_xy_m[..., 1])
    kernel = np.zeros((n, len(along_m), len(cross_m), len(spectral_grid.points)))
    coefficients = photoelectron_response(
        spectral_grid,
        acquisition,
        optical_response,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    for dx, wx in ((0, 1 - ax), (1, ax)):
        for dy, wy in ((0, 1 - ay), (1, ay)):
            np.add.at(
                kernel,
                (np.arange(n)[:, None], ix + dx, iy + dy),
                coefficients * (wx * wy)[..., None],
            )
    return kernel


def _axis(axis: NDArray[np.float64]) -> None:
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError("ground axes must be finite strictly increasing vectors")


def _positive(array: NDArray[np.float64], name: str) -> None:
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _support(
    bounds: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64], n: int
) -> None:
    if (
        bounds.shape != (n, 4)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, [1, 3]] <= bounds[:, [0, 2]])
    ):
        raise ValueError("ground support must be finite ordered (reads,4)")
    if np.any(bounds[:, [0, 2]] < [x[0], y[0]]) or np.any(bounds[:, [1, 3]] > [x[-1], y[-1]]):
        raise ValueError("entire ground support requires explicit scene coverage")


def _brackets(
    axis: NDArray[np.float64], points: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    index = np.minimum(np.searchsorted(axis, points, side="right") - 1, len(axis) - 2).astype(
        np.int64
    )
    return index, (points - axis[index]) / (axis[index + 1] - axis[index])


def _stencil(offsets: NDArray[np.float64], scene_axis: NDArray[np.float64]) -> int:
    _axis(scene_axis)
    spacing = scene_axis[1] - scene_axis[0]
    half = len(offsets) // 2
    if (
        len(offsets) % 2 != 1
        or not np.allclose(
            offsets, (np.arange(len(offsets)) - half) * spacing, rtol=1e-12, atol=1e-12
        )
        or not np.allclose(np.diff(scene_axis), spacing, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("correlation requires uniform matching centred odd stencils")
    return half
