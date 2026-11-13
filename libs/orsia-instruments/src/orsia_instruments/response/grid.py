"""Positive bilinear scene-to-photoelectron responses and explicit valid correlation."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import PointLattice, same_local_frame
from orsia_radiative import SpectralGrid
from scipy.ndimage import correlate

from orsia_instruments.acquisition.quadrature import (
    Acquisition,
    along_track_acquisition,
    read_schedule,
)
from orsia_instruments.acquisition.schedule import LinearTrajectory, ReadSchedule
from orsia_instruments.optics.response import ImagingOptics, OpticalResponse
from orsia_instruments.photodetection.charge import ExpectedElectrons
from orsia_instruments.photodetection.coefficients import (
    Efficiency,
    QuantumEfficiency,
    photoelectron_response,
    photon_conversion_weights,
    sample_optical_acceptance,
)
from orsia_instruments.sampling.surface import SurfaceRadianceSamples


@dataclass(frozen=True, eq=False)
class GridElectronResponse:
    """Own a positive radiance-to-electron map on a declared point lattice.

    Attributes:
        reads: Shared immutable native read identities and exposure intervals.
        grid: Shared spectral samples and full declared integration support, nm or cm-1.
        lattice: Shared immutable x/y point coordinates and local frame. The basis
            is bilinear interpolation; both axes require at least two knots.
        support_bounds_m: Independent writable (read,4) complete integration bounds
            (xmin,xmax,ymin,ymax), metres in lattice.frame, including unsampled support.
        coefficients: Independent writable float64 (read,y,x,spectral) electrons per
            W/m2/sr per grid coordinate, including all declared quadrature and physical
            factors once. Keep numerical buffers stable during reuse/concurrent calls.
            Construction route does not change this physical response concept.
    """

    reads: ReadSchedule
    grid: SpectralGrid
    lattice: PointLattice
    support_bounds_m: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate full physical support and snapshot the numerical response.

        Raises:
            ValueError: Coordinates, coverage, coefficient axes or physical values fail.
        """
        _snapshot_response(self)


def detector_response(
    spectral_grid: SpectralGrid,
    acquisition: Acquisition,
    optical_response: OpticalResponse,
    *,
    ground_xy_m: NDArray[np.float64],
    lattice: PointLattice,
    support_bounds_m: NDArray[np.float64],
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
) -> GridElectronResponse:
    """Project optical acquisition and photon conversion onto positive bilinear hats.

    ground_xy_m[read,node,2] supplies ground coordinates corresponding exactly to
    Acquisition nodes. Explicit support_bounds_m must enclose the entire moving
    integration support, including exposure endpoints, and fit within the supplied
    ground axes. This declaration is caller-owned; sampled nodes are also checked.
    The spectral grid has fully endpoint-sampled declared nm or cm-1 intervals. No tail extension,
    extrapolation, renormalization or optical separability is assumed. Numerical
     work and retained K are dense float64 (reads,y,x,k).
    The optical callback supplies accepted fractions; the returned detector K
    also includes collecting area, throughput, QE, photon energy, exposure and
    quadrature. This is a physical response before detector noise and digitization.
    Numerical outputs are independent; immutable lattice/reads are shared.
    Time-dependent radiance requires
    direct expected_electrons or separate responses for its actual time slices.

    Args:
        spectral_grid: Endpoint-sampled declared spectral grid in nm or cm-1.
        acquisition: Physical reads and time/direction quadrature nodes.
        optical_response: Callback supplying accepted optical fractions per node.
        ground_xy_m: (read,node,2) corresponding ground positions in metres.
        lattice: Explicit point support/frame. ground_xy_m uses its (x,y) coordinates.
        support_bounds_m: (reads,4) enclosing x/y spatial support in metres.
        collecting_area_m2: Projected collecting area in square metres.
        throughput: Fractions broadcastable across reads and spectral samples.
        quantum_efficiency: Fractions or read/wavelength QE callback.

    Returns:
        Physical detector response with owned (read,y,x,spectral) weights.

    Raises:
        ValueError: Grid, support, response fractions or geometry is invalid.
    """
    kernel = _project_coefficients(
        spectral_grid,
        acquisition,
        optical_response,
        ground_xy_m=ground_xy_m,
        lattice=lattice,
        support_bounds_m=support_bounds_m,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    return GridElectronResponse(
        read_schedule(acquisition), spectral_grid, lattice, support_bounds_m, kernel
    )


def instrument_response(
    spectral_grid: SpectralGrid,
    reads: ReadSchedule[np.float64],
    trajectory: LinearTrajectory,
    optical_factory: Callable[[Acquisition], OpticalResponse] | None = None,
    *,
    optical_response: OpticalResponse | None = None,
    target_xy_m: NDArray[np.float64],
    pixel_bounds_m: NDArray[np.float64],
    along_edges_m: NDArray[np.float64],
    cross_edges_m: NDArray[np.float64],
    lattice: PointLattice,
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
) -> GridElectronResponse:
    """Construct a local scan's positive ground response on explicit support.

    Supply optical_response directly for ordinary per-read optics. An optional
    optical_factory receives the complete Acquisition for acquisition-dependent
    preparation. Supply exactly one of these alternatives. Optical callbacks are
    borrowed, pure and stable. Ground coordinates and the complete swept support
    are derived from the same scan (metres).

    Args:
        spectral_grid: Endpoint-sampled declared spectral support in nm or cm-1.
        reads: Ordered physical read identities and exposure intervals in seconds.
        trajectory: Local constant-height/speed scan and time origin.
        optical_factory: Optional acquisition-dependent preparation callback;
            existing positional and keyword calls remain supported.
        optical_response: Direct per-read response(read,wavelengths_m,acquisition),
            excluding area, throughput and QE. Mutually exclusive with optical_factory.
        target_xy_m: (reads,2) ground target coordinates in metres.
        pixel_bounds_m: (reads,4) detector bounds in focal-plane metres.
        along_edges_m: Piecewise along-track ground support edges in metres.
        cross_edges_m: Piecewise target-relative cross-track support edges in metres.
        lattice: Explicit response point support; x is along-track and y is cross-track.
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
        Local scan response retaining immutable reads and independent positive coefficients.

    Raises:
        ValueError: Optical selection is missing or ambiguous, or scan geometry,
            spectral/ground support or optical fractions fail.
    """
    if (optical_factory is None) == (optical_response is None):
        raise ValueError("supply exactly one of optical_response or optical_factory")
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
    if optical_factory is not None:
        optical_response = optical_factory(acquisition)
    assert optical_response is not None
    kernel = _project_coefficients(
        spectral_grid,
        acquisition,
        optical_response,
        ground_xy_m=xy,
        lattice=lattice,
        support_bounds_m=support_bounds,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    return GridElectronResponse(
        read_schedule(acquisition),
        spectral_grid,
        lattice,
        support_bounds,
        kernel,
    )


def along_track_response(
    spectral_grid: SpectralGrid,
    reads: ReadSchedule[np.float64],
    trajectory: LinearTrajectory,
    optics_for_read: Callable[[int], ImagingOptics],
    *,
    pixel_bounds_m: NDArray[np.float64],
    along_edges_m: NDArray[np.float64],
    cross_edges_m: NDArray[np.float64],
    lattice: PointLattice,
    focal_length_m: float,
    pointing_law: Callable[[int, NDArray[np.float64]], NDArray[np.float64]],
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
    support_frame: str = "ground",
    cross_offset_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    angular_order: int = 5,
    cross_order: int | None = None,
) -> GridElectronResponse:
    """Stack per-read optical scan stencils into one positive physical response.

    Reads define ordered template identities and exposure intervals, with detector
    bounds supplied separately; each target is centred at ground (0,0). Both spatial
    axes of lattice are relative to that centre in metres. Each optical law
    supplied by optics_for_read(index) defines its PSF and is evaluated on every
    node of the full acquisition through optics.response. Configure the spatial
    response on the supplied optics. The per-read optical area and throughput enter once;
    QE enters once here. The supplied cross-track quadrature defines the full
    response support and projection.
    Response arrays are independent writable snapshots; physical read times remain
    in the immutable read schedule. Patterns, optics, grid and callbacks are borrowed
    and remain stable during use.
    Stacking retains a dense (template,y,x,spectral) float64 K.

    Args:
        spectral_grid: Endpoint-sampled declared spectral support in nm or cm-1.
        reads: Ordered strip read identities and timed exposures.
        trajectory: Local constant-height/speed scan and time origin.
        optics_for_read: Callback selecting an imaging law for each template index.
        pixel_bounds_m: (reads,4) detector rectangles in metres.
        along_edges_m: Piecewise along edges in ground metres.
        cross_edges_m: Target-relative cross support edges in ground metres.
        lattice: Explicit local point support/frame; x is along-track, y is cross-track.
        focal_length_m: Image-plane focal length in metres.
        pointing_law: Global read-index/time-node local-slope callback.
        quantum_efficiency: Physical fractions broadcastable to (reads,spectral),
            or a pure callback evaluated once with all original read indices and
            wavelengths in metres, returning exactly (reads,spectral).
        support_frame: 'ground' fixed or 'platform' moving along spatial support.
        cross_offset_m: Optional (reads,) cross ground offset from column origin in metres.
        time_order: Exposure Gauss order.
        angular_order: Along spatial Gauss order.
        cross_order: Optional cross spatial Gauss order.

    Returns:
        Stacked physical scan response with independent read times and coefficients.

    Raises:
        ValueError: Templates, QE, optical inputs or spatial support is invalid.
    """
    count = len(reads.sample_ids)
    if pixel_bounds_m.shape != (count, 4):
        raise ValueError("Along-track templates require (reads,4) pixel bounds")
    if cross_offset_m is not None and cross_offset_m.shape != (count,):
        raise ValueError("Along-track templates require (reads,) cross offsets")
    qe_shape = (count, len(spectral_grid.points))
    if callable(quantum_efficiency):
        qe_values = quantum_efficiency(
            np.arange(count, dtype=np.int64),
            0.01 / spectral_grid.points
            if spectral_grid.coordinate == "wavenumber_cm1"
            else spectral_grid.points * 1e-9,
        )
        if qe_values.shape != qe_shape:
            raise ValueError("QE callback must return (reads,spectral)")
    else:
        qe_values = quantum_efficiency
    qe = np.asarray(qe_values, dtype=np.float64)
    if not np.isfinite(qe).all() or np.any((qe < 0) | (qe > 1)):
        raise ValueError("QE must be finite and in [0, 1]")
    qe = np.broadcast_to(qe, qe_shape).copy()
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
        optics = optics_for_read(read)
        physical = instrument_response(
            spectral_grid,
            local_reads,
            trajectory,
            optical_response=partial(optics.response, 0),
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
            lattice=lattice,
            focal_length_m=focal_length_m,
            collecting_area_m2=optics.collecting_area_m2,
            throughput=optics.throughput,
            quantum_efficiency=qe[read : read + 1],
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
    return GridElectronResponse(
        reads,
        spectral_grid,
        lattice,
        np.stack(supports),
        np.stack(coefficients),
    )


def response_electrons(
    response: GridElectronResponse,
    scene: NDArray[np.float64] | SurfaceRadianceSamples,
    *,
    spectra: NDArray[np.float64] | None = None,
    scene_lattice: PointLattice | None = None,
    target_indices: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
    reads: ReadSchedule | None = None,
    method: str = "direct",
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Integrate scene radiance to expected electrons using a constructed response K.

    Apply K directly, or reuse a caller-declared translation-invariant stencil.
    Values have (y,x,k) axes on the response's exact lattice/grid. A supplied
    SurfaceRadianceSamples also verifies coordinate/frame and spectral identity.
    Alternatively scene[factor,y,x] and spectra[factor,k] describe
    a nonnegative mathematical factorization on the response's spectral grid.

    Supplying target_indices=(y_indices,x_indices) declares stationary
    scene, uniform translation-invariant acquisition/optics and a stencil centred
    at (0,0). Explicit scene axes must be uniform with the stencil spacing. Every
    requested target's entire stencil needs a supplied halo; padding is rejected.
    method='correlation' reuses each stencil; method='direct' sums each patch as
    an independent reference. Output scan-native read order is (template,target_x,target_y)
    flattened C-order. Caller-supplied reads are mandatory and carry real physical
    timing/pixels; setting/port IDs and durations must match their template.
    This mathematical reuse leaves temporal stationarity/optical invariance and
    spectral-grid agreement as explicit caller assertions. Input arrays are
    borrowed and must remain stable during use; returned ExpectedElectrons owns
    signal storage. No interpolation or signed mixing occurs.

    Args:
        response: Physical grid response with positive quadrature-inclusive weights.
        scene: Supported radiance record, raw (y,x,spectral) values, or (factor,y,x) maps.
            Raw arrays assert the response spectral sampling and declared point support.
        spectra: Optional nonnegative (factor,spectral) component spectra.
        scene_lattice: Explicit scene points/frame, required for translated raw arrays.
            Sampled records supply it; direct records must exactly match the response.
        target_indices: Optional (y,x) index vectors with complete stencil halos.
            Native output remains template, then x target, then y target, matching scans.
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
    if isinstance(scene, SurfaceRadianceSamples):
        if spectra is not None or not scene.spectral_grid.same_points(response.grid):
            raise ValueError(
                "sampled radiance requires matching spectral points without factor spectra"
            )
        if scene_lattice is not None and not scene_lattice.matches(scene.lattice):
            raise ValueError("explicit scene lattice differs from radiance support")
        scene_lattice = scene.lattice
        scene = scene.values
    if scene_lattice is not None and not same_local_frame(
        scene_lattice.frame, response.lattice.frame
    ):
        raise ValueError("scene and response must share their declared local frame")
    _positive(scene, "scene radiance or factor maps")
    k = len(response.grid.points)
    if spectra is None:
        if scene.ndim != 3 or scene.shape[-1] != k:
            raise ValueError("scene must be (y,x,spectral)")
        maps = np.moveaxis(scene, -1, 0)
    else:
        _positive(spectra, "component spectra")
        if scene.ndim != 3 or spectra.shape != (len(scene), k):
            raise ValueError("maps/spectra must be (factor,y,x)/(factor,spectral)")
        maps = scene
    if target_indices is None:
        if method != "direct" or reads is not None:
            raise ValueError("translation options require explicit target indices")
        if scene_lattice is not None and not scene_lattice.matches(response.lattice):
            raise ValueError("scene lattice must match the response for direct integration")
        if maps.shape[1:] != response.lattice.shape:
            raise ValueError("scene ground lattice disagrees with response")
        if spectra is None:
            counts = np.einsum("rijk,ijk->r", response.coefficients, scene)
        else:
            counts = np.einsum("rijk,cij,ck->r", response.coefficients, maps, spectra)
        return ExpectedElectrons(response.reads, {source_name: counts})
    if scene_lattice is None or reads is None:
        raise ValueError("translation requires explicit scene axes and native reads")
    rows, columns = target_indices
    half_y = _stencil(response.lattice.y_m, scene_lattice.y_m)
    half_x = _stencil(response.lattice.x_m, scene_lattice.x_m)
    if maps.shape[1:] != scene_lattice.shape:
        raise ValueError("scene axes disagree with maps")
    for indices, half, size in ((rows, half_y, maps.shape[1]), (columns, half_x, maps.shape[2])):
        if (
            indices.ndim != 1
            or not len(indices)
            or np.any(indices < half)
            or np.any(indices >= size - half)
        ):
            raise ValueError("every target requires complete explicit scene halo")
    shape = (len(response.reads.sample_ids), len(rows), len(columns))
    if len(reads.sample_ids) != np.prod(shape):
        raise ValueError("native reads must match (template,target_x,target_y)")
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

        patches = sliding_window_view(local_maps, (2 * half_y + 1, 2 * half_x + 1), axis=(1, 2))[
            :, rows[:, None] - half_y, columns[None, :] - half_x
        ]
        result[:] = np.einsum("cijxy,rcxy->rij", patches, stencils)
    return ExpectedElectrons(reads, {source_name: result.transpose(0, 2, 1).ravel()})


def _snapshot_response(response: GridElectronResponse) -> None:
    photon_conversion_weights(response.grid)
    for name in ("support_bounds_m", "coefficients"):
        object.__setattr__(
            response, name, np.array(getattr(response, name), dtype=np.float64, copy=True)
        )
    _axis(response.lattice.x_m)
    _axis(response.lattice.y_m)
    _support(
        response.support_bounds_m,
        response.lattice.x_m,
        response.lattice.y_m,
        len(response.reads.sample_ids),
    )
    shape = (len(response.reads.sample_ids), *response.lattice.shape, response.grid.points.size)
    if response.coefficients.shape != shape:
        raise ValueError("coefficients must be (reads,y,x,spectral)")
    _positive(response.coefficients, "physical coefficients")


def _project_coefficients(
    spectral_grid: SpectralGrid,
    acquisition: Acquisition,
    optical_response: OpticalResponse,
    *,
    ground_xy_m: NDArray[np.float64],
    lattice: PointLattice,
    support_bounds_m: NDArray[np.float64],
    collecting_area_m2: float,
    throughput: Efficiency,
    quantum_efficiency: Efficiency | QuantumEfficiency,
) -> NDArray[np.float64]:
    """Project acquired photoelectron weights onto the declared ground grid.

    Returns:
        Fresh dense K with axes (read,y,x,spectral), owned by the caller.

    Raises:
        ValueError: Ground nodes or declared support are invalid or lack grid coverage.
    """
    x_m, y_m = lattice.x_m, lattice.y_m
    _axis(x_m)
    _axis(y_m)
    n, nodes = acquisition.times_s.shape
    _support(support_bounds_m, x_m, y_m, n)
    if ground_xy_m.shape != (n, nodes, 2) or not np.isfinite(ground_xy_m).all():
        raise ValueError("ground coordinates must be finite (reads,nodes,2)")
    if np.any(ground_xy_m < support_bounds_m[:, None, [0, 2]]) or np.any(
        ground_xy_m > support_bounds_m[:, None, [1, 3]]
    ):
        raise ValueError("ground nodes lie outside declared support")
    ix, ax = _brackets(x_m, ground_xy_m[..., 0])
    iy, ay = _brackets(y_m, ground_xy_m[..., 1])
    kernel = np.zeros((n, len(y_m), len(x_m), len(spectral_grid.points)))
    coefficients = photoelectron_response(
        spectral_grid,
        sample_optical_acceptance(
            acquisition,
            optical_response,
            0.01 / spectral_grid.points
            if spectral_grid.coordinate == "wavenumber_cm1"
            else spectral_grid.points * 1e-9,
        ),
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    for dx, wx in ((0, 1 - ax), (1, ax)):
        for dy, wy in ((0, 1 - ay), (1, ay)):
            np.add.at(
                kernel,
                (np.arange(n)[:, None], iy + dy, ix + dx),
                coefficients * (wx * wy)[..., None],
            )
    return kernel


def _axis(axis: NDArray[np.float64]) -> None:
    with np.errstate(over="ignore"):
        intervals = np.diff(axis)
    if (
        axis.ndim != 1
        or len(axis) < 2
        or not np.isfinite(axis).all()
        or not np.isfinite(intervals).all()
        or np.any(intervals <= 0)
    ):
        raise ValueError("ground axes need finite positive intervals and increasing knots")


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
