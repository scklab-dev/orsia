"""Project native camera rays onto a supplied surface or a fixed WGS84 chart.

Camera/platform geometry can select the WGS84 chart's reference ray. Surface
projection and continuous geometric bounds remain explicit acquisition inputs.
The output sampling grid remains separate from a requested reconstruction grid.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from orsia_geometry.los import AttitudeLaw, Trajectory, compute_lines_of_sight
from orsia_geometry.surfaces import wgs84_surface_projection_from_ray

from ..acquisition import RasterReadSchedule
from ..los import (
    LineOfSightSamples,
    focal_plane_direction_bounds,
    sample_lines_of_sight,
)
from ..optics.camera import FilterCamera, FocalPlaneImager, camera_read_bands
from ..optics.spectrometer import DispersiveSpectrometer

type SurfaceProjection = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.bool_]],
]


@dataclass(frozen=True, eq=False)
class SurfaceAcquisitionGrid:
    """Instrument-selected ground quadrature with native read correspondence and support.

    The factory owns arrays; direct construction borrows them. Keep arrays and
    shared rays stable during use. Coordinates use a caller-declared metre chart.
    The surface must exist continuously on every full time/direction rectangle.
    Projection adapters certify the stated global Lipschitz bounds on that domain.

    Attributes:
        rays: GCRS rays and native sensor-plane quadrature, with UTC epoch.
        xy_m: (reads,nodes,2) surface integration coordinates, metres.
        bounds_m: (reads,4) conservative xmin,xmax,ymin,ymax continuous support, metres.
        schedule: Shared immutable read identity, timing and native raster correspondence.
        native_indices: Derived (reads,3) frame,row,column mapping in original read order.
        image_shape: Native (frames,rows,columns) dimensions.
        read_bands: Physical achieved-channel identity per read, empty for dark pixels.
        motion_bound_m_s: Certified infinity-norm surface speed bound at fixed sensor direction.
        angular_bound_m: Certified infinity-norm Lipschitz constant from sensor (u,v)
            direction cosines to surface (x,y), using the infinity norm in both spaces.
        focal_length_m: Physical projection focal length used to select rays, metres.
        psf_support_radius_m: Pre-detector camera PSF support used to select rays,
            metres; zero when a hard entrance slit bounds the full incident domain.
        entrance_bounds_m: Optional owned (4,) finite entrance slit rectangle, metres.
            None identifies the camera's detector-expanded incident domain.
    """

    rays: LineOfSightSamples
    xy_m: NDArray[np.float64]
    bounds_m: NDArray[np.float64]
    schedule: RasterReadSchedule
    read_bands: tuple[str, ...]
    motion_bound_m_s: float
    angular_bound_m: float
    focal_length_m: float
    psf_support_radius_m: float
    entrance_bounds_m: NDArray[np.float64] | None = None

    @property
    def native_indices(self) -> NDArray[np.int64]:
        """Return immutable native indices from the shared schedule."""
        return self.schedule.native_indices

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return full native shape from the shared schedule."""
        return self.schedule.native_shape


def frame_exposure_bounds(
    *, start_s: float, frame_count: int, frame_rate_hz: float, exposure_s: float
) -> NDArray[np.float64]:
    """Schedule uniformly spaced, nonoverlapping complete frame exposures.

    Args:
        start_s: Finite first start in elapsed seconds from the observation epoch.
        frame_count: Positive number of frames.
        frame_rate_hz: Positive cadence in frames per second.
        exposure_s: Positive duration at most one frame period, seconds.

    Returns:
        Owned (frames,2) start/end seconds, including complete final exposure.

    Raises:
        ValueError: Cadence, duration, count or start has invalid support.
    """
    if (
        frame_count < 1
        or not np.isfinite([start_s, frame_rate_hz, exposure_s]).all()
        or frame_rate_hz <= 0
        or exposure_s <= 0
        or exposure_s > 1 / frame_rate_hz
    ):
        raise ValueError("frames require positive count/cadence and exposure within a period")
    starts = start_s + np.arange(frame_count) / frame_rate_hz
    return np.column_stack((starts, starts + exposure_s))


def wgs84_surface_projection_from_camera(
    camera: FilterCamera | FocalPlaneImager,
    *,
    trajectory: Trajectory,
    platform_attitude: AttitudeLaw,
    instrument_attitude: AttitudeLaw,
    epoch: datetime,
    validity_s: tuple[float, float],
    reference_time_s: float = 0.0,
    mount_offset_p_m: NDArray[np.float64] | None = None,
    reference_pixel: tuple[int, int] | None = None,
) -> SurfaceProjection:
    """Fix a WGS84 east/north chart at a camera reference ray's surface intercept.

    The default reference is the midpoint of the bounding extent of all active
    detector rectangles: ((min(xmin)+max(xmax))/2, (min(ymin)+max(ymax))/2).
    This preserves physical detector offsets and the centre between pixels on
    even-sized detectors. A selected pixel uses its active rectangle's centre.
    Forward focal-plane (x,y,focal_length_m) defines the instrument unit ray,
    with columns along x, rows along y and +instrument-z toward the scene.
    Geometry composes both attitudes and the platform-frame mount offset, then
    transforms the reference GCRS ray to ITRF and intersects height-zero WGS84.

    Args:
        camera: Borrowed physical active rectangles and positive focal length, metres.
        trajectory: Pure callback returning finite (times,3) GCRS positions in
            metres and velocities in metres per second.
        platform_attitude: Pure callback returning proper (times,3,3)
            platform-to-GCRS column-vector rotations.
        instrument_attitude: Independent pure callback returning proper
            (times,3,3) instrument-to-platform rotations.
        epoch: Timezone-aware UTC reference instant with zero offset.
        validity_s: Finite closed elapsed-SI-second support for callbacks and projection.
        reference_time_s: Scalar elapsed SI seconds within validity_s; defaults to zero.
        mount_offset_p_m: Borrowed finite (3,) platform-to-optical-origin offset
            in platform-frame metres; None selects zero, as in camera_acquisition_grid.
        reference_pixel: Native zero-based (row,column) index, or None for the
            geometric detector centre. Negative and out-of-range indices raise.

    Returns:
        Plain surface(origins_gcrs_m, directions_gcrs, times_s) callback accepting
        matching (...,3) GCRS origins in metres and unit directions, and exactly
        (...) elapsed times. Owned writable float64 (...,2) coordinates are east,
        north metres; Boolean (...) hits mark nearest forward intersections.
        Misses have NaN coordinates and false hits. The chart and interval are
        snapshots. Inputs remain unchanged; the callback retains no camera,
        reference arrays or trajectory/attitude callbacks. Reuse this one surface
        for truth and estimated acquisition geometry to keep the chart fixed.
        Continuous visibility and support bounds remain caller responsibilities.
        Construction and evaluation use Geometry's required SatKit backend, public
        auxiliaries and process-global caches. Callers serialize native operations
        and auxiliary refresh. Terrain, geoid heights and ray bending are excluded.

    Raises:
        ValueError: Pixel index, reference support, epoch, state or rotations are
            invalid, or the reference ray misses WGS84 or intersects an exact pole.
            Evaluation validates ray shapes, finiteness, unit norms and time support.
        ImportError: Construction or evaluation cannot load the required SatKit installation.
        RuntimeError: Required EOP coverage or backend evaluation fails.
            Trajectory and attitude callback exceptions propagate unchanged.
    """
    pixels = camera.pixel_bounds_m
    if reference_pixel is None:
        bounds = np.array(
            [pixels[..., 0].min(), pixels[..., 1].max(), pixels[..., 2].min(), pixels[..., 3].max()]
        )
    else:
        row, column = reference_pixel
        if not (0 <= row < pixels.shape[0] and 0 <= column < pixels.shape[1]):
            raise ValueError("reference_pixel must be a native zero-based (row,column) index")
        bounds = pixels[row, column]
    direction = np.array(
        [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, camera.optics.focal_length_m]
    )
    direction /= np.linalg.norm(direction)
    origins, directions = compute_lines_of_sight(
        np.array([reference_time_s]),
        epoch=epoch,
        validity_s=validity_s,
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        directions_i=direction[None, :],
        mount_offset_p_m=np.zeros(3) if mount_offset_p_m is None else mount_offset_p_m,
    )
    return wgs84_surface_projection_from_ray(
        origin_gcrs_m=origins[0, 0],
        direction_gcrs=directions[0, 0],
        time_s=reference_time_s,
        epoch=epoch,
        validity_s=validity_s,
    )


def camera_acquisition_grid(
    camera: FilterCamera | FocalPlaneImager,
    schedule: RasterReadSchedule,
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    trajectory: Trajectory,
    platform_attitude: AttitudeLaw,
    instrument_attitude: AttitudeLaw,
    surface: SurfaceProjection,
    motion_bound_m_s: float,
    angular_bound_m: float,
    mount_offset_p_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    angular_order: int = 8,
) -> SurfaceAcquisitionGrid:
    """Project finite-pixel, PSF-expanded and exposure-integrated camera samples.

    Every pixel selects its own direction-cosine rectangle enclosing the active
    rectangle expanded by the camera's PSF radius. Gaussian/Airy truncation leaves
    the declared tail budget outside this support. Gauss orders control time and
    angular refinement; changed orders require new radiance evaluation.
    Projection callbacks borrow GCRS origins/directions (...,3) and elapsed
    times (...,), returning finite (...,2) metre coordinates and Boolean hits.
    Arbitrary surfaces can have extrema between corners. Certified Lipschitz
    bounds enlarge sampled corner envelopes by half the largest sampled time gap
    and half the direction-rectangle width. The adapter certifies continuous
    visibility and these global bounds; finite sampling cannot establish them.

    Args:
        camera: Physical detector, filter layout, focal length and PSF support.
        schedule: Complete or selected native reads with positive elapsed-second intervals.
        epoch: Timezone-aware UTC observation origin.
        validity_s: Closed interval certified for trajectory and both attitude laws.
        trajectory: Pure GCRS position/velocity callback in metres and m/s.
        platform_attitude: Pure platform-to-GCRS rotation callback.
        instrument_attitude: Independent instrument-to-platform rotation callback.
        surface: Pure projection into the declared surface metre chart, with hits.
        motion_bound_m_s: Nonnegative certified infinity-norm surface speed bound.
        angular_bound_m: Positive certified infinity-norm directional Lipschitz constant.
        mount_offset_p_m: Optional (3,) platform-to-optical-origin offset, metres.
        time_order: Positive temporal Gauss order per exposure.
        angular_order: Positive Gauss order on each local direction coordinate.

    Returns:
        Owned projected integration grid with native identities and conservative support.

    Raises:
        ValueError: Invalid camera/time/geometric support, projection miss or malformed output.
    """
    if (
        not np.isfinite([motion_bound_m_s, angular_bound_m]).all()
        or motion_bound_m_s < 0
        or angular_bound_m <= 0
    ):
        raise ValueError(
            "continuous projection requires finite nonnegative motion and positive angular bounds"
        )
    if schedule.native_shape[1:] != camera.pixel_bounds_m.shape[:2]:
        raise ValueError("native schedule and detector dimensions disagree")
    frame, row, column = schedule.native_indices.T
    reads = schedule.reads
    pixels = camera.pixel_bounds_m[row, column]
    expanded = pixels + camera.support_radius_m * np.array([-1, 1, -1, 1])
    bounds = focal_plane_direction_bounds(expanded, camera.optics.focal_length_m)
    rays = sample_lines_of_sight(
        reads,
        pixel_bounds_m=pixels,
        fov_bounds_uv=bounds,
        epoch=epoch,
        validity_s=validity_s,
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=np.zeros(3) if mount_offset_p_m is None else mount_offset_p_m,
        focal_length_m=camera.optics.focal_length_m,
        time_order=time_order,
        angular_order=angular_order,
    )
    return _surface_grid(
        rays,
        surface,
        schedule,
        camera_read_bands(camera, schedule.native_indices),
        motion_bound_m_s,
        angular_bound_m,
        camera.optics.focal_length_m,
        camera.support_radius_m,
    )


def spectrometer_acquisition_grid(
    spectrometer: DispersiveSpectrometer,
    schedule: RasterReadSchedule,
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    trajectory: Trajectory,
    platform_attitude: AttitudeLaw,
    instrument_attitude: AttitudeLaw,
    surface: SurfaceProjection,
    motion_bound_m_s: float,
    angular_bound_m: float,
    mount_offset_p_m: NDArray[np.float64] | None = None,
    time_order: int = 3,
    slit_order: int = 12,
) -> SurfaceAcquisitionGrid:
    """Project a complete finite entrance slit for each dispersed detector read.

    Every detector pixel receives the same slit/time quadrature, independently
    of detector position or wavelength mapping. Optics subsequently disperses
    entrance coordinates and integrates the PSF over each finite detector pixel.
    Full slit support includes every wavelength and all post-slit PSF tails.
    The slit is a hard field stop; pre-slit PSF blur is excluded. Gauss quadrature
    in slit x/y uses the exact focal-plane solid-angle Jacobian. Complete frames
    share exposure intervals; spatial scanning follows the supplied trajectory
    and independent attitude laws. Continuous visibility and surface Lipschitz
    bounds have the same caller-certified meaning as camera_acquisition_grid.

    Args:
        spectrometer: Physical finite slit, dispersive optics and active detector pixels.
        schedule: Complete or selected native reads with positive elapsed-second intervals.
        epoch: Timezone-aware UTC observation origin.
        validity_s: Closed elapsed-second support for geometry and projection callbacks.
        trajectory: Pure GCRS position/velocity callback in metres and m/s.
        platform_attitude: Pure platform-to-GCRS rotation callback.
        instrument_attitude: Independent pure instrument-to-platform rotation callback.
        surface: Pure GCRS ray/time projection returning metre-chart coordinates and hits.
        motion_bound_m_s: Certified nonnegative infinity-norm surface speed bound, m/s.
        angular_bound_m: Certified positive direction-cosine to surface Lipschitz bound, metres.
        mount_offset_p_m: Borrowed optional (3,) platform-frame optical-origin offset, metres.
        time_order: Positive exposure-time Gauss order.
        slit_order: Positive Gauss order on each entrance-plane coordinate.
            Refine against slit extent, detector pixels, PSF and spectral structure.

    Returns:
        Owned surface quadrature, conservative full-slit/exposure bounds and native
        (frame,row,column) correspondence. Keep returned writable buffers stable.

    Raises:
        ValueError: Invalid intervals, geometry, quadrature, projection or support bounds.
            User callback exceptions propagate unchanged.
    """
    if (
        not np.isfinite([motion_bound_m_s, angular_bound_m]).all()
        or motion_bound_m_s < 0
        or angular_bound_m <= 0
    ):
        raise ValueError("finite nonnegative motion and positive angular bounds required")
    if schedule.native_shape[1:] != spectrometer.pixel_bounds_m.shape[:2]:
        raise ValueError("native schedule and detector dimensions disagree")
    shape = schedule.native_shape
    frame, row, column = schedule.native_indices.T
    reads = schedule.reads
    bands = tuple(
        spectrometer.pixel_bands[r * shape[2] + c] for r, c in zip(row, column, strict=True)
    )
    entrance = np.broadcast_to(spectrometer.slit_bounds_m, (len(frame), 4))
    bounds = focal_plane_direction_bounds(entrance, spectrometer.optics.focal_length_m)
    rays = sample_lines_of_sight(
        reads,
        pixel_bounds_m=spectrometer.pixel_bounds_m[row, column],
        fov_bounds_uv=bounds,
        entrance_bounds_m=entrance,
        epoch=epoch,
        validity_s=validity_s,
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=np.zeros(3) if mount_offset_p_m is None else mount_offset_p_m,
        focal_length_m=spectrometer.optics.focal_length_m,
        time_order=time_order,
        angular_order=slit_order,
    )
    return _surface_grid(
        rays,
        surface,
        schedule,
        bands,
        motion_bound_m_s,
        angular_bound_m,
        spectrometer.optics.focal_length_m,
        0.0,
        spectrometer.slit_bounds_m.copy(),
    )


def ground_integration_bounds(grid: SurfaceAcquisitionGrid) -> NDArray[np.float64]:
    """Enclose all continuous instrument integration supports in the surface chart.

    Args:
        grid: Instrument-selected surface coordinates with certified motion support.

    Returns:
        Owned (xmin,xmax,ymin,ymax) enclosing bounds in surface metres.
    """
    bounds = grid.bounds_m
    return np.array(
        [bounds[:, 0].min(), bounds[:, 1].max(), bounds[:, 2].min(), bounds[:, 3].max()]
    )


def _surface_grid(
    rays: LineOfSightSamples,
    surface: SurfaceProjection,
    schedule: RasterReadSchedule,
    bands: tuple[str, ...],
    motion_bound_m_s: float,
    angular_bound_m: float,
    focal_length_m: float,
    psf_support_radius_m: float,
    entrance_bounds_m: NDArray[np.float64] | None = None,
) -> SurfaceAcquisitionGrid:
    bounds = rays.acquisition.direction_bounds_uv
    xy = _project(surface, rays.origins_gcrs_m, rays.directions_gcrs, rays.acquisition.times_s)
    edge_times = np.broadcast_to(
        rays.support_times_s[..., None], rays.support_origins_gcrs_m.shape[:-1]
    )
    edges = _project(surface, rays.support_origins_gcrs_m, rays.support_directions_gcrs, edge_times)
    time_gap = np.max(np.diff(rays.support_times_s, axis=1), axis=1) / 2
    angular_gap = np.max(bounds[:, [1, 3]] - bounds[:, [0, 2]], axis=1) / 2
    padding = motion_bound_m_s * time_gap + angular_bound_m * angular_gap
    lower = np.minimum(edges.min(axis=(1, 2)), xy.min(axis=1)) - padding[:, None]
    upper = np.maximum(edges.max(axis=(1, 2)), xy.max(axis=1)) + padding[:, None]
    ground_bounds = np.column_stack((lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1]))
    return SurfaceAcquisitionGrid(
        rays,
        xy,
        ground_bounds,
        schedule,
        bands,
        motion_bound_m_s,
        angular_bound_m,
        focal_length_m,
        psf_support_radius_m,
        entrance_bounds_m,
    )


def _project(
    surface: SurfaceProjection,
    origins: NDArray[np.float64],
    directions: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    xy, hits = surface(origins, directions, times)
    if hits.shape != times.shape or hits.dtype != np.bool_ or not hits.all():
        raise ValueError("surface must support every integration ray and exposure boundary")
    result = np.array(xy, dtype=np.float64, copy=True)
    if result.shape != (*times.shape, 2) or not np.isfinite(result).all():
        raise ValueError("surface must return finite (...,2) coordinates in metres")
    return result
