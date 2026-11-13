"""Evaluate supplied trajectories and independent pointing laws at physical elapsed times."""

from collections.abc import Callable
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from orsia_geometry.frames import Frame, frame_rotations, require_time_support, require_utc_epoch
from orsia_geometry.rays import compose_rays

type Trajectory = Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]]
type AttitudeLaw = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def compute_lines_of_sight(
    times_s: NDArray[np.float64],
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    trajectory: Trajectory,
    platform_attitude: AttitudeLaw,
    instrument_attitude: AttitudeLaw,
    directions_i: NDArray[np.float64],
    mount_offset_p_m: NDArray[np.float64],
    output_frame: Frame = "GCRS",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute dated optical origins and sight directions in the selected frame.

    Callbacks are borrowed, pure and stable for the call. They receive the same
    elapsed SI seconds from epoch, including across leap seconds. Callback owners
    supply GCRS states and attitudes and handle interpolation and continuous time.
    GCRS output uses NumPy only. ITRF/TEME output lazily uses SatKit frame rotations;
    first use can acquire public auxiliary files and populate process-global caches.
    Callers serialize native evaluation, configuration changes and auxiliary refresh.
    Both ray arrays use the same dated rotation, without velocity derivatives.
    Surface intersection is separate and requires a surface in the output frame.

    Args:
        times_s: Finite one-dimensional elapsed SI seconds; order and repeats are retained.
        epoch: Timezone-aware UTC epoch (zero UTC offset).
        validity_s: Finite closed interval valid for all three callbacks.
        trajectory: Returns GCRS (times,3) positions in m and velocities in m/s.
        platform_attitude: Returns proper (times,3,3) platform-to-GCRS rotations.
        instrument_attitude: Returns proper (times,3,3) instrument-to-platform
            rotations; a caller scan law can be supplied here independently.
        directions_i: Shared unit (rays,3) instrument directions.
        mount_offset_p_m: Fixed finite (3,) optical offset in platform metres.
        output_frame: GCRS (default), ITRF or TEME output axes. ITRF/TEME require
            observed/predicted EOP at every date through the required SatKit backend.

    Returns:
        Independent owned writable float64 origins in metres and unit directions
        in output_frame, both (times,rays,3). Empty time/ray axes remain empty.
        Input arrays and callback outputs remain unchanged.

    Raises:
        ValueError: Frame, epoch, support, trajectory arrays, rays or rotations
            are invalid. Invalid frames/times fail before callbacks or backend use.
        ImportError: ITRF/TEME output cannot load the required SatKit installation.
        RuntimeError: ITRF/TEME auxiliary EOP coverage or backend evaluation fails.
            Callback exceptions propagate unchanged for every output frame.
    """
    if output_frame not in ("GCRS", "ITRF", "TEME"):
        raise ValueError("output_frame must be GCRS, ITRF or TEME")
    require_utc_epoch(epoch)
    times = np.array(times_s, dtype=np.float64, copy=True)
    require_time_support(times, validity_s)
    position, velocity = trajectory(times)
    for array in (position, velocity):
        if array.shape != (len(times), 3) or not np.isfinite(array).all():
            raise ValueError("trajectory must return finite (times,3) position and velocity")
    origins, directions = compose_rays(
        position,
        platform_attitude(times),
        instrument_attitude(times),
        directions_i,
        mount_offset_p_m,
    )
    if output_frame == "GCRS":
        return origins, directions
    rotations = frame_rotations(
        times, epoch=epoch, validity_s=validity_s, from_frame="GCRS", to_frame=output_frame
    )
    return (
        np.einsum("tij,trj->tri", rotations, origins),
        np.einsum("tij,trj->tri", rotations, directions),
    )
