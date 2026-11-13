"""Declare local Cartesian frames and convert dated terrestrial coordinates.

SatKit owns its normal auxiliary search, lazy public-file downloads and process-global
caches. Callers serialize backend operations and refresh; Geometry does not claim an
absolute identity for the EOP file selected by that backend. Core ray operations remain
independent of SatKit and provider access.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_geometry import times as _times

type Frame = Literal["GCRS", "ITRF", "TEME"]


@dataclass(frozen=True, eq=False)
class LocalCartesianFrame:
    """Identify Cartesian x/y/z axes and their caller-declared vertical datum.

    Unanchored synthetic declarations match only by object identity. An anchored
    chart uses ``itrf = origin_itrf_m + basis_itrf @ local_xyz_m``. Columns of the
    basis are local unit axes expressed in ITRF. The caller establishes the
    datum's physical meaning; this declaration performs no datum conversion.

    Attributes:
        name: Nonempty descriptive name; equal names do not establish alignment.
        vertical_datum: Nonempty declaration of the local altitude reference.
        origin_itrf_m: Optional finite float64 (3,) ITRF origin in metres.
        basis_itrf: Optional finite float64 (3,3) right-handed orthonormal basis,
            supplied together with the origin. Numerical rotation tolerance is 1e-12.
            Both arrays are owned immutable snapshots.
    """

    name: str
    vertical_datum: str
    origin_itrf_m: NDArray[np.float64] | None = None
    basis_itrf: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate and snapshot an optional georeferenced chart.

        Raises:
            ValueError: Names, paired anchor inputs, finite axes or rotation fail.
        """
        if not self.name or not self.vertical_datum:
            raise ValueError("frame name and vertical datum must be nonempty")
        if (self.origin_itrf_m is None) != (self.basis_itrf is None):
            raise ValueError("frame origin and basis must be supplied together")
        if self.origin_itrf_m is not None and self.basis_itrf is not None:
            origin, basis = self.origin_itrf_m, self.basis_itrf
            if (
                origin.shape != (3,)
                or basis.shape != (3, 3)
                or origin.dtype != np.float64
                or basis.dtype != np.float64
                or not np.isfinite(origin).all()
                or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(3), rtol=0, atol=1e-12)
                or not np.isclose(np.linalg.det(basis), 1, rtol=0, atol=1e-12)
            ):
                raise ValueError(
                    "finite float64 origin and right-handed orthonormal basis required"
                )
            for name, values in (("origin_itrf_m", origin), ("basis_itrf", basis)):
                owned = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
                object.__setattr__(self, name, owned)


def same_local_frame(left: LocalCartesianFrame, right: LocalCartesianFrame) -> bool:
    """Compare frame identity or exact anchored coordinates and vertical datum.

    Args:
        left: First local chart declaration.
        right: Second local chart declaration.

    Returns:
        True for a shared declaration or exactly equal anchored origin, basis and
        datum. Independent synthetic charts return False even with equal names.
        No coordinate tolerance, alignment or implicit reprojection is applied.
    """
    if left is right:
        return True
    return bool(
        left.origin_itrf_m is not None
        and right.origin_itrf_m is not None
        and left.vertical_datum == right.vertical_datum
        and np.array_equal(left.origin_itrf_m, right.origin_itrf_m)
        and np.array_equal(left.basis_itrf, right.basis_itrf)
    )


def require_utc_epoch(epoch: datetime) -> None:
    """Validate a timezone-aware zero-offset UTC epoch without backend access.

    Args:
        epoch: Reference datetime for elapsed SI seconds.

    Raises:
        ValueError: Epoch is naive or has a nonzero UTC offset.
    """
    if epoch.tzinfo is None or epoch.utcoffset() != timedelta(0):
        raise ValueError("epoch must be a timezone-aware UTC datetime")


def require_time_support(times_s: NDArray[np.float64], validity_s: tuple[float, float]) -> None:
    """Check finite elapsed seconds against a finite closed interval.

    Args:
        times_s: Finite one-dimensional elapsed seconds, allowing order/repeats/empty input.
        validity_s: Finite ordered inclusive lower/upper elapsed-second bounds.

    Raises:
        ValueError: Interval, time shape, finiteness or closed support is invalid.
    """
    bounds = np.asarray(validity_s, dtype=np.float64)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[0] > bounds[1]
        or times_s.ndim != 1
        or not np.isfinite(times_s).all()
        or np.any((times_s < bounds[0]) | (times_s > bounds[1]))
    ):
        raise ValueError("times must lie in the finite closed validity interval")


def frame_rotations(
    times_s: NDArray[np.float64],
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    from_frame: Frame = "GCRS",
    to_frame: Frame = "ITRF",
) -> NDArray[np.float64]:
    """Rotate column vectors between GCRS, TEME and ITRF with full IERS 2010 reduction.

    Args:
        times_s: Finite (times,) physical elapsed SI seconds.
        epoch: Timezone-aware UTC reference instant.
        validity_s: Finite closed caller-accepted interval.
        from_frame: Source GCRS, TEME or ITRF axes; GCRS maps to SatKit GCRF axes.
        to_frame: Destination axes; ITRF uses the backend's IERS 2010 realization,
            IAU 2006/2000A reduction and available EOP, excluding solid/ocean tides.

    Returns:
        Owned writable float64 (times,3,3) source-to-destination rotations. EOP
        status must be observed or predicted at every date; missing/held values fail.
        First use can trigger SatKit's normal public auxiliary acquisition.

    Raises:
        ValueError: Frame names, epoch or support are invalid.
        RuntimeError: Auxiliary EOP coverage is absent or the backend operation fails.
        ImportError: The required SatKit installation cannot load.
    """
    require_utc_epoch(epoch)
    require_time_support(times_s, validity_s)
    instants = _times._native_instants(times_s, epoch=epoch)
    import satkit

    frames = {"GCRS": satkit.frame.GCRF, "ITRF": satkit.frame.ITRF, "TEME": satkit.frame.TEME}
    if from_frame not in frames or to_frame not in frames:
        raise ValueError("supported frames are GCRS, ITRF and TEME")
    if from_frame == to_frame:
        return np.broadcast_to(np.eye(3), (len(instants), 3, 3)).copy()
    if any(satkit.frametransform.eop_status(t) not in ("observed", "predicted") for t in instants):
        raise RuntimeError("Earth-orientation coverage is missing or extrapolated")
    rotations = np.empty((len(instants), 3, 3), dtype=np.float64)
    for index, instant in enumerate(instants):
        rotations[index] = satkit.frametransform.rotation(
            frames[from_frame], frames[to_frame], instant
        ).to_rotation_matrix()
    return rotations


def transform_states(
    times_s: NDArray[np.float64],
    positions_m: NDArray[np.float64],
    velocities_m_s: NDArray[np.float64],
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    from_frame: Frame = "GCRS",
    to_frame: Frame = "ITRF",
    derivative_step_s: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform positions and their coordinate-time derivatives consistently.

    Args:
        times_s: Finite (times,) physical elapsed SI seconds.
        positions_m: Borrowed finite (times,3) source-frame positions, metres.
        velocities_m_s: Matching coordinate derivatives, metres per second.
        epoch: Timezone-aware UTC reference instant.
        validity_s: Closed accepted interval for the supplied states.
        from_frame: GCRS, ITRF or TEME source axes.
        to_frame: GCRS, ITRF or TEME destination axes.
        derivative_step_s: Finite central-difference half-step of at least 1e-4 seconds.
            Auxiliary coverage must also include dates shifted by this amount.

    Returns:
        Owned writable destination positions and velocities, both (times,3).
        Velocity is R @ v + Rdot @ r; Rdot uses central differences of the full
        rotation, including precession/nutation and Earth rotation. Inputs stay unchanged.
        Step refinement is needed for the intended velocity error budget.

    Raises:
        ValueError: State shape, finite values, derivative step or time/frame support is invalid.
        RuntimeError: Required auxiliary coverage or backend evaluation fails.
        ImportError: The required SatKit installation cannot load.
    """
    if (
        positions_m.shape != (len(times_s), 3)
        or velocities_m_s.shape != positions_m.shape
        or not np.isfinite(positions_m).all()
        or not np.isfinite(velocities_m_s).all()
        or not np.isfinite(derivative_step_s)
        or derivative_step_s < 1e-4
    ):
        raise ValueError("finite matching states and derivative step >= 1e-4 seconds required")
    rotation = frame_rotations(
        times_s, epoch=epoch, validity_s=validity_s, from_frame=from_frame, to_frame=to_frame
    )
    padded = (validity_s[0] - derivative_step_s, validity_s[1] + derivative_step_s)
    plus = frame_rotations(
        times_s + derivative_step_s,
        epoch=epoch,
        validity_s=padded,
        from_frame=from_frame,
        to_frame=to_frame,
    )
    minus = frame_rotations(
        times_s - derivative_step_s,
        epoch=epoch,
        validity_s=padded,
        from_frame=from_frame,
        to_frame=to_frame,
    )
    derivative = (plus - minus) / (2 * derivative_step_s)
    positions = np.einsum("tij,tj->ti", rotation, positions_m)
    velocities = np.einsum("tij,tj->ti", rotation, velocities_m_s) + np.einsum(
        "tij,tj->ti", derivative, positions_m
    )
    return positions, velocities


def transform_rays(
    times_s: NDArray[np.float64],
    origins_m: NDArray[np.float64],
    directions: NDArray[np.float64],
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    from_frame: Frame = "GCRS",
    to_frame: Frame = "ITRF",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Express existing dated rays in another Earth-centred frame.

    Each origin and unit direction receives the same dated column-vector rotation.
    Distinct flattened dates are evaluated internally, then the original leading
    axes, order and repetitions are restored. Ray coordinates need no velocity
    derivative or shifted auxiliary dates. SatKit is imported lazily through
    frame_rotations; callers serialize native operations and auxiliary refresh.

    Args:
        times_s: Finite elapsed SI seconds, exactly matching the ray leading axes (...).
        origins_m: Borrowed finite (...,3) source-frame x/y/z origins in metres.
        directions: Matching finite unit (...,3) source-frame directions, with
            absolute norm tolerance 1e-10. Inputs remain unchanged.
        epoch: Timezone-aware UTC reference instant with zero offset.
        validity_s: Finite closed accepted interval in elapsed seconds.
        from_frame: GCRS, ITRF or TEME source axes.
        to_frame: GCRS, ITRF or TEME destination axes.

    Returns:
        Independent owned writable float64 origins in metres and unit directions,
        both (...,3) in to_frame. Scalar and empty leading axes are preserved.
        First use can acquire normal public SatKit auxiliary files. Non-identity
        transformations require observed/predicted EOP at every distinct date.

    Raises:
        ValueError: Ray shapes, finite values, direction norms, epoch, frames or
            time support are invalid.
        ImportError: The required SatKit installation cannot load.
        RuntimeError: Required EOP coverage or backend evaluation fails.
    """
    if (
        origins_m.ndim < 1
        or origins_m.shape[-1] != 3
        or directions.shape != origins_m.shape
        or times_s.shape != origins_m.shape[:-1]
        or not np.isfinite(origins_m).all()
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError(
            "rays need matching finite (...,3) origins/unit directions and (...) times"
        )
    dates, inverse = np.unique(times_s.reshape(-1), return_inverse=True)
    rotations = frame_rotations(
        dates, epoch=epoch, validity_s=validity_s, from_frame=from_frame, to_frame=to_frame
    )[inverse].reshape((*times_s.shape, 3, 3))
    return (
        np.einsum("...ij,...j->...i", rotations, origins_m),
        np.einsum("...ij,...j->...i", rotations, directions),
    )
