"""Construct geometric nadir, target-tracking and scan attitudes with explicit roll."""

import numpy as np
from numpy.typing import NDArray

from orsia_geometry.rays import validate_rotations


def radial_nadir(
    positions_g_m: NDArray[np.float64], velocities_g_m_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Point +platform-z toward the centre with +x along transverse inertial motion.

    Args:
        positions_g_m: Finite (...,3) nonzero GCRS positions in metres.
        velocities_g_m_s: Matching GCRS velocities in metres per second.

    Returns:
        Owned proper (...,3,3) platform-to-GCRS matrices. +y completes the
        right-handed triad. Inputs remain unchanged. This law is radial nadir.

    Raises:
        ValueError: Arrays are invalid or transverse motion cannot define roll.
    """
    return _look_rotation(-positions_g_m, velocities_g_m_s)


def platform_tracking(
    positions_g_m: NDArray[np.float64],
    targets_g_m: NDArray[np.float64],
    roll_reference_g: NDArray[np.float64],
    *,
    mount_offset_p_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aim the platform boresight at dated targets with an explicit projected +x reference.

    Args:
        positions_g_m: Finite (...,3) platform-reference GCRS positions, metres.
        targets_g_m: Matching target GCRS positions, metres; caller supplies target motion.
        roll_reference_g: Matching vectors whose transverse projection defines +platform-x.
        mount_offset_p_m: Fixed (3,) offset, required to be exactly zero for this helper.

    Returns:
        Owned proper (...,3,3) platform-to-GCRS rotations, including invisible
        targets. Call surface_visibility separately to test occlusion.

    Raises:
        ValueError: Offset is unsupported, arrays are invalid, or roll is singular.
    """
    if mount_offset_p_m.shape != (3,) or np.any(mount_offset_p_m != 0):
        raise ValueError("platform tracking requires zero mount offset")
    _matching_positions(positions_g_m, targets_g_m)
    return _look_rotation(targets_g_m - positions_g_m, roll_reference_g)


def instrument_tracking(
    optical_origins_g_m: NDArray[np.float64],
    targets_g_m: NDArray[np.float64],
    rotations_g_p: NDArray[np.float64],
    roll_reference_g: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aim +instrument-z at targets relative to an independently supplied platform attitude.

    Args:
        optical_origins_g_m: Finite (...,3) GCRS optical origins, metres; include mount offset.
        targets_g_m: Matching dated target GCRS positions, metres.
        rotations_g_p: Proper (...,3,3) nominal platform-to-GCRS rotations.
        roll_reference_g: Matching vectors whose transverse projection defines +instrument-x.

    Returns:
        Owned proper (...,3,3) instrument-to-platform rotations. The caller retains
        the nominal platform law. Visibility and mechanism limits are separate inputs.

    Raises:
        ValueError: Arrays or rotations are invalid, or the target/roll is singular.
    """
    _matching_positions(optical_origins_g_m, targets_g_m)
    validate_rotations(rotations_g_p, optical_origins_g_m.shape[:-1])
    desired = _look_rotation(targets_g_m - optical_origins_g_m, roll_reference_g)
    return np.swapaxes(rotations_g_p, -1, -2) @ desired


def axis_rotations(
    axis: NDArray[np.float64], angles_rad: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Construct right-handed axis rotations for a caller-defined scan-angle law.

    Args:
        axis: Finite unit (3,) axis in the rotation's destination frame. Norm errors
            up to 1e-10 are accepted and normalized without changing the input.
        angles_rad: Finite (...) signed angles in radians.

    Returns:
        Owned proper (...,3,3) column-vector rotations. Positive angles use the
        right-hand rule. Inputs remain unchanged; multiplication order belongs to callers.

    Raises:
        ValueError: Axis or angles are invalid.
    """
    if (
        axis.shape != (3,)
        or not np.isfinite(axis).all()
        or not np.isclose(np.linalg.norm(axis), 1, atol=1e-10, rtol=0)
        or not np.isfinite(angles_rad).all()
    ):
        raise ValueError("scan axis must be unit (3,) and angles finite")
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    cosine = np.cos(angles_rad)[..., None, None]
    sine = np.sin(angles_rad)[..., None, None]
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew


def attitude_interval_speeds(
    times_s: NDArray[np.float64], rotations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate interval-mean angular speed including roll from sampled attitudes.

    Args:
        times_s: Finite strictly increasing (times,) SI seconds, at least two samples.
        rotations: Proper (times,3,3) body-to-reference attitudes in a common frame.

    Returns:
        Owned (times-1,) principal rotation angles divided by interval durations,
        radians per second. Sampling must resolve motion below pi radians per interval.
        Refinement is required for varying rates; unresolved turns can alias. Inputs stay unchanged.

    Raises:
        ValueError: Time support or rotations are invalid.
    """
    if (
        times_s.ndim != 1
        or len(times_s) < 2
        or not np.isfinite(times_s).all()
        or np.any(np.diff(times_s) <= 0)
    ):
        raise ValueError("rates require at least two increasing finite times")
    validate_rotations(rotations, (len(times_s),))
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    sine = (
        np.linalg.norm(
            np.stack(
                (
                    relative[:, 2, 1] - relative[:, 1, 2],
                    relative[:, 0, 2] - relative[:, 2, 0],
                    relative[:, 1, 0] - relative[:, 0, 1],
                ),
                axis=-1,
            ),
            axis=-1,
        )
        / 2
    )
    return np.arctan2(sine, cosine) / np.diff(times_s)


def _matching_positions(first: NDArray[np.float64], second: NDArray[np.float64]) -> None:
    if (
        first.ndim < 1
        or first.shape[-1] != 3
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("positions must be matching finite (...,3) arrays")


def _look_rotation(
    direction: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    _matching_positions(direction, reference)
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(length == 0):
        raise ValueError("pointing direction must be nonzero")
    z = direction / length
    y = np.cross(z, reference)
    transverse = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(transverse <= 1e-12 * np.linalg.norm(reference, axis=-1, keepdims=True)):
        raise ValueError("roll reference is singular or nearly parallel to boresight")
    y = y / transverse
    x = np.cross(y, z)
    return np.stack((x, y, z), axis=-1)
