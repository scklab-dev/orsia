"""Compose independent platform and instrument attitudes into surface-directed rays."""

import numpy as np
from numpy.typing import NDArray


def compose_rays(
    positions_m: NDArray[np.float64],
    rotations_g_p: NDArray[np.float64],
    rotations_p_i: NDArray[np.float64],
    directions_i: NDArray[np.float64],
    mount_offset_p_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Locate mounted optical origins and rotate a shared set of instrument rays.

    All arrays are borrowed unchanged. Outputs own writable float64 storage.
    The destination frame can be GCRS or another explicitly caller-named frame.

    Args:
        positions_m: Finite (...,3) platform positions in destination-frame metres.
        rotations_g_p: Proper (...,3,3) platform-to-destination rotations.
        rotations_p_i: Proper (3,3) or (...,3,3) instrument-to-platform rotations.
        directions_i: Unit (rays,3) directions; +instrument-z is the boresight.
        mount_offset_p_m: Fixed finite (3,) platform-to-optical offset in metres.

    Returns:
        Independent origins and directions, each (...,rays,3), in the destination frame.

    Raises:
        ValueError: Shapes, finite values, unit directions or rotations are invalid.
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim < 1 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must be finite (...,3) metres")
    leading = positions.shape[:-1]
    platform = np.asarray(rotations_g_p, dtype=np.float64)
    relative = np.asarray(rotations_p_i, dtype=np.float64)
    validate_rotations(platform, leading)
    validate_rotations(relative, () if relative.shape == (3, 3) else leading)
    offset = np.asarray(mount_offset_p_m, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("mount offset must be finite (3,) metres")
    directions = np.asarray(directions_i, dtype=np.float64)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("directions must be finite unit (rays,3) vectors")
    centres = positions + np.einsum("...ij,j->...i", platform, offset)
    rays = np.einsum("...ij,aj->...ai", platform @ relative, directions)
    origins = np.broadcast_to(centres[..., None, :], rays.shape).copy()
    return origins, rays


def validate_rotations(rotations: NDArray[np.float64], leading: tuple[int, ...]) -> None:
    """Check proper column-vector rotations for callers composing numerical geometry.

    Args:
        rotations: Borrowed rotation matrices, unchanged by validation.
        leading: Required leading shape before the final (3,3) axes.

    Raises:
        ValueError: Matrices have invalid shape, nonfinite values or orthogonality.
    """
    if rotations.shape != (*leading, 3, 3) or not np.isfinite(rotations).all():
        raise ValueError("rotation must be finite with (...,3,3) shape")
    if not np.allclose(
        rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-10, rtol=0
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-10, rtol=0):
        raise ValueError("rotation must be proper and orthonormal")
