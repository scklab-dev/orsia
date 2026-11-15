"""Define shared flow callbacks, boundary choices and physical tensor checks."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Boundary = Literal["free", "reflecting"]
type Wind = Callable[[FloatArray, float], tuple[FloatArray, FloatArray]]
type Diffusivity = Callable[[FloatArray, float, FloatArray], FloatArray]


def validate_tensors(tensors: FloatArray, *, positive: bool) -> None:
    """Check finite symmetric three-dimensional covariance or diffusion tensors.

    PSD checks allow negative eigenvalue roundoff up to three float64 epsilons
    times each tensor's infinity norm. Inputs are unchanged.

    Args:
        tensors: Array of shape (n, 3, 3).
        positive: Require positive definiteness if true; otherwise semidefiniteness.

    Raises:
        ValueError: Shape, symmetry or definiteness is invalid.
    """
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise ValueError("tensors must have shape (n, 3, 3)")
    if not np.all(np.isfinite(tensors)):
        raise ValueError("tensors must be finite")
    if not np.allclose(tensors, tensors.swapaxes(1, 2), rtol=1e-12, atol=0):
        raise ValueError("tensors must be symmetric")
    if positive:
        try:
            np.linalg.cholesky(tensors)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariances must be positive definite") from exc
    else:
        scale = np.max(np.abs(tensors), axis=(1, 2))
        normalized = tensors / np.where(scale > 0, scale, 1)[:, None, None]
        norm = np.max(np.sum(np.abs(normalized), axis=2), axis=1)
        tolerance = 3 * np.finfo(np.float64).eps * norm
        if np.any(np.linalg.eigvalsh(normalized)[:, 0] < -tolerance):
            raise ValueError("diffusivity must be positive semidefinite")
