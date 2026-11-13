"""Assemble sparse correlated field covariance with shared low-rank modes."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, sparray, spmatrix

from orsia_uncertainty.sparse import validate_sparse_covariance

type FloatArray = NDArray[np.float64]
type ArrayInput = NDArray[np.floating | np.integer] | Sequence[float] | Sequence[Sequence[float]]


def covariance_from_marginals(
    marginal_variance: ArrayInput,
    detector_covariance: spmatrix | sparray | None = None,
    common_modes: ArrayInput | None = None,
    inflation: ArrayInput | None = None,
) -> tuple[Callable[[FloatArray], FloatArray], FloatArray, FloatArray]:
    """Assemble scalar posterior covariance without double-counting local marginals.

    Marginals and inflation are (pixel,) in explicit selected C-order. Detector
    covariance is sparse (pixel,pixel) including its diagonal. Common modes have
    (pixel,mode) axes and already include a covariance square root. Returns a
    matrix-product callable, inflated diagonal and independent residual variance.
    Negative unexplained residual variance raises ValueError. The returned action
    borrows common_modes; keep that array stable for the callable lifetime.

    Args:
        marginal_variance: Positive selected scalar-state variances (pixel,).
        detector_covariance: Optional selected sparse covariance, including diagonal.
        common_modes: Optional (pixel,mode) sensitivities already multiplied by a root.
        inflation: Optional per-state variance multipliers, each at least one.

    Returns:
        Covariance action callable, owned inflated diagonal and owned independent
        residual variance. Low-rank common errors are retained across pixels.

    Raises:
        ValueError: Covariance axes, finite/positive domains, PSD or residual variance fails.
    """
    marginal = np.asarray(marginal_variance, float)
    n = len(marginal)
    if detector_covariance is None:
        detector = csr_matrix((n, n))
    else:
        # scipy-stubs omits sparse-base conversion although every concrete format is accepted.
        detector = csr_matrix(detector_covariance)  # pyrefly: ignore[no-matching-overload]
    modes = np.zeros((n, 0)) if common_modes is None else np.asarray(common_modes, float)
    inflation = np.ones(n) if inflation is None else np.asarray(inflation, float)
    if (
        marginal.shape != (n,)
        or detector.shape != (n, n)
        or modes.ndim != 2
        or modes.shape[0] != n
        or inflation.shape != (n,)
        or not np.isfinite(modes).all()
        or not np.isfinite(marginal).all()
        or np.any(marginal <= 0)
        or not np.isfinite(inflation).all()
        or np.any(inflation < 1)
    ):
        raise ValueError("covariance axes or domains differ")
    validate_sparse_covariance(detector)
    residual = marginal - detector.diagonal() - (modes**2).sum(axis=1)
    if np.any(residual < -1e-10 * marginal):
        raise ValueError("declared cross-pixel terms exceed local posterior variance")
    residual = np.maximum(residual, 0) + (inflation - 1) * marginal
    matrix = detector + diags(residual)
    return (
        lambda vector: matrix @ vector + modes @ (modes.T @ vector),
        marginal * inflation,
        residual,
    )
