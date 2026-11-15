"""Sample standardized perturbation laws with explicit normalization and RNGs."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


def standardized_draw(
    rng: np.random.Generator,
    shape: int | tuple[int, ...],
    distribution: str,
    degrees: float = 5.0,
    *,
    shared: bool = False,
    outlier_probability: float = 0.001,
    outlier_scale: float = 5.0,
) -> FloatArray:
    """Draw zero-mean unit-variance perturbations with an explicit generator.

    Args:
        rng: Borrowed generator, advanced in the documented law's draw order.
        shape: Output dimensions, passed to the generator unchanged.
        distribution: uniform, gaussian, student_t or contaminated_gaussian.
        degrees: Finite Student degrees of freedom greater than two.
        shared: One Student chi-square scale for the entire output. Other laws
            use independent entries; contaminated Gaussian requires False.
        outlier_probability: Contaminated-Gaussian outlier probability in [0,1).
        outlier_scale: Positive outlier standard-deviation multiplier.

    Returns:
        Fresh float64 unit-variance samples. Shared Student draws its scale before
        its Gaussian entries. Contamination draws the mask before Gaussian entries.
        Independent Student uses Generator.standard_t with variance normalization.

    Raises:
        ValueError: Distribution, degrees, shape or contamination parameters fail.
    """
    if not np.isfinite(degrees) or degrees <= 2:
        raise ValueError("finite Student-t variance required")
    if distribution == "uniform":
        return np.sqrt(3) * (2 * rng.random(shape) - 1)
    if distribution == "gaussian":
        return rng.standard_normal(shape)
    if distribution == "student_t":
        if shared:
            return np.sqrt((degrees - 2) / rng.chisquare(degrees)) * rng.standard_normal(shape)
        return np.sqrt((degrees - 2) / degrees) * rng.standard_t(degrees, shape)
    if distribution == "contaminated_gaussian" and not shared:
        if not 0 <= outlier_probability < 1 or not np.isfinite(outlier_scale) or outlier_scale <= 0:
            raise ValueError("invalid contamination")
        scale = np.where(rng.random(shape) < outlier_probability, outlier_scale, 1)
        return (
            scale
            * rng.standard_normal(shape)
            / np.sqrt(1 - outlier_probability + outlier_probability * outlier_scale**2)
        )
    raise ValueError("unsupported standardized law")
