"""Share conservative Gaussian and compact ellipsoidal puff kernels.

A compact kernel preserves the population's mass, centre and unreflected spatial
covariance. Reflection folds the kernel at z=0 after this moment definition.
"""

from collections.abc import Iterator

import numpy as np
from scipy.special import erf, gammainc, gammaincc, ndtr

from orsia_dispersion.forcing import Boundary, FloatArray
from orsia_dispersion.puffs.population import GaussianPuffs


def compact_kernel_parameters(radius: float | None) -> tuple[float, float]:
    """Compute covariance inflation and normalization for a radial Gaussian cut.

    Args:
        radius: Positive finite cutoff in the inflated kernel's Mahalanobis units,
            or None for an unbounded Gaussian. Extremely small cutoffs whose
            normalization underflows float64 are unsupported.

    Returns:
        Cholesky multiplier and log retained Gaussian probability. If Sigma is
        the stored covariance, the kernel covariance is Sigma/alpha, with
        alpha=F_chi2(5)(radius**2)/F_chi2(3)(radius**2).

    Raises:
        ValueError: The radius or its normalization is numerically unsupported.
    """
    if radius is None:
        return 1.0, 0.0
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("kernel_cutoff_radius must be positive and finite")
    # Beyond this value both retained probabilities round to one. Avoid overflow
    # in radius squared while retaining the caller's finite support in evaluation.
    argument = min(radius, 40.0) ** 2 / 2
    retained = float(gammainc(1.5, argument))
    second = float(gammainc(2.5, argument))
    if retained <= 0 or second <= 0:
        raise ValueError("kernel_cutoff_radius normalization underflows float64")
    return float(np.sqrt(retained / second)), float(np.log(retained))


def puff_kernels(
    state: GaussianPuffs, boundary: Boundary, cutoff_radius: float | None = None
) -> Iterator[tuple[FloatArray, FloatArray, float]]:
    """Yield normalized kernels shared by point, path and pixel integrals.

    Args:
        state: Validated population with stable buffers.
        boundary: Free space or validated vertically separable reflection at z=0.
        cutoff_radius: Optional compact-kernel radius; see compact_kernel_parameters.

    Yields:
        Centre (3,), inflated Cholesky factor (3,3), and normalized log mass.
        Reflection adds an image kernel; callers retain only the upper half-space.

    Raises:
        ValueError: The cutoff cannot be normalized in float64.
    """
    multiplier, log_probability = compact_kernel_parameters(cutoff_radius)
    for centre, covariance, mass in zip(
        state.centres_m, state.covariances_m2, state.masses_kg, strict=True
    ):
        chol = np.linalg.cholesky(covariance) * multiplier
        log_mass = float(np.log(mass) - log_probability)
        yield centre, chol, log_mass
        if boundary == "reflecting":
            image = centre.copy()
            image[2] *= -1
            yield image, chol, log_mass


def clip_kernel_interval(
    lower: FloatArray, upper: FloatArray, perpendicular_squared: FloatArray, radius: float | None
) -> tuple[FloatArray, FloatArray]:
    """Intersect standardized line intervals with an ellipsoidal kernel.

    Args:
        lower: Ordered lower line coordinates in whitened kernel units.
        upper: Upper coordinates broadcast-compatible with lower.
        perpendicular_squared: Squared perpendicular distance in those units.
        radius: Optional compact cutoff; None leaves the intervals unchanged.

    Returns:
        Broadcast clipped endpoints. Disjoint intervals have equal endpoints,
        including lines outside the compact support.
    """
    if radius is None:
        return lower, upper
    half_width = np.sqrt(np.maximum(radius * radius - perpendicular_squared, 0.0))
    return np.clip(lower, -half_width, half_width), np.clip(upper, -half_width, half_width)


def kernel_axis_cdf(upper: FloatArray, radius: float | None) -> FloatArray:
    """Integrate a normalized kernel along one whitened marginal axis.

    Args:
        upper: Standardized upper limits in the inflated kernel coordinates.
        radius: Optional validated radial cutoff.

    Returns:
        Probabilities below each limit. Spherical symmetry gives the same
        marginal for any axis, including an arbitrarily tilted altitude axis.
    """
    if radius is None:
        return ndtr(upper)
    _, log_probability = compact_kernel_parameters(radius)
    distance = np.minimum(np.abs(upper), radius)
    if radius < 0.1:
        nodes, weights = np.polynomial.legendre.leggauss(16)
        x = distance[..., None] * (nodes + 1) / 2
        integral = (
            distance
            / 2
            * np.sum(weights * np.expm1((radius * radius - x * x) / 2), axis=-1)
            * np.exp(-radius * radius / 2)
            / np.sqrt(2 * np.pi)
        )
    else:
        integral = erf(distance / np.sqrt(2)) / 2 - distance * (
            np.exp(-(min(radius, 40.0) ** 2) / 2) / np.sqrt(2 * np.pi)
        )
    result = 0.5 + np.sign(upper) * integral / np.exp(log_probability)
    return np.where(
        upper <= -radius, 0.0, np.where(upper >= radius, 1.0, np.clip(result, 0.0, 1.0))
    )


def kernel_envelope_tail(envelope_sigma: float, radius: float | None) -> float:
    """Compute mass outside a covariance-scaled forcing ellipsoid.

    Args:
        envelope_sigma: Positive radius in stored spatial-covariance units.
        radius: Optional validated compact-kernel cutoff.

    Returns:
        Retained-kernel mass fraction outside the envelope, before reflection.
        Folding gives an upper bound for mass outside the reflected envelope.
    """
    multiplier, log_probability = compact_kernel_parameters(radius)
    if radius is None:
        return float(gammaincc(1.5, envelope_sigma**2 / 2))
    cut = min(envelope_sigma / multiplier, radius)
    a, b = min(cut, 40.0) ** 2 / 2, min(radius, 40.0) ** 2 / 2
    difference = (
        gammainc(1.5, b) - gammainc(1.5, a) if radius < 2 else gammaincc(1.5, a) - gammaincc(1.5, b)
    )
    return float(np.clip(difference / np.exp(log_probability), 0.0, 1.0))
