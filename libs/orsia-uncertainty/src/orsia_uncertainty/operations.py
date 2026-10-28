"""Propagate signed bias and combine or sample caller-labelled covariance arrays."""

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .covariance import _minimum_eigenvalue, propagate_vector, validate_covariance


def propagate_bias(bias: NDArray[np.float64], jacobian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Carry a signed source offset into target coordinates independently of noise.

    Args:
        bias: Signed float64 offset of shape (source,) in input-coordinate units.
        jacobian: Finite float64 (target, source) derivative in output/input units.

    Returns:
        Transformed signed offset (target,) in output-coordinate units.

    Raises:
        ValueError: Vector or Jacobian has invalid shape or nonfinite values.
    """
    return propagate_vector(bias, jacobian)


def combine_contributions(
    contributions: Mapping[str, NDArray[np.float64]],
    cross_covariances: Mapping[tuple[str, str], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Sum source covariances and explicit cross terms in caller-defined order.

    Values are variance vectors or dense matrices in one common numerical space.
    Keys identify sources, not correlation metadata. Require one (i,j) cross
    matrix for each pair in mapping insertion order, including explicit zeros.
    Reject impossible joint covariance even if the sum itself would be PSD.
    The returned dense matrix owns its storage; attribution stays with the caller.

    Args:
        contributions: Ordered names mapped to float64 source covariances.
        cross_covariances: One (source_i,source_j) dense term per unordered pair.

    Returns:
        Owned dense covariance including both directions of each cross term.

    Raises:
        ValueError: Sources, cross terms or their joint covariance are inconsistent.
    """
    if not contributions:
        raise ValueError("at least one contribution is required")
    ids = tuple(contributions)
    covariances = tuple(validate_covariance(value) for value in contributions.values())
    size = covariances[0].shape[0]
    if any(value.shape[0] != size for value in covariances):
        raise ValueError("contributions must share the same numerical dimensions")
    expected = {(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))}
    if set(cross_covariances) != expected:
        raise ValueError("declare exactly one cross covariance for every unordered source pair")
    independent = True
    for cross in cross_covariances.values():
        if (
            cross.dtype != np.float64
            or cross.shape != (size, size)
            or not np.all(np.isfinite(cross))
        ):
            raise ValueError("cross covariance must be a finite float64 matrix in the common space")
        independent = independent and not np.any(cross != 0)
    if not independent:
        joint = np.zeros((size * len(ids), size * len(ids)), dtype=np.float64)
        for i, covariance in enumerate(covariances):
            block = joint[i * size : (i + 1) * size, i * size : (i + 1) * size]
            if covariance.ndim == 1:
                np.fill_diagonal(block, covariance)
            else:
                block[:] = covariance
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cross = cross_covariances[(ids[i], ids[j])]
                joint[i * size : (i + 1) * size, j * size : (j + 1) * size] = cross
                joint[j * size : (j + 1) * size, i * size : (i + 1) * size] = cross.T
        if _minimum_eigenvalue(joint) < -1e-12 * np.max(np.abs(joint)):
            raise ValueError("joint source covariance must be positive semidefinite")
        del joint
    matrix = np.zeros((size, size), dtype=np.float64)
    diagonal_indices = np.diag_indices(size)
    for covariance in covariances:
        if covariance.ndim == 1:
            matrix[diagonal_indices] += covariance
        else:
            matrix += covariance
    if not independent:
        for cross in cross_covariances.values():
            matrix += cross
            matrix += cross.T
    return validate_covariance(matrix)


def sample_gaussian(
    covariance: NDArray[np.float64], rng: np.random.Generator, count: int
) -> NDArray[np.float64]:
    """Draw centred Gaussian perturbations with caller-owned RNG and declared order.

    Rows represent realisations, columns follow the covariance's vector order.
    Semidefinite covariances are supported without adding artificial variance.

    Args:
        covariance: Float64 (n,) variances or PSD (n,n) covariance.
        rng: Caller-owned NumPy generator consumed for each realization.
        count: Positive number of independent draws.

    Returns:
        Fresh float64 perturbations of shape (count,n), with draws in rows.

    Raises:
        ValueError: Covariance fails validation or the count is not positive.
    """
    return gaussian_sampler(covariance)(rng, count)


def gaussian_sampler(
    covariance: NDArray[np.float64],
) -> Callable[[np.random.Generator, int], NDArray[np.float64]]:
    """Prepare centred Gaussian draws for repeated use of one covariance.

    Return a callable accepting (caller-owned RNG, positive count). Output is a
    fresh writable float64 (count, size) array with realisations in rows and
    columns in caller-defined order. Dense preparation retains an owned square
    root factor, O(size**2); diagonal preparation retains O(size) scales. Singular
    PSD laws and the tiny-negative-eigenvalue clipping used by sample_gaussian
    are preserved, without artificial variance or regularisation. No RNG is
    consumed during preparation. Calls allocate their own work; callers manage
    RNG sharing themselves. No global cache or persistent input mutation occurs.

    Args:
        covariance: Float64 (n,) variances or positive-semidefinite (n,n) matrix.

    Returns:
        Callable accepting an explicit RNG and count, yielding owned (count,n) draws.

    Raises:
        ValueError: The covariance shape, dtype or PSD property is invalid.
    """
    covariance = validate_covariance(covariance)
    if covariance.ndim == 1:
        factor = np.sqrt(covariance)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    size = covariance.shape[0]
    diagonal = covariance.ndim == 1

    def draw(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
        if count < 1:
            raise ValueError("sample count must be positive")
        normal = rng.standard_normal((count, size))
        return normal * factor if diagonal else normal @ factor.T

    return draw


def empirical_covariance(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return unbiased dense covariance, centring finite float64 rows (N,n).

    Require N>=2 and n>=1; use N-1 denominator. Output is independent and writable.

    Args:
        samples: Finite float64 sample rows of shape (N,n), N at least two.

    Returns:
        Independent dense (n,n) unbiased sample covariance.

    Raises:
        ValueError: Samples have invalid shape, dtype or nonfinite values.
    """
    if (
        samples.dtype != np.float64
        or samples.ndim != 2
        or samples.shape[0] < 2
        or samples.shape[1] == 0
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("ensemble must have at least two finite float64 rows and one column")
    deviations = samples - np.mean(samples, axis=0)
    matrix = deviations.T @ deviations / (samples.shape[0] - 1)
    return validate_covariance(matrix)
