"""Own Gaussian puff releases, population invariants and conservative subdivision."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from orsia_dispersion.emissions import release_intervals
from orsia_dispersion.forcing import Boundary, FloatArray, validate_tensors


@dataclass(frozen=True)
class PuffReleases:
    """Scheduled finite emissions with independent writable float64 storage.

    Attributes:
        release_times_s: Nondecreasing elapsed SI seconds, shape (r,).
        masses_kg: Strictly positive masses, shape (r,).
        centres_m: East, north, up centres in metres, shape (r, 3).
        covariances_m2: Symmetric positive-definite covariances, shape (r, 3, 3).
            All arrays are copied on construction; empty populations are valid.
    """

    release_times_s: FloatArray
    masses_kg: FloatArray
    centres_m: FloatArray
    covariances_m2: FloatArray

    def __post_init__(self) -> None:
        """Copy and validate release arrays.

        Raises:
            ValueError: Arrays violate shape, finiteness, mass, covariance or time order.
        """
        _copy_population(self)
        if np.any(np.diff(self.release_times_s) < 0):
            raise ValueError("release times must be nondecreasing")


@dataclass(frozen=True)
class GaussianPuffs:
    """Active Gaussian masses at one time, with independent writable storage.

    Attributes:
        time_s: Common elapsed SI time in seconds.
        release_times_s: Original release times, shape (p,), at or before time_s.
        masses_kg: Strictly positive masses, shape (p,).
        centres_m: ENU centres in metres, shape (p, 3).
        covariances_m2: SPD spatial covariances in m2, shape (p, 3, 3).
            Construction copies all arrays to float64. Frozen fields retain writable
            buffers; callers must keep them stable during calculations. In reflecting
            mode these moments parameterize the underlying unreflected Gaussian.
    """

    time_s: float
    release_times_s: FloatArray
    masses_kg: FloatArray
    centres_m: FloatArray
    covariances_m2: FloatArray

    def __post_init__(self) -> None:
        """Copy and validate the active distribution.

        Raises:
            ValueError: Population values are invalid or a release is in the future.
        """
        _copy_population(self)
        if not np.isfinite(self.time_s) or np.any(self.release_times_s > self.time_s):
            raise ValueError("time_s must be finite and at or after all releases")


def discretize_puff_emissions(
    edges_s: FloatArray,
    rates_kg_s: FloatArray,
    centres_m: FloatArray,
    covariances_m2: FloatArray,
    *,
    partition_times_s: FloatArray | None = None,
    stop_time_s: float | None = None,
) -> PuffReleases:
    """Integrate piecewise-constant emission rates into midpoint releases.

    Args:
        edges_s: Strictly increasing finite interval edges, shape (n+1,).
        rates_kg_s: Nonnegative rates in kg/s, shape (n,).
        centres_m: Initial ENU centres per interval, shape (n, 3), metres.
        covariances_m2: Initial SPD covariances per interval, shape (n, 3, 3), m2.
        partition_times_s: Optional increasing output/acquisition times. Interior
            times split intervals so cumulative mass is exact at those times.
        stop_time_s: Optional finite end time; clips the last included interval.

    Returns:
        Independent releases with exact included masses; zero-rate intervals omitted.

    Raises:
        ValueError: Inputs violate shape, finiteness, time order or physical bounds.
    """
    times, masses, indices = release_intervals(edges_s, rates_kg_s, partition_times_s, stop_time_s)
    edges = np.asarray(edges_s, dtype=np.float64)
    n = len(edges) - 1
    # Validate every source, including sources whose rate is zero.
    sources = PuffReleases(edges[:-1], np.ones(n), centres_m, covariances_m2)
    return PuffReleases(
        times,
        masses,
        sources.centres_m[indices],
        sources.covariances_m2[indices],
    )


def split_puffs(
    state: GaussianPuffs,
    indices: NDArray[np.int64],
    axes: NDArray[np.int64],
    fractions: FloatArray,
    *,
    boundary: Boundary,
) -> GaussianPuffs:
    """Replace selected parents by equal-mass eigenvector-displaced children.

    Args:
        state: Stable active distribution; no input buffers are changed.
        indices: Unique parent indices, shape (s,), within the population.
        axes: Eigenvector indices, shape (s,). Free-space axes 0,1,2 follow
            ascending eigenvalues. Reflecting axes 0,1 use the horizontal block.
        fractions: Fractions alpha in (0,1), shape (s,), of the principal variance
            moved into child separation. Child covariance removes alpha*lambda*e*e.T.
        boundary: Free space or separable reflecting ground; vertical splitting forbidden.

    Returns:
        Independent state containing unselected parents then paired children.
        Release times, mass, mean and covariance of the mixture are conserved.
        Density and higher moments generally change.

    Raises:
        ValueError: Selection, fraction, boundary or resulting covariance is invalid.
    """
    validate_boundary(state, boundary)
    indices = np.asarray(indices)
    axes = np.asarray(axes)
    fractions = np.asarray(fractions, dtype=np.float64)
    n = len(state.masses_kg)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be an integer vector")
    if np.any((indices < 0) | (indices >= n)) or len(np.unique(indices)) != len(indices):
        raise ValueError("indices must be unique and within the population")
    if axes.shape != indices.shape or not np.issubdtype(axes.dtype, np.integer):
        raise ValueError("axes must be an integer vector matching indices")
    if np.any((axes < 0) | (axes >= (2 if boundary == "reflecting" else 3))):
        raise ValueError("split axis is unsupported for this boundary")
    if fractions.shape != indices.shape or not np.all(np.isfinite(fractions)):
        raise ValueError("fractions must be finite and match indices")
    if np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError("split fractions must lie strictly between zero and one")
    values, vectors = puff_eigenvectors(state, boundary)
    direction = vectors[indices, :, axes]
    variance = fractions * values[indices, axes]
    offset = np.sqrt(variance)[:, None] * direction
    children = state.centres_m[indices, None, :] + np.stack((-offset, offset), axis=1)
    covariance = state.covariances_m2[indices] - np.einsum(
        "p,pi,pj->pij", variance, direction, direction
    )
    keep = np.ones(n, dtype=bool)
    keep[indices] = False
    return GaussianPuffs(
        state.time_s,
        np.concatenate((state.release_times_s[keep], np.repeat(state.release_times_s[indices], 2))),
        np.concatenate((state.masses_kg[keep], np.repeat(state.masses_kg[indices] / 2, 2))),
        np.concatenate((state.centres_m[keep], children.reshape(-1, 3))),
        np.concatenate((state.covariances_m2[keep], np.repeat(covariance, 2, axis=0))),
    )


def validate_boundary(state: GaussianPuffs, boundary: Boundary) -> None:
    """Validate a stable snapshot and the separable reflecting-ground contract.

    Args:
        state: Active distribution; writable buffers are checked on every operation.
        boundary: Free space or flat reflecting ground at z=0.

    Raises:
        ValueError: Values are invalid, or reflection has tilted covariance/low centres.
    """
    _validate_population(state)
    if not np.isfinite(state.time_s) or np.any(state.release_times_s > state.time_s):
        raise ValueError("invalid active time")
    if boundary not in ("free", "reflecting"):
        raise ValueError("boundary must be 'free' or 'reflecting'")
    if boundary == "reflecting":
        if np.any(state.centres_m[:, 2] < 0):
            raise ValueError("reflecting centres must be at or above z=0")
        if np.any(state.covariances_m2[:, :2, 2] != 0):
            raise ValueError("reflecting covariance must be vertically separable")


def puff_eigenvectors(state: GaussianPuffs, boundary: Boundary) -> tuple[FloatArray, FloatArray]:
    """Return principal variances and directions for splitting and envelope probes.

    Args:
        state: Validated active distribution.
        boundary: Reflecting mode orders horizontal eigenvectors before vertical.

    Returns:
        Variances (p,3) and orthonormal column eigenvectors (p,3,3), fresh arrays.
    """
    if boundary == "free":
        return np.linalg.eigh(state.covariances_m2)
    p = len(state.masses_kg)
    values = np.zeros((p, 3))
    vectors = np.zeros((p, 3, 3))
    values[:, :2], vectors[:, :2, :2] = np.linalg.eigh(state.covariances_m2[:, :2, :2])
    values[:, 2] = state.covariances_m2[:, 2, 2]
    vectors[:, 2, 2] = 1
    return values, vectors


def _copy_population(state: PuffReleases | GaussianPuffs) -> None:
    for name in ("release_times_s", "masses_kg", "centres_m", "covariances_m2"):
        object.__setattr__(state, name, np.array(getattr(state, name), dtype=np.float64, copy=True))
    _validate_population(state)


def _validate_population(state: PuffReleases | GaussianPuffs) -> None:
    n = state.masses_kg.size
    for name, shape in (
        ("release_times_s", (n,)),
        ("masses_kg", (n,)),
        ("centres_m", (n, 3)),
        ("covariances_m2", (n, 3, 3)),
    ):
        value = getattr(state, name)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(state.masses_kg <= 0) or not np.isfinite(np.sum(state.masses_kg)):
        raise ValueError("masses must be positive with finite total")
    validate_tensors(state.covariances_m2, positive=True)
