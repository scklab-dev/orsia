"""Draw explicit finite linear state processes with caller-owned RNGs."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def sample_linear_process(
    transition: NDArray[np.float64],
    stationary_root: NDArray[np.float64],
    innovation_root: NDArray[np.float64],
    count: int,
    rng: np.random.Generator,
    *,
    innovation_draw: Callable[[np.random.Generator, int], NDArray[np.float64]] | None = None,
    initial_draw: Callable[[np.random.Generator, int], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Draw one covariance-initialized state and chronological independent innovations.

    Args:
        transition: (state,state) one-step transition matrix.
        stationary_root: (state,draw) stationary covariance root in state units.
        innovation_root: (state,draw) innovation covariance root in state units.
        count: Positive number of chronological samples, including the initial state.
        rng: Explicit generator advanced once per sample, first for stationary state.
            Default draws are standard_normal(root.shape[1]) in chronological order.
        innovation_draw: Optional callback (rng,draw_count) returning a finite vector.
            Use zero-mean identity-covariance draws to retain the supplied covariance.
        initial_draw: Separate optional law for the initial state; defaults to Gaussian
            even when innovation_draw is supplied. Non-Gaussian higher moments need
            not be stationary under the transition.

    Returns:
        Owned (time,state) float64 state sequence. Inputs remain unchanged. Roots
        are supplied explicitly so a caller can preserve historical factorizations.

    Raises:
        ValueError: Matrix axes, finite values or sample count are invalid.
    """
    if transition.ndim != 2 or count < 1:
        raise ValueError("matrix transition and positive sample count required")
    n = transition.shape[0]
    if (
        transition.shape != (n, n)
        or stationary_root.ndim != 2
        or innovation_root.ndim != 2
        or stationary_root.shape[0] != n
        or innovation_root.shape[0] != n
        or count < 1
        or not all(np.isfinite(v).all() for v in (transition, stationary_root, innovation_root))
    ):
        raise ValueError("aligned finite state matrices and positive sample count required")
    state = stationary_root @ _draw(rng, stationary_root.shape[1], initial_draw)
    process = np.empty((count, n))
    process[0] = state
    for i in range(1, count):
        state = transition @ state + innovation_root @ _draw(
            rng, innovation_root.shape[1], innovation_draw
        )
        process[i] = state
    return process


def _draw(
    rng: np.random.Generator,
    size: int,
    callback: Callable[[np.random.Generator, int], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    values = rng.standard_normal(size) if callback is None else callback(rng, size)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError("draw callback must return a finite vector of the requested size")
    return values
