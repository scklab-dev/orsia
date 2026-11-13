"""Draw explicit finite linear Gaussian state processes with caller-owned RNGs."""

import numpy as np
from numpy.typing import NDArray


def sample_linear_process(
    transition: NDArray[np.float64],
    stationary_root: NDArray[np.float64],
    innovation_root: NDArray[np.float64],
    count: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw one stationary state and chronological independent innovations.

    Args:
        transition: (state,state) one-step transition matrix.
        stationary_root: (state,draw) stationary covariance root in state units.
        innovation_root: (state,draw) innovation covariance root in state units.
        count: Positive number of chronological samples, including the initial state.
        rng: Explicit generator advanced once per sample, first for stationary state.
            Each draw is standard_normal(root.shape[1]) in chronological order.

    Returns:
        Owned (time,state) float64 state sequence. Inputs remain unchanged. Roots
        are supplied explicitly so a caller can preserve historical factorizations.

    Raises:
        ValueError: Matrix axes, finite values or sample count are invalid.
    """
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
    state = stationary_root @ rng.standard_normal(stationary_root.shape[1])
    process = np.empty((count, n))
    process[0] = state
    for i in range(1, count):
        state = transition @ state + innovation_root @ rng.standard_normal(innovation_root.shape[1])
        process[i] = state
    return process
