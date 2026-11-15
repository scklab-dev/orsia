"""Select instrument channel coordinates under caller-supplied constraints and criteria."""

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class ChannelSearchResult[T](TypedDict):
    """Initial and locally optimal selections evaluated by the supplied objective.

    Attributes:
        initial: Initial caller-defined evaluation.
        optimized: Best accepted evaluation, with no global-optimality claim.
        iterations: Number of accepted exchanges.
        converged: Whether a complete sweep found no improving exchange.
    """

    initial: T
    optimized: T
    iterations: int
    converged: bool


def refine_channels[T](
    candidates: Sequence[FloatArray],
    minimum_spacing: Sequence[float],
    initial_selection: Sequence[FloatArray],
    evaluate: Callable[[list[FloatArray]], tuple[float, T]],
    *,
    maximum_iterations: int = 25,
    improvement_rtol: float = 1e-8,
) -> ChannelSearchResult[T]:
    """Refine an initial channel selection under spacing and objective constraints.

    Search single-channel exchanges in fixed deterministic order. Termination
    establishes a local stopping condition, with no global-optimum guarantee.

    Args:
        candidates: Increasing candidate coordinates for each window, caller units.
        minimum_spacing: Minimum within-window spacing in the same units.
        initial_selection: Ordered distinct selected coordinates for each window.
        evaluate: Pure callback returning objective to minimize and caller evaluation.
            Singular candidate information may raise numpy.linalg.LinAlgError and
            is skipped. Other exceptions propagate. Callback inputs stay unchanged.
        maximum_iterations: Positive maximum accepted-exchange sweeps.
        improvement_rtol: Required relative objective improvement in [0,1).

    Returns:
        Initial and best evaluations, accepted iterations and local convergence.

    Raises:
        ValueError: Search controls or window counts are invalid.
    """
    if (
        maximum_iterations < 1
        or not 0 <= improvement_rtol < 1
        or len(candidates) != len(initial_selection)
        or len(minimum_spacing) != len(candidates)
    ):
        raise ValueError("invalid optimization controls")
    selection = [np.array(v, dtype=float, copy=True) for v in initial_selection]
    current_score, current = evaluate(selection)
    initial = current
    iterations, converged = 0, False
    for iteration in range(1, maximum_iterations + 1):
        best, best_score, best_selection = current, current_score, selection
        for w, grid in enumerate(candidates):
            selected = selection[w]
            for position in range(len(selected)):
                for candidate in grid:
                    if candidate in selected:
                        continue
                    replacement = selected.copy()
                    replacement[position] = candidate
                    replacement.sort()
                    if np.any(np.diff(replacement) < minimum_spacing[w]):
                        continue
                    proposed = [v.copy() for v in selection]
                    proposed[w] = replacement
                    try:
                        score, evaluation = evaluate(proposed)
                    except np.linalg.LinAlgError:
                        continue
                    if score < best_score * (1 - improvement_rtol):
                        best, best_score, best_selection = evaluation, score, proposed
        if best is current:
            converged = True
            break
        current, current_score, selection = best, best_score, best_selection
        iterations = iteration
    return dict(initial=initial, optimized=current, iterations=iterations, converged=converged)
