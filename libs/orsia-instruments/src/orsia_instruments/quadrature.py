"""Build coordinate-measure nodes and weights for instrument integrations."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre


def interval_quadrature(
    low: float, high: float, order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fresh writable nodes and measure weights on a finite positive-width interval.

    Nodes use the units of the bounds; weights integrate that coordinate measure,
    not a normalized probability or physical response. No input is mutated.

    Args:
        low: Finite lower bound in the integrated coordinate's unit.
        high: Finite upper bound strictly above low, in the same unit.
        order: Positive number of Gauss-Legendre nodes.

    Returns:
        Fresh writable (nodes, weights) arrays for the interval measure.

    Raises:
        ValueError: Bounds are invalid or quadrature order is nonpositive.
    """
    if not np.isfinite([low, high]).all() or high <= low or order < 1:
        raise ValueError("quadrature needs finite ordered bounds and positive order")
    x, w = roots_legendre(order)
    return low + (x + 1) * ((high - low) / 2), w * ((high - low) / 2)


def validate_quadrature_order(value: int) -> None:
    """Validate a Gauss-Legendre order before constructing interval nodes.

    Args:
        value: Positive integer number of quadrature nodes.

    Raises:
        ValueError: Order is zero or negative.
    """
    if value < 1:
        raise ValueError("quadrature order must be positive")
