"""Evaluate named perturbations of layer or spectral physical profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ProfileDomain = Literal["layer", "spectral"]


def validate_profile_values(
    values: NDArray[np.float64], n: int, name: str, *, minimum: float | None = None
) -> None:
    """Check finite (n,) physical values and an optional inclusive minimum.

    Do not mutate or retain the supplied array.

    Args:
        values: Profile vector expected to contain ``n`` finite values.
        n: Expected number of layer or spectral samples.
        name: Quantity name used to identify validation failures.
        minimum: Optional inclusive physical lower bound.

    Raises:
        ValueError: Values differ in shape, contain nonfinite entries or fall
            below the requested physical minimum.
    """
    if (
        values.shape != (n,)
        or not np.all(np.isfinite(values))
        or (minimum is not None and np.any(values < minimum))
    ):
        raise ValueError(
            f"{name} must contain finite values with shape ({n},) in its physical domain"
        )


def check_profile_bounds(
    values: NDArray[np.float64], low: float | None, high: float | None
) -> None:
    """Check inclusive physical bounds on already validated finite profile values.

    None leaves that side unbounded; do not mutate or retain the input.

    Args:
        values: Previously validated finite profile values in their physical unit.
        low: Optional inclusive lower bound in that unit.
        high: Optional inclusive upper bound in that unit.

    Raises:
        ValueError: At least one value falls outside a supplied bound.
    """
    if (low is not None and np.any(values < low)) or (high is not None and np.any(values > high)):
        raise ValueError("profile leaves declared bounds")


def evaluate_profile(
    basis: LayerProfileBasis, coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return fresh writable (N,) values, baseline + coefficients @ directions.

    Coefficients have shape (n_directions,), are dimensionless and are neither
    mutated nor retained. Reject nonfinite or out-of-domain evaluated values.

    Args:
        basis: Named baseline and additive directions with physical value bounds.
        coefficients: Dimensionless (n_directions,) multipliers in basis name order.

    Returns:
        Fresh writable (N,) profile in the basis unit and sample order.

    Raises:
        ValueError: Coefficients or evaluated profile violate shape, finiteness
            or the basis physical bounds.
    """
    validate_profile_values(coefficients, len(basis.names), "profile coefficients")
    values = basis.baseline + coefficients @ basis.directions
    validate_profile_values(values, len(basis.baseline), "evaluated profile")
    check_profile_bounds(values, basis.minimum, basis.maximum)
    return values


@dataclass(frozen=True, eq=False)
class LayerProfileBasis:
    """Additive directions in a named unit, coefficients dimensionless.

    Baseline/directions follow the atmosphere's bottom-to-top N-layer order or
    a surface's spectral axis. Rebuild and validate a new state after evaluation;
    evaluating a basis never mutates the original state or coupled columns.
    Baseline has shape (N,), directions (n_directions, N); both are writable
    float64 snapshots. Names label the first direction axis, not the sample axis.

    Attributes:
        unit: Physical unit for baseline and additive direction values.
        baseline: (N,) reference layer or spectral values.
        names: Unique names of the additive perturbation directions.
        directions: (named directions,N) values per unit dimensionless coefficient.
        minimum: Optional inclusive lower bound on evaluated values.
        maximum: Optional inclusive upper bound on evaluated values.
        domain: Layer or spectral interpretation of the last axis.
    """

    unit: str
    baseline: NDArray[np.float64]
    names: tuple[str, ...]
    directions: NDArray[np.float64]
    minimum: float | None = None
    maximum: float | None = None
    domain: ProfileDomain = "layer"
    __hash__ = None

    def __post_init__(self) -> None:
        n = self.baseline.size
        if not self.unit or n == 0 or not self.names:
            raise ValueError("profile requires a unit, baseline and named directions")
        if self.directions.shape != (len(self.names), n) or not np.all(
            np.isfinite(self.directions)
        ):
            raise ValueError("profile directions must be finite with shape (directions, N)")
        if any(not name for name in self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("profile direction names must be distinct and nonempty")
        if self.minimum is not None and not isfinite(self.minimum):
            raise ValueError("non-finite profile minimum")
        if self.maximum is not None and not isfinite(self.maximum):
            raise ValueError("non-finite profile maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("profile minimum exceeds maximum")
        if self.domain not in ("layer", "spectral"):
            raise ValueError("profile domain must be layer or spectral")
        validate_profile_values(self.baseline, n, "profile baseline")
        check_profile_bounds(self.baseline, self.minimum, self.maximum)
        object.__setattr__(self, "baseline", np.copy(self.baseline))
        object.__setattr__(self, "directions", np.copy(self.directions))
