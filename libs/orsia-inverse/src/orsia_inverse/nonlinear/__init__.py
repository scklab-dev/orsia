"""Fit shared Gaussian objectives with explicitly selected nonlinear methods."""

from orsia_inverse.nonlinear.damped import solve_damped_gaussian
from orsia_inverse.nonlinear.estimation import FitTermination, GaussianEstimate, TrialDomainError
from orsia_inverse.nonlinear.map import solve_map

__all__ = [
    "FitTermination",
    "GaussianEstimate",
    "TrialDomainError",
    "solve_damped_gaussian",
    "solve_map",
]
