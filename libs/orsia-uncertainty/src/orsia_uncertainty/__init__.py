"""Generic covariance and linear uncertainty operations."""

from orsia_uncertainty.covariance import (
    CovarianceFactor,
    covariance_product,
    factor_covariance,
    propagate_covariance,
    propagate_vector,
    sample_gaussian,
    solve_covariance,
    validate_covariance,
    whiten,
)
from orsia_uncertainty.operations import combine_contributions, empirical_covariance, propagate_bias
from orsia_uncertainty.projection import (
    ProjectedCovariance,
    materialize_covariance,
    projected_covariance_product,
    projected_variances,
    select_projected_covariance,
    solve_diagonal_low_rank,
    transport_projected_covariance,
)

__all__ = [
    "ProjectedCovariance",
    "materialize_covariance",
    "projected_covariance_product",
    "projected_variances",
    "select_projected_covariance",
    "solve_diagonal_low_rank",
    "transport_projected_covariance",
    "covariance_product",
    "combine_contributions",
    "empirical_covariance",
    "CovarianceFactor",
    "factor_covariance",
    "whiten",
    "propagate_bias",
    "propagate_covariance",
    "propagate_vector",
    "sample_gaussian",
    "solve_covariance",
    "validate_covariance",
]
