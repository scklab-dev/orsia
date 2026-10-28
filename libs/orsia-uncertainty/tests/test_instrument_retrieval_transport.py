"""Synthetic instrument-to-retrieval closure using existing public operations.

Dimensionless instrument perturbations form three DN samples; a prescribed
linear retrieval sensitivity maps DN to two dimensionless state errors. This
is not an instrument simulator or an atmospheric retrieval implementation.
"""

import numpy as np
from orsia_uncertainty import (
    combine_contributions,
    propagate_bias,
    propagate_covariance,
    validate_covariance,
)


def test_correlated_instrument_sources_to_retrieval() -> None:
    """Keep source and cross attribution; compare joint and staged transport."""

    # A joint law in dimensionless parameter coordinates, with a rectangular
    # cross block. The factor independently guarantees positive definiteness.
    factor = np.array([[0.2, 0.0, 0.0], [0.04, 0.15, 0.0], [0.08, -0.03, 0.1]])
    joint = factor @ factor.T
    ja = np.array([[2.0, -1.0], [0.5, 3.0], [-1.0, 0.4]])  # DN / 1
    jb = np.array([[1.2], [-0.7], [2.0]])  # DN / 1
    g = np.array([[0.4, -0.1, 0.2], [-0.3, 0.5, 0.1]])  # 1 / DN
    ca = validate_covariance(joint[:2, :2])
    cb = validate_covariance(joint[2:, 2:])
    contributions = (
        ("spectral", propagate_covariance(ca, ja)),
        ("gain", propagate_covariance(cb, jb)),
    )
    cross = ja @ joint[:2, 2:] @ jb.T
    combined = combine_contributions(dict(contributions), {("spectral", "gain"): cross})
    result = propagate_covariance(combined, g)
    h = np.column_stack((ja, jb))
    expected = g @ h @ joint @ h.T @ g.T
    np.testing.assert_allclose(result, expected, atol=1e-14)
    # Transport each source before combining, retaining the cross contribution.
    transported = tuple((item[0], propagate_covariance(item[1], g)) for item in contributions)
    direct = combine_contributions(dict(transported), {("spectral", "gain"): g @ cross @ g.T})
    np.testing.assert_allclose(direct, expected, atol=1e-14)
    independent = g @ (ja @ joint[:2, :2] @ ja.T + jb @ joint[2:, 2:] @ jb.T) @ g.T
    assert np.linalg.norm(expected - independent) > 0.01

    # Conditional detector noise remains a separate independent contribution.
    noise = np.diag([0.01, 0.02, 0.015])
    detector = propagate_covariance(validate_covariance(noise), g)
    total = result + detector
    bias = np.array([0.03, -0.02, 0.01])
    ba = propagate_bias(np.copy(bias[:2]), ja)
    bb = propagate_bias(np.copy(bias[2:]), jb)
    retrieved_bias = propagate_bias(np.copy(ba + bb), g)
    np.testing.assert_allclose(retrieved_bias, g @ h @ bias, atol=1e-14)

    # Independently sample the joint factor, not the propagated covariance.
    rng = np.random.default_rng(72814)
    draws = 120_000
    parameters = rng.standard_normal((draws, 3)) @ factor.T + bias
    detector_draws = rng.standard_normal((draws, 3)) * np.sqrt(np.diag(noise))
    errors = (parameters @ h.T + detector_draws) @ g.T
    empirical = np.cov(errors, rowvar=False)
    np.testing.assert_allclose(empirical, total, rtol=0.025, atol=0.0005)
    standard_error = np.sqrt(np.diag(total) / draws)
    assert np.all(np.abs(errors.mean(axis=0) - retrieved_bias) < 5 * standard_error)


def test_nonlinear_retrieval_exposes_linearization_limit() -> None:
    """A quadratic response has exact Gaussian moments beyond its Jacobian."""
    rng = np.random.default_rng(491)
    normal = rng.standard_normal(160_000)
    quadratic = 2.0
    discrepancies = []
    for sigma in (0.01, 0.3):
        perturbation = sigma * normal
        errors = perturbation + quadratic * perturbation**2
        expected_mean = quadratic * sigma**2
        expected_variance = sigma**2 + 2 * quadratic**2 * sigma**4
        np.testing.assert_allclose(
            errors.mean(), expected_mean, atol=5 * np.sqrt(expected_variance / normal.size)
        )
        np.testing.assert_allclose(errors.var(ddof=1), expected_variance, rtol=0.025)
        discrepancies.append((expected_variance - sigma**2) / sigma**2)
    assert discrepancies[0] < 0.001
    assert discrepancies[1] > 0.5
    # The derivative at zero is one: linear transport predicts sigma**2 and
    # zero bias, missing both the quadratic variance and curvature-induced mean.
