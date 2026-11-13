"""Dense reference, sparse storage, singular support and specialized solve checks."""

from typing import Any, cast

import numpy as np
import pytest
from orsia_uncertainty import (
    ProjectedCovariance,
    materialize_covariance,
    projected_covariance_product,
    projected_variances,
    propagate_covariance,
    select_projected_covariance,
    solve_diagonal_low_rank,
    transport_projected_covariance,
    validate_covariance,
)
from scipy.sparse import csc_array, csr_array, diags_array


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
def test_projection_reference_and_ownership(sparse, diagonal):
    m = np.array([[1.0, 2.0, 0.0], [-1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    q = (
        np.array([1.0, 0.0, 3.0])
        if diagonal
        else np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    u = np.array([[1.0], [2.0], [3.0]])
    reference = (m * q) @ m.T if diagonal else m @ q @ m.T
    reference += u @ u.T
    original = csr_array(m) if sparse else m
    cov = ProjectedCovariance((original, u), (q, np.ones(1)))
    m[:] = 99
    q[:] = 99
    if isinstance(original, csr_array):
        original.data[:] = 99
    np.testing.assert_allclose(materialize_covariance(cov, max_bytes=72), reference)
    np.testing.assert_allclose(projected_variances(cov), np.diag(reference))
    for rhs in (np.arange(3.0), np.arange(6.0).reshape(3, 2)):
        np.testing.assert_allclose(projected_covariance_product(cov, rhs), reference @ rhs)
    selection = np.array([2, 0, 0], dtype=np.int64)
    np.testing.assert_allclose(
        materialize_covariance(select_projected_covariance(cov, selection), max_bytes=72),
        reference[np.ix_(selection, selection)],
    )
    t = np.array([[0.25, 0.75, 0.0], [0.0, 0.0, 1.0]])
    for transform in (t, csr_array(t)):
        transported = transport_projected_covariance(cov, transform)
        np.testing.assert_allclose(
            materialize_covariance(transported, max_bytes=32), t @ reference @ t.T
        )


def test_sparse_duplicate_entries_and_large_diagonal():
    duplicated = csr_array(
        (np.array([0.25, 0.75]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1)
    )
    cov = ProjectedCovariance((duplicated,), (np.ones(1),))
    assert projected_variances(cov)[0] == 1
    # A dense identity/covariance here would require 80 GB.
    n = 100_000
    big = ProjectedCovariance((csr_array(diags_array(np.ones(n))),), (np.ones(n),))
    assert isinstance(big.projections[0], csr_array)
    assert big.projections[0].nnz == n
    np.testing.assert_array_equal(projected_covariance_product(big, np.ones(n)), np.ones(n))
    np.testing.assert_array_equal(projected_variances(big), np.ones(n))
    with pytest.raises(MemoryError):
        materialize_covariance(big, max_bytes=1_000_000)


@pytest.mark.parametrize(
    "m,q",
    [
        ((), ()),
        ((np.ones((1, 1)),), ()),
        ((np.ones((1, 2)),), (np.ones(1),)),
        ((np.ones((0, 1)),), (np.ones(1),)),
        ((np.ones((1, 1), dtype=np.float32),), (np.ones(1),)),
        ((np.array([[np.nan]]),), (np.ones(1),)),
        ((csr_array([[np.inf]]),), (np.ones(1),)),
        ((csc_array([[1.0]]),), (np.ones(1),)),
        ((np.ones((1, 2)),), (np.array([[1.0, 2.0], [2.0, 1.0]]),)),
    ],
)
def test_invalid_sources(m, q):
    with pytest.raises(ValueError):
        ProjectedCovariance(m, q)


def test_invalid_operations_and_singular_support():
    cov = ProjectedCovariance((np.ones((2, 1)),), (np.zeros(1),))
    np.testing.assert_array_equal(materialize_covariance(cov, max_bytes=32), 0)
    for x in (
        np.ones(1),
        np.array([np.inf, 1.0]),
        np.ones(2, dtype=np.float32),
        np.ones((2, 1, 1)),
    ):
        with pytest.raises(ValueError):
            projected_covariance_product(cov, cast(Any, x))
    for idx in (
        np.array([], dtype=np.int64),
        np.array([2]),
        np.array([-1]),
        np.array([0.0]),
        np.array([[0]]),
    ):
        with pytest.raises(ValueError):
            select_projected_covariance(cov, idx)
    with pytest.raises(ValueError):
        transport_projected_covariance(cov, np.ones((1, 3)))
    with pytest.raises(MemoryError):
        materialize_covariance(cov, max_bytes=-1)


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_diagonal_low_rank_solve(rank):
    rng = np.random.default_rng(27)
    d = rng.uniform(0.5, 2, 8)
    u = rng.normal(size=(8, rank))
    for rhs in (rng.normal(size=8), rng.normal(size=(8, 2))):
        solved = solve_diagonal_low_rank(d, u, rhs)
        np.testing.assert_allclose(
            solved, np.linalg.solve(np.diag(d) + u @ u.T, rhs), rtol=2e-13, atol=2e-14
        )
    for bad in (np.zeros(8), np.eye(8), -d):
        with pytest.raises(ValueError):
            solve_diagonal_low_rank(bad, u, np.ones(8))
    for bad in (np.ones((7, rank)), np.full((8, 1), np.nan), np.ones(8)):
        with pytest.raises(ValueError):
            solve_diagonal_low_rank(d, bad, np.ones(8))


def test_solve_rejects_cancellation():
    with pytest.raises(ValueError, match="accuracy"):
        solve_diagonal_low_rank(np.ones(1), np.array([[1e10]]), np.ones(1))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e7])
def test_projection_rejects_negative_marginals(sparse, scale):
    q = np.diag([1.0, -1e-13])
    m = np.array([[0.0, scale]])
    # The source-space tolerance accepts Q, but the projected variance is negative.
    np.testing.assert_array_equal(validate_covariance(q), q)
    with pytest.raises(ValueError, match="positive semidefinite"):
        propagate_covariance(q, m)
    with pytest.raises(ValueError, match="negative"):
        ProjectedCovariance((csr_array(m) if sparse else m,), (q,))


def test_transport_rechecks_marginals_without_changing_source():
    q = np.diag([1.0, -1e-13])
    m = np.array([[1.0, 0.0], [1.0, 1.0]])
    cov = ProjectedCovariance((m,), (q,))
    np.testing.assert_array_equal(cov.source_covariances[0], q)
    # Positive marginals do not certify all directions of a tolerance-accepted Q.
    for transform in (np.array([[-1e7, 1e7]]), csr_array([[-1e7, 1e7]])):
        with pytest.raises(ValueError, match="negative"):
            transport_projected_covariance(cov, transform)


def test_products_do_not_repeat_marginal_validation(monkeypatch):
    import orsia_uncertainty.projection as projection

    cov = ProjectedCovariance((np.eye(2),), (np.ones(2),))

    def unexpected(*args, **kwargs):
        pytest.fail("products must reuse construction invariants")

    monkeypatch.setattr(projection, "projected_variances", unexpected)
    np.testing.assert_array_equal(projected_covariance_product(cov, np.ones(2)), 1)
    np.testing.assert_array_equal(materialize_covariance(cov, max_bytes=32), np.eye(2))


def test_overflow_is_explicit():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            ProjectedCovariance((np.array([[1e200]]),), (np.ones(1),))
        cov = ProjectedCovariance((np.array([[1e100]]),), (np.ones(1),))
        with pytest.raises(ValueError, match="finite"):
            projected_covariance_product(cov, np.array([1e200]))
        with pytest.raises(ValueError, match="finite"):
            solve_diagonal_low_rank(np.array([1e-300]), np.empty((1, 0)), np.array([1e300]))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("weight,variance", [(1e200, 1e-300), (1e-200, 1e300)])
def test_diagonal_contraction_preserves_finite_scaled_products(sparse, weight, variance):
    m = np.array([[weight], [0.0], [-weight]])
    q = np.array([variance])
    expected = np.array([np.dot(row * q, row) for row in m])
    cov = ProjectedCovariance((csr_array(m) if sparse else m,), (q,))
    np.testing.assert_allclose(projected_variances(cov), expected, rtol=2e-15, atol=0)


def test_dense_correlated_contraction_crosses_blocks_with_strided_inputs():
    rng = np.random.default_rng(42)
    m = rng.normal(size=(65_539, 8))[:, ::2]
    root = rng.normal(size=(4, 4))
    q = root @ root.T
    expected = np.array([row @ q @ row for row in m])
    cov = ProjectedCovariance((m,), (q,))
    np.testing.assert_allclose(projected_variances(cov), expected, rtol=3e-14, atol=2e-14)
