"""Fixed-state products, ownership and explicit dense allocation budgets."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from orsia_numerics.operators import linear_operator, materialize_operator
from scipy.sparse.linalg import LinearOperator


def test_products_blocks_transpose_ownership_concurrency():
    a = np.arange(15.0).reshape(3, 5)
    op = linear_operator(a.shape, lambda v: a @ v, lambda w: a.T @ w)
    v, w = np.arange(5.0), np.arange(3.0)
    np.testing.assert_array_equal(op @ v, a @ v)
    np.testing.assert_array_equal(op.T @ w, a.T @ w)
    assert (op @ v) @ w == v @ (op.T @ w)
    np.testing.assert_array_equal(op.matmat(np.empty((5, 0))), np.empty((3, 0)))
    np.testing.assert_array_equal(materialize_operator(op, max_bytes=1000, block_size=2), a)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda _: op @ v, range(4)):
            np.testing.assert_array_equal(result, a @ v)
            assert not np.shares_memory(result, v)


def test_invalid_contract_and_budget():
    with pytest.raises(ValueError):
        linear_operator((0, 2), lambda v: v, lambda v: v)
    op = linear_operator((2, 2), lambda v: v, lambda w: w)
    for x in (np.ones(2, dtype=np.float32), np.array([np.nan, 0.0])):
        with pytest.raises(ValueError):
            op @ x
    bad = linear_operator((2, 2), lambda v: v[:1], lambda v: v)
    with pytest.raises(ValueError):
        bad @ np.ones(2)
    for budget, width, error in ((1, 1, MemoryError), (0, 1, ValueError), (100, 0, ValueError)):
        with pytest.raises(error):
            materialize_operator(op, max_bytes=budget, block_size=width)
    empty = LinearOperator((0, 0), matvec=lambda v: v, dtype=np.float64)
    with pytest.raises(ValueError):
        materialize_operator(empty, max_bytes=100)
    external = LinearOperator(
        (2, 2), matvec=lambda v: v, matmat=lambda v: np.full(v.shape, np.nan), dtype=np.float64
    )
    with pytest.raises(ValueError):
        materialize_operator(external, max_bytes=1000)
    result = op.matmat(np.ones((2, 2)))
    result[:] = 8
    np.testing.assert_array_equal(op @ np.ones(2), np.ones(2))
