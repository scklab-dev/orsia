"""Exact dependency comparison for absorption freshness."""

import numpy as np
from orsia_radiative.spectroscopy.components import values_equal


def test_exact_values_equality():
    assert values_equal((np.arange(2), (3,)), (np.arange(2), (3,)))
    assert values_equal(np.arange(2), np.arange(2))
    assert values_equal(1, 1)
    assert not values_equal(1, 2)
    assert not values_equal(np.arange(2), np.arange(3))
    assert not values_equal(np.arange(2), np.array([0, 2]))
    assert not values_equal(np.arange(2), (0, 1))
    assert not values_equal((0, 1), np.arange(2))
    assert not values_equal((1,), [1])
    assert not values_equal((1,), (1, 2))
    assert not values_equal((1,), (2,))
    assert not values_equal(np.array([np.nan]), np.array([np.nan]))
