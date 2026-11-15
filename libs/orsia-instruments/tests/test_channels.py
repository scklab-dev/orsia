"""Channel-selection contracts independent of a retrieval objective."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from orsia_instruments.channels import refine_channels


def test_deterministic_discrete_search() -> None:
    def evaluate(selection: list[NDArray[np.float64]]) -> tuple[float, list[NDArray[np.float64]]]:
        return float(1 + np.sum((selection[0] - np.array([1.0, 4.0])) ** 2)), selection

    search = refine_channels([np.arange(6.0)], [2.0], [np.array([0.0, 5.0])], evaluate)
    assert search["converged"] and search["iterations"] > 0
    assert_array_equal(search["optimized"][0], [1.0, 4.0])
    assert_array_equal(search["initial"][0], [0.0, 5.0])
    with pytest.raises(ValueError):
        refine_channels(
            [np.arange(6.0)], [2.0], [np.array([0.0, 5.0])], evaluate, maximum_iterations=0
        )
