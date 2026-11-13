"""Explicit continuation, conservative interior deposition and finite edge loss."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_scene.projection import project_layer_columns_by_centroid


def test_positive_weights_axes_and_conservation():
    amounts = np.zeros((5, 6, 2))
    amounts[2, 2] = [8.0, 4.0]
    heights = np.ones_like(amounts)
    projected = project_layer_columns_by_centroid(
        amounts, heights, (0.5, 0.25), (1.0, 1.0), continuation="finite"
    )
    expected = np.zeros_like(amounts)
    expected[2:4, 2:4, 0] = [[3.0, 1.0], [3.0, 1.0]]
    expected[2:4, 2:4, 1] = [[1.5, 0.5], [1.5, 0.5]]
    assert_array_equal(projected, expected)
    assert_array_equal(projected.sum(axis=(0, 1)), amounts.sum(axis=(0, 1)))
    assert_array_equal(amounts[2, 2], [8.0, 4.0])
    assert not np.shares_memory(projected, amounts)


def test_uniform_finite_support_stays_finite_under_small_perturbations():
    amounts = np.ones((5, 6, 1))
    heights = np.ones_like(amounts)
    finite = project_layer_columns_by_centroid(
        amounts, heights, (1.0, 0.0), (1.0, 1.0), continuation="finite"
    )
    assert_array_equal(finite[0], 0)
    assert_array_equal(finite[1:], 1)
    changed = amounts.copy()
    changed[2, 2, 0] += 1e-12
    other = project_layer_columns_by_centroid(
        changed, heights, (1.0, 0.0), (1.0, 1.0), continuation="finite"
    )
    assert_allclose(other, finite, rtol=0, atol=1.01e-12)
    assert finite.sum() == 24
    infinite = project_layer_columns_by_centroid(
        amounts, heights, (1.0, 0.0), (1.0, 1.0), continuation="infinite_uniform"
    )
    assert_array_equal(infinite, amounts)
    with pytest.raises(ValueError, match="uniform"):
        project_layer_columns_by_centroid(
            changed, heights, (1.0, 0.0), (1.0, 1.0), continuation="infinite_uniform"
        )
    heights[1, 1] = 2
    with pytest.raises(ValueError, match="uniform"):
        project_layer_columns_by_centroid(
            amounts, heights, (1.0, 0.0), (1.0, 1.0), continuation="infinite_uniform"
        )


def test_empty_layers_large_displacements_and_zero_sheet_heights():
    empty = np.empty((2, 3, 0))
    assert (
        project_layer_columns_by_centroid(
            empty, empty, (0.0, 0.0), (1.0, 1.0), continuation="finite"
        ).shape
        == empty.shape
    )
    amounts = np.ones((2, 3, 1))
    out = project_layer_columns_by_centroid(
        amounts, amounts, (1e100, -1e100), (1.0, 1.0), continuation="finite"
    )
    assert_array_equal(out, 0)
    heights = np.arange(6.0).reshape(2, 3, 1)
    assert_array_equal(
        project_layer_columns_by_centroid(
            np.zeros_like(heights), heights, (1.0, 1.0), (1.0, 1.0), continuation="infinite_uniform"
        ),
        0,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"amounts": np.ones((2, 3))},
        {"amounts": np.empty((0, 3, 1))},
        {"amounts": -np.ones((2, 3, 1))},
        {"heights_agl_m": np.ones((2, 3))},
        {"heights_agl_m": np.full((2, 3, 1), np.nan)},
        {"heights_agl_m": -np.ones((2, 3, 1))},
        {"slope_yx": (np.inf, 0.0)},
        {"spacing_yx_m": (0.0, 1.0)},
        {"continuation": "implicit"},
    ],
)
def test_invalid_inputs(override):
    args = dict(
        amounts=np.ones((2, 3, 1)),
        heights_agl_m=np.ones((2, 3, 1)),
        slope_yx=(0.0, 0.0),
        spacing_yx_m=(1.0, 1.0),
        continuation="finite",
    )
    with pytest.raises(ValueError):
        project_layer_columns_by_centroid(**(args | override))


def test_position_and_accumulation_overflow_fail():
    one = np.ones((2, 2, 1))
    with pytest.raises(ValueError, match="positions"):
        project_layer_columns_by_centroid(
            one, one * 1e308, (2.0, 0.0), (1.0, 1.0), continuation="finite"
        )
    # Both rows land on the second row, exceeding representable column storage.
    heights = np.zeros_like(one)
    heights[0] = 1
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="accumulation"):
            project_layer_columns_by_centroid(
                one * 1e308, heights, (1.0, 0.0), (1.0, 1.0), continuation="finite"
            )
