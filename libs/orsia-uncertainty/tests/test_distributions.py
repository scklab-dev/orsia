"""Normalization and shared-scale checks for explicit perturbation laws."""

from typing import Any

import numpy as np
import pytest
from orsia_uncertainty.distributions import standardized_draw


@pytest.mark.parametrize("law", ["gaussian", "uniform", "student_t", "contaminated_gaussian"])
def test_standardized_laws_have_declared_moments(law: str) -> None:
    values = standardized_draw(
        np.random.default_rng(73), 200000, law, 7.0, outlier_probability=0.01, outlier_scale=5.0
    )
    assert abs(values.mean()) < 0.012
    assert abs(values.var() - 1) < 0.035
    if law == "uniform":
        assert abs(values).max() <= np.sqrt(3)
    if law in ("student_t", "contaminated_gaussian"):
        assert np.mean(values**4) > 3.5


def test_shared_student_scale_and_invalid_laws() -> None:
    rng = np.random.default_rng(54)
    draws = np.array(
        [standardized_draw(rng, 2, "student_t", 7.0, shared=True) for _ in range(20000)]
    )
    assert np.asarray(np.corrcoef(draws**2, rowvar=False))[0, 1] > 0.12
    independent = standardized_draw(rng, (20000, 2), "student_t", 7.0)
    assert abs(np.asarray(np.corrcoef(independent**2, rowvar=False))[0, 1]) < 0.05
    with pytest.raises(ValueError):
        standardized_draw(rng, 2, "student_t", 2.0)


def test_invalid_standardized_laws():
    cases: tuple[tuple[str, dict[str, Any]], ...] = (
        ("invalid", {}),
        ("contaminated_gaussian", {"shared": True}),
        ("contaminated_gaussian", {"outlier_probability": 1.0}),
        ("contaminated_gaussian", {"outlier_scale": np.nan}),
    )
    for distribution, options in cases:
        with pytest.raises(ValueError):
            standardized_draw(np.random.default_rng(1), 2, distribution, **options)


def test_linear_process_explicit_laws_and_validation():
    from orsia_uncertainty.temporal import sample_linear_process

    matrix = np.eye(2)
    rng = np.random.default_rng(10)

    def draw(rng, size):
        return standardized_draw(rng, size, "uniform")

    actual = sample_linear_process(
        matrix * 0.5, matrix, matrix * 0.2, 5, rng, initial_draw=draw, innovation_draw=draw
    )
    oracle_rng = np.random.default_rng(10)
    expected = [draw(oracle_rng, 2)]
    for _ in range(4):
        expected.append(expected[-1] * 0.5 + draw(oracle_rng, 2) * 0.2)
    np.testing.assert_array_equal(actual, expected)
    assert rng.random() == oracle_rng.random()
    with pytest.raises(ValueError):
        sample_linear_process(
            matrix, matrix, matrix, 2, rng, initial_draw=lambda r, n: np.ones(n + 1)
        )
