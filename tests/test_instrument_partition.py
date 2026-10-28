"""Full-grid callback equivalence and scalar interpolation for physical reads."""

import numpy as np
import pytest
from instrument_cases import make_case, simulate
from orsia_instruments import expected_electrons


def test_full_grid_qe_callback_matches_constant_physical_fraction() -> None:
    first = simulate("filter")
    field, acquisition, response = make_case("filter")
    calls = []

    def qe(reads, wavelengths):
        calls.append((reads.copy(), wavelengths.copy()))
        return np.full((len(reads), len(wavelengths)), 0.7)

    second = expected_electrons(
        field,
        acquisition,
        response,
        collecting_area_m2=1e-8,
        throughput=0.8,
        quantum_efficiency=qe,
    )
    np.testing.assert_allclose(first.total, second.total, rtol=2e-13)
    assert np.all(first.total > 0)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.arange(len(acquisition.sample_ids)))
    np.testing.assert_array_equal(calls[0][1], 0.01 / field.grid.points)


def test_batched_interpolation_matches_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    import orsia_instruments.integration as integration

    original = integration.sample_radiance
    batch_sizes = []

    def compare(field, times, directions, block):
        batch = original(field, times, directions, block)
        scalar = np.concatenate(
            [
                original(field, times[i : i + 1], directions[i : i + 1], block)
                for i in range(len(times))
            ]
        )
        np.testing.assert_allclose(batch, scalar, rtol=2e-15)
        batch_sizes.append(len(times))
        return batch

    monkeypatch.setattr(integration, "sample_radiance", compare)
    simulate("filter")
    assert batch_sizes and max(batch_sizes) > 1
