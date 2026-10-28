"""Ownership and last-order allocation contracts for SOS field accumulation."""

from unittest.mock import patch

import numpy as np
import orders_reference as orders
import pytest
from orsia_radiative.transfer.formal import apply_propagation, propagation_coefficients


@pytest.mark.parametrize("retain_centres", [False, True])
@pytest.mark.parametrize("solar_source", [False, True])
@pytest.mark.parametrize("bottom_radiance", [0.0, 0.7])
def test_add_propagated_source_mutates_only_owned_field_buffers(
    retain_centres: bool, solar_source: bool, bottom_radiance: float
) -> None:
    transport = propagation_coefficients(
        np.array([0.1, 0.0, 0.3]), np.array([-0.6, 0.4]), solar_mu=0.8
    )
    source = np.array([[0.2, 0.3], [0.0, 0.1], [0.4, 0.2]])
    original_source = source.copy()
    source.flags.writeable = False
    centres = np.full_like(source, 0.5) if retain_centres else None
    exits = np.array([0.8, 0.9])
    expected_centres, expected_exits = apply_propagation(
        source,
        transport,
        solar_source=solar_source,
        bottom_radiance=bottom_radiance,
        return_centres=retain_centres,
    )

    with patch.object(orders, "apply_propagation", wraps=apply_propagation) as propagate:
        orders._add_propagated_source(
            centres,
            exits,
            source,
            transport,
            solar_source=solar_source,
            bottom_radiance=bottom_radiance,
        )

    assert propagate.call_count == 1
    assert propagate.call_args.kwargs["return_centres"] is retain_centres
    np.testing.assert_array_equal(exits, np.array([0.8, 0.9]) + expected_exits)
    if retain_centres:
        assert centres is not None and expected_centres is not None
        np.testing.assert_array_equal(centres, 0.5 + expected_centres)
        assert not np.shares_memory(centres, source)
        assert not np.shares_memory(centres, transport.transmission)
    else:
        assert centres is None
    np.testing.assert_array_equal(source, original_source)
    assert not source.flags.writeable
    assert not np.shares_memory(exits, source)
    assert not np.shares_memory(exits, transport.transmission)
