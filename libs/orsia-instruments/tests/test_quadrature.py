"""Instrument-owned interval quadrature, with no numerics compatibility facade."""

import importlib

import numpy as np
import pytest
from orsia_instruments.quadrature import interval_quadrature, validate_quadrature_order


def test_quadrature_exactness():
    nodes, weights = interval_quadrature(-2, 3, 4)
    for power in range(8):
        expected = (3 ** (power + 1) - (-2) ** (power + 1)) / (power + 1)
        np.testing.assert_allclose(weights @ nodes**power, expected, rtol=2e-14, atol=2e-14)
    assert np.all(weights > 0) and nodes.flags.writeable and weights.flags.writeable
    validate_quadrature_order(1)
    for bounds in ((0, 0, 1), (1, 0, 1), (0, np.inf, 1), (0, 1, 0)):
        with pytest.raises(ValueError):
            interval_quadrature(*bounds)
    with pytest.raises(ValueError):
        validate_quadrature_order(0)


def test_quadrature_ownership_and_consumers():
    from orsia_instruments import acquisition
    from orsia_instruments.optics import cavities, dispersion, filters, pixels, response, two_beam

    assert interval_quadrature.__module__ == "orsia_instruments.quadrature"
    assert validate_quadrature_order.__module__ == "orsia_instruments.quadrature"
    assert acquisition.interval_quadrature is pixels.interval_quadrature is interval_quadrature
    for module in (cavities, dispersion, filters, response, two_beam):
        assert module.validate_quadrature_order is validate_quadrature_order
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("orsia_numerics.quadrature")
    assert error.value.name == "orsia_numerics.quadrature"
