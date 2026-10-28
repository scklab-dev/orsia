"""Numerical test adapters for Orsia's physical Radiance API."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import ObservationGeometry, SpectralGrid, transfer
from orsia_radiative.transfer import orders


@dataclass(frozen=True, eq=False)
class SpectralTerm:
    name: str
    values: NDArray[np.float64]
    role: str | None = None
    __hash__ = None


@dataclass(frozen=True, eq=False)
class RadianceDecomposition:
    grid: SpectralGrid
    values: NDArray[np.float64]
    geometry: ObservationGeometry
    components: tuple[SpectralTerm, ...]
    __hash__ = None

    @property
    def wavenumbers_cm1(self) -> NDArray[np.float64]:
        return self.grid.points


def assert_term_equal(actual: SpectralTerm, expected: SpectralTerm) -> None:
    """Check a named test term's role and exact numerical values."""
    assert actual.name == expected.name
    assert actual.role == expected.role
    np.testing.assert_array_equal(actual.values, expected.values)


def assert_decomposition_equal(
    actual: RadianceDecomposition, expected: RadianceDecomposition
) -> None:
    """Check support, geometry, total and each ordered test term exactly."""
    assert actual.grid == expected.grid
    assert actual.geometry == expected.geometry
    np.testing.assert_array_equal(actual.values, expected.values)
    for term, reference in zip(actual.components, expected.components, strict=True):
        assert_term_equal(term, reference)


def order_zero(*args: Any, **kwargs: Any) -> RadianceDecomposition:
    return reflected_radiance(*args, max_order=0, delta_m_moments=None, **kwargs)


def order_one(*args: Any, **kwargs: Any) -> RadianceDecomposition:
    return reflected_radiance(*args, max_order=1, delta_m_moments=None, **kwargs)


def reflected_radiance(*args: Any, **kwargs: Any) -> RadianceDecomposition:
    result = transfer.solar.reflected_radiance(*args, **kwargs)
    rows = orders.scattering_orders(*args, **kwargs)
    np.testing.assert_array_equal(rows.sum(axis=0), result.values)
    prefix = "residual_order" if kwargs.get("delta_m_moments") is not None else "atmospheric_order"
    return RadianceDecomposition(
        result.grid,
        result.values,
        result.geometry,
        tuple(SpectralTerm(f"{prefix}_{n}", row) for n, row in enumerate(rows)),
    )


def schwarzschild_radiance(*args: Any, **kwargs: Any) -> RadianceDecomposition:
    result = transfer.thermal.schwarzschild_radiance(*args, **kwargs)
    terms = transfer.thermal.schwarzschild_terms(*args, **kwargs)
    np.testing.assert_array_equal(sum(terms.values()), result.values)
    return RadianceDecomposition(
        result.grid,
        result.values,
        result.geometry,
        tuple(SpectralTerm(name, value) for name, value in terms.items()),
    )
