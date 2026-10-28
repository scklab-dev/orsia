"""Public continuum decomposition and homogeneous-slab scaling, without Fortran."""

from dataclasses import replace

import numpy as np
import pytest
from orsia_radiative import (
    Absorber,
    Atmosphere,
    WaterContinuumReference,
    water_continuum_absorption,
)
from scipy.constants import k


@pytest.mark.parametrize("fraction", [0.0, 1e-6, 0.01, 0.5])
def test_public_kernel_decomposition_geometry_and_density(fraction: float) -> None:
    nodes = np.arange(0.0, 10010.0, 10.0)
    table = WaterContinuumReference(
        nodes,
        np.full_like(nodes, 2e-25),
        np.full_like(nodes, 3e-25),
        np.full_like(nodes, 5.0),
        101300.0,
        296.0,
        (250.0, 320.0),
    )
    grid = np.array([925.25, 935.0, 950.75])
    air = _layer(20000.0, fraction)
    parts = [
        water_continuum_absorption(air, grid, part).optical_depth
        for part in (
            replace(table, foreign_absco_ref=np.zeros_like(nodes)),
            replace(table, self_absco_ref=np.zeros_like(nodes)),
        )
    ]
    total = water_continuum_absorption(air, grid, table).optical_depth
    np.testing.assert_allclose(sum(parts), total, rtol=1e-15, atol=0)
    for part, expected in zip(
        (
            replace(table, foreign_absco_ref=np.zeros_like(nodes)),
            replace(table, self_absco_ref=np.zeros_like(nodes)),
        ),
        parts,
        strict=True,
    ):
        np.testing.assert_allclose(
            water_continuum_absorption(_layer(20000.0, fraction, 2000.0), grid, part).optical_depth,
            2 * expected,
            rtol=1e-15,
            atol=0,
        )
        np.testing.assert_allclose(
            water_continuum_absorption(_layer(40000.0, fraction), grid, part).optical_depth,
            4 * expected,
            rtol=1e-15,
            atol=0,
        )


def _layer(pressure: float, fraction: float, thickness: float = 1000.0) -> Atmosphere:
    """One owned 280 K layer; pressure in Pa, thickness in m, fraction wet mol/mol."""
    total = np.array([pressure / (k * 280.0) * thickness])
    water = total * fraction
    return Atmosphere(
        np.array([0.0, thickness]),
        np.array([pressure * 1.01, pressure * 0.99]),
        np.array([pressure]),
        np.array([280.0]),
        total,
        total - water,
        water,
        (Absorber("H2O", water, "dry_air"),),
    )
