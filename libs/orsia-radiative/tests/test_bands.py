"""Positive physical band means and ground-cube ownership/validity contracts."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_radiative import (
    RadianceCube,
    SpectralBand,
    SpectralDomain,
    spectral_band,
    wavenumber_grid,
)


@pytest.mark.parametrize("domain", [None, SpectralDomain("wavenumber_cm1", ((2.0, 7.0),))])
def test_band_density_on_nonuniform_nodes_and_ownership(domain):
    points = np.array([2.0, 3.0, 7.0])
    density = np.ones(3)
    band = spectral_band("energy", wavenumber_grid(points, domain=domain), density)
    assert_allclose(band.weights, [0.1, 0.5, 0.4])
    assert np.dot(points, band.weights) == pytest.approx(4.5)
    points[0] = 1
    density[0] = 8
    assert band.grid.points[0] == 2
    photon = spectral_band("photon", band.grid, 1 / band.grid.points)
    assert np.dot(band.grid.points, photon.weights) != pytest.approx(4.5)


def test_band_density_rejects_disconnected_domain():
    points = np.array([1.0, 2.0, 100.0, 102.0])
    density = np.ones(4)
    discrete = spectral_band("discrete", wavenumber_grid(points), density)
    assert_allclose(discrete.weights, np.array([0.5, 49.5, 50.0, 1.0]) / 101)
    assert discrete.grid.domain is None
    domain = SpectralDomain("wavenumber_cm1", ((1.0, 2.0), (100.0, 102.0)))
    grid = wavenumber_grid(points, domain=domain)
    with pytest.raises(ValueError, match="connected.*domain"):
        spectral_band("gap", grid, density)


@pytest.mark.parametrize(
    "points,density",
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1]),
        ([1, 2], [-1, 1]),
        ([1, 2], [0, 0]),
        ([2, 1], [1, 1]),
        ([0, 1], [1, 1]),
        ([1, np.inf], [1, 1]),
        ([1, 2], [1, np.nan]),
    ],
)
def test_bad_band_density(points, density):
    with pytest.raises(ValueError):
        spectral_band(
            "bad", wavenumber_grid(np.array(points, dtype=float)), np.array(density, dtype=float)
        )


@pytest.mark.parametrize(
    "name,points,weights",
    [
        ("", [1, 2], [0.5, 0.5]),
        ("x", [1, 2], [1, 1]),
        ("x", [2, 1], [0.5, 0.5]),
        ("x", [1, 2], [-1, 2]),
    ],
)
def test_bad_band_weights(name, points, weights):
    with pytest.raises(ValueError):
        SpectralBand(
            name, wavenumber_grid(np.array(points, dtype=float)), np.array(weights, dtype=float)
        )


def test_cube_axes_nan_validity_and_copies():
    band = spectral_band("a", wavenumber_grid(np.array([1.0, 2])), np.ones(2))
    values = np.array([[[3.0], [np.nan]]])
    coverage = np.array([[[1.0], [0.0]]])
    valid = np.array([[[True], [False]]])
    cube = RadianceCube(np.array([0.0, 1]), np.array([2.0]), (band,), values, coverage, valid)
    values[0, 0, 0] = 4
    assert cube.values[0, 0, 0] == 3
    changes: tuple[dict[str, Any], ...] = (
        dict(x_m=np.array([1.0, 0])),
        dict(y_m=np.array([])),
        dict(values=np.zeros((1, 2, 1))),
        dict(coverage=-np.ones((1, 2, 1))),
        dict(valid=np.ones((1, 2, 1), dtype=bool)),
        dict(bands=()),
    )
    for change in changes:
        with pytest.raises(ValueError):
            replace(cube, **change)
