"""Registered band-radiance product validity and owned numerical storage."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from orsia_instruments.observation.measurements import BandRadianceCube
from orsia_radiative import spectral_band, wavenumber_grid


def test_cube_axes_nan_validity_and_copies():
    band = spectral_band("a", wavenumber_grid(np.array([1.0, 2])), np.ones(2))
    values = np.array([[[3.0], [np.nan]]])
    coverage = np.array([[[1.0], [0.0]]])
    valid = np.array([[[True], [False]]])
    cube = BandRadianceCube(np.array([0.0, 1]), np.array([2.0]), (band,), values, coverage, valid)
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
