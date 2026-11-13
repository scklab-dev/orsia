"""Field-owned boundary/cutoff propagation into straight-path mass integrals."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import (
    MassParticles,
    ParticleMassField,
    concentration,
    path_columns,
    puff_path_columns,
)
from scipy.integrate import quad
from test_fields import puff, stationary


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
@pytest.mark.parametrize("cutoff", [None, 3.0])
def test_path_preserves_field_declaration(boundary, cutoff):
    field = replace(puff(boundary, centre=(0.0, 0.0, 0.0)), kernel_cutoff_radius=cutoff)
    starts = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [20.0, 0.0, 0.0]])
    ends = starts + [0.0, 0.0, 12.0]
    actual = path_columns(field, starts, ends, batch_size=1)
    assert_array_equal(
        actual,
        puff_path_columns(
            field.state, starts, ends, boundary=boundary, kernel_cutoff_radius=cutoff, batch_size=1
        ),
    )
    for a, b, column in zip(starts, ends, actual, strict=True):
        length = np.linalg.norm(b - a)
        expected = quad(
            lambda t, a=a, b=b, length=length: (
                float(concentration(field, a + t * (b - a))) * length
            ),
            0.0,
            1.0,
            epsabs=1e-10,
            points=np.linspace(0.0, 1.0, 33),
        )[0]
        assert_allclose(column, expected, rtol=2e-7, atol=1e-10)
    assert_array_equal(path_columns(field, starts, starts), 0)
    assert path_columns(field, starts[:0], ends[:0]).shape == (0,)
    if boundary == "reflecting":
        free = replace(field, boundary="free")
        assert_allclose(actual, 2 * path_columns(free, starts, ends), rtol=2e-15)
        with pytest.raises(ValueError):
            path_columns(field, starts - [0.0, 0.0, 1.0], ends)
    if cutoff is not None:
        assert actual[-1] == 0
        assert path_columns(replace(field, kernel_cutoff_radius=None), starts, ends)[-1] > 0


def test_unsupported_field_representations_fail():
    particles = ParticleMassField(
        MassParticles(0.0, np.zeros(1), np.ones(1), np.zeros((1, 3)), np.zeros((1, 3))), "free"
    )
    for field in (stationary(), particles):
        with pytest.raises(TypeError, match="unsupported"):
            path_columns(field, np.zeros((1, 3)), np.ones((1, 3)))
