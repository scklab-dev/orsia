"""Signed image registration, compact propagation and selected-state covariance."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_instruments.pointing import (
    coadd_temporal_covariance,
    pointing_covariance_parameters,
    smooth_pointing_state_space,
)
from orsia_instruments.registration.covariance import project_detector_covariance
from orsia_instruments.registration.estimation import (
    estimate_row_displacements,
    estimate_translation,
    sample_image,
)
from orsia_instruments.registration.resampling import (
    RegistrationMap,
    interpolation_stencil,
    interpolation_weights,
)
from scipy.sparse import csr_matrix, vstack


@pytest.mark.parametrize("kernel", ["keys_cubic", "lanczos3", "lanczos5"])
def test_signed_map_and_scalar_covariance_against_dense(kernel: str) -> None:
    shape = (16, 17, 2)
    rng = np.random.default_rng(87)
    support = np.ones(shape, bool)
    support[8, 8, 1] = False
    shifts = np.zeros((16, 2, 2))
    shifts[:, 1] = [0.2, -0.1]
    shifts[7, 1] = 0
    mapping = RegistrationMap(shape, shifts, support, kernel=kernel)
    matrix = csr_matrix(vstack([mapping.rows(c) for c in mapping.blocks()]))
    field = rng.normal(size=shape)
    variance = rng.uniform(1, 2, size=shape)
    assert_allclose(mapping.apply(field), (matrix @ field.ravel()).reshape(shape), atol=1e-14)
    assert_allclose(
        mapping.apply(variance, squared=True),
        (matrix.power(2) @ variance.ravel()).reshape(shape),
        atol=1e-14,
    )
    flags = field > 0
    structural = matrix.copy()
    structural.data[:] = 1
    assert_array_equal(
        mapping.propagate_any(flags), (structural @ flags.ravel()).reshape(shape) > 0
    )
    assert_array_equal(
        mapping.propagate_any(flags[..., 0]),
        (structural @ np.repeat(flags[..., 0].ravel(), 2)).reshape(shape) > 0,
    )
    gain = rng.normal(size=shape)
    selected_mask = mapping.valid.all(axis=2)
    selected_mask[:7] = False
    selected_mask[9:] = False
    projected, indices = project_detector_covariance(mapping, variance, gain, selected_mask)
    legacy, same = project_detector_covariance(matrix, variance, gain, selected_mask)
    assert_array_equal(indices, same)
    assert_allclose(projected.toarray(), legacy.toarray(), rtol=1e-13, atol=1e-14)
    assert mapping.kernel == kernel and mapping.reference_band == 0
    assert mapping.cube_shape == shape
    shifts[:] = 20
    support[:] = False
    assert np.all(mapping.corrections_pixel[:, 0] == 0)
    assert mapping.source_valid.any()
    assert not mapping.valid.flags.writeable
    assert mapping.rows(np.empty(0, dtype=int)).shape == (0, np.prod(shape))
    for cells in [np.array([-1]), np.zeros(2049, int), np.array([0.5])]:
        with pytest.raises(ValueError):
            mapping.rows(cells)
    with pytest.raises(ValueError):
        mapping.overlapping_rows(-1, 0)
    with pytest.raises(ValueError):
        mapping.apply(field[0])
    with pytest.raises(ValueError):
        mapping.propagate_any(field)
    with pytest.raises(ValueError):
        project_detector_covariance(mapping, variance[0], gain, selected_mask)
    coords = np.array([5.2, 7.1])
    stencil = interpolation_stencil(coords, 16, kernel)
    weights = interpolation_weights(coords, 16, kernel)
    assert_array_equal(weights[0], stencil[0])
    assert_array_equal(weights[1], stencil[1])
    assert_array_equal(weights[2], stencil[3])
    epsilon = 1e-6
    numerical = (
        interpolation_weights(coords + epsilon, 16, kernel)[1]
        - interpolation_weights(coords - epsilon, 16, kernel)[1]
    ) / (2 * epsilon)
    assert_allclose(stencil[2], numerical, rtol=2e-8, atol=1e-9)


def test_measured_registration_against_unchanged_julia_oracle() -> None:
    with np.load(Path(__file__).parent / "fixtures/image_registration_reference.npz") as r:
        y = r["registration.measurement.toa_radiance_w_m2_sr_nm"]
        valid = r["registration.measurement.valid"]
        result = estimate_translation(y[..., 0], y[..., 1], valid[..., 0], valid[..., 1])
        assert_allclose(
            result["correction_pixel"], r["registration.estimate.correction_pixel"][1], atol=1e-11
        )
        assert_allclose(
            result["covariance_pixel2"],
            r["registration.estimate.correction_covariance_pixel2"][1],
            rtol=1e-8,
            atol=1e-15,
        )
        y = r["joint.measurement.toa_radiance_w_m2_sr_nm"]
        valid = r["joint.measurement.valid"]
        joint = estimate_row_displacements(
            y[..., 0],
            y[..., 1],
            valid[..., 0],
            valid[..., 1],
            knot_spacing=12,
            regularization_sigma=0.03,
            window_half_width=12,
            row_minimum_samples=256,
            maximum_iterations=30,
            tolerance=1e-6,
        )
        assert_allclose(
            joint["row_correction_pixel"],
            r["joint.estimate.row_correction_pixel"][:, 1],
            atol=1e-10,
        )
        assert_allclose(
            joint["row_joint_covariance_pixel2"],
            r["joint.estimate.row_joint_covariance_pixel2"][..., 1],
            rtol=1e-8,
            atol=1e-13,
        )
    with pytest.raises(ValueError):
        estimate_translation(
            np.ones((16, 17)), np.ones((16, 17)), np.ones((16, 17), bool), np.ones((16, 17), bool)
        )
    with pytest.raises(ValueError):
        sample_image(np.ones((3, 3)), [], [], kernel="invalid")
    with pytest.raises(ValueError):
        RegistrationMap((2, 2, 2), [[1, 0], [0, 0]], np.ones((2, 2, 2), bool))


def test_pointing_and_chronological_correlations() -> None:
    rate, angular = pointing_covariance_parameters([0.01, 0.1], [1.0, 3.0], 0.3)
    stationary, transition, innovation = smooth_pointing_state_space(rate, angular, 0.01)
    assert_allclose(innovation + transition @ stationary @ transition.T, stationary, rtol=1e-15)
    ranges = [np.array([0, 1]), np.array([4, 5, 6])]
    correlation = coadd_temporal_covariance(3, ranges, 0.01, rate)
    assert_allclose(correlation, correlation.T, atol=1e-15)
    assert np.linalg.eigvalsh(correlation).min() > 0
    with pytest.raises(ValueError):
        pointing_covariance_parameters([1.0, 0.1], [1, 2])
    with pytest.raises(ValueError):
        pointing_covariance_parameters([0.1, 1.0], [1, 1])
    with pytest.raises(ValueError):
        smooth_pointing_state_space(0, angular, 0.1)
