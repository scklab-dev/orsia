"""Native cube indexing, joint calibration derivatives and fixed-support evidence."""

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from orsia_instruments import (
    RadiometricParameterMaps,
    calibrate_frames,
    calibrate_observations,
    form_observation_cube,
    raster_read_schedule,
)
from orsia_uncertainty import ProjectedCovariance, materialize_covariance
from scipy.sparse import csr_array

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/native_observation_cube.py"


@pytest.fixture
def inputs():
    return runpy.run_path(str(EXAMPLE))["example_inputs"]()


def options(inputs, **updates: Any) -> dict[str, Any]:
    result: dict[str, Any] = dict(
        detector=inputs[2],
        read_covariance=np.ones(8),
        parameters=inputs[4],
        parameter_covariance=np.array([1e-4]),
        read_parameter_cross_covariance="zero",
        read_valid=inputs[5],
    )
    result.update(updates)
    return result


def form(calibrated, inputs, **kwargs):
    return form_observation_cube(
        calibrated,
        inputs[3],
        shape=(1, 2, 2),
        band_names=("a", "b"),
        native_weights=inputs[6],
        **kwargs,
    )


def test_example_and_plot(tmp_path):
    cube = runpy.run_path(str(EXAMPLE))["run_example"](tmp_path / "example.png")
    assert (tmp_path / "example.png").stat().st_size > 1000
    assert np.isnan(cube.values[0, 1, 1])


def test_values_support_and_legacy_compatibility(inputs):
    images, response, detector, geometry, _, _, _ = inputs
    new = calibrate_observations(images, response, **options(inputs))
    old = calibrate_frames(images, response, detector=detector)
    np.testing.assert_allclose(new.values[new.valid], old.values[0, new.valid])
    cube = form(new, inputs)
    assert cube.response is response and cube.geometry is geometry
    np.testing.assert_array_equal(cube.observation_indices, [0, 1, 2])
    np.testing.assert_allclose(
        cube.native_weights @ np.nan_to_num(new.values), cube.values[cube.valid]
    )
    np.testing.assert_allclose(cube.parameter_sensitivity[:, 0], [10, 15, 20])
    assert cube.native_weights.shape == (3, 8)
    assert not np.array_equal(geometry.bounds_m[0], geometry.bounds_m[2])
    assert cube.bias is None


def test_joint_cross_terms_derivatives_and_bias(inputs):
    images, response = inputs[:2]
    n, p = 8, 3
    gain = np.zeros((n, p))
    gain[:, 0] = 2
    offset = np.zeros((n, p))
    offset[:, 1] = 1
    normalization = np.zeros((n, p))
    normalization[:, 2] = response.spectral_coefficients.sum(axis=1)
    maps = RadiometricParameterMaps(
        ("gain", "offset", "normalization"), ("1", "DN", "1"), gain, offset, normalization
    )
    cq = np.diag([1e-4, 0.01, 4e-4])
    cross = np.zeros((8, 3))
    cross[:, 0] = 0.001
    kwargs = options(
        inputs,
        parameters=maps,
        parameter_covariance=cq,
        read_parameter_cross_covariance=cross,
        read_bias=np.arange(8.0) * 0.01,
        parameter_bias=np.array([0.01, -0.02, 0.03]),
    )
    projected = calibrate_observations(images, response, **kwargs)
    dense = calibrate_observations(images, response, representation="dense", **kwargs)
    valid = projected.read_indices
    s = response.spectral_coefficients.sum(axis=1)[valid]
    a = np.zeros((len(valid), 8))
    a[np.arange(len(valid)), valid] = 2 / s
    dn = images.dn.ravel()[valid]
    b = np.column_stack((2 * (dn - 10) / s, -2 / s, -projected.values[valid]))
    np.testing.assert_allclose(projected.parameter_sensitivity, b)
    reference = a @ a.T + b @ cq @ b.T + a @ cross @ b.T + b @ cross.T @ a.T
    assert isinstance(dense.covariance, np.ndarray)
    assert isinstance(projected.covariance, ProjectedCovariance)
    assert projected.bias is not None
    np.testing.assert_allclose(dense.covariance, reference)
    np.testing.assert_allclose(
        materialize_covariance(projected.covariance, max_bytes=10000), reference
    )
    np.testing.assert_allclose(
        projected.bias, a @ kwargs["read_bias"] + b @ kwargs["parameter_bias"]
    )
    cube = form(projected, inputs)
    assert isinstance(cube.covariance, ProjectedCovariance)
    assert cube.bias is not None
    t = cast(csr_array, cube.native_weights[:, valid]).toarray()
    np.testing.assert_allclose(
        materialize_covariance(cube.covariance, max_bytes=10000), t @ reference @ t.T
    )
    dense_cube = form(dense, inputs)
    assert isinstance(dense_cube.covariance, np.ndarray)
    np.testing.assert_allclose(dense_cube.covariance, t @ reference @ t.T)
    np.testing.assert_allclose(cube.bias, t @ projected.bias)
    # Finite differences use the same physical correction with independently varied calibration.
    for j in range(p):
        h = 1e-6
        delta = np.eye(p)[j] * h

        def values(theta):
            return ((2 + gain[valid] @ theta) * (dn - 10 - offset[valid] @ theta)) / (
                s + normalization[valid] @ theta
            )

        np.testing.assert_allclose(
            (values(delta) - values(-delta)) / (2 * h), b[:, j], rtol=2e-9, atol=2e-9
        )
    # Joint small-error ensemble: fixed support, seed and population.
    # Clipping and quantization are not redrawn in this linearization check.
    rng = np.random.default_rng(8041)
    joint = np.block([[np.eye(8), cross], [cross.T, cq]]) * 1e-4
    draws = rng.multivariate_normal(np.zeros(11), joint, size=80_000)
    ds, theta = draws[:, :8][:, valid], draws[:, 8:]
    observed = ((2 + theta @ gain[valid].T) * (dn + ds - 10 - theta @ offset[valid].T)) / (
        s + theta @ normalization[valid].T
    )
    empirical = np.cov(observed, rowvar=False)
    predicted = reference * 1e-4
    assert np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted) < 0.018
    linear = ds @ a[:, valid].T + theta @ b.T
    remainder = observed - projected.values[valid] - linear
    assert np.sqrt(np.mean(remainder**2)) < 0.002 * np.sqrt(np.mean(linear**2))


def test_common_error_does_not_average_away(inputs):
    new = calibrate_observations(*inputs[:2], **options(inputs, read_covariance=np.zeros(8)))
    cube = form(new, inputs)
    assert isinstance(cube.covariance, ProjectedCovariance)
    cov = materialize_covariance(cube.covariance, max_bytes=72)
    np.testing.assert_allclose(cov, np.outer([10, 15, 20], [10, 15, 20]) * 1e-4)
    with pytest.raises(MemoryError):
        form(
            calibrate_observations(*inputs[:2], representation="dense", **options(inputs)),
            inputs,
            max_dense_bytes=1,
        )


def test_expected_charge_and_masks(inputs):
    images, response = inputs[:2]
    maps = replace(
        inputs[4],
        gain=np.zeros((8, 1)),
        normalization=response.spectral_coefficients.sum(axis=1)[:, None],
    )
    expected = replace(images, photoelectrons=np.full((2, 2, 2), 20.0))
    result = calibrate_observations(
        expected, response, **options(inputs, parameters=maps, use_expectation=True)
    )
    np.testing.assert_allclose(result.parameter_sensitivity[:, 0], -result.values[result.valid])
    result2 = calibrate_observations(
        replace(expected, dn=None), response, **options(inputs, parameters=maps)
    )
    np.testing.assert_allclose(result.values, result2.values)
    flagged = images.adc_clipped.copy()
    flagged.ravel()[0] = True
    with pytest.raises(ValueError, match="clipped"):
        calibrate_observations(replace(images, adc_clipped=flagged), response, **options(inputs))
    masked = calibrate_observations(
        replace(images, adc_clipped=flagged), response, mask_flagged=True, **options(inputs)
    )
    assert not masked.valid[0]
    with pytest.raises(ValueError, match="invalid reads"):
        form(masked, inputs)


@pytest.mark.parametrize(
    "update",
    [
        dict(representation="wrong"),
        dict(max_dense_bytes=-1),
        dict(realisation=2),
        dict(read_valid=np.zeros(8, dtype=bool)),
        dict(read_valid=np.ones(8)),
        dict(read_covariance=np.ones(7)),
        dict(parameter_covariance=np.ones(2)),
        dict(read_parameter_cross_covariance="independent"),
        dict(read_parameter_cross_covariance=np.zeros((7, 1))),
        dict(read_parameter_cross_covariance=np.ones((8, 1))),
        dict(read_bias=np.ones(7)),
        dict(parameter_bias=np.array([np.inf])),
        dict(use_expectation=True),
    ],
)
def test_invalid_calibration(inputs, update):
    with pytest.raises(ValueError):
        calibrate_observations(*inputs[:2], **options(inputs, **update))


def test_invalid_parameter_maps_and_budget(inputs):
    for maps in (
        replace(inputs[4], names=("",)),
        replace(inputs[4], units=()),
        replace(inputs[4], gain=np.zeros((7, 1))),
    ):
        with pytest.raises(ValueError):
            calibrate_observations(*inputs[:2], **options(inputs, parameters=maps))
    for extra in (
        dict(read_parameter_cross_covariance=np.zeros((8, 1))),
        dict(representation="dense"),
    ):
        with pytest.raises(MemoryError):
            calibrate_observations(*inputs[:2], **options(inputs, **extra, max_dense_bytes=1))
    # Dense read covariance and diagonal parameter covariance use the same joint contract.
    out = calibrate_observations(
        *inputs[:2],
        **options(
            inputs, read_covariance=np.eye(8), read_parameter_cross_covariance=np.zeros((8, 1))
        ),
    )
    assert out.valid.sum() == 5


@pytest.mark.parametrize(
    "update",
    [
        dict(shape=(1, 2, 3)),
        dict(band_names=("a", "a")),
        dict(response_tolerance=-1),
        dict(native_weights=np.ones((4, 8))),
        dict(native_weights=csr_array(np.ones((3, 8)))),
        dict(native_weights=csr_array(np.zeros((4, 8)))),
        dict(max_dense_bytes=-1),
    ],
)
def test_invalid_cube_arguments(inputs, update):
    new = calibrate_observations(*inputs[:2], **options(inputs))
    kwargs: dict[str, Any] = (
        dict(shape=(1, 2, 2), band_names=("a", "b"), native_weights=inputs[6]) | update
    )
    with pytest.raises(ValueError):
        form_observation_cube(new, inputs[3], **kwargs)


def test_bad_weights_response_and_geometry(inputs):
    new = calibrate_observations(*inputs[:2], **options(inputs))
    for data in (
        [-1.0, 2.0, 0.5, 0.5, 1.0],
        [np.nan, 0.5, 0.5, 0.5, 1.0],
        [0.4, 0.5, 0.5, 0.5, 1.0],
    ):
        w = inputs[6].copy()
        w.data[:] = data
        with pytest.raises(ValueError):
            form_observation_cube(
                new, inputs[3], shape=(1, 2, 2), band_names=("a", "b"), native_weights=w
            )
    w = inputs[6].toarray()
    w[0] = 0
    w[0, 2] = 1
    with pytest.raises(ValueError, match="within"):
        form_observation_cube(
            new, inputs[3], shape=(1, 2, 2), band_names=("a", "b"), native_weights=csr_array(w)
        )
    spectral = new.response.spectral_coefficients.copy()
    spectral[4] = [0.0, 1.0, 1.0]
    changed = replace(new, response=replace(new.response, spectral_coefficients=spectral))
    with pytest.raises(ValueError, match="incompatible"):
        form(changed, inputs)
    broken = list(inputs)
    with pytest.raises(ValueError, match="coordinates"):
        replace(inputs[3], xy_m=np.full_like(inputs[3].xy_m, np.nan))
    # Retained writable buffers still require validation at their consuming boundary.
    broken[3] = replace(inputs[3], xy_m=inputs[3].xy_m.copy())
    broken[3].xy_m[:] = np.nan
    with pytest.raises(ValueError, match="geometry"):
        form(new, broken)


def test_identical_large_band_avoids_quadratic_comparisons(inputs, monkeypatch):
    calibrated = calibrate_observations(*inputs[:2], **options(inputs))
    n = 4096
    schedule = raster_read_schedule(np.array([[0.0, 0.01]]), (1, n))

    def repeated(array):
        return np.repeat(array[:1], n, axis=0)

    response = replace(
        calibrated.response,
        schedule=schedule,
        read_bands=("a",) * n,
        spectral_coefficients=repeated(calibrated.response.spectral_coefficients),
        spatial_weights=repeated(calibrated.response.spatial_weights),
        node_times_s=repeated(calibrated.response.node_times_s),
        node_directions_uv=repeated(calibrated.response.node_directions_uv),
    )
    acquisition = inputs[3].rays.acquisition
    acquisition = replace(
        acquisition,
        reads=schedule.reads,
        **{
            name: repeated(getattr(acquisition, name))
            for name in (
                "times_s",
                "directions_uv",
                "image_xy_m",
                "weights_sr_s",
                "pixel_bounds_m",
                "direction_bounds_uv",
            )
        },
    )
    geometry = replace(
        inputs[3],
        rays=replace(
            inputs[3].rays,
            acquisition=acquisition,
            **{
                name: repeated(getattr(inputs[3].rays, name))
                for name in (
                    "origins_gcrs_m",
                    "directions_gcrs",
                    "support_times_s",
                    "support_origins_gcrs_m",
                    "support_directions_gcrs",
                )
            },
        ),
        schedule=schedule,
        read_bands=response.read_bands,
        xy_m=repeated(inputs[3].xy_m),
        view_local=repeated(inputs[3].view_local),
        bounds_m=repeated(inputs[3].bounds_m),
    )
    calibrated = replace(
        calibrated,
        response=response,
        values=np.ones(n),
        valid=np.ones(n, dtype=bool),
        read_indices=np.arange(n),
        parameter_sensitivity=np.zeros((n, 1)),
        covariance=ProjectedCovariance((np.zeros((n, 1)),), (np.ones(1),)),
    )
    weights = csr_array((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    absolute = np.abs
    compared_entries = 0

    def bounded_absolute(values, *args, **kwargs):
        nonlocal compared_entries
        if values.ndim == 2 and values.shape[1] == 3:
            compared_entries += values.size
            assert compared_entries <= n * 3, "quadratic spectral comparisons"
        return absolute(values, *args, **kwargs)

    monkeypatch.setattr(np, "abs", bounded_absolute)
    cube = form_observation_cube(
        calibrated,
        geometry,
        shape=(1, n, 1),
        band_names=("a",),
        native_weights=weights,
    )
    np.testing.assert_array_equal(cube.values, 1)
    assert compared_entries <= n * 3


def test_spectral_compatibility_checks_all_distinct_pairs(inputs):
    calibrated = calibrate_observations(*inputs[:2], **options(inputs))
    spectral = calibrated.response.spectral_coefficients.copy()
    tolerance = 1e-8
    delta = 0.3 * tolerance
    spectral[[0, 1, 4]] = [
        [0.5, 0.5, 0],
        [0.5 + delta, 0.5 - delta, 0],
        [0.5 - delta, 0.5 + delta, 0],
    ]
    # Both extremes agree with the first row, but their mutual L1 exceeds tolerance.
    assert np.sum(np.abs(spectral[0] - spectral[1])) < tolerance
    assert np.sum(np.abs(spectral[0] - spectral[4])) < tolerance
    assert np.sum(np.abs(spectral[1] - spectral[4])) > tolerance
    changed = replace(
        calibrated, response=replace(calibrated.response, spectral_coefficients=spectral)
    )
    with pytest.raises(ValueError, match="incompatible"):
        form(changed, inputs, response_tolerance=tolerance)
    form(changed, inputs, response_tolerance=1.3 * tolerance)


@pytest.mark.parametrize("kind", ["parameter", "read", "sum"])
def test_finite_bias_inputs_reject_output_overflow(inputs, kind):
    updates = {
        "parameter": dict(parameter_bias=np.array([1e308])),
        "read": dict(read_bias=np.full(8, 1e308), detector=replace(inputs[2], electrons_per_dn=4)),
        "sum": dict(read_bias=np.full(8, 1e308), parameter_bias=np.array([5e306])),
    }
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            calibrate_observations(*inputs[:2], **options(inputs, **updates[kind]))


def test_cube_bias_transport_rejects_finite_overflow(inputs):
    calibrated = calibrate_observations(*inputs[:2], **options(inputs))
    calibrated = replace(
        calibrated, bias=np.full(calibrated.read_indices.size, np.finfo(float).max)
    )
    # The weight-sum tolerance admits this slight excess over unity.
    weights = inputs[6].copy()
    weights.data *= 1 + 5e-13
    changed = list(inputs)
    changed[6] = weights
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            form(calibrated, changed)
