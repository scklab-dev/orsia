"""Geometry-free radiometry, compact response storage and shared-block errors."""

import runpy
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from orsia_instruments import (
    NativeBandResponse,
    RadiometricParameterMaps,
    accumulate_observations,
    calibrate_frames,
    calibrate_observations,
    calibrate_readings,
    form_observation_cube,
    group_native_bands,
    normalized_accumulation_weights,
    raster_read_schedule,
    reconstruct_radiance_cube,
)
from orsia_radiative import SpectralBand
from orsia_uncertainty import ProjectedCovariance, materialize_covariance
from scipy.sparse import csr_array, diags_array

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/native_observation_cube.py"


@pytest.fixture
def inputs():
    return runpy.run_path(str(EXAMPLE))["example_inputs"]()


def dense(covariance):
    return (
        materialize_covariance(covariance, max_bytes=10000)
        if isinstance(covariance, ProjectedCovariance)
        else covariance
    )


@pytest.mark.parametrize("explicit_empty", [False, True])
@pytest.mark.parametrize("representation", ["projected", "dense"])
def test_fixed_calibration_and_empty_parameter_contract(explicit_empty, representation):
    updates: dict[str, Any] = {}
    if explicit_empty:
        updates = dict(
            parameters=RadiometricParameterMaps(
                (), (), np.empty((3, 0)), np.empty((3, 0)), np.empty((3, 0))
            ),
            parameter_covariance=np.empty((0, 0)),
            parameter_bias=np.empty(0),
            read_parameter_cross_covariance=np.empty((3, 0)),
        )
    result = calibrate_readings(
        np.array([5.0, 1.0, np.nan]),
        sensitivity=np.array([2.0, 4.0, 0.0]),
        gain=2.0,
        offset=2.0,
        dark_electrons=1.0,
        read_covariance=np.array([1.0, 4.0, 9.0]),
        read_bias=np.array([0.2, -0.4, 0.0]),
        representation=representation,
        **updates,
    )
    np.testing.assert_allclose(result.values, [2.5, -0.75, np.nan])
    np.testing.assert_allclose(dense(result.covariance), np.diag([1, 1]))
    assert result.bias is not None
    np.testing.assert_allclose(result.bias, [0.2, -0.2])
    assert result.parameter_names == result.parameter_units == ()
    assert result.parameter_sensitivity.shape == (2, 0)
    if isinstance(result.covariance, ProjectedCovariance):
        assert len(result.covariance.projections) == 1
    output = accumulate_observations(result, csr_array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(output.values, [0.875, np.nan])
    np.testing.assert_allclose(dense(output.covariance), [[0.5]])
    assert output.bias is not None
    np.testing.assert_allclose(output.bias, [0.0])
    assert output.parameter_sensitivity.shape == (1, 0)


def test_two_read_blocks_preserve_shared_read_and_calibration_sources():
    blocks = (np.array([10.0, 12.0]), np.array([20.0, 24.0]))
    readings = np.concatenate(blocks)
    shared_read = np.ones((4, 1))
    read_covariance = ProjectedCovariance(
        (csr_array(diags_array(np.ones(4))), shared_read),
        (np.ones(4), np.array([0.25])),
    )
    maps = RadiometricParameterMaps(
        ("common-gain",), ("1",), np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))
    )
    calibrated = calibrate_readings(
        readings,
        sensitivity=np.ones(4),
        read_covariance=read_covariance,
        parameters=maps,
        parameter_covariance=np.array([0.01]),
        read_parameter_cross_covariance="zero",
        read_bias=np.array([0.1, -0.1, 0.2, -0.2]),
        parameter_bias=np.array([-0.02]),
    )
    assert isinstance(calibrated.covariance, ProjectedCovariance)
    assert len(calibrated.covariance.projections) == 3
    weights = csr_array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    output = accumulate_observations(calibrated, weights)
    expected = np.eye(2) * 0.5 + 0.25 + np.outer([11, 22], [11, 22]) * 0.01
    np.testing.assert_allclose(dense(output.covariance), expected)
    assert output.bias is not None
    np.testing.assert_allclose(output.bias, [-0.22, -0.44])
    np.testing.assert_allclose(output.parameter_sensitivity, [[11], [22]])
    # A second accumulation must still retain the same single common gain error.
    combined = accumulate_observations(output, csr_array([[0.5, 0.5]]))
    np.testing.assert_allclose(dense(combined.covariance), [[0.25 + 0.25 + 16.5**2 * 0.01]])
    assert expected[0, 1] > 0


def test_compact_response_native_paths_match_dense(inputs):
    images, response, detector, geometry, maps, read_valid, weights = inputs
    table, inverse = np.unique(response.spectral_coefficients, axis=0, return_inverse=True)
    compact = replace(response, spectral_coefficients=table, response_indices=inverse)
    np.testing.assert_allclose(compact.sensitivity, response.sensitivity)
    np.testing.assert_allclose(
        compact.coefficients_for_reads(np.arange(8)), response.spectral_coefficients
    )
    assert compact.spectral_coefficients.shape[0] < 8
    legacy = calibrate_frames(images, response, detector=detector)
    current = calibrate_frames(images, compact, detector=detector)
    np.testing.assert_allclose(current.values, legacy.values)
    options = dict(
        detector=detector,
        read_covariance=np.ones(8),
        read_valid=read_valid,
    )
    native = calibrate_observations(images, compact, **options)
    reference = calibrate_observations(images, response, **options)
    cube = form_observation_cube(
        native, geometry, shape=(1, 2, 2), band_names=("a", "b"), native_weights=weights
    )
    np.testing.assert_allclose(native.values, reference.values)
    np.testing.assert_allclose(dense(native.covariance), dense(reference.covariance))
    assert cube.parameter_sensitivity.shape == (3, 0)
    assert cube.response is compact
    assert compact.spectral_coefficients.shape == table.shape
    weights32 = csr_array(
        (weights.data, weights.indices.astype(np.int32), weights.indptr.astype(np.int32)),
        shape=weights.shape,
    )
    cube32 = form_observation_cube(
        native, geometry, shape=(1, 2, 2), band_names=("a", "b"), native_weights=weights32
    )
    np.testing.assert_allclose(cube32.values, cube.values)
    # Absolute per-read sensitivities can vary independently of shared spectral shape.
    scaled = replace(compact, read_sensitivity=2 * compact.sensitivity)
    np.testing.assert_allclose(scaled.sensitivity, 2 * compact.sensitivity)
    np.testing.assert_allclose(
        scaled.coefficients_for_reads(np.arange(8)), 2 * response.spectral_coefficients
    )
    result = calibrate_observations(images, scaled, **options)
    np.testing.assert_allclose(result.values, reference.values / 2)
    np.testing.assert_allclose(dense(result.covariance), dense(reference.covariance) / 4)
    grouped_dense, dense_bands = group_native_bands(response)
    grouped_compact, compact_bands = group_native_bands(compact)
    assert grouped_dense.read_bands == grouped_compact.read_bands
    for first, second in zip(dense_bands, compact_bands, strict=True):
        np.testing.assert_allclose(first.weights, second.weights)
    # Uncertain calibration remains additive to the fixed-calibration contract.
    uncertain = calibrate_observations(
        images,
        compact,
        parameters=maps,
        parameter_covariance=np.array([1e-4]),
        read_parameter_cross_covariance="zero",
        **options,
    )
    assert uncertain.parameter_sensitivity.shape == (5, 1)


def test_compact_reconstruction_keeps_first_native_achieved_response(inputs):
    images, response, detector, geometry = inputs[:4]
    table = np.array([[0.5, 0.5, 0], [0.5 + 1e-10, 0.5 - 1e-10, 0]])
    compact = replace(
        response,
        spectral_coefficients=table,
        response_indices=np.array([1, 0, 1, 0, 1, 0, 1, 0]),
        read_bands=("one",) * 8,
    )
    calibrated = calibrate_frames(images, compact, detector=detector)
    cube = reconstruct_radiance_cube(
        calibrated,
        geometry,
        x_edges_m=np.array([geometry.bounds_m[:, 0].min() - 1, geometry.bounds_m[:, 1].max() + 1]),
        y_edges_m=np.array([geometry.bounds_m[:, 2].min() - 1, geometry.bounds_m[:, 3].max() + 1]),
        bands=(SpectralBand("one", compact.grid, table[1]),),
    )
    assert cube.valid.all()
    np.testing.assert_array_equal(cube.bands[0].weights, compact.normalized_responses[1])


def test_compact_grouping_keeps_genuinely_distinct_shapes_and_dark_reads(inputs):
    response = inputs[1]
    table = np.array([[0.5, 0.5, 0], [0.5 + 3e-9, 0.5 - 3e-9, 0], [0.5 - 3e-9, 0.5 + 3e-9, 0]])
    indices = np.array([0, 1, 2, 0, 1, 2, 0, 0])
    sensitivity = np.ones(8)
    sensitivity[-1] = 0
    spatial = response.spatial_weights.copy()
    spatial[-1] = 0
    compact = replace(
        response,
        spectral_coefficients=table,
        response_indices=indices,
        read_sensitivity=sensitivity,
        spatial_weights=spatial,
    )
    grouped, bands = group_native_bands(compact, response_tolerance=1e-8)
    assert len(bands) == 2  # Both extremes match row zero, but not each other.
    assert grouped.read_bands[-1] == ""
    assert grouped.read_bands[1] != grouped.read_bands[2]
    np.testing.assert_array_equal(grouped.spectral_coefficients, table)
    sensitivity[:] = 8
    indices[:] = 0
    assert compact.read_sensitivity[-1] == 0  # Owned snapshots.
    assert compact.response_indices[1] == 1
    np.testing.assert_allclose(compact.coefficients_for_reads(np.array([7])), 0)


@pytest.mark.parametrize(
    "update",
    [
        dict(response_indices=np.zeros(7, dtype=np.int64)),
        dict(response_indices=np.zeros(8)),
        dict(response_indices=np.full(8, 20, dtype=np.int64)),
        dict(read_sensitivity=np.ones(7)),
        dict(read_sensitivity=np.full(8, -1.0)),
        dict(read_sensitivity=np.full(8, np.inf)),
        dict(
            spectral_coefficients=np.zeros((2, 3)),
            response_indices=np.zeros(8, dtype=np.int64),
            read_sensitivity=np.ones(8),
        ),
    ],
)
def test_invalid_compact_response(inputs, update):
    with pytest.raises(ValueError):
        replace(inputs[1], **update)


@pytest.mark.parametrize(
    "indices", [np.array([-1]), np.array([8]), np.array([0.0]), np.zeros((1, 1), dtype=np.int64)]
)
def test_invalid_response_selection(inputs, indices):
    with pytest.raises(ValueError):
        inputs[1].coefficients_for_reads(indices)


def test_compact_large_spectral_support_has_bounded_scratch():
    from orsia_radiative.spectral.grid import wavenumber_grid

    n, k = 12_000, 8192
    schedule = raster_read_schedule(np.array([[0.0, 1.0]]), (1, n))
    points = wavenumber_grid(np.linspace(1000, 2000, k))
    coefficients = np.zeros((2, k))
    coefficients[0, : k // 2] = 1 / (k // 2)
    coefficients[1, k // 2 :] = 1 / (k // 2)
    tracemalloc.start()
    try:
        response = NativeBandResponse(
            schedule,
            points,
            ("band",) * n,
            coefficients,
            np.ones((n, 1)),
            np.full((n, 1), 0.5),
            np.zeros((n, 1, 2)),
            response_indices=np.arange(n) % 2,
            read_sensitivity=np.full(n, 2.0),
        )
        grouped, bands = group_native_bands(response)
        calibrated = calibrate_readings(
            np.full(n, 6.0), sensitivity=grouped.sensitivity, read_covariance=np.ones(n)
        )
        weights = csr_array((np.full(n, 1 / n), (np.zeros(n), np.arange(n))), shape=(1, n))
        output = accumulate_observations(calibrated, weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(bands) == 2
    assert response.spectral_coefficients.nbytes == 2 * k * 8
    assert peak < 20_000_000, f"compact scratch exceeded 20 MB: {peak}"
    assert n * k * 8 > 750_000_000  # A hidden dense response would exceed the bound.
    np.testing.assert_allclose(output.values, [3])
    np.testing.assert_allclose(dense(output.covariance), [[0.25 / n]])


@pytest.mark.parametrize(
    "update",
    [
        dict(sensitivity=np.array([-1.0, 1.0])),
        dict(gain=0.0),
        dict(gain=np.ones(3)),
        dict(dark_electrons=-1.0),
        dict(read_covariance=np.ones(3)),
        dict(parameter_covariance=np.ones(1)),
        dict(read_parameter_cross_covariance=np.ones((2, 1))),
    ],
)
def test_numerical_invalid_support(update):
    options: dict[str, Any] = dict(sensitivity=np.ones(2), read_covariance=np.ones(2))
    options.update(update)
    with pytest.raises(ValueError):
        calibrate_readings(np.ones(2), **options)


def test_numerical_projected_cross_rejection_and_masked_support():
    read_covariance = ProjectedCovariance((np.ones((2, 1)),), (np.ones(1),))
    maps = RadiometricParameterMaps(
        ("gain",), ("1",), np.ones((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))
    )
    for cross in (None, np.zeros((2, 1))):
        with pytest.raises(ValueError):
            calibrate_readings(
                np.ones(2),
                sensitivity=np.ones(2),
                read_covariance=read_covariance,
                parameters=maps,
                parameter_covariance=np.ones(1),
                read_parameter_cross_covariance=cross,
            )
    result = calibrate_readings(
        np.array([np.nan, 2.0]),
        sensitivity=np.ones(2),
        read_covariance=read_covariance,
        read_valid=np.array([False, True]),
    )
    np.testing.assert_array_equal(result.indices, [1])
    with pytest.raises(ValueError, match="invalid reads"):
        accumulate_observations(result, csr_array([[0.5, 0.5]]))
    with pytest.raises(ValueError):
        accumulate_observations(replace(result, indices=np.array([0])), csr_array([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        normalized_accumulation_weights(csr_array([[0.2, 0.2]]), 2)


def test_compact_coaddition_quotient_derivative_and_shared_blocks():
    from orsia_instruments.calibration.radiometry import coadd_readings

    q = np.array([[5.0, 12.0, 4.0], [9.0, 10.0, 8.0], [np.nan, np.nan, np.nan]])
    norm = np.array([1.0, 3.0, 2.0])
    valid = np.array([[True, True, False], [True, True, True], [False, False, False]])
    weights = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    gain = np.linspace(0.1, 0.9, 9)[:, None]
    offset = np.full((9, 1), 0.2)
    ds = np.linspace(0.2, 1.0, 9)[:, None]
    maps = RadiometricParameterMaps(("calibration",), ("1",), gain, offset, ds)
    result = coadd_readings(
        q,
        sensitivity=norm,
        read_variance=np.ones(q.shape),
        read_valid=valid,
        weights=weights,
        parameters=maps,
        parameter_covariance=np.array([0.04]),
        read_parameter_cross_covariance="zero",
        read_bias=np.full(q.shape, -0.3),
        parameter_bias=np.array([0.1]),
    )
    a = np.where(valid, weights, 0)
    charge = np.nan_to_num(q)
    denominator = np.sum(a * norm, axis=1)[:2]
    expected = np.sum(a * charge, axis=1)[:2] / denominator
    np.testing.assert_allclose(result.values[:2], expected)
    assert np.isnan(result.values[2])
    epsilon = 1e-5

    def perturbed(sign):
        d = sign * epsilon
        return (
            np.sum(
                a * (charge * (1 + d * gain.reshape(q.shape)) - d * offset.reshape(q.shape)), axis=1
            )[:2]
            / np.sum(a * (norm + d * ds.reshape(q.shape)), axis=1)[:2]
        )

    finite = (perturbed(1) - perturbed(-1)) / (2 * epsilon)
    np.testing.assert_allclose(result.parameter_sensitivity[:, 0], finite, rtol=2e-9)
    independent = np.sum(a * a, axis=1)[:2] / denominator**2
    np.testing.assert_allclose(
        dense(result.covariance), np.diag(independent) + 0.04 * np.outer(finite, finite), rtol=2e-9
    )
    assert result.bias is not None
    np.testing.assert_allclose(
        result.bias, -0.3 * np.sum(a, axis=1)[:2] / denominator + 0.1 * finite, rtol=2e-9
    )
    assert isinstance(result.covariance, ProjectedCovariance)
    assert result.covariance.source_covariances[0].size == 3  # groups, never reads
    columns = []
    for index in range(2):
        block_maps = RadiometricParameterMaps(
            maps.names,
            maps.units,
            gain[index * 3 : index * 3 + 3],
            offset[index * 3 : index * 3 + 3],
            ds[index * 3 : index * 3 + 3],
        )
        block = coadd_readings(
            q[index : index + 1],
            sensitivity=norm,
            read_variance=np.ones((1, 3)),
            read_valid=valid[index : index + 1],
            weights=weights[index : index + 1],
            parameters=block_maps,
            parameter_covariance=np.array([0.04]),
            read_parameter_cross_covariance="zero",
        )
        columns.append(block.parameter_sensitivity)
    joined = np.concatenate(columns)
    np.testing.assert_allclose(joined, result.parameter_sensitivity)
    assert (joined @ np.array([[0.04]]) @ joined.T)[0, 1] != 0
    with pytest.raises(ValueError, match="cross"):
        coadd_readings(
            q,
            sensitivity=norm,
            read_variance=np.ones(q.shape),
            read_valid=valid,
            parameters=maps,
            parameter_covariance=np.array([0.04]),
        )
    with pytest.raises(ValueError):
        coadd_readings(q, sensitivity=-norm, read_variance=np.ones(q.shape), read_valid=valid)
