"""Full-domain photon quadrature and bounded native read/spectral contractions."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_instruments import frame_electron_tangents, simulate_frames
from orsia_instruments.photodetection.coefficients import (
    OpticalAcceptance,
    photoelectron_response,
    photon_conversion_weights,
    weighted_photoelectron_response,
)
from orsia_instruments.photodetection.frames import frame_response_coefficients
from orsia_instruments.response.joint import spatial_spectral_operator, spatial_spectral_response
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.constants import c, h
from test_filter_camera import acquisition, camera, connected_grid, coverage


def test_coordinate_measures_and_disconnected_intervals():
    points = np.array([1000.0, 1001.0, 1002.0, 2000.0, 2004.0])
    spectral = SpectralGrid(
        "wavenumber_cm1",
        points,
        SpectralDomain("wavenumber_cm1", ((1000.0, 1002.0), (2000.0, 2004.0))),
    )
    weights = photon_conversion_weights(spectral)
    assert_allclose(weights * (h * c * 100 * points), [0.5, 1.0, 0.5, 2.0, 2.0], rtol=2e-15)
    nm = SpectralGrid(
        "wavelength_nm",
        np.array([500.0, 600.0, 700.0]),
        SpectralDomain("wavelength_nm", ((500.0, 700.0),)),
    )
    assert_allclose(
        photon_conversion_weights(nm) * h * c / (nm.points * 1e-9), [50.0, 100.0, 50.0], rtol=2e-15
    )
    for invalid in (
        SpectralGrid("wavenumber_cm1", points),
        SpectralGrid(
            "wavenumber_cm1",
            np.array([1001.0]),
            SpectralDomain("wavenumber_cm1", ((1000.0, 1002.0),)),
        ),
        SpectralGrid(
            "wavenumber_cm1",
            np.array([1000.0, 1001.0]),
            SpectralDomain("wavenumber_cm1", ((1000.0, 1002.0),)),
        ),
    ):
        with pytest.raises(ValueError):
            photon_conversion_weights(invalid)


@pytest.mark.parametrize("block_size", [1, 3, 9])
@pytest.mark.parametrize("read_batch", [1, 3])
def test_block_coefficients_counts_normalization_and_query_shapes(block_size, read_batch):
    cam = camera(shape=(2, 2))
    grid = acquisition(cam, angular_order=2, frames=2)
    spectral = connected_grid(np.linspace(4000.0, 5000.0, 7))
    seen = []

    def radiance(xy, view, times, selected):
        assert selected.coordinate == "wavenumber_cm1" and selected.domain == spectral.domain
        seen.append((len(times), len(selected.points)))
        return (
            (2 + 0.1 * xy[:, 0, None] + times[:, None] + view[:, 2, None]) * selected.points * 1e-9
        )

    baseline = simulate_frames(cam, grid, spectral, radiance, coverage, read_batch_size=read_batch)
    seen.clear()
    blocked = simulate_frames(
        cam,
        grid,
        spectral,
        radiance,
        coverage,
        read_batch_size=read_batch,
        spectral_block_size=block_size,
    )
    assert max(k for _, k in seen) <= block_size
    assert max(q for q, _ in seen) <= read_batch * grid.xy_m.shape[1]
    assert_allclose(blocked.images.photoelectrons, baseline.images.photoelectrons, rtol=2e-14)
    assert_allclose(
        blocked.response.spectral_coefficients, baseline.response.spectral_coefficients, rtol=2e-15
    )
    assert_allclose(blocked.response.spatial_weights, baseline.response.spatial_weights, rtol=3e-15)
    assert blocked.reads.sample_ids == baseline.reads.sample_ids
    assembled = np.zeros((*grid.rays.acquisition.times_s.shape, spectral.points.size))
    visited = np.zeros((len(grid.read_bands), spectral.points.size), dtype=int)
    for reads, samples, coefficients in frame_response_coefficients(
        cam, grid, spectral, batch_size=read_batch, spectral_block_size=block_size
    ):
        assembled[reads, :, samples] = coefficients
        visited[reads, samples] += 1
    assert_array_equal(visited, 1)
    assert_allclose(assembled.sum(axis=1), baseline.response.spectral_coefficients, rtol=2e-15)


def test_weighted_kernel_uses_supplied_full_weights_for_singletons_and_qe():
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=1, angular_order=2)
    spectral = connected_grid(np.array([4000.0, 4500.0, 5000.0]))
    current = grid.rays.acquisition
    n, q = current.times_s.shape
    accepted = OpticalAcceptance(current, 0.01 / spectral.points, np.full((n, q, 3), 0.5))
    full = photoelectron_response(
        spectral, accepted, collecting_area_m2=2.0, quantum_efficiency=0.8
    )
    photon = photon_conversion_weights(spectral)
    for k in range(3):
        sliced = OpticalAcceptance(
            current, accepted.wavelengths_m[k : k + 1], accepted.fractions[:, :, k : k + 1]
        )
        actual = weighted_photoelectron_response(
            sliced,
            photon[k : k + 1],
            collecting_area_m2=2.0,
            quantum_efficiency=lambda indices, wave: np.full((len(indices), len(wave)), 0.8),
        )
        assert_array_equal(actual, full[:, :, k : k + 1])
    for bad in (np.full(3, -1.0), np.full(3, np.nan), np.ones(2)):
        with pytest.raises(ValueError, match="photon weights"):
            weighted_photoelectron_response(accepted, bad, collecting_area_m2=1.0)
    with pytest.raises(ValueError, match="QE callback"):
        weighted_photoelectron_response(
            accepted,
            photon,
            collecting_area_m2=1.0,
            quantum_efficiency=lambda indices, wave: np.ones(1),
        )
    with pytest.raises(ValueError, match="overflow"):
        weighted_photoelectron_response(accepted, photon, collecting_area_m2=1e308)


def test_operator_transpose_and_dense_response_share_spectral_blocks():
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=2, angular_order=2)
    spectral = connected_grid(np.linspace(4000.0, 5000.0, 7))
    bounds = grid.bounds_m
    options = dict(
        x_edges_m=np.linspace(bounds[:, 0].min() - 1, bounds[:, 1].max() + 1, 4),
        y_edges_m=np.linspace(bounds[:, 2].min() - 1, bounds[:, 3].max() + 1, 3),
        spectral_edges_cm1=np.array([4000.0, 4500.0, 5000.0]),
        max_bytes=10_000_000,
        read_batch_size=1,
    )
    dense = spatial_spectral_response((cam,), (grid,), spectral, **options)
    blocked = spatial_spectral_response((cam,), (grid,), spectral, **options, spectral_block_size=2)
    operator = spatial_spectral_operator(
        (cam,), (grid,), spectral, **options, spectral_block_size=1
    )
    assert_allclose(blocked.matrix, dense.matrix, rtol=3e-15)
    state = np.arange(dense.matrix.shape[1], dtype=float) * 1e-6
    cotangent = np.arange(dense.matrix.shape[0], dtype=float) + 1
    assert_allclose(operator @ state, dense.matrix @ state, rtol=3e-15)
    assert_allclose(operator.rmatvec(cotangent), dense.matrix.T @ cotangent, rtol=3e-15)
    assert_allclose(cotangent @ (operator @ state), state @ operator.rmatvec(cotangent), rtol=3e-15)
    for fn in (spatial_spectral_operator, spatial_spectral_response):
        with pytest.raises(ValueError, match="spectral_block_size"):
            fn((cam,), (grid,), spectral, **options, spectral_block_size=0)
    with pytest.raises(ValueError, match="spectral_block_size"):
        list(frame_response_coefficients(cam, grid, spectral, spectral_block_size=0))


@pytest.mark.parametrize("block_size", [1, 3, None])
def test_tangent_product_rule_and_zero_directions(block_size):
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=2, angular_order=2)
    spectral = connected_grid(np.linspace(4000.0, 5000.0, 5))
    full = np.zeros((*grid.rays.acquisition.times_s.shape, 5))
    for reads, samples, coefficients in frame_response_coefficients(cam, grid, spectral):
        full[reads, :, samples] = coefficients

    def radiance(xy, view, times, selected):
        values = np.broadcast_to(selected.points * 1e-9, (len(times), len(selected.points))).copy()
        return values, np.stack((0.2 * values, np.zeros_like(values)), axis=-1)

    def response_seed(reads, samples):
        values = full[reads, :, samples]
        return np.stack((-0.3 * values, 0.7 * values), axis=-1)

    signal, derivative = frame_electron_tangents(
        cam,
        grid,
        spectral,
        radiance,
        coverage,
        response_tangent=response_seed,
        spectral_block_size=block_size,
        read_batch_size=1,
    )
    expected = np.sum(full * (spectral.points * 1e-9), axis=(1, 2))
    assert_allclose(signal.total, expected, rtol=5e-15)
    assert_allclose(derivative, expected[:, None] * [-0.1, 0.7], rtol=2e-14)
    plain, seed = frame_electron_tangents(
        cam, grid, spectral, radiance, coverage, spectral_block_size=block_size
    )
    assert_allclose(plain.total, expected, rtol=5e-15)
    assert_allclose(seed, expected[:, None] * [0.2, 0.0], rtol=2e-14)

    def no_directions(xy, view, times, selected):
        return np.ones((len(times), len(selected.points))), np.empty(
            (len(times), len(selected.points), 0)
        )

    assert frame_electron_tangents(cam, grid, spectral, no_directions, coverage)[1].shape == (
        len(grid.read_bands),
        0,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "values",
        "seeds",
        "seed_nan",
        "varying",
        "response",
        "response_nan",
        "overflow",
        "coverage",
        "source",
    ],
)
def test_tangent_contract_failures(kind):
    cam = camera(shape=(1, 1))
    grid = acquisition(cam, frames=2, angular_order=2)
    spectral = connected_grid(np.linspace(4000.0, 5000.0, 5))
    calls = []

    def radiance(xy, view, times, selected):
        calls.append(1)
        values = np.ones((len(times), len(selected.points)))
        p = 2 if kind == "varying" and len(calls) > 1 else 1
        seeds = np.ones((*values.shape, p))
        if kind == "values":
            values[:] = -1
        if kind == "seeds":
            seeds = seeds[:, 0]
        if kind == "seed_nan":
            seeds[:] = np.nan
        if kind == "overflow":
            seeds[:] = 1e308
        return values, seeds

    def response(reads, samples):
        shape = (reads.stop - reads.start, grid.xy_m.shape[1], samples.stop - samples.start, 1)
        return np.full(shape, np.nan) if kind == "response_nan" else np.ones(1)

    options: dict[str, Any] = {"spectral_block_size": 2}
    if kind.startswith("response"):
        options["response_tangent"] = response
    if kind == "source":
        options["source_name"] = ""
    check = (
        (lambda current, spec: np.zeros(len(current.read_bands), dtype=bool))
        if kind == "coverage"
        else coverage
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            frame_electron_tangents(cam, grid, spectral, radiance, check, **options)
    if kind in ("coverage", "source"):
        assert not calls


def test_reciprocal_density_integrals_converge_under_each_quadrature_measure():
    reference = 1000.0 / (h * c * 100)
    errors = []
    for count in (9, 33, 129):
        wave = np.linspace(2000.0, 2500.0, count)
        nm = SpectralGrid(
            "wavelength_nm", wave, SpectralDomain("wavelength_nm", ((2000.0, 2500.0),))
        )
        # L_nu = nu and L_lambda = L_nu * |dnu/dlambda| describe the same source.
        per_nm = (1e7 / wave) * 1e7 / wave**2
        integrated = photon_conversion_weights(nm) @ per_nm
        errors.append(abs(integrated / reference - 1))
        points = np.linspace(4000.0, 5000.0, count)
        cm = connected_grid(points)
        assert_allclose(photon_conversion_weights(cm) @ points, reference, rtol=3e-15)
    assert errors[2] < errors[1] < errors[0] and errors[2] < 1e-5
