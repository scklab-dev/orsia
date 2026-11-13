"""Common field contracts and independent finite-support mass/moment oracles."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import (
    GaussianPuffDispersion,
    GaussianPuffField,
    GaussianPuffs,
    LayerMassMoments,
    PuffReleases,
    StationaryGaussianPlume,
    concentration,
    layer_columns,
    pixel_layer_moments,
    puff_layer_columns,
    sample_fields,
)
from orsia_geometry import LayeredCells, LocalCartesianFrame, RectangularCells
from scipy.integrate import quad
from scipy.special import ndtr

TEST_FRAME = LocalCartesianFrame("dispersion test", "synthetic altitude")


def layered_cells(lower, upper, edges, terrain=None):
    horizontal = RectangularCells(lower, upper, TEST_FRAME)
    surface = np.full(horizontal.shape, edges[0]) if terrain is None else terrain
    return LayeredCells(horizontal, edges, surface)


def stationary(**changes):
    return StationaryGaussianPlume(
        **(
            dict(
                emission_rate_kg_s=1.0,
                wind_speed_m_s=5.0,
                source_regularization_m=20.0,
                stack_height_m=10.0,
                stack_inside_diameter_m=4.0,
                stack_exit_velocity_m_s=20.0,
                stack_exit_temperature_k=290.0,
                ambient_temperature_k=290.0,
            )
            | changes
        )
    )


def puff(boundary: Literal["free", "reflecting"] = "free", covariance=None, centre=(0.0, 0.0, 2.0)):
    return GaussianPuffField(
        GaussianPuffs(
            0.0,
            np.zeros(1),
            np.array([3.0]),
            np.array([centre]),
            np.array([np.diag([4.0, 9.0, 1.0]) if covariance is None else covariance]),
        ),
        boundary,
    )


def wind(points, time):
    return np.broadcast_to([1.0, 0.0, 0.0], points.shape), np.zeros((len(points), 3, 3))


def diffusion(points, time, ages):
    return np.zeros((len(points), 3, 3))


def puff_model():
    state = puff("reflecting").state
    return GaussianPuffDispersion(
        releases=PuffReleases(
            state.release_times_s, state.masses_kg, state.centres_m, state.covariances_m2
        ),
        wind=wind,
        diffusivity=diffusion,
        boundary="reflecting",
        rtol=1e-7,
        atol_position_m=1e-9,
        atol_covariance_m2=1e-9,
        min_step_s=1e-6,
        max_step_s=1.0,
        max_steps=1000,
        envelope_sigma=3.0,
        wind_tolerance_m_s=0.1,
        diffusivity_tolerance_m2_s=0.1,
        split_fraction=0.5,
        max_puffs=100,
        max_split_rounds=10,
    )


def test_common_chronology_history_and_snapshot_isolation():
    model = puff_model()
    fields = sample_fields(model, np.array([-1.0, 0.0, 2.0]))
    empty = next(fields)
    assert isinstance(empty, GaussianPuffField) and empty.state.masses_kg.size == 0
    first = next(fields)
    assert isinstance(first, GaussianPuffField)
    saved = first.state.centres_m.copy()
    first.state.centres_m[:] = 1000
    last = next(fields)
    assert isinstance(last, GaussianPuffField)
    assert_allclose(last.state.centres_m, saved + [2, 0, 0])
    assert last.diagnostics is not None
    assert last.diagnostics.released_mass_kg == 3
    assert list(sample_fields(model, np.empty(0))) == []
    constant = stationary()
    assert list(sample_fields(constant, np.array([-1.0, 4.0]))) == [constant, constant]
    for bad in ([1.0, 1.0], [2.0, 1.0], [np.nan], [[0.0]]):
        with pytest.raises(ValueError, match="times_s"):
            list(sample_fields(constant, np.array(bad)))
    assert list(sample_fields(constant, np.empty(0))) == []


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_analytic_rectangles_clipping_moments_and_partition(boundary):
    field = puff(boundary)
    lower = np.array([[[-30.0, -30.0], [-30.0, -30.0]]])
    upper = lower + 60
    edges = np.array([0.0, 1.0, 3.0, 20.0])
    terrain = np.array([[0.0, 2.0]])
    result = pixel_layer_moments(field, layered_cells(lower, upper, edges, terrain))
    # Independent univariate truncated-normal formula, diagonal horizontal support.
    horizontal = (ndtr(15) - ndtr(-15)) * (ndtr(10) - ndtr(-10)) * 3 / 3600
    expected = np.zeros_like(result.columns_kg_m2)
    moments = np.zeros_like(expected)
    for z in [2.0] if boundary == "free" else [2.0, -2.0]:
        a = np.minimum(np.maximum(edges[:-1], terrain[..., None]), edges[1:])
        b = np.broadcast_to(edges[1:], a.shape)
        probability = ndtr(b - z) - ndtr(a - z)
        expected += horizontal * probability
        moments += horizontal * (
            z * probability
            + (np.exp(-0.5 * (a - z) ** 2) - np.exp(-0.5 * (b - z) ** 2)) / np.sqrt(2 * np.pi)
        )
    assert_allclose(result.columns_kg_m2, expected, atol=1e-18)
    assert_allclose(result.height_moments_kg_m, moments, atol=1e-18)
    assert result.columns_kg_m2[0, 1].sum() < result.columns_kg_m2[0, 0].sum()
    assert not result.columns_kg_m2.flags.writeable
    combined = pixel_layer_moments(field, layered_cells(lower, upper, edges[[0, -1]], terrain))
    assert_allclose(result.columns_kg_m2.sum(-1), combined.columns_kg_m2[..., 0])
    assert_allclose(result.height_moments_kg_m.sum(-1), combined.height_moments_kg_m[..., 0])
    assert_allclose(
        layer_columns(field, np.zeros((1, 2)), edges),
        puff_layer_columns(field.state, np.zeros((1, 2)), edges, boundary=boundary),
    )


def test_stationary_pixel_integral_matches_full_column_and_point_limit():
    field = stationary()
    east = np.array([-100.0, 0.0, 100.0, 1000.0])
    north = np.arange(500.0, -501.0, -50.0)
    x, y = np.meshgrid(east, north)
    centres = np.stack((x, y), axis=-1)
    half = np.array([50.0, 25.0])
    result = pixel_layer_moments(
        field, layered_cells(centres - half, centres + half, np.array([0.0, 5.0, 10.0, 1000.0]))
    )
    # Independent scalar adaptive integral of the full-height crosswind probability.
    expected = np.zeros_like(x)
    for row, column in np.ndindex(x.shape):
        left, right = max(0.0, x[row, column] - 50), x[row, column] + 50
        if right <= left:
            continue

        def horizontal(distance, row=row):
            effective = max(distance, 20.0)
            sigma = 0.08 * effective / np.sqrt(1 + 0.0001 * effective)
            return (ndtr((north[row] + 25) / sigma) - ndtr((north[row] - 25) / sigma)) / 5

        expected[row, column] = (
            quad(
                horizontal,
                left,
                right,
                points=[20.0] if left < 20 < right else None,
                epsabs=1e-12,
                epsrel=1e-10,
            )[0]
            / 5000
        )
    assert_allclose(result.columns_kg_m2.sum(-1), expected, rtol=1e-8, atol=1e-12)
    assert_allclose(result.columns_kg_m2[:, 2:].sum(axis=(0, 2)) * 50, 1 / 5, rtol=2e-7)
    assert_array_equal(result.columns_kg_m2[:, 0], 0)
    xy = np.array([[200.0, 0.0]])
    narrow = pixel_layer_moments(field, layered_cells(xy - 1e-3, xy + 1e-3, np.array([0.0, 20.0])))
    assert_allclose(
        narrow.columns_kg_m2, layer_columns(field, xy, np.array([0.0, 20.0])), rtol=1e-8
    )
    assert_allclose(
        concentration(field, np.array([[200.0, 0.0, 10.0]])),
        layer_columns(field, xy, np.array([10.0 - 1e-5, 10.0 + 1e-5]))[:, 0] / 2e-5,
        rtol=1e-8,
    )
    shifted = replace(field, source_east_m=50.0, source_north_m=10.0, ground_altitude_m=100.0)
    shifted_result = pixel_layer_moments(
        shifted,
        layered_cells(xy - 1e-3 + [50.0, 10.0], xy + 1e-3 + [50.0, 10.0], np.array([100.0, 120.0])),
    )
    assert_allclose(shifted_result.columns_kg_m2, narrow.columns_kg_m2, rtol=1e-10)
    assert_allclose(
        shifted_result.height_moments_kg_m,
        narrow.height_moments_kg_m + 100 * narrow.columns_kg_m2,
        rtol=1e-10,
    )
    removed = pixel_layer_moments(
        replace(field, removal_timescale_s=10.0),
        layered_cells(centres - half, centres + half, np.array([0.0, 1000.0])),
    )
    assert np.all(removed.columns_kg_m2[..., 0] <= result.columns_kg_m2.sum(-1) + 1e-16)


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[4.0, 1.0, 0.7], [1.0, 3.0, 0.8], [0.7, 0.8, 2.0]]),
        np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]),
        np.array([[4.0, 0.0, 0.7], [0.0, 3.0, 0.0], [0.7, 0.0, 2.0]]),
    ],
)
def test_tilted_puff_against_independent_three_dimensional_quadrature(covariance):
    field = puff(covariance=covariance)
    lo, hi = np.array([[-2.0, -1.0]]), np.array([[1.0, 3.0]])
    edges = np.array([0.0, 1.0, 4.0])
    result = pixel_layer_moments(field, layered_cells(lo, hi, edges, np.array([0.5])), rtol=1e-9)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    determinant = np.linalg.det(covariance)
    inverse = np.linalg.inv(covariance)
    for k, (a, b) in enumerate(zip([0.5, 1.0], [1.0, 4.0], strict=True)):
        lower, upper = np.r_[lo[0], a], np.r_[hi[0], b]
        axes = (lower[:, None] + upper[:, None]) / 2 + (upper - lower)[:, None] / 2 * nodes
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        delta = points - field.state.centres_m[0]
        density = (
            3
            * np.exp(-0.5 * np.einsum("...i,ij,...j->...", delta, inverse, delta))
            / np.sqrt((2 * np.pi) ** 3 * determinant)
        )
        w = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * np.prod((upper - lower) / 2)
            / 12
        )
        assert_allclose(result.columns_kg_m2[0, k], np.sum(density * w), rtol=1e-9)
        assert_allclose(
            result.height_moments_kg_m[0, k], np.sum(density * w * points[..., 2]), rtol=1e-9
        )


def test_empty_zero_tail_and_invalid_domains():
    field = puff()
    lo, hi, edges = np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([0.0, 1.0, 2.0])
    assert_array_equal(
        pixel_layer_moments(field, layered_cells(lo, hi, edges, np.array([3.0]))).columns_kg_m2,
        0,
    )
    assert_array_equal(
        pixel_layer_moments(
            stationary(emission_rate_kg_s=0), layered_cells(lo, hi, edges)
        ).columns_kg_m2,
        0,
    )
    with pytest.raises(ValueError):
        concentration(stationary(), np.array([[0.0, 0.0, -1.0]]))
    kwargs: dict[str, Any]
    for kwargs in (
        dict(rtol=0),
        dict(atol_kg_m2=np.nan),
        dict(batch_size=0),
        dict(max_intervals=0),
    ):
        with pytest.raises(ValueError):
            pixel_layer_moments(field, layered_cells(lo, hi, edges), **kwargs)
    for a, b, z, terrain in (
        (lo, lo, edges, None),
        (lo, hi, np.array([0.0]), None),
        (lo, hi, edges, np.zeros(2)),
        (lo, hi, edges[::-1], None),
    ):
        with pytest.raises(ValueError):
            pixel_layer_moments(field, layered_cells(a, b, z, terrain))
    with pytest.raises(ValueError):
        layer_columns(field, np.array([[np.nan, 0.0]]), edges)
    with pytest.raises(ValueError):
        layer_columns(field, lo, edges, batch_size=0)
    with pytest.raises(ValueError):
        LayerMassMoments(layered_cells(lo, hi, edges), np.ones((1, 2)), np.ones((1, 2)) * 100)
    with pytest.raises(ValueError):
        LayerMassMoments(layered_cells(lo, hi, edges), -np.ones((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        stationary(removal_timescale_s=0)
    with pytest.raises(ValueError):
        stationary(wind_speed_m_s=-1)
    # A tight budget must fail explicitly rather than returning an unqualified integral.
    with pytest.raises(RuntimeError, match="integration failed"):
        pixel_layer_moments(
            stationary(),
            layered_cells(np.array([[0.0, -100.0]]), np.array([[10000.0, 100.0]]), edges),
            max_intervals=1,
            rtol=1e-12,
            atol_kg_m2=1e-30,
        )


def test_nearly_singular_horizontal_covariance_resolves_narrow_pixel():
    rho = 1 - 1e-10
    field = puff(covariance=np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    result = pixel_layer_moments(
        field,
        layered_cells(
            np.array([[-50.0, 1.234]]), np.array([[50.0, 1.235]]), np.array([-10.0, 20.0])
        ),
        rtol=1e-8,
    )
    expected = 3 * (ndtr(1.235) - ndtr(1.234)) / 0.1
    assert_allclose(result.columns_kg_m2, expected, rtol=1e-8)
    assert_allclose(result.height_moments_kg_m, 2 * result.columns_kg_m2, rtol=1e-8)


def test_thin_tail_layer_centroid_and_zero_mass_are_stable():
    field = puff()
    edges = np.array([10.0, 10.00000001])
    result = pixel_layer_moments(
        field, layered_cells(np.array([[-30.0, -30.0]]), np.array([[30.0, 30.0]]), edges)
    )
    centroid = result.height_moments_kg_m / result.columns_kg_m2
    assert_allclose(centroid, np.mean(edges), atol=1e-14, rtol=0)
    assert np.isfinite(result.height_moments_kg_m).all()


def test_subnormal_columns_and_moments_are_zeroed_together():
    result = pixel_layer_moments(
        puff(),
        layered_cells(np.array([[-30.0, -30.0]]), np.array([[30.0, 30.0]]), np.array([40.0, 41.0])),
    )
    assert_array_equal(result.columns_kg_m2, 0)
    assert_array_equal(result.height_moments_kg_m, 0)
    with pytest.raises(ValueError):
        pixel_layer_moments(
            puff(),
            layered_cells(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([0.0, 1.0])),
            atol_kg_m2=1e-320,
        )


def test_common_puff_concentration_and_empty_population():
    from orsia_dispersion import puff_concentration

    field = puff("reflecting")
    points = np.array([[0.0, 0.0, 2.0], [1.0, 2.0, 3.0]])
    assert_array_equal(
        concentration(field, points, batch_size=1),
        puff_concentration(field.state, points, boundary="reflecting", batch_size=1),
    )
    empty = GaussianPuffField(
        GaussianPuffs(0.0, np.empty(0), np.empty(0), np.empty((0, 3)), np.empty((0, 3, 3))), "free"
    )
    result = pixel_layer_moments(
        empty, layered_cells(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([0.0, 1.0]))
    )
    assert_array_equal(result.columns_kg_m2, 0)
