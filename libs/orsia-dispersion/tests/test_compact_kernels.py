"""Independent mass/moment and support oracles for conservative compact puffs."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import (
    GaussianPuffDispersion,
    GaussianPuffField,
    GaussianPuffs,
    PuffReleases,
    concentration,
    layer_columns,
    pixel_layer_moments,
    puff_concentration,
    puff_layer_columns,
    puff_path_columns,
    sample_fields,
)
from scipy.integrate import quad
from scipy.special import gammainc
from test_fields import layered_cells


def field(
    radius=2.0,
    *,
    boundary: Literal["free", "reflecting"] = "free",
    covariance=None,
    centre=(1.0, 2.0, 3.0),
):
    covariance = np.eye(3) if covariance is None else covariance
    return GaussianPuffField(
        GaussianPuffs(
            0.0, np.zeros(1), np.array([7.0]), np.array([centre]), np.array([covariance])
        ),
        boundary,
        kernel_cutoff_radius=radius,
    )


@pytest.mark.parametrize("radius", [0.05, 2.0, 8.0])
def test_mass_centre_covariance_with_independent_spherical_quadrature(radius):
    covariance = np.array([[4.0, 1.0, 0.3], [1.0, 3.0, 0.6], [0.3, 0.6, 2.0]])
    f = field(radius, covariance=covariance)
    alpha = gammainc(2.5, radius**2 / 2) / gammainc(1.5, radius**2 / 2)
    chol = np.linalg.cholesky(covariance / alpha)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    radial = (nodes + 1) * radius / 2
    azimuth = np.arange(64) * 2 * np.pi / 64
    r, mu, phi = np.meshgrid(radial, nodes, azimuth, indexing="ij")
    unit = np.stack(
        (np.sqrt(1 - mu**2) * np.cos(phi), np.sqrt(1 - mu**2) * np.sin(phi), mu), axis=-1
    )
    delta = (r[..., None] * unit) @ chol.T
    points = f.state.centres_m[0] + delta
    volume = (
        weights[:, None, None]
        * radius
        / 2
        * weights[None, :, None]
        * 2
        * np.pi
        / 64
        * r**2
        * np.linalg.det(chol)
    )
    density = concentration(f, points)
    w = density * volume
    assert_allclose(w.sum(), 7.0, rtol=2e-13)
    assert_allclose(
        np.sum(w[..., None] * points, axis=(0, 1, 2)) / 7, f.state.centres_m[0], atol=2e-13
    )
    assert_allclose(
        np.einsum("n,ni,nj->ij", w.ravel(), delta.reshape(-1, 3), delta.reshape(-1, 3)) / 7,
        covariance,
        rtol=3e-13,
        atol=2e-13,
    )
    outside = f.state.centres_m + np.array([[radius * 1.001, 0, 0]]) @ chol.T
    assert_array_equal(concentration(f, outside), 0)


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_pixel_partition_mass_moment_and_vertical_path_identity(boundary):
    f = field(boundary=boundary, covariance=np.diag([1.0, 2.0, 1.0]), centre=(0.0, 0.0, 3.0))
    bottom = -10.0 if boundary == "free" else 0.0
    lo = np.array([[-10.0, -10.0], [0.0, -10.0]])
    hi = np.array([[0.0, 10.0], [10.0, 10.0]])
    edges = np.array([bottom, 3.0, 20.0])
    moments = pixel_layer_moments(f, layered_cells(lo, hi, edges), rtol=2e-7)
    assert_allclose(np.sum(moments.columns_kg_m2) * 200, 7, rtol=2e-7)
    # The lower support of the unreflected kernel lies above zero in this case.
    assert_allclose(np.sum(moments.height_moments_kg_m) * 200, 21, rtol=2e-7)
    xy = np.array([[0.0, 0.0], [1.0, 0.4], [20.0, 0.0]])
    direct = puff_layer_columns(f.state, xy, edges, boundary=boundary, kernel_cutoff_radius=2)
    assert_allclose(layer_columns(f, xy, edges), direct, rtol=2e-14)
    start = np.column_stack((xy, np.full(3, bottom)))
    end = np.column_stack((xy, np.full(3, 20.0)))
    path = puff_path_columns(f.state, start, end, boundary=boundary, kernel_cutoff_radius=2)
    assert_allclose(path, direct.sum(axis=-1), rtol=2e-14, atol=1e-30)
    assert_array_equal(
        puff_path_columns(f.state, start, start, boundary=boundary, kernel_cutoff_radius=2), 0
    )
    assert_array_equal(direct[-1], 0)


def test_tilted_pixels_clipping_thin_cap_and_quadrature_convergence():
    covariance = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.2], [0.3, 0.2, 1.0]])
    f = field(covariance=covariance, centre=(0.0, 0.0, 0.0))
    whole = pixel_layer_moments(
        f,
        layered_cells(
            np.array([[-10.0, -10.0]]), np.array([[10.0, 10.0]]), np.array([-10.0, 10.0])
        ),
        rtol=1e-7,
    )
    assert_allclose(whole.columns_kg_m2 * 400, 7, rtol=1e-7)
    assert_allclose(whole.height_moments_kg_m, 0, atol=1e-12)
    lo, hi, edges = np.array([[-0.4, -0.6]]), np.array([[0.7, 0.8]]), np.array([-1.0, 0.0, 1.0])
    coarse = pixel_layer_moments(f, layered_cells(lo, hi, edges, np.array([-0.5])), rtol=1e-5)
    fine = pixel_layer_moments(f, layered_cells(lo, hi, edges, np.array([-0.5])), rtol=1e-8)
    assert_allclose(coarse.columns_kg_m2, fine.columns_kg_m2, rtol=2e-5)
    assert_allclose(coarse.height_moments_kg_m, fine.height_moments_kg_m, rtol=2e-5)
    f = field(centre=(0.0, 0.0, 0.0))
    z = float(gammainc(1.5, 2.0))
    sigma = np.sqrt(z / gammainc(2.5, 2.0))
    a, b = 2.0 - 1e-3, 2.0
    expected = (
        7
        / z
        * quad(
            lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * (-np.expm1(-(4 - x * x) / 2)),
            a,
            b,
            epsabs=1e-20,
        )[0]
    )
    cap = pixel_layer_moments(
        f,
        layered_cells(
            np.array([[a * sigma, -10.0]]), np.array([[b * sigma, 10.0]]), np.array([-10.0, 10.0])
        ),
        rtol=1e-7,
        atol_kg_m2=1e-16,
    )
    assert_allclose(cap.columns_kg_m2 * (b - a) * sigma * 20, expected, rtol=2e-7)


def test_reflection_overlap_terrain_no_scene_renormalization_and_point_limit():
    f = field(boundary="reflecting", centre=(0.0, 0.0, 0.3))
    full = pixel_layer_moments(
        f,
        layered_cells(np.array([[-10.0, -10.0]]), np.array([[10.0, 10.0]]), np.array([0.0, 10.0])),
        rtol=1e-7,
    )
    assert_allclose(full.columns_kg_m2 * 400, 7, rtol=1e-7)
    clipped = pixel_layer_moments(
        f,
        layered_cells(
            np.array([[-10.0, -10.0]]),
            np.array([[10.0, 10.0]]),
            np.array([0.0, 0.5, 10.0]),
            np.array([0.4]),
        ),
    )
    assert clipped.columns_kg_m2.sum() < full.columns_kg_m2.sum()
    xy = np.array([[0.1, 0.2]])
    tiny = pixel_layer_moments(f, layered_cells(xy - 1e-4, xy + 1e-4, np.array([0.0, 1.0, 4.0])))
    assert_allclose(tiny.columns_kg_m2, layer_columns(f, xy, np.array([0.0, 1.0, 4.0])), rtol=1e-8)
    point = np.array([[0.1, 0.2, 0.3]])
    assert_allclose(
        concentration(f, point),
        layer_columns(f, xy, np.array([0.3 - 1e-6, 0.3 + 1e-6]))[:, 0] / 2e-6,
        rtol=1e-8,
    )


def model_for_field(f):
    def wind(points, time):
        return np.tile([1.0, 0.0, 0.0], (len(points), 1)), np.zeros((len(points), 3, 3))

    def diffusion(points, time, ages):
        return np.zeros((len(points), 3, 3))

    state = f.state
    return GaussianPuffDispersion(
        releases=PuffReleases(
            state.release_times_s, state.masses_kg, state.centres_m, state.covariances_m2
        ),
        wind=wind,
        diffusivity=diffusion,
        boundary=f.boundary,
        rtol=1e-6,
        atol_position_m=1e-8,
        atol_covariance_m2=1e-8,
        min_step_s=1e-6,
        max_step_s=1.0,
        max_steps=100,
        envelope_sigma=3.0,
        wind_tolerance_m_s=0.1,
        diffusivity_tolerance_m2_s=0.1,
        split_fraction=0.5,
        max_puffs=10,
        max_split_rounds=10,
        kernel_cutoff_radius=f.kernel_cutoff_radius,
    )


def test_cutoff_is_carried_by_model_and_large_radius_recovers_gaussian():
    model = model_for_field(field())
    sampled = list(sample_fields(model, np.array([0.0, 1.0])))
    assert all(isinstance(f, GaussianPuffField) and f.kernel_cutoff_radius == 2 for f in sampled)
    assert all(
        isinstance(f, GaussianPuffField)
        and f.diagnostics is not None
        and f.diagnostics.mass_residual_kg == 0
        for f in sampled
    )
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 3.0, 5.0]])
    f = field(12.0)
    assert_allclose(
        concentration(f, points),
        concentration(replace(f, kernel_cutoff_radius=None), points),
        rtol=1e-14,
    )
    for radius in (0.0, -1.0, np.nan, np.inf, 1e-150):
        with pytest.raises(ValueError, match="kernel_cutoff_radius"):
            field(radius)
        with pytest.raises(ValueError, match="kernel_cutoff_radius"):
            puff_concentration(f.state, points, boundary="free", kernel_cutoff_radius=radius)
        with pytest.raises(ValueError):
            replace(model, kernel_cutoff_radius=radius)
    # Large finite cuts retain a valid normalization without squaring overflow.
    assert_allclose(concentration(field(1e200), points), concentration(field(None), points))


@pytest.mark.parametrize("radius", [0.05, 2.0, 8.0])
def test_compact_below_ground_and_forcing_envelope_diagnostics(radius):
    f = field(radius, centre=(0.0, 0.0, 0.3))
    m = replace(model_for_field(f), envelope_sigma=1.0)
    sample = next(sample_fields(m, np.array([0.0])))
    assert isinstance(sample, GaussianPuffField) and sample.diagnostics is not None
    z = gammainc(1.5, radius * radius / 2)
    multiplier = np.sqrt(z / gammainc(2.5, radius * radius / 2))
    upper = -0.3 / multiplier
    expected_below = (
        quad(
            lambda x: (
                np.exp(-x * x / 2)
                / np.sqrt(2 * np.pi)
                * (-np.expm1(-(radius * radius - x * x) / 2))
            ),
            -radius,
            upper,
            epsabs=1e-15 * z,
        )[0]
        / z
    )
    expected_tail = (
        quad(
            lambda r: np.sqrt(2 / np.pi) * r * r * np.exp(-r * r / 2),
            1 / multiplier,
            radius,
            epsabs=1e-15 * z,
        )[0]
        / z
    )
    assert_allclose(sample.diagnostics.below_ground_mass_kg, 7 * expected_below, rtol=2e-12)
    assert_allclose(sample.diagnostics.forcing_tail_probability, expected_tail, rtol=2e-12)
    reflected = next(sample_fields(replace(m, boundary="reflecting"), np.array([0.0])))
    assert isinstance(reflected, GaussianPuffField) and reflected.diagnostics is not None
    assert reflected.diagnostics.below_ground_mass_kg == 0


def test_compact_partial_tilted_rectangle_and_oblique_path_independent_integrals():
    covariance = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 0.8]])
    f = field(covariance=covariance, centre=(0.0, 0.0, 0.0))
    z = gammainc(1.5, 2.0)
    alpha = gammainc(2.5, 2.0) / z
    b = covariance / alpha
    inverse = np.linalg.inv(b)
    amplitude = 7 / (z * np.sqrt((2 * np.pi) ** 3 * np.linalg.det(b)))
    lo, hi = np.array([[-0.4, -0.3]]), np.array([[0.5, 0.2]])
    edges = np.array([-0.2, 0.1, 0.4])
    moments = pixel_layer_moments(f, layered_cells(lo, hi, edges), rtol=1e-9)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    for i in range(2):
        lower, upper = np.r_[lo[0], edges[i]], np.r_[hi[0], edges[i + 1]]
        axes = (lower[:, None] + upper[:, None]) / 2 + (upper - lower)[:, None] / 2 * nodes
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        squared = np.einsum("...i,ij,...j->...", points, inverse, points)
        assert np.max(squared) < 4  # This rectangle is entirely inside the ellipsoid.
        density = amplitude * np.exp(-squared / 2)
        volume = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * np.prod((upper - lower) / 2)
            / np.prod(hi - lo)
        )
        assert_allclose(moments.columns_kg_m2[0, i], np.sum(density * volume), rtol=1e-10)
        assert_allclose(
            moments.height_moments_kg_m[0, i], np.sum(density * volume * points[..., 2]), rtol=1e-10
        )
    start, end = np.array([-10.0, -5.0, -2.0]), np.array([10.0, 5.0, 2.0])
    direction = (end - start) / np.linalg.norm(end - start)
    edge = 2 / np.sqrt(direction @ inverse @ direction)
    reference = quad(
        lambda s: amplitude * np.exp(-0.5 * s * s * (direction @ inverse @ direction)),
        -edge,
        edge,
        epsabs=1e-12,
    )[0]
    path = puff_path_columns(f.state, start, end, boundary="free", kernel_cutoff_radius=2.0)
    assert_allclose(path, reference, rtol=1e-12)
