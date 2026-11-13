import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_dispersion import (
    GaussianPuffs,
    PuffReleases,
    advance_puffs,
    discretize_puff_emissions,
    puff_concentration,
    puff_layer_columns,
    puff_path_columns,
    simulate_puffs,
    split_puffs,
)
from orsia_dispersion.puffs.transport import sample_forcing
from scipy.integrate import quad
from scipy.special import ndtr


def state(centre=(0.0, 0.0, 3.0), covariance=None, mass=2.0, time=0.0):
    return GaussianPuffs(
        time,
        np.array([0.0]),
        np.array([mass]),
        np.array([centre]),
        np.array([np.eye(3) if covariance is None else covariance]),
    )


def wind(points, time):
    return np.tile([1.0, 0.0, 0.0], (len(points), 1)), np.zeros((len(points), 3, 3))


def diffusion(points, time, ages):
    return np.tile(np.diag([0.2, 0.3, 0.1]), (len(points), 1, 1))


def zero_diffusion(points, time, ages):
    return np.zeros((len(points), 3, 3))


def advance(initial, target, velocity=wind, mixing=diffusion, **changes):
    boundary = changes.pop("boundary", "free")
    options = dict(
        rtol=1e-7,
        atol_position_m=1e-9,
        atol_covariance_m2=1e-9,
        min_step_s=1e-7,
        max_step_s=1.0,
        max_steps=10000,
        envelope_sigma=3.0,
    )
    options.update(changes)
    return advance_puffs(initial, target, velocity, mixing, boundary=boundary, **options)


def simulate(releases, times, velocity=wind, mixing=diffusion, **changes):
    boundary = changes.pop("boundary", "free")
    options = dict(
        rtol=1e-6,
        atol_position_m=1e-8,
        atol_covariance_m2=1e-8,
        min_step_s=1e-6,
        max_step_s=0.5,
        max_steps=10000,
        envelope_sigma=3.0,
        wind_tolerance_m_s=0.1,
        diffusivity_tolerance_m2_s=0.1,
        split_fraction=0.5,
        max_puffs=1000,
        max_split_rounds=20,
    )
    options.update(changes)
    return simulate_puffs(
        releases,
        np.asarray(times, dtype=np.float64),
        velocity,
        mixing,
        boundary=boundary,
        **options,
    )


def release(initial=None, times=None):
    s = state() if initial is None else initial
    return PuffReleases(
        s.release_times_s if times is None else np.asarray(times),
        s.masses_kg,
        s.centres_m,
        s.covariances_m2,
    )


def test_emissions_partition_clip_and_zero_rates():
    source_centres = np.zeros((3, 3))
    covariance = np.tile(np.eye(3), (3, 1, 1))
    releases = discretize_puff_emissions(
        np.array([0.0, 2.0, 3.0, 5.0]),
        np.array([2.0, 0.0, 1.0]),
        source_centres,
        covariance,
        partition_times_s=np.array([1.0, 4.0, 9.0]),
        stop_time_s=4.5,
    )
    assert_allclose(releases.release_times_s, [0.5, 1.5, 3.5, 4.25])
    assert_allclose(releases.masses_kg, [2, 2, 1, 0.5])
    assert releases.centres_m.dtype == np.float64
    assert not np.shares_memory(releases.centres_m, source_centres)
    empty = discretize_puff_emissions(
        np.array([0.0, 1.0]), np.zeros(1), source_centres[:1], covariance[:1]
    )
    assert empty.centres_m.shape == (0, 3)
    clipped = discretize_puff_emissions(
        np.array([0.0, 1.0]), np.ones(1), source_centres[:1], covariance[:1], stop_time_s=-1
    )
    assert len(clipped.masses_kg) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"edges_s": np.array([0.0])},
        {"edges_s": np.array([1.0, 0.0])},
        {"rates_kg_s": np.array([-1.0])},
        {"rates_kg_s": np.array([np.nan])},
        {"partition_times_s": np.array([1.0, 1.0])},
        {"stop_time_s": np.inf},
    ],
)
def test_invalid_emissions(change):
    kwargs = dict(
        edges_s=np.array([0.0, 1.0]),
        rates_kg_s=np.ones(1),
        centres_m=np.zeros((1, 3)),
        covariances_m2=np.eye(3)[None],
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        discretize_puff_emissions(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("masses_kg", np.array([0.0])),
        ("masses_kg", np.array([-1.0])),
        ("centres_m", np.ones((1, 2))),
        ("release_times_s", np.array([np.inf])),
        ("covariances_m2", np.zeros((1, 3, 3))),
        ("covariances_m2", np.array([[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])),
        ("covariances_m2", np.eye(2)[None]),
        ("covariances_m2", np.full((1, 3, 3), np.nan)),
        ("time_s", -1.0),
    ],
)
def test_invalid_states(field, value):
    with pytest.raises(ValueError):
        replace(state(), **{field: value})


def test_releases_order_and_snapshot_ownership():
    s = state()
    copy = replace(s)
    copy.centres_m[:] = 99
    assert_allclose(s.centres_m, [[0, 0, 3]])
    with pytest.raises(ValueError, match="nondecreasing"):
        PuffReleases(
            np.array([1.0, 0.0]), np.ones(2), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1))
        )
    s.masses_kg[0] = -1
    with pytest.raises(ValueError):
        puff_concentration(s, np.zeros((1, 3)), boundary="free")


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_constant_transport_diffusion(boundary):
    s = state(covariance=np.diag([1.0, 2.0, 3.0]))
    moved = advance(s, 4.0, boundary=boundary)
    assert_allclose(moved.centres_m, [[4.0, 0.0, 3.0]], atol=1e-14)
    assert_allclose(moved.covariances_m2[0], np.diag([2.6, 4.4, 3.8]), atol=1e-13)
    assert_allclose(s.centres_m, [[0, 0, 3]])
    assert not np.shares_memory(moved.masses_kg, s.masses_kg)
    same = advance(s, 0.0)
    assert same is not s
    assert not np.shares_memory(same.centres_m, s.centres_m)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_diffusion_accepts_rank_one_psd_at_each_tensor_scale(scale):
    from orsia_dispersion.forcing import validate_tensors

    tensor = scale * np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    tensors = np.stack((tensor, np.zeros((3, 3))))
    saved = tensors.copy()
    validate_tensors(tensors, positive=False)
    np.testing.assert_array_equal(tensors, saved)
    invalid = np.stack((tensor, scale * np.diag([1.0, 1.0, -1e-8])))
    with pytest.raises(ValueError, match="semidefinite"):
        validate_tensors(invalid, positive=False)


def test_rank_one_diffusion_propagates_covariance():
    tensor = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    def mixing(points, time, ages):
        return np.broadcast_to(tensor, (len(points), 3, 3)).copy()

    moved = advance(state(), 1.0, mixing=mixing)
    assert_allclose(moved.covariances_m2[0], np.eye(3) + 2 * tensor, rtol=1e-13)


def test_affine_vertical_shear_exact_moments():
    shear = 0.4

    def velocity(points, time):
        u = np.zeros_like(points)
        u[:, 0] = shear * points[:, 2]
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 2] = shear
        return u, a

    initial_cov = np.array([[2.0, 0.2, 0.3], [0.2, 1.0, 0.0], [0.3, 0.0, 3.0]])
    result = advance(state(covariance=initial_cov), 2.0, velocity)
    t, kx, kz = 2.0, 0.2, 0.1
    expected = initial_cov.copy()
    expected[0, 0] += (
        2 * shear * 0.3 * t + shear**2 * 3 * t * t + 2 * kx * t + 2 / 3 * shear**2 * kz * t**3
    )
    expected[0, 2] = expected[2, 0] = 0.3 + shear * 3 * t + shear * kz * t * t
    expected[1, 1] += 2 * 0.3 * t
    expected[2, 2] += 2 * kz * t
    assert_allclose(result.covariances_m2[0], expected, rtol=2e-14, atol=1e-14)
    assert_allclose(result.centres_m, [[2.4, 0, 3]])
    with pytest.raises(ValueError, match="height-independent"):
        advance(state(), 1.0, velocity, boundary="reflecting")


def test_midpoint_time_convergence_and_original_age_diffusion():
    def velocity(points, time):
        return np.tile([np.sin(time), 0, 0], (len(points), 1)), np.zeros((len(points), 3, 3))

    errors = []
    for step in (1.0, 0.5, 0.25):
        result = advance(
            state(), 2.0, velocity, zero_diffusion, max_step_s=step, rtol=1.0, atol_position_m=1.0
        )
        errors.append(abs(result.centres_m[0, 0] - (1 - np.cos(2.0))))
    assert errors[0] / errors[1] > 3.9 and errors[1] / errors[2] > 3.9

    def aging(points, time, ages):
        return ages[:, None, None] * np.eye(3)[None]

    result = advance(state(time=2.0), 3.0, mixing=aging)
    assert_allclose(result.covariances_m2[0], 6 * np.eye(3), atol=1e-13)


def test_integration_refinement_and_failures():
    def velocity(points, time):
        return np.tile([time * time, 0, 0], (len(points), 1)), np.zeros((len(points), 3, 3))

    fine = advance(state(), 1.0, velocity, rtol=1e-5)
    assert_allclose(fine.centres_m[0, 0], 1 / 3, atol=3e-6)
    with pytest.raises(RuntimeError, match="minimum step.*time=0"):
        advance(state(), 1.0, velocity, min_step_s=0.5, rtol=1e-12)
    with pytest.raises(RuntimeError, match="step budget.*time=0"):
        advance(state(), 1.0, velocity, max_steps=1)
    with pytest.raises(RuntimeError, match="time resolution"):
        advance(state(time=1e16), 1e16 + 2, max_step_s=1.0)
    with pytest.raises(RuntimeError, match="step budget"):
        advance(state(), 3.0, max_steps=1)


@pytest.mark.parametrize(
    "options",
    [
        {"rtol": 0.0},
        {"atol_position_m": np.nan},
        {"max_step_s": 1e-9},
        {"max_steps": 0},
        {"envelope_sigma": -1.0},
        {"discontinuities_s": (1.0, 1.0)},
        {"discontinuities_s": (np.nan,)},
        {"boundary": "other"},
    ],
)
def test_invalid_integration(options):
    with pytest.raises(ValueError):
        advance(state(), 1.0, **options)


def test_invalid_target_and_reflection_states():
    with pytest.raises(ValueError):
        advance(state(), -1.0)
    with pytest.raises(ValueError, match="at or above"):
        advance(state(centre=(0, 0, -1)), 1.0, boundary="reflecting")
    tilted = np.array([[2.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]])
    with pytest.raises(ValueError, match="separable"):
        advance(state(covariance=tilted), 1.0, boundary="reflecting")


@pytest.mark.parametrize(
    "kind", ["velocity", "gradient", "compressible", "negative", "shape", "nonfinite", "asymmetric"]
)
def test_invalid_forcing(kind):
    def velocity(points, time):
        u, a = wind(points, time)
        if kind == "velocity":
            u = u[:, :2]
        if kind == "gradient":
            a = np.full_like(a, np.nan)
        if kind == "compressible":
            a[:, 0, 0] = 1
        return u, a

    def mixing(points, time, ages):
        k = diffusion(points, time, ages)
        if kind == "negative":
            k[:, 0, 0] = -1
        if kind == "shape":
            k = k[:, :2]
        if kind == "nonfinite":
            k[:] = np.inf
        if kind == "asymmetric":
            k[:, 0, 1] = 1
        return k

    with pytest.raises(ValueError):
        advance(state(), 1.0, velocity, mixing)


@pytest.mark.parametrize("kind", ["kzz", "kxy_z", "kxz", "u_z", "uz"])
def test_reflecting_forcing_restrictions(kind):
    def velocity(points, time):
        u, a = wind(points, time)
        if kind == "u_z":
            u[:, 0] += points[:, 2] ** 2  # Deliberately inconsistent gradient is still probed.
        if kind == "uz":
            u[:, 2] = 1
        return u, a

    def mixing(points, time, ages):
        k = diffusion(points, time, ages)
        if kind == "kzz":
            k[:, 2, 2] = 1 + points[:, 0] ** 2
        if kind == "kxy_z":
            k[:, 0, 0] = 1 + points[:, 2] ** 2
        if kind == "kxz":
            k[:, 0, 2] = k[:, 2, 0] = 0.01
        return k

    with pytest.raises(ValueError):
        advance(state(), 1.0, velocity, mixing, boundary="reflecting")


def test_forcing_envelope_coverage_and_probe_blind_spot():
    def limited(points, time):
        if np.any(points[:, 0] < -2):
            raise ValueError("outside field coverage")
        return wind(points, time)

    with pytest.raises(ValueError, match="coverage"):
        advance(state(), 1.0, limited)

    def mixed(points, time):
        u = np.zeros_like(points)
        u[:, 0] = points[:, 1] * points[:, 2]
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 1], a[:, 0, 2] = points[:, 2], points[:, 1]
        return u, a

    _, _, _, residual, _ = sample_forcing(
        state(centre=(0, 0, 0)), mixed, zero_diffusion, "free", 3.0
    )
    assert_allclose(residual, 0)
    assert mixed(np.array([[0.0, 1.0, 1.0]]), 0)[0][0, 0] == 1


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_split_preserves_moments_age_and_density_changes(boundary):
    initial = state(covariance=np.diag([2.0, 2.0, 2.0]), time=1.0)
    children = split_puffs(
        initial, np.array([0]), np.array([1]), np.array([0.5]), boundary=boundary
    )
    mean = np.average(children.centres_m, weights=children.masses_kg, axis=0)
    delta = children.centres_m - mean
    moment = np.average(
        children.covariances_m2 + np.einsum("pi,pj->pij", delta, delta),
        weights=children.masses_kg,
        axis=0,
    )
    assert_allclose(children.masses_kg.sum(), initial.masses_kg.sum())
    assert_allclose(mean, initial.centres_m[0])
    assert_allclose(moment, initial.covariances_m2[0])
    assert_allclose(children.release_times_s, 0)
    assert_allclose(children.centres_m[:, 2], 3)
    before = puff_concentration(initial, initial.centres_m, boundary=boundary)
    after = puff_concentration(children, initial.centres_m, boundary=boundary)
    assert abs(float(after[0] - before[0])) > 0.001
    smaller = split_puffs(
        initial, np.array([0]), np.array([1]), np.array([0.05]), boundary=boundary
    )
    delta_small = abs(puff_concentration(smaller, initial.centres_m, boundary=boundary) - before)
    assert np.all(delta_small < abs(after - before))
    assert not np.shares_memory(children.covariances_m2, initial.covariances_m2)


@pytest.mark.parametrize(
    "indices,axes,fractions,boundary",
    [
        ([0, 0], [0, 1], [0.5, 0.5], "free"),
        ([1], [0], [0.5], "free"),
        ([0.5], [0], [0.5], "free"),
        ([0], [0.5], [0.5], "free"),
        ([0], [3], [0.5], "free"),
        ([0], [2], [0.5], "reflecting"),
        ([0], [0], [0.0], "free"),
        ([0], [0], [1.0], "free"),
        ([0], [0], [np.nan], "free"),
        ([0], [0], [0.1, 0.2], "free"),
    ],
)
def test_invalid_splits(indices, axes, fractions, boundary):
    with pytest.raises(ValueError):
        split_puffs(
            state(), np.array(indices), np.array(axes), np.array(fractions), boundary=boundary
        )


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_density_layer_path_oracles_and_batching(boundary):
    covariance = np.array([[2.0, 0.3, 0.2], [0.3, 1.5, -0.1], [0.2, -0.1, 1.0]])
    if boundary == "reflecting":
        covariance[:2, 2] = covariance[2, :2] = 0
    initial = state(covariance=covariance)
    point = np.array([0.4, -0.2, 1.5])

    def reference(z):
        x = np.array([point[0], point[1], z]) - initial.centres_m[0]
        density = np.exp(-0.5 * x @ np.linalg.solve(covariance, x))
        if boundary == "reflecting":
            x[2] = z + initial.centres_m[0, 2]
            density += np.exp(-0.5 * x @ np.linalg.solve(covariance, x))
        return 2 * density / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(covariance))

    concentration = puff_concentration(initial, point, boundary=boundary)
    assert_allclose(concentration, reference(point[2]), rtol=2e-14)
    edges = np.array([0.0, 1.0, 2.0, 5.0, 8.0])
    columns = puff_layer_columns(initial, point[:2], edges, boundary=boundary)
    numerical = np.array(
        [
            quad(reference, lo, hi, epsabs=1e-13)[0]
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    assert_allclose(columns, numerical, rtol=2e-13)
    whole = puff_layer_columns(initial, point[:2], edges[[0, -1]], boundary=boundary)
    assert_allclose(columns.sum(), whole[0], rtol=2e-14)
    start, end = np.array([0.4, -0.2, 0.0]), np.array([0.4, -0.2, 8.0])
    path = puff_path_columns(initial, start, end, boundary=boundary)
    assert_allclose(path, whole[0], rtol=2e-14)
    assert_allclose(puff_path_columns(initial, end, start, boundary=boundary), path, rtol=2e-14)
    middle = (start + end) / 2
    parts = puff_path_columns(
        initial, np.stack([start, middle]), np.stack([middle, end]), boundary=boundary, batch_size=1
    )
    assert_allclose(parts.sum(), path, rtol=2e-14)
    assert puff_path_columns(initial, start, start, boundary=boundary) == 0
    points = np.tile(point, (5, 1))
    assert_allclose(
        puff_concentration(initial, points, boundary=boundary, batch_size=2), concentration
    )
    assert_allclose(
        puff_layer_columns(initial, points[:, :2], edges, boundary=boundary, batch_size=2),
        np.tile(columns, (5, 1)),
    )


def test_oblique_path_independent_quadrature_and_geometry_factor():
    s = state(covariance=np.array([[2.0, 0.2, 0.6], [0.2, 1.0, 0.1], [0.6, 0.1, 1.0]]))
    a, b = np.array([-2.0, -1.0, 0.0]), np.array([3.0, 2.0, 6.0])
    length = np.linalg.norm(b - a)
    reference = quad(
        lambda t: float(puff_concentration(s, a + t * (b - a), boundary="free")) * length,
        0,
        1,
        epsabs=1e-13,
    )[0]
    value = puff_path_columns(s, a, b, boundary="free")
    assert_allclose(value, reference, rtol=2e-13)
    vertical = puff_layer_columns(s, np.array([0.0, 0.0]), np.array([0.0, 6.0]), boundary="free")[0]
    # A slant column differs from a single vertical column multiplied by path/height.
    assert abs(value - vertical * length / 6) > 0.01


@pytest.mark.parametrize("lo,width", [(10.0, 0.01), (-10.01, 0.01), (8.0, 1e-9), (-8.0, 1e-9)])
def test_remote_narrow_tails(lo, width):
    s = state(centre=(0, 0, 0), mass=1.0)
    hi = lo + width
    reference = quad(
        lambda z: np.exp(-0.5 * z * z) / (2 * np.pi) ** 1.5, lo, hi, epsabs=1e-100, epsrel=1e-10
    )[0]
    column = puff_layer_columns(s, np.zeros(2), np.array([lo, hi]), boundary="free")[0]
    path = puff_path_columns(s, np.array([0.0, 0.0, lo]), np.array([0.0, 0.0, hi]), boundary="free")
    assert column > 0
    assert_allclose(column, reference, rtol=5e-12)
    assert_allclose(path, reference, rtol=5e-12)


def test_reflection_mass_flux_and_finite_top():
    s = state(centre=(0, 0, 0.3), mass=1.0)

    def vertical_density(z):
        return (
            float(puff_concentration(s, np.array([0.0, 0.0, z]), boundary="reflecting")) * 2 * np.pi
        )

    assert_allclose(quad(vertical_density, 0, 12, epsabs=1e-12)[0], 1, atol=1e-12)
    epsilon = 1e-5
    # Right derivative converges to zero at the reflecting boundary.
    d1 = (vertical_density(epsilon) - vertical_density(0)) / epsilon
    d2 = (vertical_density(epsilon / 2) - vertical_density(0)) / (epsilon / 2)
    assert abs(d2) < 0.6 * abs(d1) and abs(d1) < 1e-5
    finite = puff_layer_columns(s, np.zeros(2), np.array([0.0, 0.5]), boundary="reflecting")[0]
    assert_allclose(finite * 2 * np.pi, ndtr(0.2) - ndtr(-0.3) + ndtr(0.8) - ndtr(0.3))


@pytest.mark.parametrize(
    "kind",
    [
        "point_shape",
        "point_nan",
        "below",
        "batch",
        "edges_shape",
        "edges_nan",
        "edges_order",
        "edges_below",
        "pairs",
    ],
)
def test_invalid_evaluation(kind):
    s = state()
    with pytest.raises(ValueError):
        if kind == "point_shape":
            puff_concentration(s, np.zeros(2), boundary="free")
        elif kind == "point_nan":
            puff_concentration(s, np.full(3, np.nan), boundary="free")
        elif kind == "below":
            puff_concentration(s, np.array([0.0, 0.0, -1.0]), boundary="reflecting")
        elif kind == "batch":
            puff_concentration(s, np.zeros(3), boundary="free", batch_size=0)
        elif kind == "pairs":
            puff_path_columns(s, np.zeros(3), np.zeros((2, 3)), boundary="free")
        else:
            edges = {
                "edges_shape": np.zeros((2, 2)),
                "edges_nan": np.array([0.0, np.nan]),
                "edges_order": np.array([1.0, 0.0]),
                "edges_below": np.array([-1.0, 1.0]),
            }[kind]
            puff_layer_columns(s, np.zeros(2), edges, boundary="reflecting")


def test_empty_states_shapes_and_no_aliases():
    empty = GaussianPuffs(0.0, np.empty(0), np.empty(0), np.empty((0, 3)), np.empty((0, 3, 3)))
    assert puff_concentration(empty, np.empty((0, 3)), boundary="free").shape == (0,)
    assert_allclose(puff_concentration(empty, np.zeros((2, 3)), boundary="free"), 0)
    assert puff_layer_columns(
        empty, np.empty((0, 2)), np.array([0.0, 1.0]), boundary="free"
    ).shape == (0, 1)
    assert puff_path_columns(empty, np.empty((0, 3)), np.empty((0, 3)), boundary="free").shape == (
        0,
    )
    assert advance(empty, 1.0).time_s == 1.0
    split = split_puffs(
        empty,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.empty(0),
        boundary="free",
    )
    assert split.centres_m.shape == (0, 3)


def test_event_order_budget_diagnostics_and_snapshot_mutation():
    releases = PuffReleases(
        np.array([1.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.tile([0.0, 0.0, 0.0], (3, 1)),
        np.tile(np.eye(3), (3, 1, 1)),
    )
    iterator = simulate(releases, [0, 1, 1.5, 2])
    initial, diagnostics = next(iterator)
    assert len(initial.masses_kg) == 0 and diagnostics.released_mass_kg == 0
    first, diagnostics = next(iterator)
    assert len(first.masses_kg) == 2 and diagnostics.released_mass_kg == 3
    assert diagnostics.below_ground_mass_kg == 1.5
    assert 0 < diagnostics.forcing_tail_probability < 0.04
    first.centres_m[:] = 999
    first.masses_kg[:] = 999
    releases.masses_kg[:] = 888
    later, _ = next(iterator)
    assert_allclose(later.centres_m[:, 0], 0.5)
    final, diagnostics = next(iterator)
    assert diagnostics.active_mass_kg == 6 and diagnostics.mass_residual_kg == 0
    assert final.time_s == 2
    with pytest.raises(StopIteration):
        next(iterator)
    assert first.masses_kg[0] == 999


def test_discontinuity_and_same_time_release():
    def velocity(points, time):
        return np.tile([1.0 if time < 0.5 else 3.0, 0.0, 0.0], (len(points), 1)), np.zeros(
            (len(points), 3, 3)
        )

    result = advance(state(), 1.0, velocity, zero_diffusion, discontinuities_s=(0.5,))
    assert_allclose(result.centres_m[0, 0], 2)
    results = list(simulate(release(), [0, 1], velocity, zero_diffusion, discontinuities_s=(0.5,)))
    assert results[0][1].released_mass_kg == 2
    assert_allclose(results[-1][0].centres_m, result.centres_m)
    assert results[-1][1].max_integration_error_ratio < 1


def test_adaptive_nonlinear_splitting_and_midpoint_check():
    def velocity(points, time):
        u = np.zeros_like(points)
        # Vanishes at t=0 and t=1, but violates the affine model at midpoint.
        u[:, 0] = 4 * time * (1 - time) * points[:, 1] ** 2
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 1] = 8 * time * (1 - time) * points[:, 1]
        return u, a

    results = list(
        simulate(
            release(),
            [0, 1],
            velocity,
            zero_diffusion,
            max_step_s=1.0,
            wind_tolerance_m_s=2.0,
            max_split_rounds=10,
        )
    )
    assert len(results[0][0].masses_kg) == 1
    assert results[-1][1].split_count > 0
    assert results[-1][1].max_sampled_variation_ratio > 1
    assert results[-1][1].sampled_variation_ratio <= 1
    assert_allclose(results[-1][1].active_mass_kg, 2)
    assert_allclose(results[-1][0].release_times_s, 0)


def test_simulation_resource_failures_preserve_previous_outputs():
    iterator = simulate(release(), [0, 2], max_steps=1)
    first, _ = next(iterator)
    with pytest.raises(RuntimeError, match="time="):
        next(iterator)
    assert_allclose(first.centres_m, [[0, 0, 3]])
    many = PuffReleases(
        np.array([0.0, 1.0]), np.ones(2), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1))
    )
    iterator = simulate(many, [0, 1], max_puffs=1)
    next(iterator)
    with pytest.raises(RuntimeError, match="release.*event time=1"):
        next(iterator)

    def nonlinear(points, time):
        u = np.zeros_like(points)
        u[:, 0] = points[:, 1] ** 2
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 1] = 2 * points[:, 1]
        return u, a

    for limits in ({"max_puffs": 1}, {"max_split_rounds": 1}):
        with pytest.raises(RuntimeError, match="split budget.*unresolved"):
            list(simulate(release(), [0], nonlinear, **limits))


@pytest.mark.parametrize(
    "options",
    [
        {"max_puffs": 0},
        {"max_split_rounds": 0},
        {"wind_tolerance_m_s": 0},
        {"diffusivity_tolerance_m2_s": np.nan},
        {"split_fraction": 1},
        {"discontinuities_s": (2.0, 1.0)},
    ],
)
def test_invalid_simulation_controls(options):
    with pytest.raises(ValueError):
        list(simulate(release(), [0, 1], **options))


def test_output_times_empty_invalid_and_reflected_diagnostics():
    assert list(simulate(release(), [])) == []
    for times in ([0, 0], [np.nan], [[0, 1]]):
        with pytest.raises(ValueError):
            list(simulate(release(), times))
    _, diagnostic = next(simulate(release(), [0], boundary="reflecting"))
    assert diagnostic.below_ground_mass_kg == 0


def test_concurrent_reordered_queries_linearity_and_native_gil():
    assert not sys._is_gil_enabled()
    s = state()
    points = np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
    expected = puff_concentration(s, points, boundary="free")

    def evaluate(index):
        return puff_concentration(s, points[::-1], boundary="free", batch_size=index + 1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for actual in executor.map(evaluate, [0, 1, 0, 1]):
            assert_allclose(actual[::-1], expected)
    doubled = replace(s, masses_kg=2 * s.masses_kg)
    assert_allclose(puff_concentration(doubled, points, boundary="free"), 2 * expected)
    assert not sys._is_gil_enabled()


def test_emission_time_convergence_and_zero_background_recovery():
    columns = []
    for n in (4, 8, 16, 128):
        edges = np.linspace(0.0, 2.0, n + 1)
        releases = discretize_puff_emissions(
            edges, np.ones(n), np.tile([0.0, 0.0, 3.0], (n, 1)), np.tile(np.eye(3), (n, 1, 1))
        )
        final, diagnostic = list(simulate(releases, [2.0], mixing=zero_diffusion))[-1]
        assert diagnostic.released_mass_kg == 2
        columns.append(
            puff_layer_columns(final, np.zeros(2), np.array([0.0, 6.0]), boundary="free")[0]
        )
    errors = abs(np.array(columns[:3]) - columns[-1])
    assert errors[0] / errors[1] > 3.9 and errors[1] / errors[2] > 3.9
    empty = discretize_puff_emissions(
        np.array([0.0, 1.0]), np.zeros(1), np.zeros((1, 3)), np.eye(3)[None]
    )
    s, _ = next(simulate(empty, [1.0]))
    mass = puff_layer_columns(s, np.zeros(2), np.array([0.0, 1.0, 2.0]), boundary="free")
    background = np.array([1e20, 2e20])
    assert_allclose(background + mass * 6.02214076e23 / 0.044, background, rtol=0, atol=0)


def test_diffusion_floor_coordinates_spatial_and_temporal_refinement():
    def velocity(points, time):
        u = np.zeros_like(points)
        u[:, 0] = points[:, 1] ** 2
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 1] = 2 * points[:, 1]
        return u, a

    def mixing(points, time, ages):
        return np.tile(np.diag([0.0, 0.1, 0.0]), (len(points), 1, 1))

    initial = state(centre=(0.0, 0.0, 0.0), covariance=np.diag([1.0, 0.025, 1.0]))
    outputs = list(
        simulate(
            release(initial),
            [0.0, 1.0],
            velocity,
            mixing,
            boundary="reflecting",
            max_step_s=1.0,
            envelope_sigma=1.0,
            wind_tolerance_m_s=0.1,
            split_fraction=0.5,
            max_split_rounds=3,
            max_puffs=100,
        )
    )
    final, diagnostic = outputs[-1]
    assert final.time_s == 1.0
    assert diagnostic.split_count > 0
    assert diagnostic.sampled_variation_ratio <= 1
    assert_allclose(final.masses_kg.sum(), initial.masses_kg.sum())
    assert len(final.masses_kg) <= 100


def test_discontinuous_event_splits_accepted_physical_axis_after_eigenvalue_crossing():
    rate = np.log(4.0)

    def velocity(points, time):
        nonlinear = float(time >= 1.0)
        u = np.zeros_like(points)
        u[:, 0] = rate * points[:, 0] + nonlinear * points[:, 1] ** 2
        u[:, 1] = -rate * points[:, 1]
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 0], a[:, 1, 1] = rate, -rate
        a[:, 0, 1] = 2 * nonlinear * points[:, 1]
        return u, a

    initial = state(centre=(0.0, 0.0, 0.0), covariance=np.diag([1.0, 4.0, 1.0]))
    final, diagnostic = list(
        simulate(
            release(initial),
            [0.0, 1.0],
            velocity,
            zero_diffusion,
            boundary="reflecting",
            discontinuities_s=(1.0,),
            max_step_s=1.0,
            envelope_sigma=1.0,
            wind_tolerance_m_s=0.1,
            split_fraction=0.5,
            max_split_rounds=4,
            max_puffs=100,
            rtol=1.0,
            atol_position_m=1.0,
            atol_covariance_m2=1.0,
        )
    )[-1]
    assert len(final.masses_kg) == 4
    assert diagnostic.sampled_variation_ratio <= 1
    assert_allclose(final.centres_m[:, 0], 0, atol=1e-14)
    assert np.any(final.centres_m[:, 1] != 0)
    assert_allclose(final.covariances_m2[:, 0, 0], 16, rtol=1e-13)
    assert_allclose(final.covariances_m2[:, 1, 1], 0.0625, rtol=1e-13)


def test_empty_population_fast_forwards_without_step_budget_or_forcing_calls():
    def forbidden(*args):
        raise AssertionError("empty populations must not sample forcing")

    empty = GaussianPuffs(0.0, np.empty(0), np.empty(0), np.empty((0, 3)), np.empty((0, 3, 3)))
    advanced = advance(empty, 86400.0, forbidden, forbidden, max_steps=1)
    assert advanced.time_s == 86400.0
    releases = PuffReleases(
        empty.release_times_s, empty.masses_kg, empty.centres_m, empty.covariances_m2
    )
    final, diagnostic = list(simulate(releases, [86400.0], forbidden, forbidden, max_steps=1))[-1]
    assert final.time_s == 86400.0
    assert diagnostic.attempted_steps == 0


def test_continuous_trial_eigenvalue_crossing_keeps_the_physical_split_direction():
    rate = np.log(4.0)

    def velocity(points, time):
        nonlinear = min(time, 1.0) ** 6
        u = np.zeros_like(points)
        u[:, 0] = rate * points[:, 0] + nonlinear * points[:, 1] ** 2
        u[:, 1] = -rate * points[:, 1]
        a = np.zeros((len(points), 3, 3))
        a[:, 0, 0], a[:, 1, 1] = rate, -rate
        a[:, 0, 1] = 2 * nonlinear * points[:, 1]
        return u, a

    initial = state(centre=(0.0, 0.0, 0.0), covariance=np.diag([1.0, 4.0, 1.0]))
    final, diagnostic = list(
        simulate(
            release(initial),
            [0.0, 1.1],
            velocity,
            zero_diffusion,
            boundary="reflecting",
            max_step_s=1.0,
            envelope_sigma=1.0,
            wind_tolerance_m_s=0.1,
            split_fraction=0.5,
            max_split_rounds=4,
            max_puffs=100,
            rtol=1.0,
            atol_position_m=1.0,
            atol_covariance_m2=1.0,
        )
    )[-1]
    assert diagnostic.split_count > 0
    assert diagnostic.sampled_variation_ratio <= 1
    # At the first trial endpoint the y eigenvector changes from ordinal 1 to 0.
    # Splitting upstream x instead would create centres several metres from x=0.
    assert np.max(np.abs(final.centres_m[:, 0])) < 0.5
