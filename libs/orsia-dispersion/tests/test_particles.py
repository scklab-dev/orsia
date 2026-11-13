"""Bounded flights, independent output clocks and conservative point deposition."""

from dataclasses import replace
from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import (
    BoundedParticleDispersion,
    MassParticles,
    ParticleMassField,
    ParticleReleases,
    concentration,
    discretize_particle_emissions,
    discretize_puff_emissions,
    layer_columns,
    pixel_layer_moments,
    sample_fields,
    simulate_particles,
)
from orsia_scene import gas_layer_columns
from test_fields import layered_cells


def wind(points, time):
    return np.broadcast_to([2.0, 0.0, 0.0], points.shape), np.zeros((len(points), 3, 3))


def model(**changes):
    parameters: dict[str, Any] = dict(
        releases=ParticleReleases(
            np.array([0.0, 0.0, 0.7, 2.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([[0.0, 0.0, 1.0]] * 4),
        ),
        wind=wind,
        wind_speed_bound_m_s=2.0,
        turbulent_covariance_m2_s2=np.diag([1.0, 2.0, 3.0]),
        correlation_time_s=0.5,
        boundary="free",
        max_step_s=0.2,
        max_steps=10000,
        max_particles=100,
    )
    return BoundedParticleDispersion(**(parameters | changes))


def test_reproducibility_output_insertion_does_not_change_paths_or_rng():
    assert find_spec("orsia_dispersion.particle_transport") is None
    m = model(discontinuities_s=(0.3, 1.2))
    a, b = np.random.default_rng(173), np.random.default_rng(173)
    sparse = list(sample_fields(m, np.array([-1.0, 0.0, 1.0, 2.0, 4.0]), rng=a))
    dense = list(sample_fields(m, np.array([-1.0, 0.0, 0.1, 0.5, 1.0, 1.1, 2.0, 3.0, 4.0]), rng=b))
    for i, j in ((0, 0), (1, 1), (2, 4), (3, 6), (4, 8)):
        f, g = sparse[i], dense[j]
        assert isinstance(f, ParticleMassField) and isinstance(g, ParticleMassField)
        assert_array_equal(f.state.positions_m, g.state.positions_m)
        assert_array_equal(f.state.turbulent_velocities_m_s, g.state.turbulent_velocities_m_s)
        assert f.diagnostics == g.diagnostics
        assert f.diagnostics is not None and f.diagnostics.mass_residual_kg == 0
        assert_allclose(
            f.state.masses_kg.sum(),
            m.releases.masses_kg[m.releases.release_times_s <= f.state.time_s].sum(),
        )
        assert not f.state.positions_m.flags.writeable
    assert a.bit_generator.state == b.bit_generator.state
    assert isinstance(sparse[0], ParticleMassField) and len(sparse[0].state.masses_kg) == 0
    # Writable mutations of caller-owned inputs never reach the model or snapshots.
    supplied = np.eye(3)
    independent = model(turbulent_covariance_m2_s2=supplied)
    supplied[:] = 100
    assert_array_equal(independent.turbulent_covariance_m2_s2, np.eye(3))
    with pytest.raises(ValueError, match="explicit rng"):
        list(sample_fields(m, np.array([0.0])))
    assert list(sample_fields(m, np.empty(0), rng=np.random.default_rng(1))) == []
    for times in ([1.0, 0.0], [0.0, 0.0], [np.nan], [[0.0]]):
        with pytest.raises(ValueError):
            list(simulate_particles(m, np.array(times), rng=np.random.default_rng(1)))


@pytest.mark.parametrize("boundary", ["free", "reflecting"])
def test_mass_and_per_release_propagation_bound_even_outside_scene(boundary):
    m = model(boundary=boundary)
    fields = list(simulate_particles(m, np.linspace(0, 5, 31), rng=np.random.default_rng(92)))
    vmax = 2 + np.sqrt(9)
    for f in fields:
        n = len(f.state.masses_kg)
        age = f.state.time_s - f.state.release_times_s
        distance = np.linalg.norm(f.state.positions_m - m.releases.positions_m[:n], axis=1)
        assert np.all(distance <= vmax * age + 1e-12)
        assert_allclose(f.state.masses_kg, m.releases.masses_kg[:n])
        assert f.diagnostics is not None
        assert f.diagnostics.mass_residual_kg == 0
        assert np.all(np.linalg.norm(f.state.turbulent_velocities_m_s, axis=1) <= 3 + 1e-14)
        if boundary == "reflecting":
            assert np.all(f.state.positions_m[:, 2] >= 0)
    outside = pixel_layer_moments(
        fields[-1],
        layered_cells(
            np.array([[100.0, 100.0]]), np.array([[101.0, 101.0]]), np.array([0.0, 20.0])
        ),
    )
    assert_array_equal(outside.columns_kg_m2, 0)
    assert fields[-1].state.masses_kg.sum() == 10


def test_uniform_advection_reflection_and_wind_step_convergence():
    m = model(turbulent_covariance_m2_s2=np.zeros((3, 3)))
    f = list(simulate_particles(m, np.array([3.0]), rng=np.random.default_rng(5)))[0]
    assert_allclose(
        f.state.positions_m,
        m.releases.positions_m + (3 - m.releases.release_times_s[:, None]) * [2, 0, 0],
        atol=1e-14,
    )
    # Compare folded free paths with specular reflection using identical clocks.
    free = model(correlation_time_s=1000.0, max_step_s=10.0)
    reflected = replace(free, boundary="reflecting")
    a = list(simulate_particles(free, np.array([0.0, 2.0, 10.0]), rng=np.random.default_rng(7)))
    b = list(
        simulate_particles(reflected, np.array([0.0, 2.0, 10.0]), rng=np.random.default_rng(7))
    )
    for x, y in zip(a, b, strict=True):
        assert_allclose(y.state.positions_m[:, :2], x.state.positions_m[:, :2])
        assert_allclose(y.state.positions_m[:, 2], np.abs(x.state.positions_m[:, 2]))
        sign = np.where(x.state.positions_m[:, 2] < 0, -1, 1)
        assert_allclose(
            y.state.turbulent_velocities_m_s[:, 2], sign * x.state.turbulent_velocities_m_s[:, 2]
        )

    def varying(points, time):
        return np.broadcast_to([1 + time, 0.0, 0.0], points.shape), np.zeros((len(points), 3, 3))

    # Euler integration of u=1+t has first-order step convergence, without moving
    # turbulent renewal events when the transport step changes.
    single = ParticleReleases(np.array([0.0]), np.array([1.0]), np.zeros((1, 3)))
    errors = []
    for step in (0.2, 0.1, 0.05):
        varied = replace(
            m, releases=single, wind=varying, max_step_s=step, wind_speed_bound_m_s=3.0
        )
        result = list(simulate_particles(varied, np.array([1.0]), rng=np.random.default_rng(1)))[0]
        errors.append(abs(result.state.positions_m[0, 0] - 1.5))
    assert errors[1] < errors[0] * 0.65 and errors[2] < errors[1] * 0.65


def test_homogeneous_velocity_and_displacement_statistics_against_exact_closure():
    # Fixed ensemble/seed; tolerances cover Monte Carlo error, not integration error.
    n, tau, time = 4000, 0.7, 3.0
    releases = ParticleReleases(np.zeros(n), np.ones(n) / n, np.zeros((n, 3)))
    covariance = np.diag([1.0, 2.0, 0.5])
    m = model(
        releases=releases,
        turbulent_covariance_m2_s2=covariance,
        correlation_time_s=tau,
        max_step_s=10.0,
        max_steps=100000,
        max_particles=n,
    )
    initial, final = list(
        simulate_particles(m, np.array([0.0, time]), rng=np.random.default_rng(1234))
    )
    v = initial.state.turbulent_velocities_m_s
    assert_allclose(v.mean(axis=0), 0.0, atol=0.05)
    assert_allclose(v.T @ v / n, covariance, atol=0.055)
    displacement = final.state.positions_m - np.array([2 * time, 0.0, 0.0])
    variance = 2 * tau * (time - tau * (-np.expm1(-time / tau))) * np.diag(covariance)
    assert_allclose(displacement.mean(axis=0), 0.0, atol=0.1)
    assert_allclose(np.mean(displacement**2, axis=0), variance, rtol=0.065)
    assert final.diagnostics is not None and final.diagnostics.renewals > n


def test_particle_ensemble_refinement_across_independent_realizations():
    errors = []
    for n in (100, 1600):
        moments = []
        m = model(
            releases=ParticleReleases(np.zeros(n), np.ones(n) / n, np.zeros((n, 3))),
            turbulent_covariance_m2_s2=np.eye(3),
            max_particles=n,
            max_steps=50000,
            max_step_s=10.0,
            correlation_time_s=1.0,
        )
        for seed in (11, 27, 49, 83):
            f = next(simulate_particles(m, np.array([2.0]), rng=np.random.default_rng(seed)))
            displacement = f.state.positions_m - [4.0, 0.0, 0.0]
            expected = 2 * (2 - (1 - np.exp(-2)))
            moments.append(np.mean(displacement**2, axis=0) - expected)
        errors.append(np.sqrt(np.mean(np.square(moments))))
    assert errors[1] < errors[0] * 0.6


def test_discrete_projection_faces_partition_moments_terrain_and_scene_conversion():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 0.5],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [-1.0, 0.0, 1.0],
            [0.5, 0.5, 0.25],
        ]
    )
    masses = np.arange(1.0, len(points) + 1)
    field = ParticleMassField(
        MassParticles(0.0, np.zeros(len(points)), masses, points, np.zeros_like(points)),
        "reflecting",
    )
    lo = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
    hi, edges = lo + 1, np.array([0.0, 1.0, 2.0, 3.0])
    result = pixel_layer_moments(field, layered_cells(lo, hi, edges))
    expected = np.array([[[9.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]])
    assert_array_equal(result.columns_kg_m2, expected)
    assert_array_equal(
        result.height_moments_kg_m,
        np.array([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [[0.0, 0.0, 6.0], [2.0, 0.0, 0.0]]]),
    )
    combined = pixel_layer_moments(field, layered_cells(lo, hi, edges[[0, -1]]))
    assert_array_equal(result.columns_kg_m2.sum(-1), combined.columns_kg_m2[..., 0])
    assert_array_equal(result.height_moments_kg_m.sum(-1), combined.height_moments_kg_m[..., 0])
    clipped = pixel_layer_moments(field, layered_cells(lo, hi, edges, np.full((2, 2), 0.5)))
    assert clipped.columns_kg_m2.sum() == 9
    assert clipped.columns_kg_m2.sum() < masses.sum()
    gas = gas_layer_columns(result, species="CO2", molar_mass_kg_mol=0.044)
    assert_allclose(gas.columns_molecules_m2 * 0.044 / 6.02214076e23, expected)
    assert not result.columns_kg_m2.flags.writeable
    for evaluator, coordinates in ((concentration, points), (layer_columns, points[:, :2])):
        with pytest.raises(TypeError, match="pixel_layer_moments"):
            if evaluator is concentration:
                evaluator(field, coordinates)
            else:
                layer_columns(field, coordinates, edges)
    with pytest.raises(ValueError, match="overlap"):
        pixel_layer_moments(field, layered_cells(np.zeros((2, 2)), np.ones((2, 2)), edges))
    empty = ParticleMassField(
        MassParticles(0.0, np.empty(0), np.empty(0), np.empty((0, 3)), np.empty((0, 3))), "free"
    )
    assert_array_equal(pixel_layer_moments(empty, layered_cells(lo, hi, edges)).columns_kg_m2, 0)


def test_particle_emissions_reuse_interval_mass_rule_and_validate_sources():
    edges, rates = np.array([0.0, 1.0, 3.0]), np.array([2.0, 0.0])
    positions = np.zeros((2, 3))
    r = discretize_particle_emissions(
        edges,
        rates,
        positions,
        particles_per_release=4,
        partition_times_s=np.array([0.5]),
        stop_time_s=2.0,
    )
    p = discretize_puff_emissions(
        edges,
        rates,
        positions,
        np.tile(np.eye(3), (2, 1, 1)),
        partition_times_s=np.array([0.5]),
        stop_time_s=2.0,
    )
    assert_array_equal(r.release_times_s[::4], p.release_times_s)
    assert_array_equal(r.masses_kg.reshape(-1, 4).sum(-1), p.masses_kg)
    assert r.masses_kg.sum() == 2
    assert not r.positions_m.flags.writeable
    before = discretize_particle_emissions(
        edges, rates, positions, particles_per_release=1, stop_time_s=-1.0
    )
    assert len(before.masses_kg) == 0
    for kwargs in (
        dict(particles_per_release=0),
        dict(stop_time_s=np.nan),
        dict(partition_times_s=np.array([1.0, 1.0])),
    ):
        arguments: dict[str, Any] = dict(particles_per_release=2) | kwargs
        with pytest.raises(ValueError):
            discretize_particle_emissions(edges, rates, positions, **arguments)
    with pytest.raises(ValueError):
        discretize_particle_emissions(
            edges, rates, np.full((2, 3), np.nan), particles_per_release=2
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(correlation_time_s=0),
        dict(max_step_s=0),
        dict(wind_speed_bound_m_s=-1),
        dict(max_steps=0),
        dict(max_particles=0),
        dict(discontinuities_s=(1.0, 0.0)),
        dict(turbulent_covariance_m2_s2=-np.eye(3)),
        dict(turbulent_covariance_m2_s2=np.ones((2, 2))),
        dict(boundary="unknown"),
        dict(
            boundary="reflecting",
            turbulent_covariance_m2_s2=np.array(
                [[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]]
            ),
        ),
    ],
)
def test_invalid_particle_models(changes):
    with pytest.raises(ValueError):
        model(**changes)


def test_explicit_budget_and_wind_bound_failures():
    for m, match in ((model(max_particles=1), "population"), (model(max_steps=1), "segment")):
        with pytest.raises(RuntimeError, match=match):
            list(simulate_particles(m, np.array([2.0]), rng=np.random.default_rng(1)))
    with pytest.raises(ValueError, match="exceeds"):
        list(
            simulate_particles(
                model(wind_speed_bound_m_s=1), np.array([0.0]), rng=np.random.default_rng(1)
            )
        )

    def vertical(points, time):
        return np.tile([0.0, 0.0, 1.0], (len(points), 1)), np.zeros((len(points), 3, 3))

    def invalid(points, time):
        return np.full((len(points), 3), np.nan), np.zeros((len(points), 3, 3))

    for w, match in ((vertical, "zero vertical"), (invalid, "finite")):
        with pytest.raises(ValueError, match=match):
            list(
                simulate_particles(
                    model(wind=w, boundary="reflecting"),
                    np.array([0.0]),
                    rng=np.random.default_rng(1),
                )
            )
    huge = ParticleReleases(np.array([1e100]), np.ones(1), np.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="clock"):
        list(
            simulate_particles(
                model(releases=huge), np.array([1e100]), rng=np.random.default_rng(1)
            )
        )


def test_invalid_particle_snapshots():
    for times, mass, pos in (
        (np.array([1.0, 0.0]), np.ones(2), np.zeros((2, 3))),
        (np.zeros(1), np.array([-1.0]), np.zeros((1, 3))),
        (np.zeros(1), np.ones(1), np.zeros((2, 3))),
    ):
        with pytest.raises(ValueError):
            ParticleReleases(times, mass, pos)
    with pytest.raises(ValueError):
        MassParticles(0.0, np.array([1.0]), np.ones(1), np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        MassParticles(0.0, np.zeros(1), np.ones(1), np.zeros((1, 3)), np.zeros((1, 2)))
    below = MassParticles(
        0.0, np.zeros(1), np.ones(1), np.array([[0.0, 0.0, -1.0]]), np.zeros((1, 3))
    )
    with pytest.raises(ValueError):
        ParticleMassField(below, "reflecting")
    with pytest.raises(ValueError):
        bad_boundary: Any = "bad"
        ParticleMassField(below, bad_boundary)
