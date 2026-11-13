"""Fixed-support SOS derivatives against independent identities and differences."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import cos, pi, radians
from typing import Any

import numpy as np
import pytest
from orsia_numerics.tangents import jacobian
from orsia_radiative import AerosolComponent, RayleighComponent, SurfaceState
from orsia_radiative.transfer.orders import (
    angular_grid,
    scattering_orders,
    scattering_orders_tangent,
)
from orsia_radiative.transfer.solar import reflected_radiance_tangent
from test_absorption_solar_tangents import inputs, layout_copy

SEED_NAMES = (
    "optical_depth_tangent",
    "rayleigh_optical_depth_tangent",
    "aerosol_optical_depth_tangent",
    "aerosol_scattering_depth_tangent",
    "aerosol_asymmetry_tangent",
    "reflectance_tangent",
)
CONTROLS: dict[str, Any] = dict(max_order=3, angular_order=3, layer_subdivisions=3)


def state(arrays=None, *, g=0.35, rho=0.3, layers=2, points=2):
    if arrays is None:
        arrays = [np.full((layers, points), a) for a in (0.04, 0.02, 0.07, 0.045, g)]
        arrays.append(np.full(points, rho))
    scene, source, optics = inputs(arrays[0])
    scene = replace(scene, surface=replace(scene.surface, reflectance=arrays[5]))
    optics = replace(
        optics,
        rayleigh=RayleighComponent(optics.grid, arrays[1]),
        aerosol=AerosolComponent(optics.grid, *arrays[2:5]),
    )
    return scene, source, optics, arrays


def seed_arguments(seeds) -> dict[str, Any]:
    return dict(zip(SEED_NAMES, seeds, strict=True))


def call(arrays, seeds, controls: dict[str, Any] | None = None, total=False):
    scene, source, optics, _ = state(arrays)
    arguments = seed_arguments(seeds) | (CONTROLS if controls is None else controls)
    if total:
        value, tangent = reflected_radiance_tangent(
            scene, source=source, optics=optics, **arguments
        )
        return value.values, tangent
    return scattering_orders_tangent(scene, source=source, optics=optics, **arguments)


def direct(arrays, controls):
    scene, source, optics, _ = state(arrays)
    return scattering_orders(scene, source=source, optics=optics, **controls)


@pytest.mark.parametrize(
    "g,moments", [(-0.3, None), (-0.3, 4), (0.0, None), (0.0, 2), (0.35, 2), (0.55, 4), (0.55, 6)]
)
@pytest.mark.parametrize("layers", [1, 3])
def test_multistep_constituent_and_mixed_directions(g, moments, layers):
    _, _, _, arrays = state(g=g, layers=layers)
    # Each constituent has its own direction; the final direction mixes them.
    rng = np.random.default_rng(2301)
    seeds = [rng.uniform(-0.06, 0.06, (*a.shape, 7)) for a in arrays]
    for i, seed in enumerate(seeds):
        for j in range(6):
            if i != j:
                seed[..., j] = 0
    controls = dict(CONTROLS, delta_m_moments=moments)
    value, tangent = call(arrays, seeds, controls)
    np.testing.assert_allclose(value, direct(arrays, controls), rtol=8e-14, atol=1e-16)
    errors = []
    for h in (0.02, 0.005, 0.001):
        numerical = np.stack(
            [
                (
                    direct(
                        [a + h * s[..., d] for a, s in zip(arrays, seeds, strict=True)], controls
                    )
                    - direct(
                        [a - h * s[..., d] for a, s in zip(arrays, seeds, strict=True)], controls
                    )
                )
                / (2 * h)
                for d in range(7)
            ],
            axis=-1,
        )
        errors.append(np.max(np.abs(numerical - tangent)))
    # K=2 is C1, but has different second derivatives at g=0.
    if g != 0 or moments is None:
        assert errors[1] < errors[0] / 10
        assert errors[2] < errors[1] / 15
    else:
        assert errors[1] < errors[0] / 3
    np.testing.assert_allclose(
        numerical,
        tangent,
        rtol=8e-4 if g == 0 and moments else 3e-7,
        atol=2e-7 if g == 0 and moments else 3e-10,
    )
    total, dt = call(arrays, seeds, controls, total=True)
    np.testing.assert_array_equal(total, value.sum(axis=0))
    np.testing.assert_array_equal(dt, tangent.sum(axis=0))


@pytest.mark.parametrize("g", [-0.3, 0.0, 0.4])
def test_black_homogeneous_single_event_analytic(g):
    _, _, _, arrays = state(g=g, rho=0.0, layers=1)
    rng = np.random.default_rng(241)
    seeds = [rng.normal(0, 0.03, (*a.shape, 4)) for a in arrays]
    seeds[5][:] = 0
    value, tangent = call(arrays, seeds, dict(CONTROLS, max_order=1))
    mu0, mu = cos(radians(30)), cos(radians(20))
    cosine = np.sin(radians(30)) * np.sin(radians(20)) * cos(radians(20)) - mu0 * mu
    ray_phase = 0.75 * (1 + cosine**2)
    q = 1 + g * g - 2 * g * cosine
    phase = (1 - g * g) / q**1.5
    phase_g = -2 * g / q**1.5 - 3 * (1 - g * g) * (g - cosine) / q**2.5
    tau = (arrays[0] + arrays[1] + arrays[2])[0]
    dtau = (seeds[0] + seeds[1] + seeds[2])[0]
    x = tau * (1 / mu0 + 1 / mu)
    integral = -np.expm1(-x) / x
    derivative = ((x + 1) * np.exp(-x) - 1) / x**2
    source = arrays[1][0] * ray_phase + arrays[3][0] * phase
    dsource = (
        seeds[1][0] * ray_phase
        + seeds[3][0] * phase
        + arrays[3][0, :, None] * phase_g * seeds[4][0]
    )
    scale = 8 / (4 * pi * mu)
    expected = scale * (
        dsource * integral[:, None]
        + source[:, None] * derivative[:, None] * (1 / mu0 + 1 / mu) * dtau
    )
    np.testing.assert_allclose(value[1], scale * source * integral, rtol=5e-15)
    np.testing.assert_allclose(tangent[1], expected, rtol=1e-14, atol=5e-17)


@pytest.mark.parametrize("depth", [1e160, 1e200, 1e300, 1e308])
def test_opaque_absorption_scaling_identity(depth):
    _, _, _, arrays = state(g=0.0, rho=0.0, layers=1, points=1)
    arrays[0][:] = depth
    arrays[1][:] = 0.0
    arrays[2][:] = arrays[3][:] = 1.0
    seeds = [np.zeros((*a.shape, 2)) for a in arrays]
    seeds[0][..., 0] = depth
    seeds[0][..., 1] = -depth
    value, tangent = call(arrays, seeds, dict(max_order=1, angular_order=2, layer_subdivisions=1))
    # Opaque, isotropic single scattering is inversely proportional to total
    # extinction. Unit aerosol depth is negligible here; scaling absorption by
    # (1 + epsilon) therefore gives dI = -I, independently of phi derivatives.
    mu0, mu = cos(radians(30)), cos(radians(20))
    expected = (8 / (4 * pi)) * (mu0 / (mu0 + mu)) / depth
    np.testing.assert_allclose(value[1], expected, rtol=5e-15, atol=0)
    np.testing.assert_allclose(tangent[1, :, 0], -value[1], rtol=5e-15, atol=0)
    np.testing.assert_allclose(tangent[1, :, 1], value[1], rtol=5e-15, atol=0)


def test_independent_scattering_order_homogeneity():
    _, _, _, arrays = state(layers=1)
    arrays[1][:] = 0
    seeds = [np.zeros((*a.shape, 1)) for a in arrays]
    seeds[3][..., 0] = arrays[3]  # scale s at fixed total extinction and g
    value, tangent = call(arrays, seeds)
    np.testing.assert_allclose(
        tangent[..., 0], np.arange(4)[:, None] * value, rtol=2e-14, atol=1e-17
    )


def test_transparent_black_surface_first_order_and_zero_rho_derivative():
    scene, source, optics = inputs(np.zeros((2, 2)), rho=0.0)
    zeros = np.zeros((2, 2, 2))
    dray = zeros.copy()
    dray[..., 0] = 1.0
    drho = np.zeros((2, 2))
    drho[:, 1] = 1.0
    values, tangent = scattering_orders_tangent(
        scene,
        source=source,
        optics=optics,
        optical_depth_tangent=zeros,
        reflectance_tangent=drho,
        rayleigh_optical_depth_tangent=dray,
        **CONTROLS,
    )
    mu0, mu = cos(radians(30)), cos(radians(20))
    # Beam photon direction has z=-mu0; outgoing view has z=mu.
    cosine = np.sin(radians(30)) * np.sin(radians(20)) * cos(radians(20)) - mu0 * mu
    expected = 8.0 / (4 * pi) * 0.75 * (1 + cosine**2) * 2 / mu
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_allclose(tangent[1, :, 0], expected, rtol=3e-15)
    np.testing.assert_allclose(tangent[0, :, 1], 8.0 * mu0 / pi, rtol=3e-15)
    np.testing.assert_array_equal(tangent[2:], 0)
    np.testing.assert_array_equal(tangent[1:, :, 1], 0)


@pytest.mark.parametrize(
    "endpoint",
    [
        "zero_scattering",
        "zero_extinction",
        "black",
        "white",
        "zero_absorption",
        "unit_ssa",
        "zero_irradiance",
    ],
)
def test_feasible_endpoint_directions(endpoint):
    scene, source, optics, arrays = state(g=0.2)
    seeds = [np.zeros((*a.shape, 1)) for a in arrays]
    if endpoint == "zero_scattering":
        arrays[1][:] = arrays[3][:] = 0
        seeds[1][:] = seeds[3][:] = 0.04
    elif endpoint == "zero_extinction":
        for a in arrays[:4]:
            a[:] = 0
        for s in seeds[:4]:
            s[:] = 0.04
    elif endpoint == "unit_ssa":
        arrays[3][:] = arrays[2]
        seeds[3][:] = -0.04
    elif endpoint == "zero_absorption":
        arrays[0][:] = 0
        seeds[0][:] = 0.04
    elif endpoint in ("black", "white"):
        arrays[5][:] = 0 if endpoint == "black" else 1
        seeds[5][:] = 0.1 if endpoint == "black" else -0.1
    else:
        source = replace(source, irradiance=replace(source.irradiance, values=np.zeros(2)))
        value, tangent = scattering_orders_tangent(
            scene,
            source=source,
            optics=optics,
            **seed_arguments([np.ones_like(s) for s in seeds]),
            **CONTROLS,
        )
        np.testing.assert_array_equal(value, 0)
        np.testing.assert_array_equal(tangent, 0)
        return
    value, tangent = call(arrays, seeds)
    errors = []
    for h in (0.01, 0.001, 0.0001, 0.00001):
        one = direct([a + h * s[..., 0] for a, s in zip(arrays, seeds, strict=True)], CONTROLS)
        two = direct([a + 2 * h * s[..., 0] for a, s in zip(arrays, seeds, strict=True)], CONTROLS)
        numerical = (-3 * value + 4 * one - two) / (2 * h)
        errors.append(np.max(np.abs(numerical - tangent[..., 0])))
    # A second-order difference eventually reaches its cancellation floor.
    assert min(errors[1:]) < max(errors[0] / 1000, 2e-11)
    np.testing.assert_allclose(numerical, tangent[..., 0], rtol=2e-7, atol=3e-10)


def test_delta_m_order_zero_explicit_chain_rule():
    _, _, _, arrays = state(g=0.6)
    rng = np.random.default_rng(72)
    seeds = [rng.normal(size=(*a.shape, 3)) for a in arrays]
    controls = dict(CONTROLS, max_order=0, delta_m_moments=4)
    value, tangent = call(arrays, seeds, controls)
    a, r, ext, scatter, g, rho = arrays
    da, dr, de, ds, dg, drho = seeds
    tau = (a + r + ext - g**4 * scatter).sum(axis=0)
    dtau = (
        da + dr + de - g[..., None] ** 4 * ds - 4 * g[..., None] ** 3 * scatter[..., None] * dg
    ).sum(axis=0)
    mu0, mu = cos(radians(30)), cos(radians(20))
    gain = 8 * mu0 / pi * np.exp(-tau * (1 / mu0 + 1 / mu))
    np.testing.assert_allclose(value[0], gain * rho, rtol=3e-15)
    np.testing.assert_allclose(
        tangent[0],
        gain[:, None] * (drho - rho[:, None] * (1 / mu0 + 1 / mu) * dtau),
        rtol=5e-15,
        atol=2e-16,
    )


def test_block_directions_projection_and_equal_primal_g_different_seeds():
    _, _, _, arrays = state()
    rng = np.random.default_rng(930)
    coefficients = [rng.normal(0, 0.02, (*a.shape, 4)) for a in arrays]

    def model(x, d):
        current = [a + c @ x for a, c in zip(arrays, coefficients, strict=True)]
        seeds = [c @ d for c in coefficients]
        value, tangent = call(current, seeds, total=True)
        return value, tangent

    x = np.zeros(4)
    value, full = jacobian(model, x)
    _, blocked = jacobian(model, x, block_size=1)
    np.testing.assert_array_equal(full, blocked)
    directions = rng.normal(size=(4, 3))
    v, mixed = model(x, directions)
    np.testing.assert_array_equal(value, v)
    np.testing.assert_allclose(mixed, full @ directions, rtol=5e-14, atol=1e-16)
    projection = np.array([[0.3, 0.7], [-0.4, 0.4]])
    np.testing.assert_allclose(
        projection @ mixed, (projection @ full) @ directions, rtol=2e-14, atol=1e-16
    )
    empty, d0 = model(x, np.empty((4, 0)))
    np.testing.assert_array_equal(empty, value)
    assert d0.shape == (2, 0)


@pytest.mark.parametrize("layout", ["F", "reversed", "unaligned"])
def test_layouts_ownership_and_concurrent_calls(layout):
    scene, source, optics, arrays = state()
    rng = np.random.default_rng(771)
    seeds = [rng.normal(0, 0.02, (*a.shape, 2)) for a in arrays]
    expected = call(arrays, seeds)
    owners = (
        optics.absorption,
        optics.rayleigh,
        optics.aerosol,
        optics.aerosol,
        optics.aerosol,
        scene.surface,
    )
    names = (
        "optical_depth",
        "optical_depth",
        "optical_depth",
        "scattering_depth",
        "asymmetry",
        "reflectance",
    )
    borrowed = []
    for owner, name, a in zip(owners, names, arrays, strict=True):
        array = layout_copy(a, layout)
        object.__setattr__(owner, name, array)
        borrowed.append(array)
    seeds = [layout_copy(s, layout) for s in seeds]
    borrowed += seeds
    snapshots = [a.tobytes() for a in borrowed]

    def evaluate():
        return scattering_orders_tangent(
            scene,
            source=source,
            optics=optics,
            **seed_arguments(seeds),
            **CONTROLS,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: evaluate(), range(4)))
    for result in results:
        for a, e in zip(result, expected, strict=True):
            np.testing.assert_array_equal(a, e)
            assert a.flags.owndata and a.flags.writeable and a.flags.c_contiguous
            assert all(not np.shares_memory(a, other) for other in borrowed)
        assert not np.shares_memory(*result)
    assert [a.tobytes() for a in borrowed] == snapshots


@pytest.mark.parametrize("role", range(6))
@pytest.mark.parametrize("invalid", ["dtype", "shape", "nan", "endian"])
def test_invalid_seeds(role, invalid):
    _, _, _, arrays = state()
    seeds = [np.zeros((*a.shape, 2)) for a in arrays]
    if invalid == "dtype":
        seeds[role] = seeds[role].astype(np.float32)
    elif invalid == "shape":
        seeds[role] = seeds[role][..., :1]
    elif invalid == "endian":
        seeds[role] = seeds[role].astype(">f8")
    else:
        seeds[role].flat[0] = np.nan
    with pytest.raises(ValueError):
        call(arrays, seeds)


@pytest.mark.parametrize(
    "control",
    [
        dict(max_order=-1),
        dict(angular_order=1),
        dict(layer_subdivisions=0),
        dict(delta_m_moments=1),
        dict(delta_m_moments=7),
    ],
)
def test_invalid_controls(control):
    _, _, _, arrays = state()
    seeds = [np.zeros((*a.shape, 0)) for a in arrays]
    with pytest.raises(ValueError):
        call(arrays, seeds, dict(CONTROLS, **control))


@pytest.mark.parametrize(
    "name", ["max_order", "angular_order", "layer_subdivisions", "delta_m_moments"]
)
@pytest.mark.parametrize("scattering", [False, True])
def test_integer_control_contract_before_dispatch(name, scattering):
    scene, source, optics, arrays = state()
    if not scattering:
        optics = replace(optics, rayleigh=None, aerosol=None)
    arguments: dict[str, Any] = dict(max_order=0, angular_order=3, layer_subdivisions=3)
    arguments[name] = 0.0 if name == "max_order" else 2.0
    seeds = [np.zeros((*a.shape, 1)) for a in arrays]
    with pytest.raises(TypeError):
        reflected_radiance_tangent(
            scene,
            source=source,
            optics=optics,
            optical_depth_tangent=seeds[0],
            reflectance_tangent=seeds[5],
            **arguments,
        )


def test_scalar_and_blas_tangent_recurrences_agree(monkeypatch):
    from orsia_radiative.transfer import orders
    from orsia_radiative.transfer.native import transfer_native

    _, _, _, arrays = state(layers=3)
    rng = np.random.default_rng(751)
    seeds = [rng.normal(0, 0.02, (*a.shape, 3)) for a in arrays]
    expected = call(arrays, seeds, dict(CONTROLS, delta_m_moments=4))
    monkeypatch.setattr(
        orders, "_native_scattering_orders_tangent", transfer_native.scattering_orders_tangent
    )
    actual = call(arrays, seeds, dict(CONTROLS, delta_m_moments=4))
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, rtol=3e-13, atol=1e-17)


def test_absent_optional_seeds_and_absent_scattering_roles():
    scene, source, optics = inputs(np.full((2, 2), 0.1))
    seeds = np.full((2, 2, 1), 0.03)
    base = dict(
        source=source,
        optics=optics,
        optical_depth_tangent=seeds,
        reflectance_tangent=np.full((2, 1), 0.02),
    )
    value, tangent = scattering_orders_tangent(scene, **base, **CONTROLS)
    direct_value, direct_tangent = reflected_radiance_tangent(scene, **base)
    np.testing.assert_allclose(value[0], direct_value.values, rtol=3e-15)
    np.testing.assert_allclose(tangent[0], direct_tangent, rtol=3e-15)
    np.testing.assert_array_equal(value[1:], 0)
    np.testing.assert_array_equal(tangent[1:], 0)
    with pytest.raises(ValueError, match="reflectance"):
        scattering_orders_tangent(
            replace(
                scene, surface=SurfaceState(optics.grid, emissivity=np.ones(2), temperature_k=280.0)
            ),
            **base,
        )


@pytest.mark.parametrize("g", [-0.8, 0.8])
def test_hg_supported_endpoints(g):
    _, _, _, arrays = state(g=g)
    seeds = [np.zeros((*a.shape, 1)) for a in arrays]
    seeds[4][:] = -np.sign(g) * 0.1
    value, tangent = call(arrays, seeds)
    h = 1e-5
    one = direct([a + h * s[..., 0] for a, s in zip(arrays, seeds, strict=True)], CONTROLS)
    two = direct([a + 2 * h * s[..., 0] for a, s in zip(arrays, seeds, strict=True)], CONTROLS)
    np.testing.assert_allclose(
        (-3 * value + 4 * one - two) / (2 * h), tangent[..., 0], rtol=1e-7, atol=2e-10
    )


def test_equal_beam_exponents_and_overlying_depth():
    scene, source, optics, arrays = state()
    vectors, _ = angular_grid(3)
    source = replace(source, zenith_deg=float(np.degrees(np.arccos(abs(vectors[0, 2])))))
    seeds = [np.zeros((*a.shape, 1)) for a in arrays]
    seeds[0][-1, :, 0] = 0.1
    _, tangent = scattering_orders_tangent(
        scene, source=source, optics=optics, **seed_arguments(seeds), **CONTROLS
    )
    results = []
    assert optics.absorption is not None
    for sign in (1, -1):
        perturbed = replace(
            optics,
            absorption=replace(
                optics.absorption, optical_depth=arrays[0] + sign * 1e-5 * seeds[0][..., 0]
            ),
        )
        results.append(scattering_orders(scene, source=source, optics=perturbed, **CONTROLS))
    np.testing.assert_allclose(
        (results[0] - results[1]) / 2e-5, tangent[..., 0], rtol=2e-9, atol=2e-11
    )


def test_extreme_attenuation_boundary_sweep():
    scene, source, optics, arrays = state()
    scene = replace(scene, geometry=replace(scene.geometry, view_zenith_deg=89.0))
    source = replace(source, irradiance=replace(source.irradiance, values=np.full(2, 1e200)))
    assert optics.absorption is not None
    optics = replace(optics, absorption=replace(optics.absorption, optical_depth=arrays[0] + 100))
    seeds = [np.full((*a.shape, 1), 0.01) for a in arrays]
    value, tangent = scattering_orders_tangent(
        scene, source=source, optics=optics, **seed_arguments(seeds), **CONTROLS
    )
    expected = scattering_orders(scene, source=source, optics=optics, **CONTROLS)
    np.testing.assert_allclose(value, expected, rtol=3e-14)
    assert np.isfinite(tangent).all()
    # A source-amplitude scaling identity checks the derivative exponent range.
    scaled = replace(source, irradiance=replace(source.irradiance, values=np.ones(2)))
    _, reference = scattering_orders_tangent(
        scene, source=scaled, optics=optics, **seed_arguments(seeds), **CONTROLS
    )
    np.testing.assert_allclose(tangent / 1e200, reference, rtol=4e-14, atol=1e-300)


def test_negative_delta_m_order_is_rejected():
    _, _, _, arrays = state(g=0.8, rho=0.0, layers=1)
    arrays[1][:] = 0
    seeds = [np.zeros((*a.shape, 0)) for a in arrays]
    scene, source, optics, _ = state(arrays)
    scene = replace(scene, geometry=replace(scene.geometry, view_zenith_deg=0.0))
    source = replace(source, zenith_deg=0.0)
    controls: dict[str, Any] = dict(CONTROLS, delta_m_moments=2)
    with pytest.raises(ValueError, match="negative order"):
        scattering_orders_tangent(
            scene,
            source=source,
            optics=optics,
            **seed_arguments(seeds),
            **controls,
        )


def test_selected_synthetic_refinement():
    _, _, _, arrays = state(g=0.15, rho=0.1, layers=1, points=1)
    arrays[1][:] = 0.01
    arrays[2][:] = 0.03
    arrays[3][:] = 0.02
    seeds = [np.full((*a.shape, 1), 0.01) for a in arrays]
    # Separate vertical, angular and event-order refinements; this optically thin
    # case only checks a selected discretization, not an accuracy envelope.
    for name, levels, controls in (
        ("layer_subdivisions", (2, 4, 8, 16), dict(max_order=4, angular_order=5)),
        ("angular_order", (3, 5, 7, 9), dict(max_order=4, layer_subdivisions=12)),
        ("max_order", (1, 2, 3, 4), dict(angular_order=5, layer_subdivisions=12)),
    ):
        results = []
        for level in levels:
            value, tangent = call(arrays, seeds, dict(controls, **{name: level}), total=True)
            results.append(np.array([value[0], tangent[0, 0]]))
        differences = np.abs(np.diff(results, axis=0))
        assert np.all(differences[-1] < differences[0] / 3)
        assert np.all(differences[-1] < 3e-5)
