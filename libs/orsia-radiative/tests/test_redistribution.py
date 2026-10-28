"""Layer-local redistribution against independent full-recomputation expressions."""

from dataclasses import replace
from typing import Literal
from unittest.mock import patch
from weakref import ref

import numpy as np
import orders_reference as orders
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    AerosolComponent,
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    SpectralIrradiance,
    SurfaceState,
    compose_atmospheric_optics,
    solar_source,
    wavenumber_grid,
)
from orsia_radiative.transfer import orders as production_orders


@pytest.mark.parametrize("max_order", [1, 2, 4])
@pytest.mark.parametrize("reflectance", [0.0, 0.3])
@pytest.mark.parametrize("constituents", ["rayleigh", "hg", "mixed", "zero"])
@pytest.mark.parametrize("direction", [(30.0, 20.0), (60.0, 130.0)])
@pytest.mark.parametrize("repeat_layers", [False, True])
def test_every_component_matches_full_recomputation(
    max_order: int,
    reflectance: float,
    constituents: str,
    direction: tuple[float, float],
    repeat_layers: bool,
) -> None:
    scene, sun, molecular, aerosol = _inputs()
    scene = replace(
        scene, surface=SurfaceState(wavenumber_grid(sun.wavenumbers_cm1), np.full(2, reflectance))
    )
    # Distinct layer/spectral asymmetries, including both allowed endpoints;
    # no assumption that a phase matrix is interchangeable between layers.
    aerosol = replace(aerosol, asymmetry=np.array([[-0.8, 0.8], [0.0, -0.2]]))
    if repeat_layers:
        scene = replace(
            scene,
            atmosphere=Atmosphere(
                np.arange(5, dtype=np.float64) * 1000,
                np.linspace(100000, 80000, 5),
                np.linspace(97500, 82500, 4),
                np.full(4, 280.0),
                np.full(4, 2e29),
                np.full(4, 1.9e29),
                np.full(4, 1e28),
            ),
        )
        molecular = replace(molecular, optical_depth=np.tile(molecular.optical_depth, (2, 1)))
        aerosol = replace(
            aerosol,
            optical_depth=np.tile(aerosol.optical_depth, (2, 1)),
            scattering_depth=np.tile(aerosol.scattering_depth, (2, 1)),
            asymmetry=np.tile(aerosol.asymmetry, (2, 1)),
        )
    if constituents == "zero":
        molecular = replace(molecular, optical_depth=np.zeros_like(molecular.optical_depth))
        aerosol = replace(aerosol, scattering_depth=np.zeros_like(aerosol.scattering_depth))
    source = solar_source(sun, direction=direction)
    optics = compose_atmospheric_optics(
        scene.atmosphere,
        source,
        rayleigh=None if constituents == "hg" else molecular,
        aerosol=None if constituents == "rayleigh" else aerosol,
    )

    def calculate() -> terms_support.RadianceDecomposition:
        return terms_support.reflected_radiance(
            scene,
            source=source,
            optics=optics,
            max_order=max_order,
            angular_order=3,
            layer_subdivisions=3,
        )

    actual = calculate()

    def reference_scatter(
        field: NDArray[np.float64],
        component: RayleighComponent | AerosolComponent,
        point: int,
        subdivisions: int,
        cosines: NDArray[np.float64],
        weights: NDArray[np.float64],
        phase_cache: dict[orders.PhaseKey, NDArray[np.float64]],
        *,
        out: NDArray[np.float64],
        delta_m_moments: int | None = None,
        rayleigh_factors: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> None:
        result = _full_recomputation((field,), component, point, subdivisions, cosines, weights)[0]
        out += result

    with patch.object(orders, "_scatter", side_effect=reference_scatter):
        expected = orders.scattering_orders(
            scene,
            source=source,
            optics=optics,
            max_order=max_order,
            angular_order=3,
            layer_subdivisions=3,
        )
    np.testing.assert_allclose(actual.values, expected.sum(axis=0), rtol=3e-13, atol=1e-15)
    for index, (component, row) in enumerate(zip(actual.components, expected, strict=True)):
        np.testing.assert_allclose(component.values, row, rtol=3e-13, atol=1e-15)
        assert component.name == f"atmospheric_order_{index}"
        assert component.role is None
    assert actual.grid == source.irradiance.grid
    assert actual.geometry == scene.geometry


@pytest.mark.parametrize("phase", ["rayleigh", "henyey_greenstein"])
def test_preparation_is_layer_local_skips_zeros_and_outputs_own_buffers(
    phase: Literal["rayleigh", "henyey_greenstein"],
) -> None:
    directions, weights = orders.angular_grid(2)
    outgoing = np.vstack((directions, [0.0, 0.0, 1.0]))
    cosines = directions @ outgoing.T
    factors = orders._prepare_rayleigh_factors(directions, outgoing, weights)
    grid = np.array([12000.0, 12001.0])
    depth = np.array([[0.1, 0.0], [0.0, 0.0], [0.2, 0.0], [0.3, 0.0]])
    component = (
        RayleighComponent(wavenumber_grid(grid), depth)
        if phase == "rayleigh"
        else AerosolComponent(
            wavenumber_grid(grid),
            np.array([[0.2, 0.0], [0.0, 0.0], [0.3, 0.0], [0.4, 0.0]]),
            depth,
            np.array([[-0.8, 0.0], [0.0, 0.0], [0.8, 0.0], [0.2, 0.0]]),
        )
    )
    fields = tuple(np.full((8, weights.size), value) for value in (1.0, 2.0, 0.0))
    for field in fields:
        field[6:] = 0.0  # positive depth, but all fields zero in this layer
        field.flags.writeable = False
    saved = tuple(field.copy() for field in fields)
    matrices: list[ref[NDArray[np.float64]]] = []
    prepare = orders._prepare_redistribution

    def observe(
        cosine: NDArray[np.float64],
        measure: NDArray[np.float64],
        g: float | None,
        moments: int | None,
    ) -> NDArray[np.float64]:
        assert all(matrix() is None for matrix in matrices)
        matrix = prepare(cosine, measure, g, moments)
        matrices.append(ref(matrix))
        return matrix

    actual = tuple(np.zeros((8, cosines.shape[1])) for _ in fields)
    with (
        patch.object(orders, "_prepare_redistribution", side_effect=observe) as preparation,
        patch.object(orders, "_PHASE_CACHE_BYTES", 0),
    ):
        for field, out in zip(fields, actual, strict=True):
            orders._scatter(
                field, component, 0, 2, cosines, weights, {}, out=out, rayleigh_factors=factors
            )
    assert preparation.call_count == (0 if phase == "rayleigh" else 4)
    assert all(matrix() is None for matrix in matrices)
    expected = _full_recomputation(fields, component, 0, 2, cosines, weights)
    for i, (source, reference) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(source, reference, rtol=3e-13, atol=1e-15)
        assert source.dtype == np.float64 and source.flags.writeable
        assert all(not np.shares_memory(source, field) for field in fields)
        assert all(not np.shares_memory(source, other) for other in actual[i + 1 :])
    with patch.object(orders, "_prepare_redistribution", wraps=prepare) as preparation:
        empty = tuple(np.zeros_like(out) for out in actual)
        for field, out in zip(fields, empty, strict=True):
            orders._scatter(
                field, component, 1, 2, cosines, weights, {}, out=out, rayleigh_factors=factors
            )
        zeros = (np.zeros_like(actual[0]),)
        orders._scatter(
            fields[-1],
            component,
            0,
            2,
            cosines,
            weights,
            {},
            out=zeros[0],
            rayleigh_factors=factors,
        )
    preparation.assert_not_called()
    assert all(not np.any(source) for source in (*empty, *zeros))
    for field, snapshot in zip(fields, saved, strict=True):
        np.testing.assert_array_equal(field, snapshot)
    repeated = np.zeros_like(actual[0])
    orders._scatter(
        fields[0], component, 0, 2, cosines, weights, {}, out=repeated, rayleigh_factors=factors
    )
    np.testing.assert_array_equal(repeated, actual[0])


@pytest.mark.parametrize("budget_matrices", [0, 1, 2, 20])
def test_call_local_reuse_is_bounded_and_keeps_distinct_phase_parameters(
    budget_matrices: int,
) -> None:
    directions, weights = orders.angular_grid(2)
    cosines = directions @ np.vstack((directions, [0.0, 0.0, 1.0])).T
    axis = np.array([12000.0, 12001.0])
    component = AerosolComponent(
        wavenumber_grid(axis),
        np.full((4, 2), 0.2),
        np.full((4, 2), 0.1),
        asymmetry=np.array([[0.2, 0.2], [0.4, 0.4], [0.2, 0.2], [0.4, 0.4]]),
    )
    field = np.ones((8, weights.size))
    cache: dict[orders.PhaseKey, NDArray[np.float64]] = {}
    budget = budget_matrices * cosines.nbytes
    prepare = orders._prepare_redistribution
    with (
        patch.object(orders, "_PHASE_CACHE_BYTES", budget),
        patch.object(orders, "_prepare_redistribution", wraps=prepare) as preparation,
    ):
        for point in (0, 1):
            actual = np.zeros((field.shape[0], cosines.shape[1]))
            orders._scatter(field, component, point, 2, cosines, weights, cache, out=actual)
            expected = _full_recomputation((field,), component, point, 2, cosines, weights)[0]
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-15)
            assert sum(matrix.nbytes for matrix in cache.values()) <= budget
    assert preparation.call_count == (2 if budget_matrices >= 2 else 8)


@pytest.mark.parametrize("moments", [None, 2, 4])
@pytest.mark.parametrize("budget_matrices", [0, 1, 20])
@pytest.mark.parametrize("rayleigh", [False, True])
@pytest.mark.parametrize("angular_order", [3, 9])
def test_grouped_strided_signed_fields_and_additive_output(
    moments: int | None, budget_matrices: int, rayleigh: bool, angular_order: int
) -> None:
    directions, weights = orders.angular_grid(angular_order)
    outgoing = np.vstack((directions, [0.0, 0.0, 1.0]))
    cosines = directions @ outgoing.T
    factors = orders._prepare_rayleigh_factors(directions, outgoing, weights)
    depth = np.array([[0.1], [0.0], [0.2], [0.3], [0.4], [0.5], [0.6]])
    g_values = np.array([[0.8], [-0.8], [0.8], [0.0], [-0.8], [0.4], [0.8]])
    grid = wavenumber_grid(np.array([12000.0]))
    component = (
        RayleighComponent(grid, depth)
        if rayleigh
        else AerosolComponent(grid, depth, depth, g_values)
    )
    subdivisions = 3
    storage = np.random.default_rng(782).normal(size=(depth.size * subdivisions, weights.size + 1))
    storage[15:18] = 0  # nonzero scattering depth, zero field
    field = storage[:, :-1]  # actual SOS incoming field is a strided centre view
    snapshot = storage.copy()
    field.flags.writeable = False
    cosines.flags.writeable = weights.flags.writeable = False
    expected = np.zeros((field.shape[0], cosines.shape[1]))
    for layer in range(depth.size):
        g = None if rayleigh else float(g_values[layer, 0])
        phase = (
            orders.phase_function(cosines, g)
            if moments is None
            else orders._transport_phase(cosines, g, moments)
        ) * weights[:, None]
        fraction = 0.0 if g is None or g <= 0 or moments is None else g**moments
        cells = slice(layer * subdivisions, (layer + 1) * subdivisions)
        expected[cells] = (field[cells] @ phase) * (depth[layer, 0] / subdivisions * (1 - fraction))
    cache: dict[orders.PhaseKey, NDArray[np.float64]] = {}
    out = np.full_like(expected, 0.125)
    fresh = np.zeros_like(expected)
    with patch.object(orders, "_PHASE_CACHE_BYTES", budget_matrices * cosines.nbytes):
        orders._scatter(
            field,
            component,
            0,
            subdivisions,
            cosines,
            weights,
            cache,
            delta_m_moments=moments,
            out=fresh,
            rayleigh_factors=factors,
        )
        returned = orders._scatter(
            field,
            component,
            0,
            subdivisions,
            cosines,
            weights,
            cache,
            delta_m_moments=moments,
            out=out,
            rayleigh_factors=factors,
        )
    assert returned is None
    assert not np.shares_memory(fresh, field)
    assert sum(matrix.nbytes for matrix in cache.values()) <= budget_matrices * cosines.nbytes
    np.testing.assert_allclose(fresh, expected, rtol=3e-13, atol=1e-15)
    np.testing.assert_allclose(out, expected + 0.125, rtol=3e-13, atol=1e-15)
    np.testing.assert_array_equal(storage, snapshot)


def test_all_active_contiguous_layers_use_factors_and_empty_addition_is_noop() -> None:
    directions, weights = orders.angular_grid(2)
    cosines = directions @ directions.T
    component = RayleighComponent(wavenumber_grid(np.array([12000.0])), np.full((5, 1), 0.1))
    field = np.ones((15, weights.size))
    factors = orders._prepare_rayleigh_factors(directions, directions, weights)
    actual = np.zeros_like(field)
    with patch.object(
        orders, "_prepare_redistribution", wraps=orders._prepare_redistribution
    ) as prep:
        orders._scatter(
            field, component, 0, 3, cosines, weights, {}, out=actual, rayleigh_factors=factors
        )
    prep.assert_not_called()
    expected = _full_recomputation((field,), component, 0, 3, cosines, weights)[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-15)
    original = actual.copy()
    with patch.object(
        orders, "_prepare_redistribution", side_effect=AssertionError("unused phase")
    ):
        returned = orders._scatter(
            np.zeros_like(field),
            component,
            0,
            3,
            cosines,
            weights,
            {},
            out=actual,
            rayleigh_factors=factors,
        )
    assert returned is None
    np.testing.assert_array_equal(actual, original)


def test_reference_calls_reuse_rayleigh_but_never_share_geometry_or_storage() -> None:
    scene, sun, molecular, _ = _inputs()
    scene = replace(
        scene,
        atmosphere=Atmosphere(
            np.arange(5, dtype=np.float64) * 1000,
            np.linspace(100000, 80000, 5),
            np.linspace(97500, 82500, 4),
            np.full(4, 280.0),
            np.full(4, 2e29),
            np.full(4, 1.9e29),
            np.full(4, 1e28),
        ),
    )
    molecular = replace(molecular, optical_depth=np.tile(molecular.optical_depth, (2, 1)))
    matrices: list[ref[NDArray[np.float64]]] = []
    prepare = orders._prepare_rayleigh_factors

    def observe(directions, outgoing, weights):
        factors = prepare(directions, outgoing, weights)
        matrices.extend(ref(matrix) for matrix in factors)
        return factors

    with patch.object(orders, "_prepare_rayleigh_factors", side_effect=observe) as preparation:
        for azimuth in (20.0, 130.0):
            orders.scattering_orders(
                scene,
                source=solar_source(sun, direction=(30.0, azimuth)),
                optics=compose_atmospheric_optics(
                    scene.atmosphere,
                    solar_source(sun, direction=(30.0, azimuth)),
                    rayleigh=molecular,
                ),
                max_order=4,
                angular_order=3,
            )
            assert all(matrix() is None for matrix in matrices)
    assert preparation.call_count == 2


@pytest.mark.parametrize("angular_order", [2, 3, 8, 12])
@pytest.mark.parametrize("view", [(0.0, 0.0, 1.0), (0.8, 0.0, 0.6), (0.0, -1.0, 0.0)])
def test_exact_rayleigh_factors_match_dense_operator_for_signed_strided_fields(
    angular_order: int, view: tuple[float, float, float]
) -> None:
    directions, weights = orders.angular_grid(angular_order)
    outgoing = np.vstack((directions, view))
    cosines = directions @ outgoing.T
    factors = orders._prepare_rayleigh_factors(directions, outgoing, weights)
    dense = orders._prepare_redistribution(cosines, weights, None)
    np.testing.assert_allclose(factors[0] @ factors[1], dense, rtol=3e-13, atol=1e-15)
    assert all(not factor.flags.writeable for factor in factors)
    assert all(not np.shares_memory(factor, directions) for factor in factors)
    field = np.random.default_rng(274).normal(size=(21, weights.size + 1))[:, :-1]
    field[3:6] = 0
    snapshot = field.copy()
    component = RayleighComponent(
        wavenumber_grid(np.array([12000.0])),
        np.array([[0.1], [0.2], [0.0], [0.3], [0.4], [0.5], [0.6]]),
    )
    with patch.object(
        orders, "_prepare_redistribution", side_effect=AssertionError("no dense Rayleigh")
    ):
        actual = np.zeros((field.shape[0], cosines.shape[1]))
        delta = np.zeros_like(actual)
        orders._scatter(
            field, component, 0, 3, cosines, weights, {}, out=actual, rayleigh_factors=factors
        )
        orders._scatter(
            field,
            component,
            0,
            3,
            cosines,
            weights,
            {},
            out=delta,
            rayleigh_factors=factors,
            delta_m_moments=4,
        )
    expected = _full_recomputation((field,), component, 0, 3, cosines, weights)[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-15)
    np.testing.assert_array_equal(delta, actual)
    np.testing.assert_array_equal(field, snapshot)


@pytest.mark.parametrize("max_order,zero", [(0, False), (3, True)])
def test_inactive_orders_skip_angular_allocation_and_match_public_radiance(
    max_order: int, zero: bool
) -> None:
    scene, sun, molecular, _ = _inputs()
    if zero:
        molecular = replace(molecular, optical_depth=np.zeros((2, 2)))
    source = solar_source(sun, direction=(30.0, 20.0))
    optics = compose_atmospheric_optics(scene.atmosphere, source, rayleigh=molecular)
    with patch.object(
        production_orders, "angular_grid", side_effect=AssertionError("unused angular allocation")
    ):
        rows = production_orders.scattering_orders(
            scene,
            source=source,
            optics=optics,
            max_order=max_order,
            angular_order=24,
        )
    result = terms_support.reflected_radiance(
        scene, source=source, optics=optics, max_order=max_order, angular_order=3
    )
    direct = terms_support.order_zero(
        scene,
        source=source,
        optics=compose_atmospheric_optics(scene.atmosphere, source, rayleigh=molecular),
    )
    np.testing.assert_array_equal(rows.sum(axis=0), result.values)
    np.testing.assert_array_equal(result.values, direct.values)
    assert len(result.components) == max_order + 1


def _inputs() -> tuple[RadiativeScene, SpectralIrradiance, RayleighComponent, AerosolComponent]:
    axis = np.array([12000.0, 12001.0], dtype=np.float64)
    scene = RadiativeScene(
        Atmosphere(
            np.array([0.0, 1000.0, 2000.0]),
            np.array([100000.0, 90000.0, 80000.0]),
            np.array([95000.0, 85000.0]),
            np.array([290.0, 280.0]),
            np.array([2e29, 1.8e29]),
            np.array([1.9e29, 1.7e29]),
            np.array([1e28, 1e28]),
        ),
        SurfaceState(wavenumber_grid(axis), reflectance=np.array([0.2, 0.0])),
        ObservationGeometry(20.0),
    )
    sun = SpectralIrradiance(wavenumber_grid(axis), np.array([8.0, 10.0]))
    molecular = RayleighComponent(wavenumber_grid(axis), np.array([[0.04, 0.03], [0.01, 0.02]]))
    aerosol = AerosolComponent(
        wavenumber_grid(axis),
        np.array([[0.02, 0.03], [0.06, 0.05]]),
        np.array([[0.01, 0.01], [0.04, 0.03]]),
        np.array([[0.2, -0.1], [0.4, 0.0]]),
    )
    return scene, sun, molecular, aerosol


def _full_recomputation(
    fields: tuple[NDArray[np.float64], ...],
    component: RayleighComponent | AerosolComponent,
    point: int,
    subdivisions: int,
    cosines: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    # Previous _scatter loop, separately executed for each field. Use the
    # legacy phase expressions, not the preparation helper under test.
    sources = []
    for field in fields:
        source = np.zeros((field.shape[0], cosines.shape[1]), dtype=np.float64)
        depth = (
            component.optical_depth
            if isinstance(component, RayleighComponent)
            else component.scattering_depth
        )
        for layer, row in enumerate(depth):
            cells = slice(layer * subdivisions, (layer + 1) * subdivisions)
            if row[point] == 0 or not np.any(field[cells]):
                continue
            cosine = np.clip(cosines, -1, 1)
            if isinstance(component, RayleighComponent):
                phase = 0.75 * (1 + cosine**2)
            else:
                assert component.asymmetry is not None
                g = component.asymmetry[layer][point]
                phase = (1 - g**2) / (1 + g**2 - 2 * g * cosine) ** 1.5
            phase = phase * weights[:, None]
            source[cells] = (field[cells] @ phase) * (row[point] / subdivisions)
        sources.append(source)
    return tuple(sources)
