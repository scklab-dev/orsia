"""Physical cell radiance, explicit geometry, stable binding and derivative oracles."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells
from orsia_radiative import LayerCrossSections, LevelAtmosphere, SpectralGrid
from orsia_scene import DirectSolarScene, GasLayerColumns, SpatialOpticalDepth
from orsia_scene.surface import affine_reflectance_weights, affine_surface_reflectance


def physical_inputs(
    *,
    coordinate: Literal["wavelength_nm", "wavenumber_cm1"] = "wavelength_nm",
    physical_mask=None,
) -> tuple[SpatialOpticalDepth, dict[str, Any]]:
    frame = LocalCartesianFrame("physical test", "synthetic datum")
    points = PointLattice.regular((4, 5), (10.0, 10.0), (0.0, 0.0), frame)
    cells = LayeredCells(
        RectangularCells.around_points(points, (10.0, 10.0)),
        np.array([0.0, 20.0, 40.0]),
        np.zeros((4, 5)),
    )
    atmosphere = LevelAtmosphere(
        np.array([0.0, 20.0, 40.0]),
        np.array([1e5, 9e4, 8e4]),
        np.full(3, 280.0),
        np.zeros(3),
        ("CH4", "CO"),
        np.full((3, 2), 1e-6),
    )
    samples = np.array([500.0, 600.0, 700.0])
    solar = np.array([1.0, 2.0, 3.0])
    sigma = np.array([[1e-23, 2e-23, 3e-23], [2e-23, 3e-23, 4e-23]])
    if coordinate == "wavenumber_cm1":
        solar = solar[::-1] * samples[::-1] ** 2 / 1e7
        sigma = sigma[:, ::-1]
        samples = 1e7 / samples[::-1]
    spectral = SpectralGrid(coordinate, samples)
    tables = tuple(
        LayerCrossSections(
            name,
            "all",
            "cross_section",
            spectral,
            np.full(2, 280.0),
            np.array([9.5e4, 8.5e4]),
            sigma * (i + 1),
            height_edges_m=cells.height_edges_m,
        )
        for i, name in enumerate(atmosphere.species)
    )
    optics = SpatialOpticalDepth(
        atmosphere, tables, spectral=spectral, support=cells, physical_mask=physical_mask
    )
    options: dict[str, Any] = dict(
        anchor_wavelengths_nm=(500.0, 700.0),
        surface_reflectance=np.broadcast_to([0.2, 0.4], (4, 5, 2)).copy(),
        beam_normal_irradiance=solar,
        solar_direction=np.array([0.6, 0.0, 0.8]),
        epoch=datetime(2026, 1, 1, tzinfo=UTC),
        validity_s=(0.0, 10.0),
        method="layer_centroid",
        continuation="finite",
        visibility="assume_clear",
    )
    return optics, options


def gas_input(scene, species="CH4"):
    amounts = np.zeros(scene.support.shape)
    amounts[1:3, 1:4, 0] = 1e21
    amounts[1:3, 1:4, 1] = 2e21
    centroids = np.where(amounts > 0, np.array([10.0, 30.0]), 0.0)
    return GasLayerColumns(scene.support, species, amounts, centroids)


def query(scene):
    xy = scene.support.horizontal.centre_xy_m.reshape(-1, 2)
    views = np.broadcast_to([0.0, 0.6, 0.8], (len(xy), 3)).copy()
    return xy, views, np.linspace(0.0, 10.0, len(xy)), scene.spectral


@pytest.mark.parametrize("method", ["layer_centroid", "independent_column"])
def test_transparent_and_background_spectral_formula(method):
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **(options | {"method": method}))
    xy, views, times, spectral = query(scene)
    value = scene.radiance(xy, views, times, spectral)
    depth = optics.spectral_block(0, 3).reshape(-1, 3)
    reflectance = np.array([0.2, 0.3, 0.4])
    expected = options["beam_normal_irradiance"] * reflectance * 0.8 / np.pi * np.exp(-depth * 2.5)
    assert_allclose(value, expected, rtol=3e-15)
    empty = scene.radiance(xy[:0], views[:0], times[:0], spectral)
    assert empty.shape == (0, 3)
    assert scene.epoch == options["epoch"] and scene.validity_s == (0.0, 10.0)
    gas = gas_input(scene)
    bound = scene.with_gas((gas,))
    result = bound.radiance(*query(scene))
    assert np.all(result <= value) and np.any(result < value)
    assert_array_equal(scene.radiance(*query(scene)), value)
    assert_array_equal(bound.with_gas(()).radiance(*query(scene)), value)
    if method == "independent_column":
        excess = gas.columns_molecules_m2.reshape(-1, 2) @ optics.species_cross_sections("CH4")
        assert_allclose(result, expected * np.exp(-excess * 2.5), rtol=3e-15)


def test_outward_view_sign_and_cell_constant_reconstruction():
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **(options | {"solar_direction": np.array([0.0, 0.0, 1.0])}))
    gas = gas_input(scene)
    amount = np.zeros_like(gas.columns_molecules_m2)
    amount[2, 2, 0] = 1e22
    gas = replace(
        gas, columns_molecules_m2=amount, centroid_altitude_m=np.where(amount > 0, 10.0, 0.0)
    )
    bound = scene.with_gas((gas,))
    xy, views, times, spectral = query(scene)
    # Outward +x slope one maps the gas footprint from x=20 to x=10.
    views[:] = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    ratio = bound.radiance(xy, views, times, spectral) / scene.radiance(xy, views, times, spectral)
    sigma = optics.species_cross_sections("CH4")[0]
    expected = np.ones((4, 5, 3))
    expected[2, 1] = np.exp(-1e22 * sigma * np.sqrt(2))
    expected[2, 2] = np.exp(-1e22 * sigma)
    assert_allclose(ratio.reshape(4, 5, 3), expected, rtol=3e-15)
    assert_array_equal(
        bound.radiance(xy + [1.0, 2.0], views, times, spectral),
        bound.radiance(xy, views, times, spectral),
    )


@pytest.mark.parametrize("method", ["layer_centroid", "independent_column"])
def test_multispecies_tangents_with_finite_differences(method):
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **(options | {"method": method}))
    gases = (gas_input(scene), gas_input(scene, "CO"))
    scales = np.array([[1.0, -0.4], [-0.2, 0.7]])
    anchors = np.zeros((4, 5, 2, 2))
    anchors[..., 0, 0], anchors[..., 1, 1] = 0.01, -0.02
    bound = scene.with_gas(gases).with_tangents(scales, anchors)
    xy, views, times, spectrum = query(scene)
    views[::2] = [0.0, 0.0, 1.0]
    value, tangent = bound.radiance_tangent(xy, views, times, spectrum)
    assert_array_equal(value, bound.radiance(xy, views, times, spectrum))
    for direction in range(2):
        values = []
        for step in (-1e-5, 1e-5):
            perturbed = DirectSolarScene(
                optics,
                **(
                    options
                    | {
                        "method": method,
                        "surface_reflectance": options["surface_reflectance"]
                        + step * anchors[..., direction],
                    }
                ),
            ).with_gas(
                tuple(
                    replace(gas, columns_molecules_m2=gas.columns_molecules_m2 * (1 + step * scale))
                    for gas, scale in zip(gases, scales[:, direction], strict=True)
                )
            )
            values.append(perturbed.radiance(xy, views, times, spectrum))
        assert_allclose(
            tangent[..., direction], (values[1] - values[0]) / 2e-5, rtol=2e-7, atol=2e-11
        )
    for indices in (np.array([0]), np.array([1, 2]), np.array([0, 2])):
        selected = SpectralGrid(spectrum.coordinate, spectrum.points[indices])
        v, d = bound.radiance_tangent(xy, views, times, selected)
        # Different spectral widths can change the small matrix-product rounding.
        assert_allclose(v, value[:, indices], rtol=3e-15, atol=2e-18)
        assert_allclose(d, tangent[:, indices], rtol=3e-15, atol=2e-18)
    no_seeds = scene.with_tangents(np.empty((0, 0)), np.empty((4, 5, 2, 0)))
    assert no_seeds.radiance_tangent(*query(scene))[1].shape == (20, 3, 0)
    assert bound.with_tangents(np.empty((2, 0)), np.empty((4, 5, 2, 0))).radiance_tangent(
        xy[:0], views[:0], times[:0], spectrum
    )[1].shape == (0, 3, 0)
    with pytest.raises(ValueError, match="bind"):
        bound.with_gas(gases).radiance_tangent(*query(scene))


def test_ownership_concurrency_and_reciprocal_density():
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **options)
    expected = scene.radiance(*query(scene))
    options["surface_reflectance"][:] = 0
    options["beam_normal_irradiance"][:] = 0
    options["solar_direction"][:] = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: scene.radiance(*query(scene)), range(4)))
    for result in results:
        assert_array_equal(result, expected)
        assert not np.shares_memory(result, expected)
    optics, options = physical_inputs(coordinate="wavenumber_cm1")
    reciprocal = DirectSolarScene(optics, **options)
    assert_allclose(
        reciprocal.radiance(*query(reciprocal)),
        expected[:, ::-1] * np.array([700.0, 600.0, 500.0]) ** 2 / 1e7,
        rtol=3e-15,
    )


@pytest.mark.parametrize(
    "change", ["translated", "resized", "reversed", "frame", "layers", "terrain", "species"]
)
def test_gas_support_failures(change):
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **options)
    gas = gas_input(scene)
    cells = scene.support.horizontal
    if change == "translated":
        support = replace(
            scene.support,
            horizontal=replace(
                cells, lower_xy_m=cells.lower_xy_m + 1, upper_xy_m=cells.upper_xy_m + 1
            ),
        )
    elif change == "resized":
        support = replace(scene.support, horizontal=replace(cells, upper_xy_m=cells.upper_xy_m + 1))
    elif change == "reversed":
        support = replace(
            scene.support,
            horizontal=replace(
                cells, lower_xy_m=cells.lower_xy_m[:, ::-1], upper_xy_m=cells.upper_xy_m[:, ::-1]
            ),
        )
    elif change == "frame":
        support = replace(
            scene.support,
            horizontal=replace(
                cells, frame=LocalCartesianFrame("physical test", "synthetic datum")
            ),
        )
    elif change == "layers":
        support = replace(scene.support, height_edges_m=scene.support.height_edges_m + 1)
    elif change == "terrain":
        support = replace(scene.support, surface_altitude_m=scene.support.surface_altitude_m + 1)
    else:
        support = scene.support
    with pytest.raises(ValueError):
        scene.with_gas(
            (
                replace(
                    gas, support=support, species="unknown" if change == "species" else gas.species
                ),
            )
        )


@pytest.mark.parametrize(
    "override",
    [
        {"method": "unknown"},
        {"continuation": "auto"},
        {"visibility": "unknown"},
        {"minimum_upward_cosine": 0.0},
        {"epoch": datetime(2026, 1, 1)},
        {"validity_s": (2.0, 1.0)},
        {"surface_reflectance": np.zeros((4, 5))},
        {"surface_reflectance": np.full((4, 5, 2), 1.1)},
        {"beam_normal_irradiance": np.full(3, -1.0)},
        {"solar_direction": np.zeros(3)},
        {"solar_direction": np.zeros((1, 3))},
        {"solar_direction": np.array([1.0, 0.0, 0.0])},
        {"anchor_wavelengths_nm": (600.0, 700.0)},
    ],
)
def test_invalid_scene_inputs(override):
    optics, options = physical_inputs()
    with pytest.raises(ValueError):
        DirectSolarScene(optics, **(options | override))


def test_queries_masks_and_binding_errors():
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **options)
    xy, views, times, spectral = query(scene)
    for args in (
        (xy[:, :1], views, times, spectral),
        (xy, views, times + 11, spectral),
        (xy + 100, views, times, spectral),
        (xy, -views, times, spectral),
        (xy, views * 2, times, spectral),
        (xy, views, times, SpectralGrid("wavelength_nm", np.array([550.0]))),
        (xy, views, times, SpectralGrid("wavenumber_cm1", np.array([1e7 / 600.0]))),
    ):
        with pytest.raises(ValueError):
            scene.radiance(*args)
    with pytest.raises(ValueError, match="bind"):
        scene.radiance_tangent(*query(scene))
    with pytest.raises(ValueError, match="directions"):
        scene.with_tangents(np.zeros((1, 2)), np.zeros((4, 5, 2, 2)))
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False
    optics, options = physical_inputs(physical_mask=mask)
    with pytest.raises(ValueError, match="physical support"):
        DirectSolarScene(optics, **options).radiance(*query(scene))
    infinite = DirectSolarScene(optics, **(options | {"continuation": "infinite_uniform"}))
    with pytest.raises(ValueError, match="uniform"):
        infinite.with_gas((gas_input(infinite),))


def test_surface_basis_and_invalid_reflectance():
    spectrum = SpectralGrid("wavelength_nm", np.array([500.0, 600.0, 700.0]))
    assert_array_equal(
        affine_reflectance_weights(spectrum, (500.0, 700.0)), [[1, 0], [0.5, 0.5], [0, 1]]
    )
    assert_allclose(
        affine_surface_reflectance(np.array([0.2, 0.4]), spectrum, (500.0, 700.0)), [0.2, 0.3, 0.4]
    )
    for anchors in ((700.0, 500.0), (0.0, 700.0), (500.0, np.inf)):
        with pytest.raises(ValueError):
            affine_reflectance_weights(spectrum, anchors)
    for values in (np.ones(1), np.array([-0.1, 0.2]), np.array([np.nan, 0.2])):
        with pytest.raises(ValueError):
            affine_surface_reflectance(values, spectrum, (500.0, 700.0))


def test_physical_scene_with_instrument_coefficients_and_full_quadrature_blocks():
    from orsia_instruments.acquisition.quadrature import Acquisition
    from orsia_instruments.acquisition.schedule import raster_read_schedule
    from orsia_instruments.photodetection.charge import ExpectedElectrons
    from orsia_instruments.photodetection.coefficients import (
        OpticalAcceptance,
        photoelectron_response,
    )
    from orsia_radiative import SpectralDomain
    from scipy.constants import c, h

    optics, options = physical_inputs(coordinate="wavenumber_cm1")
    scene = DirectSolarScene(optics, **options)
    spectral = SpectralGrid(
        scene.spectral.coordinate,
        scene.spectral.points,
        SpectralDomain(
            scene.spectral.coordinate, ((scene.spectral.points[0], scene.spectral.points[-1]),)
        ),
    )
    reads = raster_read_schedule(np.array([[0.0, 1.0], [1.0, 2.0]]), (1, 1)).reads
    times = np.array([[0.2, 0.5], [1.2, 1.5]])
    angular_weights = np.array([[0.1, 0.2], [0.2, 0.3]])
    acquisition = Acquisition(
        reads,
        times,
        np.zeros((2, 2, 2)),
        np.zeros((2, 2, 2)),
        angular_weights,
        np.tile([-0.1, 0.1, -0.1, 0.1], (2, 1)),
        np.tile([-0.1, 0.1, -0.1, 0.1], (2, 1)),
    )
    acceptance = OpticalAcceptance(acquisition, 0.01 / spectral.points, np.full((2, 2, 3), 0.5))
    response = photoelectron_response(
        spectral, acceptance, collecting_area_m2=2.0, quantum_efficiency=0.8
    )
    xy = np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]])
    view = np.broadcast_to([0.0, 0.0, 1.0], (4, 3))
    values = scene.radiance(xy, view, times.ravel(), spectral).reshape(2, 2, 3)
    counts = np.einsum("rqk,rqk->r", response, values)
    charge = ExpectedElectrons(reads, {"scene": counts})
    assert charge.reads is reads
    widths = np.diff(spectral.points)
    quadrature = np.array([widths[0] / 2, widths.sum() / 2, widths[1] / 2])
    independent = (
        values
        * angular_weights[..., None]
        * (2.0 * 0.5 * 0.8 * quadrature / (h * c * 100 * spectral.points))
    ).sum(axis=(1, 2))
    assert_allclose(charge.total, independent, rtol=3e-15)
    for block_size in (1, 2, 3):
        accumulated = np.zeros(2)
        for start in range(0, 3, block_size):
            stop = min(3, start + block_size)
            selected = SpectralGrid(spectral.coordinate, spectral.points[start:stop])
            radiance = scene.radiance(xy, view, times.ravel(), selected).reshape(2, 2, -1)
            accumulated += np.einsum("rqk,rqk->r", response[:, :, start:stop], radiance)
        assert_allclose(accumulated, counts, rtol=3e-15)


def test_additional_batched_optical_support_and_scene_geometry_failures():
    optics, options = physical_inputs()
    for pixels in (np.zeros((1, 1), dtype=np.int64), np.array([[0, -1]]), np.array([[4, 0]])):
        with pytest.raises(ValueError):
            optics.at_pixels(pixels, optics.spectral)
    with pytest.raises(ValueError, match="exact"):
        optics.spectral_indices(SpectralGrid("wavelength_nm", np.array([800.0])))
    pixels = np.array([[0, 0], [2, 3]], dtype=np.int64)
    species = optics.at_pixels(pixels, optics.spectral, species="CH4")
    assert_array_equal(species[0], optics.at_pixel((0, 0), species="CH4"))
    cells = optics.support.horizontal
    nonregular = replace(optics.support, horizontal=replace(cells, upper_xy_m=cells.upper_xy_m + 1))
    changed = SpatialOpticalDepth(
        optics.atmosphere, (), spectral=optics.spectral, support=nonregular
    )
    with pytest.raises(ValueError, match="regular"):
        DirectSolarScene(changed, **options)
    one_dimensional = LayeredCells(
        RectangularCells(
            cells.lower_xy_m.reshape(-1, 2), cells.upper_xy_m.reshape(-1, 2), cells.frame
        ),
        optics.support.height_edges_m,
        np.zeros(20),
    )
    changed = SpatialOpticalDepth(
        optics.atmosphere, (), spectral=optics.spectral, support=one_dimensional
    )
    with pytest.raises(ValueError, match="regular"):
        DirectSolarScene(changed, **options)


def test_water_excess_requires_updated_continuum():
    from orsia_radiative import WaterContinuumCoefficients

    optics, options = physical_inputs()
    continuum = WaterContinuumCoefficients(
        optics.spectral,
        optics.support.height_edges_m,
        np.full(2, 280.0),
        np.array([9.5e4, 8.5e4]),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        "mt_ckd_local_25",
    )
    atmosphere = replace(
        optics.atmosphere,
        species=(*optics.atmosphere.species, "H2O"),
        species_vmr=np.column_stack((optics.atmosphere.species_vmr, np.zeros(3))),
    )
    wet = SpatialOpticalDepth(
        atmosphere, (), spectral=optics.spectral, support=optics.support, continuum=continuum
    )
    scene = DirectSolarScene(wet, **options)
    with pytest.raises(ValueError, match="updated continuum"):
        scene.with_gas((replace(gas_input(scene), species="H2O"),))


def test_uniform_continuation_and_tangent_snapshot_ownership():
    optics, options = physical_inputs()
    infinite = DirectSolarScene(optics, **(options | {"continuation": "infinite_uniform"}))
    amounts = np.full(infinite.support.shape, 1e21)
    heights = np.broadcast_to([10.0, 30.0], amounts.shape).copy()
    gas = GasLayerColumns(infinite.support, "CH4", amounts, heights)
    bound = infinite.with_gas((gas,))
    independent = DirectSolarScene(optics, **(options | {"method": "independent_column"})).with_gas(
        (gas,)
    )
    assert_allclose(bound.radiance(*query(bound)), independent.radiance(*query(bound)), rtol=3e-15)
    gas_seed = np.ones((1, 1))
    surface_seed = np.zeros((4, 5, 2, 1))
    tangent_scene = bound.with_tangents(gas_seed, surface_seed)
    expected = tangent_scene.radiance_tangent(*query(bound))
    gas_seed[:] = 99
    surface_seed[:] = 99
    actual = tangent_scene.radiance_tangent(*query(bound))
    assert_array_equal(actual[0], expected[0])
    assert_array_equal(actual[1], expected[1])
    assert not np.shares_memory(actual[1], expected[1])


def test_physical_scene_guide_uses_public_contracts():
    import re
    from pathlib import Path

    guide = Path(__file__).parents[1] / "docs/direct-solar.md"
    blocks = re.findall(r"^```python\n(.*?)^```", guide.read_text(), re.M | re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)


@pytest.mark.parametrize("method", ["layer_centroid", "independent_column"])
def test_null_gas_amplitude_retains_a_nonzero_physical_direction(method):
    optics, options = physical_inputs()
    scene = DirectSolarScene(optics, **(options | {"method": method}))
    gas = gas_input(scene)
    scales = np.zeros(1)
    null = scene.with_gas((gas,), scales=scales).with_tangents(
        np.ones((1, 1)), np.zeros((4, 5, 2, 1))
    )
    values, derivative = null.radiance_tangent(*query(scene))
    assert_array_equal(values, scene.radiance(*query(scene)))
    assert np.any(derivative < 0)
    h = 1e-3
    full = scene.with_gas((gas,), scales=np.array([h])).radiance(*query(scene))
    half = scene.with_gas((gas,), scales=np.array([h / 2])).radiance(*query(scene))
    oracle = (4 * (half - values) - (full - values)) / h
    assert_allclose(derivative[..., 0], oracle, rtol=2e-7, atol=2e-11)
    scales[:] = 9
    assert_array_equal(null.radiance(*query(scene)), values)
    for bad in (np.ones(2), np.array([-1.0]), np.array([np.nan])):
        with pytest.raises(ValueError, match="scales"):
            scene.with_gas((gas,), scales=bad)
