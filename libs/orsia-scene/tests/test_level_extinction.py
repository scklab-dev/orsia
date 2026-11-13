"""Bounded original-method regression and repeated-render ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_dispersion import GaussianPuffField, GaussianPuffs, LayerMassMoments, pixel_layer_moments
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells
from orsia_instruments.response.spatial import apply_spatial_response, spatial_response_support
from orsia_radiative import (
    LayerCrossSections,
    LevelAtmosphere,
    SpectralGrid,
    WaterContinuumCoefficients,
    column_absorption_optical_depth,
    integrate_log_linear_columns,
)
from orsia_scene import GasLayerColumns, gas_layer_columns
from orsia_scene.optical_depth import SpatialOpticalDepth
from orsia_scene.projection import project_layer_columns_by_centroid
from reference_log_tangent import ReferenceLogTangentRenderer


def fixture() -> tuple[LevelAtmosphere, dict[str, Any]]:
    with np.load(Path(__file__).parent / "fixtures/level_extinction_reference.npz") as source:
        arrays = {k: source[k] for k in source.files}
    values = {k.removeprefix("column_"): v for k, v in arrays.items() if k.startswith("column_")}
    values["species_vmr"] = values["species_vmr"].T
    atmosphere = LevelAtmosphere(**values, species=tuple(str(v) for v in arrays["species"]))
    return atmosphere, arrays


def coefficients(atmosphere, arrays):
    # The retained file is a numerical reference, without coefficient-condition provenance.
    # These lower-endpoint T/p labels are explicit synthetic test declarations, not recovered
    # scientific evaluation conditions. Production readers require actual supplied conditions.
    spectral = SpectralGrid("wavelength_nm", arrays["wavelength_nm"])
    tables = tuple(
        LayerCrossSections(
            name,
            "all",
            "cross_section",
            spectral,
            atmosphere.level_temperature_k[:-1],
            atmosphere.level_pressure_pa[:-1],
            sigma,
            height_edges_m=atmosphere.level_altitude_m,
            line_wing_convention="mt_ckd_local_25" if name == "H2O" else None,
            includes_water_continuum=False if name == "H2O" else None,
        )
        for name, sigma in zip(
            atmosphere.species, arrays["cross_sections_m2_molecule"], strict=True
        )
    )
    continuum = WaterContinuumCoefficients(
        spectral,
        atmosphere.level_altitude_m,
        atmosphere.level_temperature_k[:-1],
        atmosphere.level_pressure_pa[:-1],
        arrays["self_coefficients_m5_molecule2"],
        arrays["foreign_coefficients_m5_molecule2"],
        "mt_ckd_local_25",
    )
    return spectral, tables, continuum


FRAME = LocalCartesianFrame("level scene test", "synthetic altitude")


def cell_support(atmosphere, arrays):
    spacing = (float(arrays["spacing"][1]), float(arrays["spacing"][0]))
    points = PointLattice.regular(arrays["terrain"].shape, spacing, (0.0, 0.0), FRAME)
    return LayeredCells(
        RectangularCells.around_points(points, spacing),
        atmosphere.level_altitude_m,
        arrays["terrain"],
    )


def optical_preparation(atmosphere, arrays, **options):
    spectral, tables, continuum = coefficients(atmosphere, arrays)
    shapes = np.array(
        [
            np.interp(spectral.points, arrays["anchors"], row)
            for row in np.eye(len(arrays["anchors"]))
        ]
    )
    shapes = np.concatenate((shapes, arrays["industrial_extinction"][None, :]))
    amplitudes = np.concatenate((arrays["background"], arrays["industrial"][..., None]), axis=-1)
    return SpatialOpticalDepth(
        atmosphere,
        tables,
        spectral=spectral,
        support=cell_support(atmosphere, arrays),
        continuum=continuum,
        extinction_amplitudes=amplitudes,
        extinction_shapes=shapes,
        **options,
    )


def preparation(atmosphere: LevelAtmosphere, a: dict[str, Any]) -> ReferenceLogTangentRenderer:
    optical_depth = optical_preparation(atmosphere, a)
    spectral, tables, continuum = coefficients(atmosphere, a)
    base = integrate_log_linear_columns(
        atmosphere, surface_altitude_m=float(atmosphere.level_altitude_m[0])
    )
    reference = column_absorption_optical_depth(
        base, tables, spectral=spectral, continuum=continuum
    )
    reference += np.interp(spectral.points, a["anchors"], a["background"][0, 0])
    return ReferenceLogTangentRenderer(
        optical_depth,
        reference_optical_depth=reference,
        support=a["support"],
        valid_mask=a["valid_mask"],
        solar=a["solar"],
        lower_nm=float(a["lower_nm"]),
        upper_nm=float(a["upper_nm"]),
        b11=a["b11"],
        b12=a["b12"],
        ranges=[a[f"range_{i}"] for i in range(len(a["kernel_0"]))],
        views=a["views"],
        azimuth_deg=a["azimuth_deg"],
        solar_angle=float(a["solar_angle"]),
        kernels=(a["kernel_0"], a["kernel_1"], a["kernel_2"]),
        mtf=float(a["mtf"]),
        pixel_integration=bool(a["pixel_integration"]),
    )


@pytest.mark.parametrize(
    "label,plume,azimuth",
    [("reference", False, None), ("plume", True, None), ("two_path", True, 20.0)],
)
def test_render_matches_retained_pretransfer_method(
    label: str, plume: bool, azimuth: float | None
) -> None:
    atmosphere, arrays = fixture()
    scene = preparation(atmosphere, arrays)
    assert arrays["plume_plume_layers_mol_m2"].min() > -1e-12
    columns = np.moveaxis(np.maximum(arrays["plume_plume_layers_mol_m2"], 0), 0, -1)
    heights = np.moveaxis(arrays["plume_plume_heights_agl_m"], 0, -1)
    gas = GasLayerColumns(
        scene.cells,
        "CO2",
        columns * 6.02214076e23,
        np.where(columns > 0, heights + arrays["terrain"][..., None], 0),
    )
    result = scene.render(
        continuation="finite", gas_layers=gas if plume else None, solar_azimuth_deg=azimuth
    )
    for name, value in result.items():
        key = "plume_layers_mol_m2" if name == "plume_layers_molecules_m2" else name
        expected = arrays[label + "_" + key]
        actual = np.asarray(value)
        if name == "plume_layers_molecules_m2":
            # Keep the historical absolute tolerance in its original mol/m2 units.
            # The fixture contains tiny negative rounding residuals clipped on input.
            actual = actual / 6.02214076e23
        if name in (
            "centre_wavelength_sensitivity_electrons_per_nm",
            "filter_fwhm_sensitivity_electrons_per_nm",
        ):
            # Positive-basis reordering changes final rounding. Signed response
            # cancellations need an amplitude-scaled absolute allowance near zero;
            # this is a bounded numerical parity criterion, not physical uncertainty.
            scale = np.max(np.abs(expected), axis=(0, 1), keepdims=True)
            allowance = 1e-12 + 16 * np.finfo(float).eps * scale + 2e-14 * np.abs(expected)
            assert np.all(np.abs(actual - expected) <= allowance)
        else:
            assert_allclose(actual, expected, rtol=2e-14, atol=1e-12)
    with ThreadPoolExecutor(max_workers=2) as executor:
        again = scene.render(
            continuation="finite",
            gas_layers=gas if plume else None,
            solar_azimuth_deg=azimuth,
            executor=executor,
        )
    for (name, value), (other_name, fresh) in zip(result.items(), again.items(), strict=True):
        assert name == other_name
        assert_array_equal(fresh, value)
        assert not np.shares_memory(fresh, value)
    result["signal_electrons"][:] = -1
    assert np.all(again["signal_electrons"] >= 0)
    with pytest.raises(ValueError, match="spectral_block_size"):
        scene.render(continuation="finite", spectral_block_size=0)


def test_expansion_cube_equivalence_and_parallel_rows() -> None:
    atmosphere, a = fixture()
    spectral, tables, continuum = coefficients(atmosphere, a)
    with ThreadPoolExecutor(max_workers=2) as executor:
        expansion = optical_preparation(atmosphere, a, executor=executor, batch_size=13)
    serial = optical_preparation(atmosphere, a)
    total = expansion.spectral_block(0, spectral.points.size)
    co2 = expansion.spectral_block(0, spectral.points.size, species="CO2")
    baseline = expansion.column_mixing_ratio("CO2") * 1e6
    sigma_co2 = expansion.species_cross_sections("CO2")
    for pixel in [(0, 0), (8, 9), (15, 16)]:
        path = integrate_log_linear_columns(
            atmosphere, surface_altitude_m=float(a["terrain"][pixel])
        )
        aerosol = (
            np.interp(spectral.points, a["anchors"], a["background"][pixel])
            + a["industrial"][pixel] * a["industrial_extinction"]
        )
        expected = (
            column_absorption_optical_depth(path, tables, spectral=spectral, continuum=continuum)
            + aerosol
        )
        expected_co2 = (
            path.layer_species_column_molecules_m2[:, atmosphere.species.index("CO2")] @ sigma_co2
        )
        assert_allclose(expansion.at_pixel(pixel), expected, rtol=2e-15, atol=1e-16)
        assert_allclose(
            expansion.at_pixel(pixel, species="CO2"), expected_co2, rtol=2e-15, atol=1e-16
        )
        assert_array_equal(expansion.at_pixel(pixel), total[pixel])
        assert_array_equal(expansion.at_pixel(pixel, species="CO2"), co2[pixel])
        assert_allclose(
            baseline[pixel],
            1e6
            * path.layer_species_column_molecules_m2[:, atmosphere.species.index("CO2")].sum()
            / path.layer_dry_air_column_molecules_m2.sum(),
        )
    assert_array_equal(total, serial.spectral_block(0, spectral.points.size))
    assert np.all(expansion.dry_air_columns_molecules_m2 > 0)
    with pytest.raises(ValueError, match="batch_size"):
        optical_preparation(atmosphere, a, batch_size=0)
    negative = a | {"background": -a["background"]}
    with pytest.raises(ValueError, match="extinction"):
        optical_preparation(atmosphere, negative)


def test_projection_support_bounds_and_changed_absorption() -> None:
    atmosphere, a = fixture()
    scene = preparation(atmosphere, a)
    changed = a | {"cross_sections_m2_molecule": a["cross_sections_m2_molecule"] * 1.01}
    other = preparation(atmosphere, changed)
    assert not np.array_equal(
        scene.render(
            continuation="finite",
        )["signal_electrons"],
        other.render(
            continuation="finite",
        )["signal_electrons"],
    )
    amounts = np.ones((2, 5, 6))
    assert_array_equal(
        project_layer_columns_by_centroid(
            amounts, amounts, (3, 4), (1, 1), continuation="infinite_uniform"
        ),
        amounts,
    )
    with pytest.raises(ValueError):
        project_layer_columns_by_centroid(
            amounts, amounts[0], (0, 0), (1, 1), continuation="finite"
        )
    assert np.isfinite(apply_spatial_response(amounts[0], upper_bound=True)).all()
    with pytest.raises(ValueError):
        apply_spatial_response(np.ones(3))
    with pytest.raises(ValueError):
        spatial_response_support(np.ones((3, 3)))


def test_conversion_units_ownership_and_empty_layers():
    support = LayeredCells(
        RectangularCells(np.zeros((1, 1, 2)), np.ones((1, 1, 2)), FRAME),
        np.array([0.0, 10.0, 20.0]),
        np.array([[3.0]]),
    )
    mass = LayerMassMoments(support, np.array([[[2.0, 0.0]]]), np.array([[[10.0, 0.0]]]))
    gas = gas_layer_columns(mass, species="CO2", molar_mass_kg_mol=0.044)
    assert_allclose(gas.columns_molecules_m2, mass.columns_kg_m2 / 0.044 * 6.02214076e23)
    assert_array_equal(gas.centroid_altitude_m, [[[5.0, 0.0]]])
    assert not gas.columns_molecules_m2.flags.writeable
    assert gas.support is mass.support is support
    with pytest.raises(ValueError):
        gas_layer_columns(mass, species="CO2", molar_mass_kg_mol=0)
    with pytest.raises(ValueError):
        replace(gas, species="")
    with pytest.raises(ValueError):
        replace(gas, columns_molecules_m2=-gas.columns_molecules_m2)
    with pytest.raises(ValueError):
        replace(gas, centroid_altitude_m=np.ones_like(gas.centroid_altitude_m))
    with pytest.raises(ValueError):
        replace(gas, support=replace(gas.support, height_edges_m=np.array([0.0, 0.0, 20.0])))


def test_renderer_updates_only_layers_and_checks_support():
    atmosphere, a = fixture()
    renderer = preparation(atmosphere, a)
    state = GaussianPuffs(
        0.0,
        np.zeros(1),
        np.array([1000.0]),
        np.array([[500.0, 400.0, 200.0]]),
        np.array([np.diag([10000.0, 10000.0, 2500.0])]),
    )
    mass = pixel_layer_moments(
        GaussianPuffField(state, "free"),
        renderer.cells,
    )
    gas = gas_layer_columns(mass, species="CO2", molar_mass_kg_mol=0.0440095)
    baseline = renderer.render(
        continuation="finite",
    )
    first = renderer.render(continuation="finite", gas_layers=gas)
    second = renderer.render(
        continuation="finite",
        gas_layers=replace(gas, columns_molecules_m2=gas.columns_molecules_m2 * 2),
    )
    assert np.max(np.abs(first["signal_electrons"] - baseline["signal_electrons"])) > 0
    assert np.max(np.abs(first["signal_electrons"] - second["signal_electrons"])) > 0
    assert_array_equal(
        first["plume_layers_molecules_m2"], np.moveaxis(gas.columns_molecules_m2, -1, 0)
    )
    assert_array_equal(
        renderer.render(
            continuation="finite",
        )["signal_electrons"],
        baseline["signal_electrons"],
    )
    assert not np.shares_memory(first["plume_layers_molecules_m2"], gas.columns_molecules_m2)
    with pytest.raises(ValueError, match="match prepared"):
        renderer.render(
            continuation="finite",
            gas_layers=replace(
                gas,
                support=replace(gas.support, surface_altitude_m=gas.support.surface_altitude_m - 1),
            ),
        )
    with pytest.raises(ValueError, match="match prepared"):
        renderer.render(
            continuation="finite",
            gas_layers=replace(
                gas, support=replace(gas.support, height_edges_m=gas.support.height_edges_m + 1)
            ),
        )
    with pytest.raises(ValueError):
        renderer.render(continuation="finite", gas_layers=replace(gas, species="unknown"))
    # The core renderer selects supplied species; the CO2 application enforces CO2 separately.
    methane = renderer.render(continuation="finite", gas_layers=replace(gas, species="CH4"))
    assert not np.array_equal(first["signal_electrons"], methane["signal_electrons"])


@pytest.mark.parametrize("change", ["translated", "resized", "reversed", "frame"])
def test_renderer_rejects_equal_shape_different_horizontal_support(change):
    atmosphere, arrays = fixture()
    renderer = preparation(atmosphere, arrays)
    cells = renderer.cells.horizontal
    if change == "translated":
        other = replace(
            cells,
            lower_xy_m=cells.lower_xy_m + [1.0, 0.0],
            upper_xy_m=cells.upper_xy_m + [1.0, 0.0],
        )
    elif change == "resized":
        other = replace(cells, upper_xy_m=cells.upper_xy_m + [1.0, 0.0])
    elif change == "reversed":
        other = replace(
            cells, lower_xy_m=cells.lower_xy_m[:, ::-1], upper_xy_m=cells.upper_xy_m[:, ::-1]
        )
    else:
        other = replace(cells, frame=LocalCartesianFrame(FRAME.name, FRAME.vertical_datum))
    support = replace(renderer.cells, horizontal=other)
    gas = GasLayerColumns(support, "CH4", np.zeros(support.shape), np.zeros(support.shape))
    with pytest.raises(ValueError, match="horizontal cells, frame"):
        renderer.render(continuation="finite", gas_layers=gas)
    admitted = replace(gas, support=renderer.cells)
    assert_array_equal(
        renderer.render(continuation="finite", gas_layers=admitted)["signal_electrons"],
        renderer.render(
            continuation="finite",
        )["signal_electrons"],
    )
