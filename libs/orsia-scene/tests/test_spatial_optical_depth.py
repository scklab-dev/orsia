"""Species-generic preparation, exact support and settled explicit parallel work."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import sleep
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells
from orsia_radiative import (
    LayerCrossSections,
    LevelAtmosphere,
    SpectralGrid,
    WaterContinuumCoefficients,
    column_absorption_optical_depth,
    integrate_log_linear_columns,
)
from orsia_scene import SpatialOpticalDepth


def inputs():
    z = np.array([0.0, 100.0, 200.0, 300.0])
    atmosphere = LevelAtmosphere(
        z,
        1e5 * np.exp(-z / 8000),
        np.full(4, 280.0),
        np.full(4, 0.01),
        ("CH4",),
        np.full((4, 1), 1e-6),
    )
    spectral = SpectralGrid("wavelength_nm", np.array([1600.0, 1601.0]))
    table = LayerCrossSections(
        "CH4",
        "all",
        "cross_section",
        spectral,
        np.full(3, 280.0),
        atmosphere.level_pressure_pa[:-1],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) * 1e-25,
        height_edges_m=z,
    )
    points = PointLattice.regular(
        (2, 2), (50.0, 50.0), (0.0, 0.0), LocalCartesianFrame("optics test", "synthetic altitude")
    )
    cells = LayeredCells(
        RectangularCells.around_points(points, (50.0, 50.0)),
        z,
        np.array([[0.0, 50.0], [150.0, 100.0]]),
    )
    return atmosphere, spectral, table, cells


def test_no_co2_material_and_optional_water_contribution():
    atmosphere, spectral, table, cells = inputs()
    continuum = WaterContinuumCoefficients(
        spectral,
        cells.height_edges_m,
        table.temperature_k,
        table.pressure_pa,
        np.full((3, 2), 1e-54),
        np.full((3, 2), 1e-55),
        "mt_ckd_local_25",
    )
    prepared = SpatialOpticalDepth(
        atmosphere, (table,), spectral=spectral, support=cells, continuum=continuum
    )
    assert prepared.support is cells and prepared.atmosphere is atmosphere
    assert prepared.spectral is spectral and prepared.continuum is continuum
    for pixel in np.ndindex(cells.horizontal.shape):
        columns = integrate_log_linear_columns(
            atmosphere, surface_altitude_m=float(cells.surface_altitude_m[pixel])
        )
        expected = column_absorption_optical_depth(
            columns, (table,), spectral=spectral, continuum=continuum
        )
        water = column_absorption_optical_depth(columns, (), spectral=spectral, continuum=continuum)
        assert_allclose(prepared.at_pixel(pixel), expected, rtol=2e-15)
        assert_allclose(prepared.at_pixel(pixel, species="H2O"), water, rtol=2e-15)
        assert_allclose(prepared.at_pixel(pixel, species="CH4") + water, expected, rtol=2e-15)
    # Use the level integrator's independent exponential-column tolerance;
    # separate logarithmic means need not preserve a scalar ratio bitwise.
    assert_allclose(prepared.column_mixing_ratio("CH4"), 1e-6, rtol=2e-12)
    assert_array_equal(prepared.species_cross_sections("CH4"), table.cross_sections_m2_per_molecule)
    with pytest.raises(ValueError, match="undeclared"):
        prepared.at_pixel((0, 0), species="CO2")
    with pytest.raises(ValueError, match="undeclared"):
        prepared.species_cross_sections("CO2")


def test_blocking_masks_owned_extinction_and_species_permutations():
    atmosphere, spectral, table, cells = inputs()
    second = replace(
        table,
        species="CO2",
        cross_sections_m2_per_molecule=table.cross_sections_m2_per_molecule * 2,
    )
    state = replace(
        atmosphere,
        species=("CH4", "CO2"),
        species_vmr=np.column_stack((atmosphere.species_vmr[:, 0], np.full(4, 400e-6))),
    )
    amplitudes, shapes = np.full((2, 2, 1), 0.2), np.array([[1.0, 2.0]])
    mask = np.array([[True, False], [True, True]])
    prep = SpatialOpticalDepth(
        state,
        (table, second),
        spectral=spectral,
        support=cells,
        extinction_amplitudes=amplitudes,
        extinction_shapes=shapes,
        physical_mask=mask,
    )
    permuted = SpatialOpticalDepth(
        replace(state, species=state.species[::-1], species_vmr=state.species_vmr[:, ::-1]),
        (second, table),
        spectral=spectral,
        support=cells,
        extinction_amplitudes=amplitudes,
        extinction_shapes=shapes,
        physical_mask=mask,
    )
    total = prep.spectral_block(0, 2)
    assert_allclose(total, permuted.spectral_block(0, 2), rtol=2e-15, equal_nan=True)
    assert_allclose(
        total,
        np.concatenate((prep.spectral_block(0, 1), prep.spectral_block(1, 2)), axis=-1),
        rtol=2e-15,
        equal_nan=True,
    )
    amplitudes.fill(99)
    shapes.fill(99)
    mask[:] = True
    assert_allclose(prep.spectral_block(0, 2), total, equal_nan=True)
    assert np.isnan(prep.column_mixing_ratio("CH4")[0, 1])
    with pytest.raises(ValueError, match="admitted"):
        prep.at_pixel((0, 1))
    for pixel in ((-1, 0), (2, 0), (0,)):
        with pytest.raises(ValueError, match="outside spatial"):
            prep.at_pixel(pixel)
    for start, stop in ((-1, 1), (0, 3), (1, 1)):
        with pytest.raises(ValueError, match="block"):
            prep.spectral_block(start, stop)
    for values in (prep.physical_mask, prep.dry_air_columns_molecules_m2):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_invalid_preparation_and_empty_absorption():
    atmosphere, spectral, table, cells = inputs()
    options: dict[str, Any]
    for options in (
        {"support": replace(cells, height_edges_m=cells.height_edges_m + 1)},
        {"extinction_amplitudes": np.ones((2, 2, 1))},
        {"extinction_amplitudes": -np.ones((2, 2, 1)), "extinction_shapes": np.ones((1, 2))},
        {"extinction_amplitudes": np.ones((2, 2, 1)), "extinction_shapes": np.ones(2)},
        {"physical_mask": np.ones((2, 2))},
        {"physical_mask": np.ones((1, 2), bool)},
        {"batch_size": 0},
    ):
        with pytest.raises(ValueError):
            arguments: dict[str, Any] = dict(spectral=spectral, support=cells) | options
            SpatialOpticalDepth(atmosphere, (table,), **arguments)
    empty_state = replace(atmosphere, species=(), species_vmr=np.empty((4, 0)))
    empty = SpatialOpticalDepth(empty_state, (), spectral=spectral, support=cells)
    assert_array_equal(empty.spectral_block(0, 2), 0)
    transparent = SpatialOpticalDepth(atmosphere, (), spectral=spectral, support=cells)
    assert_array_equal(transparent.species_cross_sections("CH4"), 0)
    thin = replace(
        atmosphere,
        level_pressure_pa=atmosphere.level_pressure_pa * 1e-300,
        level_temperature_k=np.full(4, 1e300),
    )
    no_air = SpatialOpticalDepth(thin, (), spectral=spectral, support=cells)
    with pytest.raises(ValueError, match="positive retained"):
        no_air.column_mixing_ratio("CH4")
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="overflow"):
        SpatialOpticalDepth(
            atmosphere,
            (),
            spectral=spectral,
            support=cells,
            extinction_amplitudes=np.full((2, 2, 1), 1e300),
            extinction_shapes=np.full((1, 2), 1e300),
        ).at_pixel((0, 0))


def test_parallel_failure_settles_submitted_work(monkeypatch):
    import orsia_scene.optical_depth as owner

    atmosphere, spectral, table, cells = inputs()
    original = owner.integrate_log_linear_columns
    started = Event()

    def controlled(atmosphere, *, surface_altitude_m):
        if surface_altitude_m == 150:
            started.set()
            sleep(0.05)
        if surface_altitude_m == 50:
            assert started.wait(1)
            raise ValueError("selected failed column")
        return original(atmosphere, surface_altitude_m=surface_altitude_m)

    monkeypatch.setattr(owner, "integrate_log_linear_columns", controlled)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = []
        submit = pool.submit

        def record(*args, **kwargs):
            future = submit(*args, **kwargs)
            submitted.append(future)
            return future

        monkeypatch.setattr(pool, "submit", record)
        with pytest.raises(ValueError, match="selected failed"):
            SpatialOpticalDepth(
                atmosphere, (table,), spectral=spectral, support=cells, executor=pool, batch_size=1
            )
        assert submitted and all(future.done() for future in submitted)


def test_species_water_pairing_is_not_inferred_from_table_name():
    atmosphere, spectral, table, cells = inputs()
    state = replace(atmosphere, species=("H2O",))
    water_table = replace(table, species="H2O")
    continuum = WaterContinuumCoefficients(
        spectral,
        cells.height_edges_m,
        table.temperature_k,
        table.pressure_pa,
        np.ones((3, 2)) * 1e-54,
        np.ones((3, 2)) * 1e-55,
        "mt_ckd_local_25",
    )
    for include in (None, True):
        with pytest.raises(ValueError, match="explicitly exclude"):
            SpatialOpticalDepth(
                state,
                (replace(water_table, includes_water_continuum=include),),
                spectral=spectral,
                support=cells,
                continuum=continuum,
            )
    with pytest.raises(ValueError, match="matching local line"):
        SpatialOpticalDepth(
            state,
            (replace(water_table, includes_water_continuum=False),),
            spectral=spectral,
            support=cells,
            continuum=continuum,
        )
    paired = replace(
        water_table, includes_water_continuum=False, line_wing_convention="mt_ckd_local_25"
    )
    prep = SpatialOpticalDepth(
        state, (paired,), spectral=spectral, support=cells, continuum=continuum
    )
    assert_allclose(prep.at_pixel((0, 0)), prep.at_pixel((0, 0), species="H2O"))
    with pytest.raises(ValueError, match="conditions"):
        SpatialOpticalDepth(
            state,
            (paired,),
            spectral=spectral,
            support=cells,
            continuum=replace(continuum, temperature_k=continuum.temperature_k + 1),
        )
