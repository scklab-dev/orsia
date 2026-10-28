"""Bounded arithmetic/support checks, not external MT_CKD qualification."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from orsia_radiative import (
    Absorber,
    Atmosphere,
    Line,
    LineSet,
    WaterContinuumReference,
    compose_absorption_on_grid,
    line_absorption,
    water_continuum_absorption,
    wavenumber_grid,
)
from scipy.constants import c, h, k


def atmosphere() -> Atmosphere:
    return Atmosphere(
        np.array([0.0, 1000.0, 2000.0]),
        np.array([100000.0, 90000.0, 80000.0]),
        np.array([95000.0, 85000.0]),
        np.array([296.0, 280.0]),
        np.array([2e28, 1e28]),
        np.array([1.9e28, 0.95e28]),
        np.array([1e27, 5e26]),
        (Absorber("H2O", np.array([1e27, 5e26]), "dry_air"),),
    )


def reference() -> WaterContinuumReference:
    return WaterContinuumReference(
        np.arange(800.0, 1101.0, 50.0),
        np.full(7, 1e-24),
        np.full(7, 2e-24),
        np.full(7, 3.0),
        101300.0,
        296.0,
        (250.0, 320.0),
    )


def test_reference_node_density_units_and_temperature_scaling() -> None:
    air, ref = atmosphere(), reference()
    grid = np.array([900.0, 1000.0])
    result = water_continuum_absorption(air, grid, ref)
    density = 101300.0 / (k * 296.0) / 1e6
    expected = np.empty((2, 2))
    for layer, t in enumerate(air.temperature_k):
        water = air.water_vapour_columns_molecules_m2[layer]
        dry = air.dry_air_columns_molecules_m2[layer]
        radiation = grid * np.tanh((100 * h * c / k) * grid / (2 * t))
        expected[layer] = (
            radiation
            * 1e-10
            / density
            / 1000
            * (1e-24 * (296 / t) ** 3 * water**2 + 2e-24 * water * dry)
        )
    np.testing.assert_allclose(result.optical_depth, expected, rtol=2e-15)
    assert result.line_wing_convention == "mt_ckd_local_25"
    assert result.optical_depth.flags.writeable
    np.testing.assert_equal(asdict(reference()), asdict(ref))
    changed = replace(ref, reference_temperature_k=300.0)
    assert changed.reference_temperature_k == 300.0
    assert ref.reference_temperature_k == 296.0


def test_negative_reference_padding_is_not_negative_evaluation_support() -> None:
    ref = replace(reference(), wavenumbers_cm1=np.arange(-20.0, 41.0, 10.0))
    result = water_continuum_absorption(atmosphere(), np.array([10.0, 20.0]), ref)
    assert np.all(result.optical_depth > 0)
    with pytest.raises(ValueError):
        water_continuum_absorption(atmosphere(), np.array([-10.0, 0.0]), ref)


def test_scaling_before_interpolation_and_full_stencil() -> None:
    ref = replace(reference(), self_temperature_exponent=np.arange(7, dtype=np.float64))
    air = atmosphere()
    grid = np.array([925.0])
    result = water_continuum_absorption(air, grid, ref)
    # Independent half-cell cubic weights for reference nodes 850,900,950,1000.
    weights = np.array([-1.0, 9.0, 9.0, -1.0]) / 16
    nodes = ref.wavenumbers_cm1[1:5]
    t = 280.0
    radiation = nodes * np.tanh((100 * h * c / k) * nodes / (2 * t))
    self_coeff = float(weights @ (1e-24 * (296 / t) ** np.arange(1.0, 5.0) * radiation))
    foreign_coeff = float(weights @ (2e-24 * radiation))
    expected = (
        1e-10
        / (101300 / (k * 296) / 1e6)
        * (self_coeff * (5e26) ** 2 / 1000 + foreign_coeff * 5e26 * 0.95e28 / 1000)
    )
    assert result.optical_depth[1, 0] == pytest.approx(expected, rel=2e-15)
    for point in (800.0, 1100.0):
        with pytest.raises(ValueError, match="stencil"):
            water_continuum_absorption(air, np.array([point]), ref)
    water_continuum_absorption(air, np.array([850.0, 1050.0]), ref)


def test_water_line_local_pedestal_and_pairing() -> None:
    air = atmosphere()
    inputs = LineSet(
        "H2O",
        "1",
        (Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7),),
        np.ones(2),
        (250.0, 320.0),
        (80000.0, 100000.0),
    )
    grid = np.array([974.0, 975.0, 999.0, 1000.0, 1001.0, 1025.0, 1026.0])
    unbounded = line_absorption(air, grid, inputs)
    truncated = line_absorption(air, grid, replace(inputs, wing_cutoff_cm1=25.0))
    local = line_absorption(
        air, grid, replace(inputs, wing_cutoff_cm1=25.0, subtract_lorentz_pedestal=True)
    )
    assert np.all(truncated.optical_depth[:, [0, -1]] == 0)
    assert np.all(local.optical_depth[:, [0, 1, -2, -1]] == 0)
    assert np.all(local.optical_depth[:, 2:5] < truncated.optical_depth[:, 2:5])
    np.testing.assert_array_equal(unbounded.optical_depth[:, 2:5], truncated.optical_depth[:, 2:5])
    continuum = water_continuum_absorption(air, grid, reference())
    compose_absorption_on_grid(air, wavenumber_grid(grid), (local, continuum))
    for incompatible in (unbounded, truncated):
        with pytest.raises(ValueError, match="pedestal"):
            compose_absorption_on_grid(air, wavenumber_grid(grid), (incompatible, continuum))
    with pytest.raises(ValueError, match="25"):
        replace(inputs, wing_cutoff_cm1=7.0, subtract_lorentz_pedestal=True)
    with pytest.raises(ValueError, match="cutoff"):
        replace(inputs, wing_cutoff_cm1=-1.0)


def test_continuum_freshness_tracks_foreign_density() -> None:
    air = atmosphere()
    grid = np.array([950.0, 1000.0])
    component = water_continuum_absorption(air, grid, reference())
    changed = replace(
        air,
        total_air_columns_molecules_m2=air.total_air_columns_molecules_m2 * 2,
        dry_air_columns_molecules_m2=air.total_air_columns_molecules_m2 * 2
        - air.water_vapour_columns_molecules_m2,
    )
    with pytest.raises(ValueError, match="stale"):
        compose_absorption_on_grid(
            changed,
            wavenumber_grid(grid),
            (component,),
            reference_atmospheres=(air,),
        )
    assert np.all(
        water_continuum_absorption(changed, grid, reference()).optical_depth
        > component.optical_depth
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"wavenumbers_cm1": np.array([0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0])},
        {"self_absco_ref": -np.ones(7)},
        {"reference_pressure_pa": 0.0},
        {"temperature_range_k": (300.0, 200.0)},
        {"reference_temperature_k": 0.0},
        {"wavenumbers_cm1": np.array([1.0, 2.0, 3.0])},
    ],
)
def test_invalid_reference(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(reference(), **changes)


def test_missing_water_and_temperature_support() -> None:
    air, grid = atmosphere(), np.array([1000.0])
    with pytest.raises(ValueError, match="temperature"):
        water_continuum_absorption(
            air, grid, replace(reference(), temperature_range_k=(290.0, 310.0))
        )
    with pytest.raises(ValueError, match="H2O"):
        water_continuum_absorption(replace(air, absorbers=()), grid, reference())
    with pytest.raises(ValueError, match="water vapour"):
        water_continuum_absorption(
            replace(
                air,
                absorbers=(
                    replace(
                        air.absorbers[0],
                        columns_molecules_m2=air.water_vapour_columns_molecules_m2 * 0.5,
                    ),
                ),
            ),
            grid,
            reference(),
        )
