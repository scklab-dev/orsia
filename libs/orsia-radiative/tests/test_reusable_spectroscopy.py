"""Reusable spectroscopy contracts on synthetic transitions and laboratory spectra."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from orsia_radiative import (
    CrossSectionMeasurements,
    CrossSectionSpectrum,
    Line,
    LineDatabase,
    LineSpectroscopy,
    PartitionFunctionTable,
    evaluate_line_absorption,
    hydrostatic_atmosphere,
    interpolate_cross_section_measurements,
    line_absorption,
    line_set,
    measurement_absorption,
    wavenumber_grid,
)


def atmosphere(temperatures=(250.0, 300.0), species="CO2"):
    pressure = np.array([100000.0, 50000.0, 10000.0])
    return hydrostatic_atmosphere(
        pressure,
        np.array(temperatures),
        np.zeros(2),
        target_pressure_boundaries_pa=pressure,
        gravity_m_s2=9.80665,
        bottom_height_m=0,
        gas_columns_molecules_m2={species: np.array([1e22, 2e22])},
        air_broadener="dry_air",
    )


def reference():
    line = Line(1000, 1e-23, 100, 0.07, 0.1, 0.7, 0.001)
    table = PartitionFunctionTable(
        2, 1, np.array([200.0, 296.0, 350.0]), np.array([100.0, 148.0, 175.0])
    )
    return LineSpectroscopy(
        "CO2",
        (LineDatabase((line,), 2, 1, (970, 1030)), LineDatabase((), 2, 1, (1970, 2030))),
        table,
        44 * 1.660539e-27,
    )


def measurement():
    spectra = []
    for t, axis in (
        (200.0, np.array([319.0, 322.0, 326.0])),
        (300.0, np.array([318.0, 321.0, 327.0])),
    ):
        spectra.append(
            CrossSectionSpectrum(
                axis,
                (axis + t) * 1e-25,
                t,
                None,
                ((float(axis[0]), float(axis[-1])),),
                "0.2 nm FWHM",
            )
        )
    return CrossSectionMeasurements("CO2", "all", tuple(spectra))


@pytest.mark.parametrize("profile", ["voigt", "lorentz"])
def test_reference_evaluation_matches_detailed_kernels(profile):
    ref, atm = reference(), atmosphere()
    grid = wavenumber_grid(np.array([999.0, 1000.0, 1001.0]))
    actual = evaluate_line_absorption(
        atm,
        grid,
        ref,
        profile=profile,
        pressure_range_pa=(100, 105000),
        wing_cutoff_cm1=25.0,
        subtract_lorentz_pedestal=False,
    )
    inputs = line_set(
        ref.databases[0],
        ref.partition_table,
        atm,
        grid,
        species="CO2",
        molecular_mass_kg=ref.molecular_mass_kg,
        pressure_range_pa=(100, 105000),
        wing_cutoff_cm1=25.0,
    )
    if profile == "lorentz":
        inputs = replace(inputs, molecular_mass_kg=None)
    np.testing.assert_array_equal(
        actual.optical_depth, line_absorption(atm, grid, inputs).optical_depth
    )
    from orsia_radiative import EmptyLineSelection

    with pytest.raises(EmptyLineSelection):
        evaluate_line_absorption(
            atm,
            wavenumber_grid(np.array([1999.0, 2000.0])),
            ref,
            profile=profile,
            pressure_range_pa=(100, 105000),
            wing_cutoff_cm1=25.0,
            subtract_lorentz_pedestal=False,
        )


def test_reference_snapshots_and_invalid_identity_support():
    ref = reference()
    for values in (ref.partition_table.temperatures_k, ref.partition_table.partition_sums):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
    for changes in (
        {"species": ""},
        {"databases": ()},
        {"molecular_mass_kg": np.nan},
        {"intensity_convention": "unweighted"},
        {"databases": (replace(ref.databases[0], molecule_id=1),)},
        {"databases": (ref.databases[0], ref.databases[0])},
    ):
        with pytest.raises(ValueError):
            invalid_replace(ref, **changes)
    atm = atmosphere()
    common: dict[str, Any] = dict(
        profile="voigt",
        pressure_range_pa=(100, 105000),
        wing_cutoff_cm1=25.0,
        subtract_lorentz_pedestal=False,
    )
    for points in ([1500.0, 1501.0], [999.0, 2001.0], [975.0, 976.0]):
        with pytest.raises(ValueError):
            evaluate_line_absorption(atm, wavenumber_grid(np.array(points)), ref, **common)
    common["profile"] = "guess"
    with pytest.raises(ValueError, match="profile"):
        evaluate_line_absorption(atm, wavenumber_grid(np.array([999.0, 1000.0])), ref, **common)


def test_measurement_bilinear_interpolation_column_application_and_diagnostics():
    ref, atm = measurement(), atmosphere((190.0, 310.0))
    grid = wavenumber_grid(np.array([1e7 / 325.0, 1e7 / 320.0]))
    component, diagnostic = measurement_absorption(
        atm,
        grid,
        ref,
        spectral_interpolation="linear_native_wavelength",
        temperature_interpolation="linear",
        temperature_outside="hold_nearest",
        pressure_model="independent",
    )
    expected = ((1e7 / grid.points)[None, :] + np.array([200.0, 300.0])[:, None]) * 1e-25
    np.testing.assert_allclose(
        component.optical_depth, expected * np.array([1e22, 2e22])[:, None], atol=0
    )
    np.testing.assert_array_equal(diagnostic.requested_k, [190, 310])
    np.testing.assert_array_equal(diagnostic.effective_k, [200, 300])
    np.testing.assert_array_equal(diagnostic.held_layers, [True, True])
    for array in (
        diagnostic.requested_k,
        diagnostic.effective_k,
        diagnostic.held_layers,
        ref.spectra[0].wavelength_nm,
        ref.spectra[0].cross_sections_m2_per_molecule,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    values, _ = interpolate(ref, np.array([250.0]), np.array([320.0, 325.0]))
    np.testing.assert_allclose(values, (np.array([[320.0, 325.0]]) + 250) * 1e-25, atol=0)


def interpolate(ref, temps, target, **changes):
    options: dict[str, Any] = dict(
        spectral_interpolation="linear_native_wavelength",
        temperature_interpolation="linear",
        temperature_outside="hold_nearest",
        pressure_model="independent",
    )
    return interpolate_cross_section_measurements(ref, temps, target, **{**options, **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature_outside": "guess"},
        {"temperature_interpolation": "cubic"},
        {"spectral_interpolation": "wavenumber"},
        {"pressure_model": "fixed"},
    ],
)
def test_measurement_policy_errors(changes):
    with pytest.raises(ValueError):
        interpolate(measurement(), np.array([250.0]), np.array([320.0]), **changes)


def test_measurement_support_and_exact_temperature_selection():
    ref = measurement()
    gap = CrossSectionSpectrum(
        np.array([319.0, 320.0, 325.0, 326.0]),
        np.ones(4),
        200.0,
        100000.0,
        ((319.0, 320.0), (325.0, 326.0)),
    )
    ref = replace(ref, spectra=(gap, ref.spectra[1]))
    # At an exact temperature only the participating spectrum needs support.
    interpolate(ref, np.array([300.0]), np.array([322.0]))
    with pytest.raises(ValueError, match="gap"):
        interpolate(ref, np.array([250.0]), np.array([322.0]))
    for temps, target in (
        ([0], [320]),
        ([np.nan], [320]),
        ([], [320]),
        ([250], [0]),
        ([250], [330]),
    ):
        with pytest.raises(ValueError):
            interpolate(ref, np.array(temps, dtype=float), np.array(target, dtype=float))
    with pytest.raises(ValueError, match="temperature outside"):
        interpolate(ref, np.array([190.0]), np.array([320.0]), temperature_outside="error")
    one = replace(ref, spectra=(gap,))
    values, _ = interpolate(one, np.array([300.0]), np.array([320.0]))
    np.testing.assert_array_equal(values, 1)


def test_measurement_invalid_records():
    spectrum = measurement().spectra[0]
    for changes in (
        {"temperature_k": 0},
        {"pressure_pa": 0},
        {"wavelength_nm": np.array([1.0, 1.0, 2.0])},
        {"cross_sections_m2_per_molecule": -np.ones(3)},
        {"support_nm": ()},
        {"support_nm": ((319.0, 321.0),)},
        {"support_nm": ((319.0, 326.0), (319.0, 326.0))},
    ):
        with pytest.raises(ValueError):
            invalid_replace(spectrum, **changes)
    ref = measurement()
    for changes in (
        {"species": ""},
        {"isotopologue": ""},
        {"spectra": ()},
        {"spectra": ref.spectra[::-1]},
    ):
        with pytest.raises(ValueError):
            invalid_replace(ref, **changes)


def invalid_replace(record: Any, **changes: Any):
    """Exercise constructor validation with intentionally invalid dynamic fields."""
    return replace(record, **changes)
