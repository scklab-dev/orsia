import sys
from dataclasses import replace
from math import exp, expm1, pi, sqrt
from typing import Any, cast

import numpy as np
import orsia_radiative
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    Atmosphere,
    CrossSectionField,
    Line,
    LineSet,
    ObservationGeometry,
    RadiativeScene,
    RayleighComponent,
    ScatteringProperties,
    SolarSource,
    SpectralIrradiance,
    SurfaceState,
    compose_absorption,
    compose_absorption_on_grid,
    line_absorption,
    rayleigh_component,
    solar_source,
    table_absorption,
    validate_optical_inputs,
    wavelength_to_wavenumber,
    wavenumber_grid,
)
from orsia_radiative.spectroscopy.components import validate_absorption_freshness
from scipy import constants
from scipy.integrate import quad


def atmosphere(**changes: Any) -> Atmosphere:
    values: dict[str, Any] = dict(
        height_boundaries_m=np.array([0, 1000, 2000], dtype=np.float64),
        pressure_boundaries_pa=np.array([101325, 90000, 80000], dtype=np.float64),
        pressure_pa=np.array([95000, 85000], dtype=np.float64),
        temperature_k=np.array([296, 310], dtype=np.float64),
        total_air_columns_molecules_m2=np.array([2e29, 1.8e29]),
        dry_air_columns_molecules_m2=np.array([1.9e29, 1.7e29]),
        water_vapour_columns_molecules_m2=np.array([1e28, 1e28]),
        absorbers=(Absorber("X", np.array([1e24, 2e24]), "dry_air"),),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return Atmosphere(**values)


def axis() -> NDArray[np.float64]:
    return np.array([999.0, 1000.0, 1001.0], dtype=np.float64)


def lines(**changes: Any) -> LineSet:
    values: dict[str, Any] = dict(
        species="X",
        isotopologue="1",
        lines=(Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7),),
        partition_ratios=np.array([1.0, 0.9]),
        temperature_range_k=(270, 320),
        pressure_range_pa=(80000, 100000),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return LineSet(**values)


def table(**changes: Any) -> CrossSectionField:
    values: dict[str, Any] = dict(
        species="X",
        isotopologue="1",
        kind="cross_section",
        wavenumbers_cm1=axis(),
        temperature_k=np.array([296, 310], dtype=np.float64),
        pressure_pa=np.array([95000, 85000], dtype=np.float64),
        cross_sections_m2_per_molecule=np.array([[1e-25, 2e-25, 3e-25], [2e-25, 4e-25, 6e-25]]),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return CrossSectionField(**values)


def scene(**changes: Any) -> RadiativeScene:
    values: dict[str, Any] = dict(
        atmosphere=atmosphere(),
        surface=SurfaceState(wavenumber_grid(axis()), reflectance=np.array([0.2, 0.2, 0.2])),
        geometry=ObservationGeometry(10),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return RadiativeScene(**values)


def source() -> SolarSource:
    return solar_source(
        SpectralIrradiance(wavenumber_grid(axis()), np.ones(3)), direction=(30.0, 20.0)
    )


def test_line_center_strength_reference_and_wing_are_analytic() -> None:
    layer = atmosphere()
    computed = line_absorption(layer, axis(), lines())
    p = 95000 / 101325
    self_fraction = 1e24 / 2e29
    foreign_fraction = (1.9e29 - 1e24) / 2e29
    gamma = p * (0.1 * foreign_fraction + 0.2 * self_fraction)
    expected = 1e24 * 1e-4 * 1e-25 / (pi * gamma)
    assert computed.optical_depth[0][1] == pytest.approx(expected)
    assert computed.optical_depth[0][0] == pytest.approx(expected * gamma**2 / (1 + gamma**2))
    assert computed.optical_depth[0][0] == computed.optical_depth[0][2]
    assert computed.optical_depth[1][1] > computed.optical_depth[0][1]
    assert computed.kind == "lorentz_lines" and computed.species == "X"


def test_temperature_pressure_partition_and_self_broadening_have_declared_effects() -> None:
    base = line_absorption(atmosphere(), axis(), lines()).optical_depth[0][1]
    half_partition = line_absorption(
        atmosphere(), axis(), lines(partition_ratios=np.array([0.5, 0.9]))
    )
    assert half_partition.optical_depth[0][1] == pytest.approx(base / 2)
    different_air = atmosphere(absorbers=(Absorber("X", np.array([1e24, 2e24]), "total_air"),))
    assert line_absorption(different_air, axis(), lines()).optical_depth[0][1] != base
    shift = line_absorption(
        atmosphere(), axis(), lines(lines=(replace(lines().lines[0], pressure_shift_ref_cm1=1.0),))
    )
    assert shift.optical_depth[0][1] < base
    # Lower-state population and stimulated-emission corrections evaluated independently.
    high = replace(lines().lines[0], lower_energy_cm1=200.0)
    modified = line_absorption(atmosphere(), axis(), lines(lines=(high,)))
    c2 = 100 * 6.62607015e-34 * 299792458 / 1.380649e-23
    ratio = exp(-c2 * 200 * (1 / 310 - 1 / 296))
    assert modified.optical_depth[1][1] == pytest.approx(
        line_absorption(atmosphere(), axis(), lines()).optical_depth[1][1] * ratio
    )


def test_wavelength_nu_mapping_preserves_input_order_without_density_conversion() -> None:
    wavelength = np.array([1e7 / 1000, 1e7 / 999], dtype=np.float64)
    converted = wavelength_to_wavenumber(wavelength)
    mapped = line_absorption(atmosphere(), converted, lines())
    reference = line_absorption(atmosphere(), axis(), lines())
    assert mapped.optical_depth[0] == pytest.approx(reference.optical_depth[0][:2])
    np.testing.assert_array_equal(mapped.wavenumbers_cm1, converted)


def test_cross_section_interpolation_is_per_layer_and_no_extrapolation() -> None:
    target = np.array([999.5, 1000.5], dtype=np.float64)
    result = table_absorption(atmosphere(), target, table())
    assert result.optical_depth[0] == pytest.approx((0.15, 0.25))
    assert result.optical_depth[1] == pytest.approx((0.6, 1.0))
    with pytest.raises(ValueError, match="extrapolation"):
        table_absorption(atmosphere(), np.array([998.0]), table())
    with pytest.raises(ValueError, match="conditions"):
        table_absorption(
            atmosphere(), target, table(temperature_k=np.array([295, 310], dtype=np.float64))
        )


def test_many_lines_vectorised_spectrum_matches_scalar_lorentz_reference() -> None:
    spectral_axis = 999.0 + np.arange(257, dtype=np.float64) * 0.01
    entries = (
        Line(999.3, 1e-25, 0.0, 0.1, 0.2, 0.7),
        Line(1000.7, 2e-25, 120.0, 0.06, 0.12, 0.9, pressure_shift_ref_cm1=0.02),
    )
    input_lines = lines(lines=entries)
    result = line_absorption(atmosphere(), spectral_axis, input_lines)
    air = atmosphere()
    for i, (temperature, pressure, amount) in enumerate(
        zip(air.temperature_k, air.pressure_pa, air.absorbers[0].columns_molecules_m2, strict=True)
    ):
        self_fraction = amount / air.total_air_columns_molecules_m2[i]
        foreign_fraction = (air.dry_air_columns_molecules_m2[i] - amount) / (
            air.total_air_columns_molecules_m2[i]
        )
        for j, nu in enumerate(spectral_axis):
            expected = 0.0
            for line in entries:
                c2 = 100 * 6.62607015e-34 * 299792458 / 1.380649e-23
                strength = (
                    line.strength_ref_cm_per_molecule
                    * input_lines.partition_ratios[i]
                    * exp(-c2 * line.lower_energy_cm1 * (1 / temperature - 1 / 296))
                    * (
                        expm1(-c2 * line.centre_cm1 / temperature)
                        / expm1(-c2 * line.centre_cm1 / 296)
                    )
                )
                gamma = (
                    (296 / temperature) ** line.temperature_exponent
                    * pressure
                    / 101325
                    * (
                        line.air_hwhm_ref_cm1 * foreign_fraction
                        + line.self_hwhm_ref_cm1 * self_fraction
                    )
                )
                centre = line.centre_cm1 + line.pressure_shift_ref_cm1 * pressure / 101325
                expected += (
                    amount * 1e-4 * strength * gamma / (pi * (gamma**2 + (nu - centre) ** 2))
                )
            assert result.optical_depth[i][j] == pytest.approx(expected, rel=2e-14, abs=1e-16)
    assert result.optical_depth.dtype == np.float64


def test_line_profile_has_no_wing_cutoff_across_spectral_blocks() -> None:
    spectral_axis = 999.0 + np.arange(8193, dtype=np.float64) * 0.001
    result = line_absorption(atmosphere(), spectral_axis, lines())
    p = 95000 / 101325
    self_fraction = 1e24 / 2e29
    foreign_fraction = (1.9e29 - 1e24) / 2e29
    gamma = p * (0.1 * foreign_fraction + 0.2 * self_fraction)
    for j in (0, 1000, 8191, 8192):
        nu = spectral_axis[j]
        expected = 1e24 * 1e-4 * 1e-25 * gamma / (pi * (gamma**2 + (nu - 1000) ** 2))
        assert result.optical_depth[0][j] == pytest.approx(expected, rel=2e-14)


def test_interpolation_grid_plan_preserves_boundaries_and_canonical_order() -> None:
    target = wavelength_to_wavenumber(np.array([1e7 / 1001, 10000, 1e7 / 999]))
    values = table_absorption(atmosphere(), target, table())
    assert values.optical_depth[0] == pytest.approx((0.1, 0.2, 0.3))
    assert values.optical_depth[1] == pytest.approx((0.4, 0.8, 1.2))
    at_boundaries = table_absorption(atmosphere(), axis(), table())
    assert at_boundaries.optical_depth[0] == pytest.approx((0.1, 0.2, 0.3))
    with pytest.raises(ValueError, match="extrapolation"):
        table_absorption(
            atmosphere(),
            np.array([999.0, 1001.0001]),
            table(),
        )


def test_optical_inputs_remain_separate_and_validate() -> None:
    material = scene()
    gas = line_absorption(material.atmosphere, axis(), lines())
    continuum = table_absorption(
        material.atmosphere, axis(), table(kind="continuum_excluding_lines", isotopologue=None)
    )
    moments = (((1.0, 0.0, 0.1),) * 3,) * 2
    scatter = ScatteringProperties(
        wavenumber_grid(axis()),
        np.array([[0.1, 0.0, 0.1], [0.2, 0.2, 0.2]]),
        np.asarray(moments, dtype=np.float64),
        "source_untruncated",
    )
    assert (
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), (gas, continuum)),
            rayleigh=rayleigh_component(scatter),
        )
        is None
    )
    assert gas.optical_depth[0][0] > 0
    np.testing.assert_array_equal(scatter.optical_depth[0], [0.1, 0.0, 0.1])
    assert scatter.truncation == "source_untruncated"
    assert (
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), (gas,)),
        )
        is None
    )


def test_no_speculative_scene_wide_or_solver_contract_is_exported() -> None:
    for name in ("OpticalState", "SolverAdaptation", "build_optical_state"):
        assert name not in orsia_radiative.__all__
        assert not hasattr(orsia_radiative, name)


@pytest.mark.parametrize(
    ("input_change", "message"),
    [
        ({"partition_ratios": np.array([1.0])}, "partition ratios"),
        ({"temperature_range_k": (270, 290)}, "temperature outside"),
        ({"pressure_range_pa": (90000, 100000)}, "pressure outside"),
        ({"species": "OTHER"}, "no declared absorber"),
    ],
)
def test_incompatible_line_inputs_fail(input_change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        line_absorption(atmosphere(), axis(), lines(**input_change))


def test_invalid_inputs_and_duplicate_or_misaligned_components_fail() -> None:
    with pytest.raises(ValueError, match="cross section"):
        table(cross_sections_m2_per_molecule=np.array([[1e-25, -1e-25, 1e-25]] * 2))
    with pytest.raises(ValueError, match="shape"):
        table(cross_sections_m2_per_molecule=np.array([[1e-25]] * 2))
    with pytest.raises(ValueError, match="wavenumbers_cm1"):
        line_absorption(atmosphere(), np.array([0.0]), lines())
    with pytest.raises(ValueError, match="float64"):
        line_absorption(
            atmosphere(), cast(NDArray[np.float64], np.array([100000], dtype=np.int64)), lines()
        )
    with pytest.raises(ValueError, match="positive"):
        Line(1000, 1e-25, 0, 0, 0.1, 0.7)
    with pytest.raises(ValueError, match="positive"):
        lines(partition_ratios=np.array([1.0, -1.0]))
    gas = line_absorption(atmosphere(), axis(), lines())
    material = scene()
    with pytest.raises(ValueError, match="duplicate"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), (gas, gas)),
        )
    with pytest.raises(ValueError, match="overlap"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(
                material.atmosphere,
                source(),
                (gas, table_absorption(atmosphere(), axis(), table())),
            ),
        )
    with pytest.raises(ValueError, match="axis and layer"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(
                material.atmosphere,
                replace(
                    source(),
                    irradiance=replace(
                        source().irradiance, grid=wavenumber_grid(np.array([999.0, 1000.0, 1002.0]))
                    ),
                ),
                (gas,),
            ),
        )
    with pytest.raises(ValueError, match="layer order"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(
                material.atmosphere, source(), (replace(gas, optical_depth=gas.optical_depth[:1]),)
            ),
        )
    assert (
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), ()),
        )
        is None
    )
    with pytest.raises(ValueError, match="undeclared"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(
                material.atmosphere, source(), (replace(gas, species="OTHER"),)
            ),
        )
    with pytest.raises(ValueError, match="scattering axis and layer order"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), (gas,)),
            rayleigh=RayleighComponent(wavenumber_grid(axis()), np.full((1, 3), 0.1)),
        )
    huge = replace(gas, optical_depth=np.full((2, 3), 1e308))
    other = replace(huge, kind="continuum_excluding_lines")
    with pytest.raises(ValueError, match="overflow"):
        validate_optical_inputs(
            material.atmosphere,
            source(),
            absorption=compose_absorption(material.atmosphere, source(), (huge, other)),
        )
    with pytest.raises(ValueError, match="zeroth"):
        ScatteringProperties(
            wavenumber_grid(axis()),
            np.full((2, 3), 0.1),
            np.full((2, 3, 1), 0.5),
            "source_untruncated",
        )
    with pytest.raises(ValueError, match="finite"):
        AbsorptionComponent(
            wavenumber_grid(axis()), "X", "1", "lorentz_lines", np.full((1, 3), np.nan)
        )


def test_voigt_mass_is_explicit_and_profile_matches_gaussian_lorentz_convolution() -> None:
    air = atmosphere()
    line = replace(lines().lines[0], air_hwhm_ref_cm1=0.001, self_hwhm_ref_cm1=0.002)
    inputs = lines(lines=(line,), molecular_mass_kg=44 * constants.atomic_mass)
    grid = 1000 + np.array([-0.01, -0.001, 0.0, 0.001, 0.01])
    saved = grid.copy()
    result = line_absorption(air, grid, inputs)
    assert result.kind == "voigt_lines" and result.species == "X"
    sigma = 1000 / constants.c * sqrt(constants.k * 296 / (44 * constants.atomic_mass))
    gamma = 95000 / 101325 * (0.001 * (1.9e29 - 1e24) / 2e29 + 0.002 * 1e24 / 2e29)
    for index, offset in enumerate(grid - 1000):
        profile = quad(
            lambda y, offset=offset: (
                exp(-y * y / 2)
                / sqrt(2 * pi)
                * gamma
                / (pi * ((offset - sigma * y) ** 2 + gamma**2))
            ),
            -12,
            12,
            epsabs=1e-11,
            epsrel=2e-13,
        )[0]
        assert result.optical_depth[0, index] == pytest.approx(1e-5 * profile, rel=3e-13)
    lorentz = line_absorption(air, grid, replace(inputs, molecular_mass_kg=None))
    assert result.optical_depth[0, 2] < lorentz.optical_depth[0, 2]
    np.testing.assert_allclose(
        line_absorption(air, grid, replace(inputs, molecular_mass_kg=1e20)).optical_depth,
        lorentz.optical_depth,
        rtol=3e-14,
    )
    np.testing.assert_array_equal(grid, saved)
    assert result.optical_depth.flags.writeable
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan])
def test_voigt_invalid_mass_is_rejected(mass: float) -> None:
    with pytest.raises(ValueError, match="molecular mass"):
        lines(molecular_mass_kg=mass)


def test_voigt_overlap_freshness_and_numerical_width_guard() -> None:
    air = atmosphere()
    voigt = line_absorption(air, axis(), lines(molecular_mass_kg=44 * constants.atomic_mass))
    lorentz = line_absorption(air, axis(), lines())
    cross = table_absorption(air, axis(), table())
    for extra in (lorentz, cross):
        with pytest.raises(ValueError, match="duplicate|overlap"):
            compose_absorption_on_grid(air, wavenumber_grid(axis()), (voigt, extra))
    modified = replace(air, absorbers=(Absorber("X", np.array([1e24, 2e24]), "total_air"),))
    with pytest.raises(ValueError, match="stale"):
        validate_absorption_freshness(modified, voigt, reference_atmosphere=air)
    np.testing.assert_array_equal(
        compose_absorption_on_grid(
            air, wavenumber_grid(axis()), (voigt,), reference_atmospheres=(air,)
        ).optical_depth,
        voigt.optical_depth,
    )
    with pytest.raises(ValueError, match="Doppler width"):
        line_absorption(
            air,
            axis(),
            lines(lines=(replace(lines().lines[0], centre_cm1=1e308),), molecular_mass_kg=1e-320),
        )
