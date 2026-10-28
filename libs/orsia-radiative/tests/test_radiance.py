from dataclasses import replace
from math import cos, exp, expm1, pi, radians, sin
from typing import Any

import numpy as np
import pytest
import radiance_terms_support as terms_support
from numpy.typing import NDArray
from orsia_radiative import (
    Absorber,
    AbsorptionComponent,
    AerosolOptics,
    Atmosphere,
    Line,
    LineSet,
    ObservationGeometry,
    RadiativeScene,
    ScatteringProperties,
    SpectralIrradiance,
    SurfaceState,
    aerosol_component,
    compose_absorption,
    compose_atmospheric_optics,
    density_per_nm_to_per_cm1,
    henyey_greenstein_phase,
    line_absorption,
    rayleigh_component,
    solar_source,
    wavelength_to_wavenumber,
    wavenumber_grid,
)


def atmosphere() -> Atmosphere:
    return Atmosphere(
        height_boundaries_m=np.array([0.0, 1000.0]),
        pressure_boundaries_pa=np.array([100000.0, 90000.0]),
        pressure_pa=np.array([95000.0]),
        temperature_k=np.array([290.0]),
        total_air_columns_molecules_m2=np.array([2.0e29]),
        dry_air_columns_molecules_m2=np.array([1.9e29]),
        water_vapour_columns_molecules_m2=np.array([1.0e28]),
        absorbers=(Absorber("X", np.array([1.0e25]), "dry_air"),),
    )


def axis() -> NDArray[np.float64]:
    return np.array([1000.0, 1001.0], dtype=np.float64)


def scene(**changes: Any) -> RadiativeScene:
    values: dict[str, Any] = dict(
        atmosphere=atmosphere(),
        surface=SurfaceState(wavenumber_grid(axis()), reflectance=np.array([0.2, 0.4])),
        geometry=ObservationGeometry(10.0),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return RadiativeScene(**values)


def absorption(optical_depth: tuple[tuple[float, ...], ...]) -> AbsorptionComponent:
    return AbsorptionComponent(
        wavenumber_grid(axis()),
        "X",
        "synthetic",
        "lorentz_lines",
        np.asarray(optical_depth, dtype=np.float64),
    )


def solar(**changes: Any) -> SpectralIrradiance:
    values: dict[str, Any] = dict(
        wavenumbers_cm1=axis(),
        values=np.array([10.0, 20.0]),
    )
    values.update(changes)
    if "wavenumbers_cm1" in values:
        values["grid"] = wavenumber_grid(values.pop("wavenumbers_cm1"))
    return SpectralIrradiance(**values)


def test_absorption_only_radiance_contract_and_reference_formula() -> None:
    result = terms_support.order_zero(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (absorption(((0.1, 0.2),)),),
            ),
        ),
    )
    mu0 = cos(radians(30.0))
    mu = cos(radians(10.0))
    expected = tuple(
        irradiance * reflectance * mu0 / pi * exp(-depth * (1 / mu0 + 1 / mu))
        for irradiance, reflectance, depth in zip((10.0, 20.0), (0.2, 0.4), (0.1, 0.2), strict=True)
    )
    assert result.values == pytest.approx(expected)
    np.testing.assert_array_equal(result.wavenumbers_cm1, axis())
    assert result.geometry == scene().geometry
    np.testing.assert_array_equal(result.components[0].values, result.values)


def test_transparent_case_reduces_to_lambertian_surface_radiance() -> None:
    result = terms_support.order_zero(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (absorption(((0.0, 0.0),)),),
            ),
        ),
    )
    mu0 = cos(radians(30.0))
    assert result.values == pytest.approx((10.0 * 0.2 * mu0 / pi, 20.0 * 0.4 * mu0 / pi))


def test_first_materialised_spectroscopy_to_radiance_path() -> None:
    material = scene()
    line_input = LineSet(
        "X",
        "synthetic",
        (Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7),),
        np.array([1.0]),
        (280.0, 300.0),
        (90000.0, 100000.0),
    )
    optical = line_absorption(material.atmosphere, axis(), line_input)
    result = terms_support.order_zero(
        material,
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            material.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                material.atmosphere, solar_source(solar(), direction=(30.0, 20.0)), (optical,)
            ),
        ),
    )
    assert optical.optical_depth[0][0] > optical.optical_depth[0][1]
    assert all(value > 0 for value in result.values)


def test_wavelength_density_requires_explicit_canonical_conversion() -> None:
    wavelength_nm = np.array([9990.0, 10000.0])
    wavenumbers_cm1 = wavelength_to_wavenumber(wavelength_nm)
    material = scene(
        surface=SurfaceState(wavenumber_grid(wavenumbers_cm1), reflectance=np.array([0.4, 0.2]))
    )
    irradiance = solar(
        wavenumbers_cm1=wavenumbers_cm1,
        values=density_per_nm_to_per_cm1(wavelength_nm, np.array([10.0, 20.0])),
    )
    optical = replace(absorption(((0.0, 0.0),)), grid=wavenumber_grid(wavenumbers_cm1))
    result = terms_support.order_zero(
        material,
        source=solar_source(irradiance, direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            material.atmosphere,
            solar_source(irradiance, direction=(30.0, 20.0)),
            absorption=compose_absorption(
                material.atmosphere, solar_source(irradiance, direction=(30.0, 20.0)), (optical,)
            ),
        ),
    )
    np.testing.assert_array_equal(result.wavenumbers_cm1, wavenumbers_cm1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("axis", "axis and layer order"),
        ("reflectance", "requires surface reflectance"),
    ],
)
def test_first_model_rejects_incompatible_inputs(change: str, message: str) -> None:
    if change == "axis":
        other_axis = np.array([1000.0, 1002.0])
        irradiance = solar(wavenumbers_cm1=other_axis)
        with pytest.raises(ValueError, match=message):
            terms_support.order_zero(
                (_scene := scene()),
                source=solar_source(irradiance, direction=(30.0, 20.0)),
                optics=compose_atmospheric_optics(
                    _scene.atmosphere,
                    solar_source(irradiance, direction=(30.0, 20.0)),
                    absorption=compose_absorption(
                        _scene.atmosphere,
                        solar_source(irradiance, direction=(30.0, 20.0)),
                        (absorption(((0.0, 0.0),)),),
                    ),
                ),
            )
    elif change == "reflectance":
        thermal_surface = SurfaceState(
            wavenumber_grid(axis()), emissivity=np.array([0.8, 0.7]), temperature_k=285.0
        )
        with pytest.raises(ValueError, match=message):
            terms_support.order_zero(
                (_scene := scene(surface=thermal_surface)),
                source=solar_source(solar(), direction=(30.0, 20.0)),
                optics=compose_atmospheric_optics(
                    _scene.atmosphere,
                    solar_source(solar(), direction=(30.0, 20.0)),
                    absorption=compose_absorption(
                        _scene.atmosphere,
                        solar_source(solar(), direction=(30.0, 20.0)),
                        (absorption(((0.0, 0.0),)),),
                    ),
                ),
            )


def test_radiance_contract_rejects_bad_grid_and_retains_typed_values() -> None:
    values = (10.0, 20.0)
    with pytest.raises(ValueError, match="wavenumbers_cm1"):
        solar(wavenumbers_cm1=np.array([1001.0, 1000.0]))
    irradiance = solar(values=np.asarray(values, dtype=np.float64))
    np.testing.assert_array_equal(irradiance.values, values)
    with pytest.raises(ValueError, match="float64"):
        solar(wavenumbers_cm1=np.array([1000, 1001], dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative"):
        solar(values=np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        solar(values=np.array([np.nan, 2.0]))


def test_radiative_grid_snapshots_prevent_input_mutation() -> None:
    grid = axis()
    surface = SurfaceState(wavenumber_grid(grid), reflectance=np.array([0.2, 0.4]))
    illumination = SpectralIrradiance(wavenumber_grid(grid), np.array([10.0, 20.0]))
    optical = AbsorptionComponent(
        wavenumber_grid(grid), "X", None, "lorentz_lines", np.array([[0.1, 0.2]])
    )
    grid[0] = 999.0
    for captured in (
        surface.wavenumbers_cm1,
        illumination.wavenumbers_cm1,
        optical.wavenumbers_cm1,
    ):
        np.testing.assert_array_equal(captured, [1000.0, 1001.0])
        with pytest.raises(ValueError):
            captured.flags.writeable = True


def test_independent_radiative_records_use_identity_but_produce_equal_values() -> None:
    input_a = solar()
    input_b = solar()
    assert input_a != input_b
    assert input_a.grid == input_b.grid
    np.testing.assert_array_equal(input_a.values, input_b.values)
    assert input_a != solar(wavenumbers_cm1=np.array([1000.0, 1002.0]))
    assert SurfaceState(wavenumber_grid(axis()), reflectance=np.array([0.2, 0.4])) != SurfaceState(
        wavenumber_grid(axis()), reflectance=np.array([0.2, 0.4])
    )
    first = absorption(((0.1, 0.2),))
    assert first != absorption(((0.1, 0.2),))
    assert first != absorption(((0.1, 0.3),))
    result_a = terms_support.order_zero(
        (_scene := scene()),
        source=solar_source(input_a, direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(input_a, direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere, solar_source(input_a, direction=(30.0, 20.0)), (first,)
            ),
        ),
    )
    result_b = terms_support.order_zero(
        (_scene := scene()),
        source=solar_source(input_b, direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(input_b, direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere, solar_source(input_b, direction=(30.0, 20.0)), (first,)
            ),
        ),
    )
    terms_support.assert_decomposition_equal(result_a, result_b)
    with pytest.raises(TypeError):
        hash(input_a)


def test_absorption_components_are_summed_over_layers() -> None:
    two_layer = replace(
        scene(),
        atmosphere=Atmosphere(
            height_boundaries_m=np.array([0.0, 1000.0, 2000.0]),
            pressure_boundaries_pa=np.array([100000.0, 90000.0, 80000.0]),
            pressure_pa=np.array([95000.0, 85000.0]),
            temperature_k=np.array([290.0, 280.0]),
            total_air_columns_molecules_m2=np.array([2.0e29, 1.8e29]),
            dry_air_columns_molecules_m2=np.array([1.9e29, 1.7e29]),
            water_vapour_columns_molecules_m2=np.array([1.0e28, 1.0e28]),
            absorbers=(Absorber("X", np.array([1.0e25, 1.0e25]), "dry_air"),),
        ),
    )
    result = terms_support.order_zero(
        two_layer,
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            two_layer.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                two_layer.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (absorption(((0.1, 0.2), (0.3, 0.4))),),
            ),
        ),
    )
    mu0 = cos(radians(30.0))
    mu = cos(radians(10.0))
    assert result.values[0] == pytest.approx(10.0 * 0.2 * mu0 / pi * exp(-0.4 * (1 / mu0 + 1 / mu)))
    line_component = absorption(((0.1, 0.2), (0.3, 0.4)))
    continuum = replace(
        line_component,
        kind="continuum_excluding_lines",
        isotopologue=None,
        optical_depth=np.full((2, 2), 0.05),
    )
    combined = terms_support.order_zero(
        two_layer,
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            two_layer.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                two_layer.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (line_component, continuum),
            ),
        ),
    )
    assert combined.values[0] == pytest.approx(
        10.0 * 0.2 * mu0 / pi * exp(-0.5 * (1 / mu0 + 1 / mu))
    )


def rayleigh(optical_depth: tuple[tuple[float, ...], ...]) -> ScatteringProperties:
    return ScatteringProperties(
        grid=wavenumber_grid(axis()),
        optical_depth=np.asarray(optical_depth, dtype=np.float64),
        phase_moments=np.tile([1.0, 0.0, 0.1], (len(optical_depth), len(optical_depth[0]), 1)),
        truncation="source_untruncated",
    )


def aerosol(
    optical_depth: tuple[tuple[float, ...], ...],
    *,
    albedo: float = 1.0,
    asymmetry: float = 0.5,
) -> AerosolOptics:
    return AerosolOptics(
        grid=wavenumber_grid(axis()),
        optical_depth=np.asarray(optical_depth, dtype=np.float64),
        single_scattering_albedo=np.full(np.shape(optical_depth), albedo, dtype=np.float64),
        asymmetry=np.full(np.shape(optical_depth), asymmetry, dtype=np.float64),
    )


def test_first_rayleigh_order_above_black_ground_matches_analytic_path() -> None:
    result = terms_support.order_one(
        (_scene := scene(surface=SurfaceState(wavenumber_grid(axis()), reflectance=np.zeros(2)))),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh(((0.01, 0.02),))),
        ),
    )
    mu0 = cos(radians(30.0))
    mu = cos(radians(10.0))
    phase = 0.75 * (
        1 + (-mu0 * mu + (1 - mu0**2) ** 0.5 * (1 - mu**2) ** 0.5 * cos(radians(20.0))) ** 2
    )
    for irradiance, reflectance, scattering_depth, value, direct, path in zip(
        (10.0, 20.0),
        (0.0, 0.0),
        (0.01, 0.02),
        result.values,
        result.components[0].values,
        result.components[1].values,
        strict=True,
    ):
        transmission = exp(-scattering_depth * (1 / mu0 + 1 / mu))
        expected_direct = irradiance * reflectance * mu0 / pi * transmission
        expected_path = irradiance * phase / (4 * pi * mu) * (1 - transmission) / (1 / mu0 + 1 / mu)
        assert direct == pytest.approx(expected_direct)
        assert path == pytest.approx(expected_path)
        assert value == pytest.approx(expected_direct + expected_path)
    assert result.components[1].name == "atmospheric_order_1"


def test_single_rayleigh_reduces_to_absorption_only_when_scattering_is_zero() -> None:
    scattering = rayleigh(((0.0, 0.0),))
    result = terms_support.order_one(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (absorption(((0.1, 0.2),)),),
            ),
            rayleigh=rayleigh_component(scattering),
        ),
    )
    reference = terms_support.order_zero(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            absorption=compose_absorption(
                _scene.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                (absorption(((0.1, 0.2),)),),
            ),
        ),
    )
    assert result.values == pytest.approx(reference.values)
    assert result.components[1].values == pytest.approx((0.0, 0.0))


def test_vectorised_rayleigh_matches_scalar_layer_integral_at_many_points() -> None:
    spectral_axis = 1000.0 + np.arange(128, dtype=np.float64) * 0.01
    material = scene(
        surface=SurfaceState(
            wavenumber_grid(spectral_axis), reflectance=np.zeros(spectral_axis.size)
        ),
        atmosphere=Atmosphere(
            np.array([0.0, 1000.0, 2000.0, 3000.0]),
            np.array([100000.0, 90000.0, 80000.0, 70000.0]),
            np.array([95000.0, 85000.0, 75000.0]),
            np.array([290.0, 280.0, 270.0]),
            np.array([2e29, 1.8e29, 1.6e29]),
            np.array([1.9e29, 1.7e29, 1.5e29]),
            np.array([1e28, 1e28, 1e28]),
            (Absorber("X", np.full(3, 1e25), "dry_air"),),
        ),
    )
    irradiance = solar(wavenumbers_cm1=spectral_axis, values=np.full(spectral_axis.size, 10.0))
    optical = AbsorptionComponent(
        wavenumber_grid(spectral_axis),
        "X",
        "synthetic",
        "lorentz_lines",
        0.001 * np.arange(1, 4, dtype=np.float64)[:, None] * np.arange(1, spectral_axis.size + 1),
    )
    scatter = ScatteringProperties(
        wavenumber_grid(spectral_axis),
        np.array([0.0, 1e-12, 0.04])[:, None] * (1 + np.arange(spectral_axis.size) / 500),
        np.tile([1.0, 0.0, 0.1], (3, spectral_axis.size, 1)),
        "source_untruncated",
    )
    result = terms_support.order_one(
        material,
        source=solar_source(irradiance, direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            material.atmosphere,
            solar_source(irradiance, direction=(30.0, 20.0)),
            absorption=compose_absorption(
                material.atmosphere, solar_source(irradiance, direction=(30.0, 20.0)), (optical,)
            ),
            rayleigh=rayleigh_component(scatter),
        ),
    )
    mu0 = cos(radians(30))
    mu = cos(radians(10))
    phase = 0.75 * (
        1 + (-mu0 * mu + (1 - mu0**2) ** 0.5 * (1 - mu**2) ** 0.5 * cos(radians(20))) ** 2
    )
    air_mass = 1 / mu0 + 1 / mu
    for point in range(spectral_axis.size):
        above = 0.0
        path_sum = 0.0
        for layer in range(2, -1, -1):
            depth = optical.optical_depth[layer][point] + scatter.optical_depth[layer][point]
            integrated = exp(-above * air_mass) * -expm1(-depth * air_mass) / (depth * air_mass)
            path_sum += scatter.optical_depth[layer][point] * integrated
            above += depth
        expected_direct = 0.0
        expected_path = 10 * phase / (4 * pi * mu) * path_sum
        assert result.components[0].values[point] == pytest.approx(expected_direct, rel=2e-14)
        assert result.components[1].values[point] == pytest.approx(expected_path, rel=2e-14)
        assert result.values[point] == pytest.approx(expected_direct + expected_path, rel=2e-14)
    assert result.values.dtype == np.float64


def test_single_rayleigh_rejects_truncated_or_non_rayleigh_moments() -> None:
    with pytest.raises(ValueError, match="untruncated"):
        rayleigh_component(
            replace(rayleigh(((0.01, 0.02),)), truncation="source_truncated"),
        )
    with pytest.raises(ValueError, match=r"\(1, 0, 0.1\)"):
        rayleigh_component(
            replace(
                rayleigh(((0.01, 0.02),)),
                phase_moments=np.array([[[1.0, 0.2, 0.1], [1.0, 0.2, 0.1]]]),
            ),
        )


def test_henyey_greenstein_phase_is_normalised_and_nonnegative() -> None:
    cosine = tuple(-1.0 + 2.0 * index / 20000 for index in range(20001))
    phase = tuple(henyey_greenstein_phase(0.6, value) for value in cosine)
    integral = sum(
        pi * (left + right) * (upper - lower)
        for left, right, lower, upper in zip(
            phase[:-1], phase[1:], cosine[:-1], cosine[1:], strict=True
        )
    )
    assert integral == pytest.approx(4 * pi, rel=2e-6)
    assert min(phase) > 0


def test_zero_aerosol_preserves_rayleigh_order_rows() -> None:
    rayleigh_input = rayleigh(((0.01, 0.02),))
    aerosol_input = aerosol(((0.0, 0.0),))
    rayleigh_only = terms_support.order_one(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh_input),
        ),
    )
    combined = terms_support.order_one(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh_input),
            aerosol=aerosol_component(aerosol_input),
        ),
    )
    np.testing.assert_array_equal(combined.values, rayleigh_only.values)
    assert combined.components[1].values == pytest.approx(rayleigh_only.components[1].values)
    assert len(combined.components) == 2
    np.testing.assert_array_equal(
        terms_support.order_one(
            (_scene := scene()),
            source=solar_source(solar(), direction=(30.0, 20.0)),
            optics=compose_atmospheric_optics(
                _scene.atmosphere,
                solar_source(solar(), direction=(30.0, 20.0)),
                rayleigh=rayleigh_component(rayleigh_input),
            ),
        ).values,
        rayleigh_only.values,
    )


def test_aerosol_first_order_above_black_surface_is_positive() -> None:
    black = scene(surface=SurfaceState(wavenumber_grid(axis()), reflectance=np.zeros(2)))
    result = terms_support.order_one(
        black,
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            black.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh(((0.0, 0.0),))),
            aerosol=aerosol_component(aerosol(((0.02, 0.03),))),
        ),
    )
    assert result.components[0].values == pytest.approx((0.0, 0.0))
    assert result.components[1].values[0] > 0
    assert result.values == pytest.approx(result.components[1].values)


def test_aerosol_surface_atmosphere_coupling_is_positive_for_reflective_surface() -> None:
    result = terms_support.order_one(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh(((0.0, 0.0),))),
            aerosol=aerosol_component(aerosol(((0.02, 0.03),))),
        ),
    )
    assert all(value > 0 for value in result.components[1].values)
    assert result.values == pytest.approx(
        tuple(sum(component.values[index] for component in result.components) for index in range(2))
    )


def test_aerosol_extinction_attenuates_direct_surface_and_validates_support() -> None:
    no_scatter = aerosol(((0.1, 0.1),), albedo=0.0)
    result = terms_support.order_one(
        (_scene := scene()),
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            _scene.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(rayleigh(((0.0, 0.0),))),
            aerosol=aerosol_component(no_scatter),
        ),
    )
    mu0 = cos(radians(30.0))
    mu = cos(radians(10.0))
    expected = tuple(
        irradiance * reflectance * mu0 / pi * exp(-0.1 * (1 / mu0 + 1 / mu))
        for irradiance, reflectance in zip((10.0, 20.0), (0.2, 0.4), strict=True)
    )
    assert result.values == pytest.approx(expected)
    with pytest.raises(ValueError, match="albedo"):
        aerosol(((0.1, 0.1),), albedo=1.1)
    with pytest.raises(ValueError, match="HG asymmetry"):
        aerosol(((0.1, 0.1),), asymmetry=0.9)


def test_aerosol_scattering_nonzero_asymmetry_and_explicit_coordinate_conversion() -> None:
    black = scene(surface=SurfaceState(wavenumber_grid(axis()), reflectance=np.zeros(2)))
    molecular = rayleigh(((0.0, 0.0),))
    particulate = aerosol(((0.02, 0.03),), asymmetry=0.6)
    result = terms_support.order_one(
        black,
        source=solar_source(solar(), direction=(30.0, 20.0)),
        optics=compose_atmospheric_optics(
            black.atmosphere,
            solar_source(solar(), direction=(30.0, 20.0)),
            rayleigh=rayleigh_component(molecular),
            aerosol=aerosol_component(particulate),
        ),
    )
    mu0 = cos(radians(30.0))
    mu = cos(radians(10.0))
    cosine = -mu0 * mu + sin(radians(30.0)) * sin(radians(10.0)) * cos(radians(20.0))
    phase = henyey_greenstein_phase(0.6, cosine)
    assert result.values == pytest.approx(
        tuple(
            irradiance
            * phase
            / (4 * pi * mu)
            * -expm1(-tau * (1 / mu0 + 1 / mu))
            / (1 / mu0 + 1 / mu)
            for irradiance, tau in zip(solar().values, particulate.optical_depth[0], strict=True)
        )
    )
    assert result.wavenumbers_cm1.flags.writeable is False
    np.testing.assert_array_equal(result.wavenumbers_cm1, axis())
