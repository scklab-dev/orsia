"""Separate physical output and numerical decompositions, with explicit ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

import numpy as np
import orsia_radiative
import pytest
from orsia_radiative import compose_atmospheric_optics, solar_source, transfer
from orsia_radiative.transfer import orders
from test_composable_transfer import case
from test_thermal import case as thermal_case


def test_removed_output_records_have_no_exports_or_compatibility_aliases() -> None:
    for name in ("RadianceComponent", "RadianceDiagnostic"):
        assert name not in orsia_radiative.__all__
        assert not hasattr(orsia_radiative, name)


def test_physical_radiance_fields_with_independent_equal_inputs() -> None:
    scene, illumination, rayleigh, aerosol = case()
    source = solar_source(illumination, direction=(30.0, 70.0))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    first = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    independent = replace(source, irradiance=replace(illumination))
    second = transfer.solar.reflected_radiance(
        replace(scene), max_order=0, delta_m_moments=None, source=independent, optics=optics
    )
    assert first.grid == second.grid
    assert first.geometry == second.geometry
    np.testing.assert_array_equal(first.values, second.values)
    assert [field.name for field in fields(first)] == ["grid", "values", "geometry"]


@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("max_order", [0, 1, 3])
def test_sos_rows_are_fresh_writable_arrays_and_sum_to_physical_output(
    moments: int | None, max_order: int
) -> None:
    scene, illumination, rayleigh, aerosol = case()
    source = solar_source(illumination, direction=(30.0, 70.0))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    kwargs = dict(
        source=source,
        optics=optics,
        max_order=max_order,
        angular_order=3,
        layer_subdivisions=2,
        delta_m_moments=moments,
    )
    result = transfer.solar.reflected_radiance(scene, **kwargs)
    rows = orders.scattering_orders(scene, **kwargs)
    assert rows.shape == (max_order + 1, illumination.values.size) and rows.dtype == np.float64
    assert rows.flags.writeable and np.all(np.isfinite(rows)) and np.all(rows >= 0)
    for borrowed in (
        illumination.values,
        result.values,
        rayleigh.optical_depth,
        aerosol.optical_depth,
        aerosol.scattering_depth,
        aerosol.asymmetry,
    ):
        assert not np.shares_memory(rows, borrowed)
    np.testing.assert_array_equal(rows.sum(axis=0), result.values)
    with ThreadPoolExecutor(max_workers=2) as pool:
        repeated = list(pool.map(lambda _: orders.scattering_orders(scene, **kwargs), range(2)))
    for values in repeated:
        np.testing.assert_array_equal(values, rows)
        assert not np.shares_memory(values, rows)
    assert not np.shares_memory(repeated[0], repeated[1])
    rows.fill(0)
    np.testing.assert_array_equal(repeated[0].sum(axis=0), result.values)


@pytest.mark.parametrize("method", ["quadrature", "expn"])
def test_thermal_terms_share_physics_but_not_storage_with_physical_radiance(method: str) -> None:
    scene, optics = thermal_case(np.array([[0.1, 0.4, 0.7]]), emissivity=0.4)
    # Public annotations remain authoritative; this parametrisation exercises the
    # two supported method literals without introducing runtime dispatch guards.
    from typing import Literal, cast

    selected = cast(Literal["quadrature", "expn"], method)
    result = transfer.thermal.schwarzschild_radiance(
        scene, optics=optics, hemispheric_method=selected
    )
    first = transfer.thermal.schwarzschild_terms(scene, optics=optics, hemispheric_method=selected)
    second = transfer.thermal.schwarzschild_terms(scene, optics=optics, hemispheric_method=selected)
    assert list(first) == [
        "surface_emission",
        "atmospheric_emission",
        "reflected_atmospheric_emission",
    ]
    np.testing.assert_array_equal(sum(first.values()), result.values)
    for name, values in first.items():
        assert values.flags.writeable and values.dtype == np.float64
        assert values.shape == result.values.shape
        assert not np.shares_memory(values, result.values)
        assert not np.shares_memory(values, second[name])
        values.fill(0)
    np.testing.assert_array_equal(sum(second.values()), result.values)
