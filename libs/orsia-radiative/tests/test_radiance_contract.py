"""Rejected physical/support/provenance contracts at radiance exchange boundaries."""

from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_radiative import (
    ObservationGeometry,
    Radiance,
    SolarSource,
    SpectralIrradiance,
    solar_source,
    wavenumber_grid,
)


def irradiance() -> SpectralIrradiance:
    return SpectralIrradiance(wavenumber_grid(np.array([1000.0, 1001.0])), np.array([2.0, 3.0]))


def radiance() -> Radiance:
    source = solar_source(irradiance(), direction=(30.0, 70.0))
    return Radiance(
        wavenumber_grid(source.irradiance.wavenumbers_cm1),
        np.array([1.0, 2.0]),
        ObservationGeometry(20.0),
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"values": np.array([1.0])}, "spectral axis"),
        ({"values": np.array([-1.0, 2.0])}, "nonnegative"),
        ({"values": np.array([np.nan, 2.0])}, "finite"),
    ],
)
def test_invalid_irradiance(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(irradiance(), **changes)


@pytest.mark.parametrize(
    "direction",
    [(float("inf"), 0.0), (0.0, float("nan")), (0.0, float("inf")), (0.0, -1.0)],
)
def test_invalid_solar_direction(direction: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="degrees"):
        solar_source(irradiance(), direction=direction)


def test_solar_source_preserves_irradiance_and_explicit_angles() -> None:
    incident = irradiance()
    source = solar_source(incident, direction=(0.0, 359.0))
    assert source != SolarSource(incident, 0.0, 359.0)
    assert source.irradiance is incident
    assert source.zenith_deg == 0.0 and source.relative_azimuth_deg == 359.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"values": np.array([1.0])}, "spectral axis"),
        ({"values": np.array([-1.0, 2.0])}, "nonnegative"),
        ({"values": np.array([np.nan, 2.0])}, "finite"),
        ({"values": np.array([np.inf, 2.0])}, "finite"),
    ],
)
def test_invalid_radiance_support(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(radiance(), **changes)


def test_radiance_contains_only_physical_value_and_support() -> None:
    assert [field.name for field in fields(radiance())] == ["grid", "values", "geometry"]


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "source_id",
        "input_source_ids",
        "components",
        "diagnostics",
        "radiative_source",
        "surface_coupling_source_ids",
    ],
)
def test_report_and_provenance_fields_are_not_accepted(field: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        replace(radiance(), **cast(Any, {field: None}))
