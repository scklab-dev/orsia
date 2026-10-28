from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_radiative import (
    Absorber,
    Atmosphere,
    LayerProfileBasis,
    ObservationGeometry,
    RadiativeScene,
    SurfaceState,
    evaluate_profile,
    wavenumber_grid,
)


def atmosphere(**changes: Any) -> Atmosphere:
    defaults: dict[str, Any] = dict(
        height_boundaries_m=np.array([0.0, 1000.0, 2000.0]),
        pressure_boundaries_pa=np.array([100000.0, 90000.0, 80000.0]),
        pressure_pa=np.array([95000.0, 85000.0]),
        temperature_k=np.array([290.0, 280.0]),
        total_air_columns_molecules_m2=np.array([2.0e29, 1.8e29]),
        dry_air_columns_molecules_m2=np.array([1.9e29, 1.75e29]),
        water_vapour_columns_molecules_m2=np.array([1.0e28, 5.0e27]),
        absorbers=(Absorber("X", np.array([1.0e25, 2.0e25]), "dry_air"),),
    )
    defaults.update(changes)
    return Atmosphere(**defaults)


def axis() -> NDArray[np.float64]:
    return np.array([900.0, 901.0], dtype=np.float64)


def test_declared_layers_gases_and_broadener_are_preserved() -> None:
    state = atmosphere(absorbers=(Absorber("H2O", np.array([1.0e28, 5.0e27]), "total_air"),))
    assert len(state.pressure_boundaries_pa) == 3
    assert state.absorbers[0].air_broadener == "total_air"
    assert state.total_air_columns_molecules_m2[0] != state.dry_air_columns_molecules_m2[0]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"height_boundaries_m": np.array([0, 0, 2000], dtype=np.float64)}, "height boundaries"),
        (
            {"pressure_boundaries_pa": np.array([100000, 110000, 80000], dtype=np.float64)},
            "pressure boundaries",
        ),
        ({"pressure_pa": np.array([95000, 95000], dtype=np.float64)}, "layer pressure"),
        ({"temperature_k": np.array([290, -1], dtype=np.float64)}, "positive"),
        ({"water_vapour_columns_molecules_m2": np.array([0.0, 5.0e27])}, "total air"),
        ({"dry_air_columns_molecules_m2": np.array([1.9e29])}, "shape"),
        ({"pressure_pa": np.array([np.nan, 85000])}, "finite"),
        ({"absorbers": (Absorber("X", np.array([1e25, 2e25]), "dry_air"),) * 2}, "unique"),
        ({"absorbers": (Absorber("X", np.array([1e25]), "dry_air"),)}, "absorber columns"),
        ({"absorbers": (Absorber("H2O", np.array([1e25, 2e25]), "total_air"),)}, "conflicts"),
        ({"absorbers": (Absorber("X", np.array([2e29, 2e25]), "dry_air"),)}, "conflicts"),
    ],
)
def test_invalid_atmosphere_fails(changes: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        atmosphere(**changes)


def test_profile_basis_attaches_to_materialised_state_and_rejects_mismatch() -> None:
    state = atmosphere()
    profile = LayerProfileBasis("K", state.temperature_k, ("lapse",), np.array([[2.0, -2.0]]), 1)
    attached = replace(state, temperature_basis=profile)
    np.testing.assert_array_equal(evaluate_profile(profile, np.array([0.5])), [291.0, 279.0])
    new_state = replace(
        attached, temperature_k=evaluate_profile(profile, np.array([0.5])), temperature_basis=None
    )
    np.testing.assert_array_equal(state.temperature_k, [290.0, 280.0])
    np.testing.assert_array_equal(new_state.temperature_k, [291.0, 279.0])
    with pytest.raises(ValueError, match="basis"):
        replace(attached, temperature_k=np.array([291.0, 279.0]))
    with pytest.raises(ValueError, match="bounds"):
        evaluate_profile(profile, np.array([-200.0]))
    with pytest.raises(ValueError, match="coefficients"):
        evaluate_profile(profile, np.empty(0))
    with pytest.raises(ValueError, match="direction"):
        LayerProfileBasis("K", np.array([290.0, 280.0]), ("bad",), np.array([[1.0]]))
    with pytest.raises(ValueError, match="basis"):
        atmosphere(absorbers=(Absorber("X", np.array([1e25, 2e25]), "dry_air", profile),))


def test_surface_properties_do_not_select_illumination_or_observation_geometry() -> None:
    solar = SurfaceState(wavenumber_grid(axis()), reflectance=np.array([0.2, 0.3]))
    thermal = SurfaceState(
        wavenumber_grid(axis()), emissivity=np.array([0.8, 0.7]), temperature_k=285.0
    )
    mixed = SurfaceState(
        wavenumber_grid(axis()),
        reflectance=np.array([0.2, 0.3]),
        emissivity=np.array([0.8, 0.7]),
        temperature_k=285.0,
    )
    geometry = ObservationGeometry(10.0)
    for surface in (solar, thermal, mixed):
        scene = RadiativeScene(atmosphere(), surface, geometry)
        assert scene.surface == surface
        assert scene.geometry is geometry
    assert not hasattr(geometry, "solar_zenith_deg")
    assert not hasattr(geometry, "relative_azimuth_deg")


def test_surface_basis_spectral_domain_and_owned_array_values() -> None:
    values = np.array([0.2, 0.3])
    basis = LayerProfileBasis("1", values, ("tilt",), np.array([[0.1, -0.1]]), 0, 1, "spectral")
    surface = SurfaceState(wavenumber_grid(axis()), reflectance=values, reflectance_basis=basis)
    assert surface.reflectance is not values
    np.testing.assert_array_equal(
        evaluate_profile(basis, np.array([2.0])), [0.4, 0.09999999999999998]
    )
    with pytest.raises(ValueError, match="materialised state"):
        SurfaceState(
            wavenumber_grid(axis()), reflectance=np.array([0.1, 0.3]), reflectance_basis=basis
        )
    np.testing.assert_array_equal(surface.reflectance, [0.2, 0.3])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"reflectance": np.array([0.2])}, "reflectance"),
        ({"reflectance": np.array([1.1, 0.0])}, "bounds"),
        ({"emissivity": np.array([0.8, 0.7])}, "together"),
        ({"emissivity": np.array([0.8, 0.7]), "temperature_k": -1}, "positive"),
        (
            {
                "reflectance": np.array([0.2, 0.3]),
                "emissivity": np.array([0.8, 0.8]),
                "temperature_k": 280,
            },
            "opaque",
        ),
    ],
)
def test_invalid_surface_rejected(kwargs: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        SurfaceState(wavenumber_grid(axis()), **kwargs)


def test_geometry_and_support_reject_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="wavenumbers_cm1"):
        SurfaceState(wavenumber_grid(np.array([0.0, 1.0])), reflectance=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="positive"):
        SurfaceState(wavenumber_grid(np.array([0.0, 1.0])), reflectance=np.array([0.1, 0.2]))
    for invalid in (-1.0, 90.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="view zenith"):
            ObservationGeometry(invalid)
