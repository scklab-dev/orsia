"""Direct initialization, physical support and the solar radiance boundary."""

from dataclasses import replace
from math import cos, pi, radians
from unittest.mock import patch

import numpy as np
import orders_reference as reference
import pytest
from numpy.typing import NDArray
from orsia_radiative import Absorption, Radiance, compose_atmospheric_optics, wavenumber_grid
from orsia_radiative.transfer import orders, solar
from test_sos_delta_m import _inputs


@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("max_order", [0, 1, 4])
def test_public_radiance_and_orders_share_direct_initialization(
    moments: int | None, max_order: int
) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    gas = Absorption(source.irradiance.grid, np.full((2, 2), 0.02))
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, absorption=gas, rayleigh=rayleigh, aerosol=aerosol
    )
    options = dict(max_order=max_order, angular_order=3, layer_subdivisions=2)
    contributions = orders.scattering_orders(
        scene, source=source, optics=optics, delta_m_moments=moments, **options
    )
    with patch.object(solar, "Radiance", wraps=Radiance) as construction:
        result = solar.reflected_radiance(
            scene, source=source, optics=optics, delta_m_moments=moments, **options
        )
    construction.assert_called_once()
    assert isinstance(result, Radiance)
    assert result.grid == source.irradiance.grid and result.geometry == scene.geometry
    assert contributions.shape == (max_order + 1, 2)
    assert contributions.dtype == np.float64 and contributions.flags.writeable
    assert result.values.flags.writeable
    assert not np.shares_memory(result.values, contributions)
    np.testing.assert_array_equal(result.values, contributions.sum(axis=0))
    extinction = gas.optical_depth + rayleigh.optical_depth + aerosol.optical_depth
    if moments is not None:
        extinction -= np.maximum(aerosol.asymmetry, 0) ** moments * aerosol.scattering_depth
    mu0 = cos(radians(source.zenith_deg))
    mu = cos(radians(scene.geometry.view_zenith_deg))
    assert scene.surface.reflectance is not None
    expected = source.irradiance.values * scene.surface.reflectance * mu0 / pi
    expected *= np.exp(-extinction.sum(axis=0) * (1 / mu0 + 1 / mu))
    np.testing.assert_allclose(contributions[0], expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize("moments", [None, 4])
@pytest.mark.parametrize("across", ["layers", "components"])
def test_physical_summed_overflow_precedes_delta_m(across: str, moments: int | None) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    # Caller-owned optical arrays can change after composition. Exercise the
    # transfer boundary itself, including a finite transformed layer sum.
    rayleigh.optical_depth.fill(0)
    aerosol.asymmetry.fill(0.8)
    if across == "layers":
        aerosol.optical_depth.fill(9e307)
        aerosol.scattering_depth.fill(9e307)
    else:
        rayleigh.optical_depth[0] = 9e307
        aerosol.optical_depth[0] = 9e307
        aerosol.scattering_depth[0] = 9e307
    with pytest.raises(ValueError, match="summed optical depth overflow"):
        solar.reflected_radiance(
            scene, source=source, optics=optics, max_order=0, delta_m_moments=moments
        )


@pytest.mark.parametrize(
    "case",
    [
        "source_grid",
        "surface_grid",
        "layers",
        "reflectance",
        "depth_nan",
        "source_nan",
        "source_negative",
        "source_shape",
    ],
)
@pytest.mark.parametrize("moments", [None, 4])
def test_initialization_preserves_support_and_finiteness_errors(
    case: str, moments: int | None
) -> None:
    scene, source, rayleigh, aerosol = _inputs()
    optics = compose_atmospheric_optics(
        scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol
    )
    message = "spectral axes"
    shifted = wavenumber_grid(source.irradiance.grid.points + 1)
    if case == "source_grid":
        source = replace(source, irradiance=replace(source.irradiance, grid=shifted))
    elif case == "surface_grid":
        scene = replace(scene, surface=replace(scene.surface, grid=shifted))
    elif case == "layers":
        optics = replace(
            optics, rayleigh=replace(rayleigh, optical_depth=np.ones((1, 2))), aerosol=None
        )
        message = "layer order"
    elif case == "reflectance":
        scene = replace(
            scene,
            surface=replace(
                scene.surface, reflectance=None, emissivity=np.ones(2), temperature_k=280.0
            ),
        )
        message = "requires surface reflectance"
    elif case == "depth_nan":
        aerosol.optical_depth[0, 0] = np.nan
        message = "summed optical depth overflow"
    elif case == "source_nan":
        source.irradiance.values[0] = np.nan
        message = "radiance values must be finite"
    elif case == "source_negative":
        source.irradiance.values[0] = -1.0
        message = "radiance values must be finite and nonnegative"
    else:
        source.irradiance.values.resize((2, 1), refcheck=False)
        message = "radiance values must match the spectral axis"
    with pytest.raises(ValueError, match=message):
        solar.reflected_radiance(
            scene, source=source, optics=optics, max_order=0, delta_m_moments=moments
        )


@pytest.mark.parametrize("invalid", [-1.0, np.inf, np.nan])
def test_accumulated_order_validation_remains_at_numerical_boundary(invalid: float) -> None:
    scene, source, rayleigh, _ = _inputs()
    optics = compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh)

    def inject(
        scene, source, extinction, scattering, contributions: NDArray[np.float64], **options
    ):
        contributions[1, 0] = invalid

    with (
        patch.object(reference, "_accumulate_orders", side_effect=inject),
        pytest.raises(ValueError),
    ):
        reference.scattering_orders(scene, source=source, optics=optics, max_order=1)


def test_public_radiance_validates_summed_orders() -> None:
    scene, source, rayleigh, _ = _inputs()
    optics = compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh)
    finite_orders = np.full((2, 2), 1e308)
    with (
        patch.object(orders, "scattering_orders", return_value=finite_orders),
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="radiance values must be finite"),
    ):
        solar.reflected_radiance(scene, source=source, optics=optics, max_order=1)
