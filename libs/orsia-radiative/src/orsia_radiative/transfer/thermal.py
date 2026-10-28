"""Compute thermal top-of-atmosphere radiance and its emitting contributions."""

from __future__ import annotations

from importlib import import_module
from math import cos, radians
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import constants
from scipy.special import roots_legendre

from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_scene_optics
from orsia_radiative.radiance import Radiance
from orsia_radiative.scene import RadiativeScene

from .native import propagation_native as _native

_native_terms, _native_intervals = _native.bind_thermal_expn(
    import_module("scipy.special.cython_special"),
    100.0 * constants.h * constants.c / constants.k,
    2.0 * constants.k * constants.c * 1e6,
)


def schwarzschild_radiance(
    scene: RadiativeScene,
    *,
    optics: AtmosphericOptics,
    angular_order: int = 32,
    hemispheric_method: Literal["quadrature", "expn"] = "quadrature",
) -> Radiance:
    """Top-of-atmosphere LTE radiance; see ``schwarzschild_terms`` for physics.

    Return owned spectral radiance values on the optics grid and scene geometry,
    without a retained source, method identity, decomposition or diagnostics.

    Args:
        scene: Layer temperatures and emitting surface with viewing geometry.
        optics: Atmospheric absorption on the scene spectral and layer grid.
        angular_order: Angular integration order for reflected atmospheric emission.
        hemispheric_method: Quadrature or analytic exponential-integral treatment
            of downward emission incident on the surface.

    Returns:
        Radiance on the optics grid and scene viewing geometry, in
        W m-2 sr-1 per cm-1.

    Raises:
        ValueError: Scene or optics are incompatible, atmospheric scattering is
            present, surface emission is unspecified or integration controls are invalid.
    """
    terms = schwarzschild_terms(
        scene,
        optics=optics,
        angular_order=angular_order,
        hemispheric_method=hemispheric_method,
    )
    return Radiance(
        optics.grid,
        terms["surface_emission"]
        + terms["atmospheric_emission"]
        + terms["reflected_atmospheric_emission"],
        scene.geometry,
    )


def schwarzschild_terms(
    scene: RadiativeScene,
    *,
    optics: AtmosphericOptics,
    angular_order: int = 32,
    hemispheric_method: Literal["quadrature", "expn"] = "quadrature",
) -> dict[str, NDArray[np.float64]]:
    """Top-of-atmosphere upward radiance from isothermal absorbing layers.

    Sampling comes from precomposed optics, exactly matching the surface; layer
    T comes from the bound atmosphere. All atmospheric scattering depths must
    be zero (absorbing aerosols are allowed and emit at layer T). The surface
    requires emissivity and temperature; its opaque reflectivity is 1-emissivity,
    including emissivity-only surfaces. Top incident radiance is zero.

    Return separate additive surface_emission, atmospheric_emission (directly
    observed), and reflected_atmospheric_emission arrays. Each is fresh writable
    float64 (points,) in W m-2 sr-1 per (cm-1) on the optics grid; inputs are
    read-only and not retained. For
    the last term, angular_order >= 2 Gauss-Legendre nodes on (0, 1) integrate
    2*integral(mu*I_down(mu), dmu), the downward irradiance divided by pi.
    Refine this control independently; the default has no general accuracy
    envelope. ``hemispheric_method='expn'`` instead integrates each layer using
    2*[E3(below)-E3(below+depth)]. Thin intervals use a midpoint expansion
    of E2; intervals near zero use its integrated logarithmic series.
    angular_order is validated for both methods but only used by quadrature.
    Ray transport is exact within the supplied isothermal layers.
    Python validates physical inputs and supplies angular nodes; the serial C++
    kernel evaluates Planck, depth assembly, transport and reflection and checks
    each output term's finite nonnegative support in one detached call.
    Stable contiguous aligned inputs are borrowed read-only;
    other layouts receive temporary copies. Each invocation owns its scratch.

    Args:
        scene: Temperature-bearing atmosphere, opaque emitting surface and view.
        optics: Absorption-bearing atmospheric optics on the scene spectral grid.
        angular_order: Positive hemispheric quadrature order of at least two.
        hemispheric_method: ``quadrature`` or ``expn`` for the reflected term.

    Returns:
        Named surface, atmospheric and surface-reflected atmospheric emission
        arrays (points,) in W m-2 sr-1 per cm-1; summing them gives TOA radiance.

    Raises:
        ValueError: Incompatible scene support, invalid controls or temperature,
            absent surface emission, or atmospheric scattering.
    """
    validate_scene_optics(scene, optics)
    if isinstance(angular_order, bool) or not isinstance(angular_order, int) or angular_order < 2:
        raise ValueError("thermal angular_order must be an integer >= 2")
    if hemispheric_method not in ("quadrature", "expn"):
        raise ValueError("thermal hemispheric_method must be 'quadrature' or 'expn'")
    if optics.rayleigh is not None and np.any(optics.rayleigh.optical_depth > 0):
        raise ValueError("Schwarzschild thermal transfer does not support atmospheric scattering")
    if optics.aerosol is not None and np.any(optics.aerosol.scattering_depth > 0):
        raise ValueError("Schwarzschild thermal transfer does not support atmospheric scattering")
    emissivity = scene.surface.emissivity
    temperature = scene.surface.temperature_k
    if emissivity is None or temperature is None:
        raise ValueError("thermal surface requires emissivity and temperature")
    temperatures = scene.atmosphere.temperature_k
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("temperature must be positive finite scalar or layer values")
    mu = weights = np.empty(0, dtype=np.float64)
    if (
        hemispheric_method == "quadrature"
        and np.any(emissivity < 1)
        and any(
            component is not None and np.any(component.optical_depth > 0)
            for component in (optics.absorption, optics.aerosol)
        )
    ):
        nodes, weights = roots_legendre(angular_order)
        mu = (nodes + 1) / 2
    view_mu = cos(radians(scene.geometry.view_zenith_deg))
    surface_emission, atmospheric, reflected = _native_terms(
        _thermal_buffer(optics.wavenumbers_cm1),
        _thermal_buffer(temperatures),
        _thermal_buffer(emissivity),
        None if optics.absorption is None else _thermal_buffer(optics.absorption.optical_depth),
        None if optics.aerosol is None else _thermal_buffer(optics.aerosol.optical_depth),
        mu,
        weights,
        temperature,
        view_mu,
        hemispheric_method == "expn",
    )
    return {
        "surface_emission": surface_emission,
        "atmospheric_emission": atmospheric,
        "reflected_atmospheric_emission": reflected,
    }


def _thermal_buffer(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Supply aligned C-contiguous binary64 storage to the thermal native kernel.

    Returns:
        The input array when already suitable, otherwise a normalized array.
    """
    if values.flags.c_contiguous and values.flags.aligned and values.dtype == np.float64:
        return values
    return np.require(values, dtype=np.float64, requirements=["C", "A"])


def _e3_difference(below: NDArray[np.float64], depth: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate positive E2 intervals while retaining separate sub-ulp depths.

    Returns:
        Native interval integral for each supplied depth and lower boundary.
    """
    return _native_intervals(
        np.require(below, dtype=np.float64, requirements=["C", "A"]),
        np.require(depth, dtype=np.float64, requirements=["C", "A"]),
    )


__all__ = ["schwarzschild_radiance", "schwarzschild_terms"]
