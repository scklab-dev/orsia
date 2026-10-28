"""Frozen pre-port thermal calculation from Orsia revision 8af080c.

This test-only NumPy/SciPy reference retains the original arithmetic and branch
partition for bounded comparisons with the native implementation.
"""

from __future__ import annotations

from math import cos, radians
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_scene_optics
from orsia_radiative.radiance import Radiance
from orsia_radiative.radiometry import planck_wavenumber
from orsia_radiative.scene import RadiativeScene
from orsia_radiative.transfer.formal import schwarzschild
from scipy.special import expn, roots_legendre


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
    Inputs are borrowed without mutation; numerical buffers are call-local, without caches or
    parallel policy. Derivatives are unavailable. No fictitious Sun is retained.
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
    grid = optics.wavenumbers_cm1
    depths = np.zeros((scene.atmosphere.temperature_k.size, grid.size), dtype=np.float64)
    if optics.absorption is not None:
        depths += optics.absorption.optical_depth
    if optics.aerosol is not None:
        depths += optics.aerosol.optical_depth
    layer_planck = planck_wavenumber(grid, scene.atmosphere.temperature_k)
    surface_planck = planck_wavenumber(grid, temperature)
    zero = np.zeros(grid.size, dtype=np.float64)
    view_mu = cos(radians(scene.geometry.view_zenith_deg))
    atmospheric = schwarzschild(layer_planck, depths, view_mu, zero)
    with np.errstate(over="ignore", under="ignore"):
        transmission = np.exp(-np.sum(depths / view_mu, axis=0))
    surface_emission = emissivity * surface_planck * transmission
    reflected = zero.copy()
    reflectivity = 1 - emissivity
    if np.any(reflectivity > 0) and np.any(depths > 0):
        if hemispheric_method == "expn":
            below = np.zeros_like(depths)
            with np.errstate(over="ignore"):
                np.cumsum(depths[:-1], axis=0, out=below[1:])
            # Batch layers as well as spectral points; d remains separate from
            # cumulative depth so sub-ulp intervals retain their contribution.
            reflected = 2 * np.sum(layer_planck * _e3_difference(below, depths), axis=0)
        else:
            nodes, weights = roots_legendre(angular_order)
            mu = (nodes + 1) / 2
            # Mapping weights to [0,1] cancels the factor 2 in irradiance/pi.
            for direction, weight in zip(mu, weights * mu, strict=True):
                reflected += weight * schwarzschild(layer_planck, depths, -float(direction), zero)
        reflected *= reflectivity * transmission
    terms = {
        "surface_emission": surface_emission,
        "atmospheric_emission": atmospheric,
        "reflected_atmospheric_emission": reflected,
    }
    for values in terms.values():
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("thermal radiance terms must be finite and nonnegative")
    return terms


def _e3_difference(below: NDArray[np.float64], depth: NDArray[np.float64]) -> NDArray[np.float64]:
    """Positive integral of E2 from a=below to a+d, keeping d separately.

    For d <= .02*min(a, 1), integrate the midpoint Taylor series through
    fourth order. Its next term is d**7*E_-4(midpoint)/322560; the two
    interval bounds control both the exponential scale and distance from the
    logarithmic singularity. At the origin, use the E3 recurrence with expm1.
    Remaining intervals ending below .5 use the convergent integrated E2
    series through degree 16 (next term < 2e-21 relative to d).
    Elsewhere direct subtraction is well conditioned by these interval bounds.

    This partition avoids angular/depth quadrature and retains sub-ulp depths.
    Opaque underflow and zero depth return zero. Buffers are call-local.
    """
    result = np.zeros_like(depth)
    active = (depth > 0) & (below < 745)
    if not np.any(active):
        return result
    with np.errstate(over="ignore"):
        upper = below + depth
    thin = active & (below > 0) & (depth <= 0.02 * np.minimum(below, 1))
    # Preserve the cheap path for wholly well-separated / opaque intervals.
    if np.all(~active | (~thin & (upper > 0.5))):
        result[active] = _thermal_expn(3, below[active]) - _thermal_expn(
            3, np.minimum(upper[active], np.finfo(np.float64).max)
        )
        return result
    origin = active & (below == 0) & (depth <= 1)
    if np.any(origin):
        d = depth[origin]
        result[origin] = (-np.expm1(-d) + d * (np.exp(-d) - d * expn(1, d))) / 2

    if np.any(thin):
        a, d = below[thin], depth[thin]
        midpoint = a + d / 2
        ratio = d / midpoint
        # Factored powers keep the derivatives finite as a approaches zero.
        correction = (
            np.exp(-midpoint)
            * (d * ratio)
            * (1 / 24 + (d * d + 2 * d * ratio + 2 * ratio * ratio) / 1920)
        )
        result[thin] = d * (_thermal_expn(2, midpoint) + correction)

    series = active & ~origin & ~thin & (upper <= 0.5)
    if np.any(series):
        a, d, b = below[series], depth[series], upper[series]
        # Here d > .02*a and a > 0, so a/d and log(b)-log(a) are safe.
        average = (
            1
            + (a + b) * (0.5 * np.log(b) + np.euler_gamma / 2 - 0.75)
            + 0.5 * a * (a / d) * (np.log(b) - np.log(a))
        )
        # q_n = (b**n-a**n)/d, computed without subtracting powers.
        divided_power = a + b
        a_power = a.copy()
        coefficient = -0.5
        for degree in range(2, 17):
            a_power *= a
            divided_power = b * divided_power + a_power
            average += (coefficient / (degree + 1)) * divided_power
            coefficient *= -(degree - 1) / (degree * (degree + 1))
        result[series] = d * average

    direct = active & ~origin & ~thin & ~series
    if np.any(direct):
        result[direct] = _thermal_expn(3, below[direct]) - _thermal_expn(
            3, np.minimum(upper[direct], np.finfo(np.float64).max)
        )
    return result


def _thermal_expn(order: Literal[2, 3], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """E2/E3 with the representable opaque tail retained beyond SciPy's cutoff.

    For x >= 700 the eight-term asymptotic series has next relative term
    below 6e-19 for these orders. Final binary64 subnormal rounding dominates.
    """
    result = expn(order, x)
    opaque = (x >= 700) & (x < 745)
    if np.any(opaque):
        a = x[opaque]
        term = np.ones_like(a)
        total = term.copy()
        for degree in range(1, 9):
            term *= -(order + degree - 1) / a
            total += term
        result[opaque] = np.exp(-a) * (total / a)
    return result


__all__ = ["schwarzschild_radiance", "schwarzschild_terms"]
