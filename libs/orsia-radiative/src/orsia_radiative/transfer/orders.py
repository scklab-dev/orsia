"""Calculate fixed-order reflected solar radiance from atmospheric scattering."""

from importlib import import_module
from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_atmospheric_optics
from orsia_radiative.optics.phases import henyey_greenstein_phase, rayleigh_phase
from orsia_radiative.scene import RadiativeScene
from orsia_radiative.solar.source import SolarSource

from .native import propagation_native as _native

_native_spectral_orders = _native.bind_spectral_blas(import_module("scipy.linalg.cython_blas"))


def angular_grid(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Photon directions (positive z upward) and weights for the 4pi mean.

    Use order-point Gauss-Legendre on each open hemisphere and 2*order periodic
    azimuth points. Weights sum to one and represent dOmega/(4*pi).

    Args:
        order: Number of Gauss-Legendre nodes per open hemisphere.

    Returns:
        Fresh (directions,3) unit vectors and (directions,) solid-angle weights
        for the 4pi mean, with upward directions having positive z components.
    """
    nodes, weights = roots_legendre(order)
    hemisphere = (nodes + 1) / 2
    mu = np.repeat(np.concatenate((-hemisphere, hemisphere)), 2 * order)
    phi = np.tile(2 * pi * (np.arange(2 * order) + 0.5) / (2 * order), 2 * order)
    radial = np.sqrt(1 - mu**2)
    directions = np.column_stack((radial * np.cos(phi), radial * np.sin(phi), mu))
    measure = np.repeat(np.tile(weights / (8 * order), 2), 2 * order)
    return directions, measure


def phase_function(cosine: NDArray[np.float64], g: float | None) -> NDArray[np.float64]:
    """Evaluate scattering phase values for geometric angle cosines.

    Args:
        cosine: Array of scattering-angle cosines, clipped to [-1, 1].
        g: HG aerosol asymmetry coefficient, or None for Rayleigh scattering.

    Returns:
        Fresh dimensionless phase values with the broadcast cosine shape.

    Raises:
        ValueError: HG asymmetry lies outside its supported physical interval.
    """
    cosine = np.clip(cosine, -1, 1)
    if g is None:
        return rayleigh_phase(cosine)
    return henyey_greenstein_phase(g, cosine)


def scattering_orders(
    scene: RadiativeScene,
    *,
    source: SolarSource,
    optics: AtmosphericOptics,
    max_order: int = 2,
    angular_order: int = 8,
    layer_subdivisions: int = 8,
    delta_m_moments: int | None = None,
) -> NDArray[np.float64]:
    """Return orders 0..N with a Lambertian boundary at every atmospheric order.

    Physical orders count atmospheric scattering events. Order zero is direct
    surface reflection. Order one scatters the beam and order-zero surface field;
    later orders scatter the preceding field. Downward diffuse exits reflect at
    the same order, including repeated ground encounters separated by scattering.

    Return fresh writable float64 [order, spectral point], shape (N+1, points),
    in W m-2 sr-1 per (cm-1) on the source grid. Constituents sum within each row.
    The caller selects fixed N and angular/vertical refinement; application-level
    convergence and the omitted tail require separate assessment.

    Optional Delta-M removes the g>0 aerosol forward fraction g**K and retains
    K Legendre moments. Beam and diffuse transport share transformed extinction;
    absorption and Rayleigh stay physical. Orders count residual events, including
    transformed order zero. Signed truncated phases retain their values; negative
    order radiance raises ValueError. Intensity correction remains a separate
    scientific extension.

    Keep input arrays stable throughout each call, including concurrent calls.
    F-layout and strided arrays are supported. Spectral points run serially with
    invocation-owned scratch; BLAS thread policy belongs to the caller. The
    supported environment supplies the compiled extension and SciPy LP64 BLAS,
    bound when this module imports.

    Args:
        scene: Atmospheric, surface and viewing state on the spectral grid.
        source: Solar illumination with irradiance on the scene grid.
        optics: Precomposed absorption and scattering on the scene layers.
        max_order: Greatest atmospheric scattering order, including order zero.
        angular_order: Gauss-Legendre nodes per hemisphere for diffuse directions.
        layer_subdivisions: Vertical subcells used for the scattering calculation.
        delta_m_moments: Optional retained Legendre moments for Delta-M scaling.

    Returns:
        Fresh float64 radiance (max_order + 1, spectral_points) by order,
        in W m-2 sr-1 per cm-1 on the source grid.

    Raises:
        ValueError: Scene, source or optics are incompatible, the surface lacks
            reflectance, controls are out of range, or an order produces negative radiance.
    """
    validate_atmospheric_optics(scene, source, optics)
    active = bool(max_order) and (
        (optics.rayleigh is not None and bool(np.any(optics.rayleigh.optical_depth)))
        or (optics.aerosol is not None and bool(np.any(optics.aerosol.scattering_depth)))
    )
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if angular_order < 2 or layer_subdivisions < 1:
        raise ValueError("angular_order must be >= 2 and layer_subdivisions must be >= 1")
    if delta_m_moments is not None and not 2 <= delta_m_moments <= 2 * angular_order:
        raise ValueError("delta_m_moments must satisfy 2 <= K <= 2*angular_order")
    if not source.irradiance.grid.same_points(scene.surface.grid):
        raise ValueError("source and scene spectral axes must agree exactly")
    reflectance = scene.surface.reflectance
    if reflectance is None:
        raise ValueError("the reflected solar calculation requires surface reflectance")
    if source.irradiance.values.shape != (source.irradiance.wavenumbers_cm1.size,):
        raise ValueError("radiance values must match the spectral axis")
    directions, weights = (
        angular_grid(angular_order)
        if active
        else (np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64))
    )
    optical_arrays = (
        None if optics.absorption is None else optics.absorption.optical_depth,
        None if optics.rayleigh is None else optics.rayleigh.optical_depth,
        None if optics.aerosol is None else optics.aerosol.optical_depth,
        None if optics.aerosol is None else optics.aerosol.scattering_depth,
        None if optics.aerosol is None else optics.aerosol.asymmetry,
    )
    return _native_spectral_orders(
        tuple(None if a is None else np.asarray(a, dtype=np.float64) for a in optical_arrays),
        np.asarray(source.irradiance.values, dtype=np.float64),
        np.asarray(reflectance, dtype=np.float64),
        directions,
        weights,
        len(scene.atmosphere.pressure_pa),
        layer_subdivisions,
        max_order,
        0 if delta_m_moments is None else delta_m_moments,
        source.zenith_deg,
        scene.geometry.view_zenith_deg,
        source.relative_azimuth_deg,
    )


__all__ = [
    "angular_grid",
    "phase_function",
    "scattering_orders",
]
