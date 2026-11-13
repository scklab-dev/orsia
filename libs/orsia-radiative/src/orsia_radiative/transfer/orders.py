"""Calculate fixed-order reflected solar radiance from atmospheric scattering."""

from importlib import import_module
from math import pi
from operator import index

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from orsia_radiative.columns import ColumnScene
from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_atmospheric_optics
from orsia_radiative.optics.phases import henyey_greenstein_phase, rayleigh_phase
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.transfer.native import transfer_native as _native

_native_spectral_orders = _native.bind_spectral_blas(import_module("scipy.linalg.cython_blas"))
_native_scattering_orders_tangent = _native.bind_scattering_tangent_blas(
    import_module("scipy.linalg.cython_blas")
)


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
    scene: ColumnScene,
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


def scattering_orders_tangent(
    scene: ColumnScene,
    *,
    source: SolarSource,
    optics: AtmosphericOptics,
    optical_depth_tangent: NDArray[np.float64],
    reflectance_tangent: NDArray[np.float64],
    rayleigh_optical_depth_tangent: NDArray[np.float64] | None = None,
    aerosol_optical_depth_tangent: NDArray[np.float64] | None = None,
    aerosol_scattering_depth_tangent: NDArray[np.float64] | None = None,
    aerosol_asymmetry_tangent: NDArray[np.float64] | None = None,
    max_order: int = 2,
    angular_order: int = 8,
    layer_subdivisions: int = 8,
    delta_m_moments: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute solar scattering orders and analytical constituent directions.

    Differentiate the same fixed-order recurrence as ``scattering_orders``,
    including beam attenuation, diffuse transport and every Lambertian boundary.
    Geometry, source irradiance, grids and numerical controls stay fixed.
    ``optical_depth_tangent`` retains its absorption-depth meaning. Total physical
    extinction directions are absorption + Rayleigh + aerosol extinction, each
    counted once. Aerosol scattering depth is independent of aerosol extinction.
    For scattering depth s = omega*tau, callers supply ds = omega*dtau + tau*domega.
    Optional absent seeds are zero; absent optical roles have zero primal values.

    Delta-M differentiates the transformed extinction, scattering depth and phase
    moments on the g>0 branch. At g=0 the first derivatives agree across branches
    for K>=2. Directional seeds may be signed at physical endpoints; interpretation
    there is a derivative of the smooth extension, with one-sided feasible checks.
    Signed truncated phases and negative-order errors follow ``scattering_orders``.

    All inputs are borrowed read-only and must stay stable during a call, including
    concurrent calls. Native float64 strided, unaligned and read-only arrays work.
    Serial native directional sweeps use call-owned scratch and no finite
    differences. Cost grows linearly with k; each sweep recomputes primal fields.
    The value and derivative describe the selected discretization and finite order;
    callers must assess convergence for their application.

    Args:
        scene: Fixed bottom-to-top atmospheric layers, Lambertian surface and view.
        source: Fixed beam-normal irradiance per cm-1 and incident direction.
        optics: Physical absorption, Rayleigh and HG aerosol optics on scene support.
        optical_depth_tangent: Finite signed dimensionless absorption-depth seeds,
            shape (layers, points, k). This is not total-extinction depth.
        reflectance_tangent: Finite signed dimensionless seeds, shape (points, k).
        rayleigh_optical_depth_tangent: Optional (layers, points, k) seeds that
            change both Rayleigh extinction and Rayleigh scattering depth.
        aerosol_optical_depth_tangent: Optional (layers, points, k) aerosol
            extinction seeds. These do not implicitly change scattering depth.
        aerosol_scattering_depth_tangent: Optional (layers, points, k) independent
            aerosol scattering-depth seeds, dimensionless per direction.
        aerosol_asymmetry_tangent: Optional (layers, points, k) dimensionless HG g seeds.
        max_order: Greatest atmospheric event order, including order zero.
        angular_order: Fixed Gauss-Legendre nodes per hemisphere.
        layer_subdivisions: Fixed vertical subcells per atmospheric layer.
        delta_m_moments: Optional fixed retained Legendre moment count for Delta-M.

    Returns:
        Owned writable C-contiguous float64 values (max_order+1, points) and
        tangents (max_order+1, points, k), where arbitrary k may be zero. Units are
        W m-2 sr-1 per (cm-1), and those radiance units per direction, respectively.
        Outputs preserve inputs and share storage with neither inputs nor each other.

    Raises:
        ValueError: Support, native float64 dtype, finite values, physical bounds
            or numerical controls are invalid, reflectance is missing, arithmetic
            overflows, or Delta-M produces a negative order radiance.
        OverflowError: Output or workspace dimensions exceed addressable storage.
        TypeError: A numerical count control does not support integer indexing.
    """
    validate_atmospheric_optics(scene, source, optics)
    max_order, angular_order, layer_subdivisions = (
        index(max_order),
        index(angular_order),
        index(layer_subdivisions),
    )
    if delta_m_moments is not None:
        delta_m_moments = index(delta_m_moments)
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if angular_order < 2 or layer_subdivisions < 1:
        raise ValueError("angular_order must be >= 2 and layer_subdivisions must be >= 1")
    if delta_m_moments is not None and not 2 <= delta_m_moments <= 2 * angular_order:
        raise ValueError("delta_m_moments must satisfy 2 <= K <= 2*angular_order")
    reflectance = scene.surface.reflectance
    if reflectance is None:
        raise ValueError("the reflected solar tangent requires surface reflectance")
    # Prepare angular support even for zero primal scattering: its derivative
    # can supply a first-order source, including when the optical role is absent.
    directions, weights = (
        angular_grid(angular_order)
        if max_order
        else (np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64))
    )
    return _native_scattering_orders_tangent(
        (
            None if optics.absorption is None else optics.absorption.optical_depth,
            None if optics.rayleigh is None else optics.rayleigh.optical_depth,
            None if optics.aerosol is None else optics.aerosol.optical_depth,
            None if optics.aerosol is None else optics.aerosol.scattering_depth,
            None if optics.aerosol is None else optics.aerosol.asymmetry,
        ),
        (
            optical_depth_tangent,
            rayleigh_optical_depth_tangent,
            aerosol_optical_depth_tangent,
            aerosol_scattering_depth_tangent,
            aerosol_asymmetry_tangent,
        ),
        source.irradiance.values,
        reflectance,
        reflectance_tangent,
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
    "scattering_orders_tangent",
]
