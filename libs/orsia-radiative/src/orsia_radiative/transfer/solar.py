"""Reflected-solar closure over one incident direction and atmospheric optics.

Optical composition and spectroscopy are upstream. Each call owns its numerical
scratch and returns independently owned radiance values.
"""

from __future__ import annotations

from operator import index

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.columns import ColumnScene
from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_atmospheric_optics
from orsia_radiative.radiance import Radiance
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.transfer import orders
from orsia_radiative.transfer.native import transfer_native


def reflected_radiance(
    scene: ColumnScene,
    *,
    source: SolarSource,
    optics: AtmosphericOptics,
    max_order: int = 2,
    angular_order: int = 8,
    layer_subdivisions: int = 8,
    delta_m_moments: int | None = None,
) -> Radiance:
    """Compute reflected-solar radiance with experimental fixed-order scalar SOS.

    Reflect each order's downward field at the Lambertian ground. N counts
    atmospheric events; ``orders.scattering_orders`` aggregates constituents by order.
    Physical direct reflection uses ``max_order=0, delta_m_moments=None``.
    Order one includes Rayleigh and HG ground coupling, including two ground
    encounters separated by one atmospheric event. Rayleigh and HG act on the
    same preceding field under common extinction. Angular and vertical controls
    require convergence assessment for the caller's application. Inputs are
    borrowed read-only; work buffers and returned radiance are call-local.
    Each call evaluates current source, surface, geometry and optical values.
    Positive orders prepare angular support when scattering is active.

    ``delta_m_moments=K`` opts into experimental aerosol forward-peak removal,
    retaining Legendre degrees 0..K-1 (2 <= K <= 2*angular_order). Only g>0
    is transformed; g<=0 retains exact HG. Orders then count residual events,
    and N=0 returns transformed, not physical, direct reflection. Physical
    inputs are unchanged. No directional intensity correction is applied.
    A signed truncated phase yielding negative order radiance raises ValueError;
    refine K/angular sampling rather than clipping or renormalizing the phase.

    Args:
        scene: Geometry, surface and atmospheric state for the current evaluation.
        source: Incident solar irradiance and direction on its spectral grid.
        optics: Composed atmospheric extinction and scattering on the source grid.
        max_order: Highest atmospheric scattering-event order to include.
        angular_order: Angular quadrature order used for scattering integration.
        layer_subdivisions: Vertical integration subdivisions per layer.
        delta_m_moments: Optional retained forward-peak Legendre moment count.

    Returns:
        Independently owned reflected-solar radiance on the source spectral grid
        and scene geometry, summed through the selected atmospheric order.

    Raises:
        ValueError: If numerical support is invalid or a truncated signed phase
            produces negative order radiance.
    """
    contributions = orders.scattering_orders(
        scene,
        source=source,
        optics=optics,
        max_order=max_order,
        angular_order=angular_order,
        layer_subdivisions=layer_subdivisions,
        delta_m_moments=delta_m_moments,
    )
    return Radiance(source.irradiance.grid, contributions.sum(axis=0), scene.geometry)


def reflected_radiance_tangent(
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
    max_order: int = 0,
    angular_order: int = 8,
    layer_subdivisions: int = 8,
    delta_m_moments: int | None = None,
) -> tuple[Radiance, NDArray[np.float64]]:
    """Compute reflected solar radiance and analytical constituent directions.

    Default ``max_order=0`` preserves direct absorption-only calls. Positive
    orders differentiate the fixed-order scalar SOS recurrence, including all
    surface couplings and optional Delta-M transformations. Geometry, irradiance,
    spectral sampling, layer support and numerical controls stay fixed.
    ``optical_depth_tangent`` remains absorption depth. Total physical extinction
    directions sum absorption, Rayleigh and aerosol-extinction seeds once.
    Aerosol scattering-depth seeds are independent; for s=omega*tau supply
    ds=omega*dtau+tau*domega. See ``orders.scattering_orders_tangent`` for branch,
    endpoint and numerical-convergence semantics and order-resolved outputs.

    Inputs are borrowed read-only and must stay stable during concurrent calls.
    Native float64 strided, unaligned and read-only arrays are supported.
    Computation is serial native code with call-owned scratch and no finite
    differences. Optical freshness remains the caller's responsibility.

    Args:
        scene: Fixed bottom-to-top atmospheric layers, Lambertian surface and view.
        source: Fixed beam-normal irradiance per cm-1 and solar direction.
        optics: Physical absorption, Rayleigh and HG aerosol optics on exactly
            the scene/source support. Absent roles supply zero primal values.
        optical_depth_tangent: Finite signed dimensionless (layers, points, k)
            absorption directions. The last axis labels caller-defined directions.
        reflectance_tangent: Finite signed dimensionless (points, k) reflectance
            directions with the same runtime k, which may be zero.
        rayleigh_optical_depth_tangent: Optional (layers, points, k) dimensionless
            seeds for both Rayleigh extinction and scattering depth.
        aerosol_optical_depth_tangent: Optional (layers, points, k) dimensionless
            aerosol extinction seeds, independent of scattering depth.
        aerosol_scattering_depth_tangent: Optional (layers, points, k) dimensionless
            aerosol scattering-depth seeds. Absent optional seeds are zero.
        aerosol_asymmetry_tangent: Optional (layers, points, k) dimensionless HG g seeds.
        max_order: Highest fixed atmospheric event order; default zero is direct.
        angular_order: Fixed Gauss-Legendre nodes per hemisphere for scattering.
        layer_subdivisions: Fixed vertical subcells per atmospheric layer.
        delta_m_moments: Optional fixed retained Legendre count for Delta-M scaling.

    Returns:
        Owned radiance and owned writable C-contiguous float64 (points, k)
        tangents, in W m-2 sr-1 per (cm-1) per direction. Outputs preserve inputs
        and share storage with neither inputs nor each other.

    Raises:
        ValueError: Support, native float64 dtype, finite values or physical
            bounds are invalid, reflectance is missing, or arithmetic overflows.
        OverflowError: Output dimensions exceed addressable storage.
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
    if (
        max_order != 0
        or delta_m_moments is not None
        or optics.rayleigh is not None
        or optics.aerosol is not None
        or any(
            seed is not None
            for seed in (
                rayleigh_optical_depth_tangent,
                aerosol_optical_depth_tangent,
                aerosol_scattering_depth_tangent,
                aerosol_asymmetry_tangent,
            )
        )
    ):
        values, tangent = orders.scattering_orders_tangent(
            scene,
            source=source,
            optics=optics,
            optical_depth_tangent=optical_depth_tangent,
            reflectance_tangent=reflectance_tangent,
            rayleigh_optical_depth_tangent=rayleigh_optical_depth_tangent,
            aerosol_optical_depth_tangent=aerosol_optical_depth_tangent,
            aerosol_scattering_depth_tangent=aerosol_scattering_depth_tangent,
            aerosol_asymmetry_tangent=aerosol_asymmetry_tangent,
            max_order=max_order,
            angular_order=angular_order,
            layer_subdivisions=layer_subdivisions,
            delta_m_moments=delta_m_moments,
        )
        return Radiance(source.irradiance.grid, values.sum(axis=0), scene.geometry), tangent.sum(
            axis=0
        )
    if angular_order < 2 or layer_subdivisions < 1:
        raise ValueError("angular_order must be >= 2 and layer_subdivisions must be >= 1")
    reflectance = scene.surface.reflectance
    if reflectance is None:
        raise ValueError("the reflected solar tangent requires surface reflectance")
    depth = (
        np.zeros((len(scene.atmosphere.pressure_pa), optics.grid.points.size), dtype=np.float64)
        if optics.absorption is None
        else optics.absorption.optical_depth
    )
    values, tangent = transfer_native.reflected_radiance_tangent(
        depth,
        optical_depth_tangent,
        source.irradiance.values,
        reflectance,
        reflectance_tangent,
        source.zenith_deg,
        scene.geometry.view_zenith_deg,
    )
    return Radiance(source.irradiance.grid, values, scene.geometry), tangent


__all__ = ["reflected_radiance", "reflected_radiance_tangent"]
