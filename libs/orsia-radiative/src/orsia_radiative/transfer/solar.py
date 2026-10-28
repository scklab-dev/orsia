"""Reflected-solar closure over one incident direction and atmospheric optics.

Optical composition and spectroscopy are upstream. Each call owns its numerical
scratch and returns independently owned radiance values.
"""

from __future__ import annotations

from orsia_radiative.optics.atmosphere import AtmosphericOptics
from orsia_radiative.radiance import Radiance
from orsia_radiative.scene import RadiativeScene
from orsia_radiative.solar.source import SolarSource

from . import orders


def reflected_radiance(
    scene: RadiativeScene,
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


__all__ = ["reflected_radiance"]
