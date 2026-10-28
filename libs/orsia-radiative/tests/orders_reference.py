"""Frozen pre-full-spectrum SOS reference and benchmark baseline.

This test-local copy preserves the original Python preparation/general recurrence
and its point-local native Rayleigh dispatch. Independent scalar-ray oracles in
other tests additionally check the physical discretization.

The direct-beam source is integrated analytically inside each cell. Diffuse
sources use cell-centre collocation and piecewise-constant formal transport;
angular and vertical refinement are caller-controlled, not an accuracy promise.
All angular fields, quadratures and work buffers are private to one evaluation.
"""

from __future__ import annotations

from math import cos, pi, radians, sin

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.optics.atmosphere import AtmosphericOptics, validate_atmospheric_optics
from orsia_radiative.optics.phases import henyey_greenstein_phase, rayleigh_phase
from orsia_radiative.optics.scattering import AerosolComponent, RayleighComponent
from orsia_radiative.scene import RadiativeScene
from orsia_radiative.solar.source import SolarSource
from orsia_radiative.transfer.formal import (
    PropagationCoefficients,
    apply_propagation,
    propagation_coefficients,
)
from scipy.special import roots_legendre

try:
    from orsia_radiative.transfer.native.propagation_native import bind_rayleigh_blas
    from orsia_radiative.transfer.native.propagation_native import (
        rayleigh_orders as _native_rayleigh_orders,
    )
except ModuleNotFoundError as error:
    if error.name != "orsia_radiative.transfer.native.propagation_native":
        raise
    _native_rayleigh_orders = None
else:
    from importlib import import_module

    try:
        _native_rayleigh_orders = bind_rayleigh_blas(import_module("scipy.linalg.cython_blas"))
    except ValueError:
        # Other capsule ABIs retain the portable pointer-only native products.
        pass

# Retained operators only; phase construction and transport have separate work
# buffers. Oversized operators are applied once, never retained. No global cache.
_PHASE_CACHE_BYTES = 16 * 1024 * 1024
PhaseKey = float | None | tuple[float, int]


def angular_grid(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Photon directions (positive z upward) and weights for the 4pi mean.

    Use order-point Gauss-Legendre on each open hemisphere and 2*order periodic
    azimuth points. Weights sum to one: they represent dOmega/(4*pi), not dOmega.
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
    """Dispatch to optical phase laws for SOS direction cosines; fresh output.

    Retain this entry's geometric clipping and g=None Rayleigh selection; the
    optical evaluators own the laws and enforce their physical parameter bounds.
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
    """Sum atmospheric orders 0..N, reflecting each order at the Lambertian ground.

    Without Delta-M, N counts physical atmospheric events, not reflections.
    Order zero is the unscattered surface contribution. At order one the beam is scattered
    in addition to the order-zero surface field; later orders scatter the entire
    preceding field. Downward diffuse exits are reflected at the same atmospheric
    order. Thus repeated ground encounters are included, separated by scattering.

    Return fresh writable float64 [order, spectral point] values, shape
    (N+1, points), in W m-2 sr-1 per (cm-1) on the source grid. Constituents sum
    within each order; no path provenance or diagnostics are returned. Row N
    is not an omitted-tail error bound.
    No automatic stopping, tail closure, thermal term or derivative is evaluated.
    Optional Delta-M removes only the g>0 aerosol forward peak, retaining K
    Legendre moments with f=g**K. It transforms both beam and diffuse extinction,
    leaves absorption/Rayleigh unchanged, and counts residual events including
    transformed order zero. No intensity correction is applied. Truncated phases
    may be signed; negative order radiances are rejected, never clipped.
    Inputs are read-only; spectral points are serial.
    Angular/vertical defaults have no qualified accuracy envelope.
    """
    if max_order < 0:
        raise ValueError("max_order must be >= 0")
    if angular_order < 2 or layer_subdivisions < 1:
        raise ValueError("angular_order must be >= 2 and layer_subdivisions must be >= 1")
    if delta_m_moments is not None and not 2 <= delta_m_moments <= 2 * angular_order:
        raise ValueError("delta_m_moments must satisfy 2 <= K <= 2*angular_order")
    zero_order, extinction = _direct_order(scene, source, optics, delta_m_moments)
    scattering = tuple(
        component for component in (optics.rayleigh, optics.aerosol) if component is not None
    )
    contributions = np.zeros((max_order + 1, zero_order.size), dtype=np.float64)
    contributions[0] = zero_order
    _accumulate_orders(
        scene,
        source,
        extinction,
        scattering,
        contributions,
        angular_order=angular_order,
        layer_subdivisions=layer_subdivisions,
        delta_m_moments=delta_m_moments,
    )
    if np.any(contributions < 0):
        raise ValueError(
            "Delta-M produced negative order radiance; refine moments/angular sampling "
            "or disable Delta-M (no intensity correction is implemented)"
        )
    if not np.all(np.isfinite(contributions)):
        raise ValueError("scattering order radiance must be finite")
    return contributions


def _direct_order(
    scene: RadiativeScene,
    source: SolarSource,
    optics: AtmosphericOptics,
    delta_m_moments: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return fresh direct radiance [point] and extinction [layer, point].

    Validate physical support and summed depth before optional Delta-M removal.
    Inputs are borrowed read-only; both float64 outputs belong to this call.
    Radiance uses W m-2 sr-1 per (cm-1). Extinction is dimensionless and remains
    common to beam and diffuse transport.
    """
    illumination = source.irradiance
    validate_atmospheric_optics(scene, source, optics)
    reflectance = scene.surface.reflectance
    if reflectance is None:
        raise ValueError("the reflected solar calculation requires surface reflectance")
    extinction = np.zeros(
        (len(scene.atmosphere.pressure_pa), illumination.wavenumbers_cm1.size), dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        for component in (optics.absorption, optics.rayleigh, optics.aerosol):
            if component is not None:
                extinction += component.optical_depth
        total = extinction.sum(axis=0, dtype=np.float64)
    if not np.all(np.isfinite(total)):
        raise ValueError("summed optical depth overflow")
    mu0 = cos(radians(source.zenith_deg))
    mu = cos(radians(scene.geometry.view_zenith_deg))
    aerosol = optics.aerosol
    if delta_m_moments is not None and aerosol is not None:
        # Physical optics and aerosol absorption are unchanged:
        # tau_e* - tau_s* = tau_e - tau_s.
        removed = np.maximum(aerosol.asymmetry, 0.0) ** delta_m_moments
        extinction -= removed * aerosol.scattering_depth
        del removed
        total = extinction.sum(axis=0)
    transmission = np.exp(-total * (1 / mu0 + 1 / mu))
    zero_order = illumination.values * reflectance * mu0 / pi * transmission
    if zero_order.shape != (illumination.wavenumbers_cm1.size,):
        raise ValueError("radiance values must match the spectral axis")
    if not np.all(np.isfinite(zero_order)) or np.any(zero_order < 0):
        raise ValueError("radiance values must be finite and nonnegative")
    return zero_order, extinction


def _accumulate_orders(
    scene: RadiativeScene,
    source: SolarSource,
    extinction: NDArray[np.float64],
    scattering: tuple[RayleighComponent | AerosolComponent, ...],
    contributions: NDArray[np.float64],
    *,
    angular_order: int,
    layer_subdivisions: int,
    delta_m_moments: int | None,
) -> None:
    """Fill orders 1..N of a writable [order, spectral point] radiance buffer.

    Row zero is already prepared and stays unchanged. Other inputs are read-only.
    Geometry and the bounded phase cache are local to this call; only the preceding
    angular field is retained between orders, and inactive points are skipped.
    """
    max_order = contributions.shape[0] - 1
    if max_order == 0:
        return
    active = np.zeros(contributions.shape[1], dtype=np.bool_)
    for component in scattering:
        scattering_depth = (
            component.optical_depth
            if isinstance(component, RayleighComponent)
            else component.scattering_depth
        )
        active |= np.any(scattering_depth != 0, axis=0)
    if not np.any(active):
        return
    illumination = source.irradiance
    zenith, azimuth = source.zenith_deg, source.relative_azimuth_deg
    reflectance = scene.surface.reflectance
    assert reflectance is not None
    mu0 = cos(radians(zenith))
    view_zenith = radians(scene.geometry.view_zenith_deg)
    view = np.array(
        [
            sin(view_zenith) * cos(radians(azimuth)),
            sin(view_zenith) * sin(radians(azimuth)),
            cos(view_zenith),
        ],
        dtype=np.float64,
    )
    sun = np.array([sin(radians(zenith)), 0.0, -mu0], dtype=np.float64)
    directions, weights = angular_grid(angular_order)
    outgoing = np.vstack((directions, view))
    mu = outgoing[:, 2]
    cosines = directions @ outgoing.T
    rayleigh_factors = (
        _prepare_rayleigh_factors(directions, outgoing, weights)
        if extinction.shape[0] > 2
        and any(isinstance(component, RayleighComponent) for component in scattering)
        else None
    )
    beam_cosines = outgoing @ sun
    # Geometry is fixed throughout this evaluation. None identifies Rayleigh;
    # float g identifies HG; (g, K) identifies its Delta-M residual expansion.
    # Depth is applied separately. No cache survives this evaluation.
    phase_cache: dict[PhaseKey, NDArray[np.float64]] = {}
    for point in range(contributions.shape[1]):
        if not active[point]:
            continue
        depth = np.repeat(extinction[:, point] / layer_subdivisions, layer_subdivisions)
        transport = propagation_coefficients(depth, mu, solar_mu=mu0)
        zero_source = np.zeros((depth.size, mu.size), dtype=np.float64)
        # The Lambertian response is linear in its amplitude and fixed for all
        # orders at this point. Retain the discrete recurrence's unit response,
        # including its cell-centre convention, rather than sweeping zero sources
        # again for each reflected field. These buffers stay local to this point.
        boundary_centres, boundary_exits = apply_propagation(
            zero_source, transport, bottom_radiance=1.0
        )
        # A unit response can lose range before a large actual amplitude would.
        # Preserve the original sweeps wherever any upward response is subnormal
        # or zero; monotone attenuation makes the exits the smallest responses.
        reuse_boundary = bool(np.all(boundary_exits[mu > 0] >= np.finfo(np.float64).tiny))
        direct_boundary = (
            illumination.values[point] * reflectance[point] * mu0 / pi * np.exp(-depth.sum() / mu0)
        )
        if (
            reuse_boundary
            and rayleigh_factors is not None
            and len(scattering) == 1
            and isinstance(scattering[0], RayleighComponent)
            and np.all(scattering[0].optical_depth[:, point] > 0)
        ):
            if _native_rayleigh_orders is None:
                raise ImportError(
                    "Eligible Rayleigh scattering orders require the compiled "
                    "orsia_radiative.transfer.native.propagation_native extension; "
                    "install or rebuild orsia-radiative for this interpreter."
                )
            # The point-local recurrence owns its scratch and runs detached;
            # all prepared inputs remain read-only, including concurrent calls.
            contributions[1:, point] = _native_rayleigh_orders(
                np.repeat(
                    scattering[0].optical_depth[:, point] / layer_subdivisions,
                    layer_subdivisions,
                ),
                rayleigh_factors,
                _beam_source(
                    scattering,
                    point,
                    illumination.values[point],
                    layer_subdivisions,
                    beam_cosines,
                    depth.size,
                    delta_m_moments,
                ),
                (
                    transport.mu,
                    transport.transmission,
                    transport.centre_transmission,
                    transport.source_weight,
                    transport.centre_source_weight,
                    transport.solar_source_weight,
                    transport.solar_centre_source_weight,
                ),
                boundary_centres,
                boundary_exits,
                weights,
                float(direct_boundary),
                float(reflectance[point]),
                max_order,
            )
            del transport, zero_source, boundary_centres, boundary_exits
            continue
        if reuse_boundary:
            field = boundary_centres * direct_boundary
        else:
            field, _ = apply_propagation(
                zero_source, transport, bottom_radiance=float(direct_boundary)
            )
        for order in range(1, max_order + 1):
            redistributed = np.zeros_like(zero_source)
            for component in scattering:
                _scatter(
                    field[:, :-1],
                    component,
                    point,
                    layer_subdivisions,
                    cosines,
                    weights,
                    phase_cache,
                    out=redistributed,
                    rayleigh_factors=rayleigh_factors,
                    delta_m_moments=delta_m_moments,
                )
            retain_centres = order < max_order
            centres, exits = apply_propagation(
                redistributed, transport, return_centres=retain_centres
            )
            if order == 1:
                beam_source = _beam_source(
                    scattering,
                    point,
                    illumination.values[point],
                    layer_subdivisions,
                    beam_cosines,
                    depth.size,
                    delta_m_moments,
                )
                _add_propagated_source(centres, exits, beam_source, transport, solar_source=True)
                del beam_source
            boundary = _reflected(exits[:-1], mu[:-1], weights, reflectance[point])
            if boundary != 0:
                if reuse_boundary:
                    exits += boundary_exits * boundary
                    if centres is not None:
                        centres += boundary_centres * boundary
                else:
                    _add_propagated_source(
                        centres, exits, zero_source, transport, bottom_radiance=boundary
                    )
            contributions[order, point] = exits[-1]
            # Keep only the preceding field, not an angular history growing with N.
            if retain_centres:
                assert centres is not None
                field = centres
            del centres, exits, redistributed
        del transport, field, zero_source, boundary_centres, boundary_exits
    phase_cache.clear()


def _beam_source(
    scattering: tuple[RayleighComponent | AerosolComponent, ...],
    point: int,
    irradiance: float,
    subdivisions: int,
    beam_cosines: NDArray[np.float64],
    cell_count: int,
    delta_m_moments: int | None,
) -> NDArray[np.float64]:
    """Fresh direct-beam scattering source [cell, outgoing direction] for order one.

    Depth and phase inputs are read-only. Solar attenuation is applied separately
    by formal propagation, using the same extinction as the diffuse field.
    """
    beam_source = np.zeros((cell_count, beam_cosines.size), dtype=np.float64)
    for component in scattering:
        if isinstance(component, RayleighComponent):
            # Every Rayleigh layer shares this phase. Broadcast the layer depths
            # over directions, then repeat cells, preserving scalar association.
            layer_source = irradiance / (4 * pi) * component.optical_depth[:, point]
            layer_source /= subdivisions
            beam_source += np.repeat(layer_source, subdivisions)[:, None] * phase_function(
                beam_cosines, None
            )
            continue
        for layer, row in enumerate(component.scattering_depth):
            g = float(component.asymmetry[layer, point])
            cells = slice(layer * subdivisions, (layer + 1) * subdivisions)
            fraction = _forward_fraction(g, delta_m_moments)
            phase = (
                phase_function(beam_cosines, g)
                if delta_m_moments is None
                else _transport_phase(beam_cosines, g, delta_m_moments)
            )
            beam_source[cells] += (
                irradiance / (4 * pi) * row[point] * (1 - fraction) / subdivisions * phase
            )
    return beam_source


def _add_propagated_source(
    centres: NDArray[np.float64] | None,
    exits: NDArray[np.float64],
    source_depth: NDArray[np.float64],
    transport: PropagationCoefficients,
    *,
    solar_source: bool = False,
    bottom_radiance: float = 0.0,
) -> None:
    """Add fresh propagated fields to owned centre/exit buffers in place.

    Source and coefficients are read-only. A missing centre buffer denotes the
    last order: calculate exits only, with no centre allocation or arithmetic.
    """
    added_centres, added_exits = apply_propagation(
        source_depth,
        transport,
        bottom_radiance=bottom_radiance,
        solar_source=solar_source,
        return_centres=centres is not None,
    )
    exits += added_exits
    if centres is not None:
        assert added_centres is not None
        centres += added_centres


def _scatter(
    field: NDArray[np.float64],
    component: RayleighComponent | AerosolComponent,
    point: int,
    subdivisions: int,
    cosines: NDArray[np.float64],
    weights: NDArray[np.float64],
    phase_cache: dict[PhaseKey, NDArray[np.float64]],
    *,
    out: NDArray[np.float64],
    delta_m_moments: int | None = None,
    rayleigh_factors: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> None:
    """Add redistributed light in place, batching layers with the same phase.

    Field and out have shape (cells, incoming/outgoing directions). Skip zero
    depth/fields. The call-owned cache must belong to this exact geometry
    and quadrature; retained operators are FIFO-bounded by _PHASE_CACHE_BYTES.
    Matmul precedes depth scaling. Out is an owned float64 buffer with storage
    separate from inputs and cached operators. Other arrays are borrowed read-only.
    Rayleigh components with more than two layers require factors prepared for
    this geometry/quadrature. They
    exactly expand 3/4*(1+cosine**2), including with Delta-M (Rayleigh unchanged).
    """
    scattering_depth = (
        component.optical_depth
        if isinstance(component, RayleighComponent)
        else component.scattering_depth
    )
    depth = scattering_depth[:, point]
    active = (depth != 0) & np.any(
        field.reshape(depth.size, subdivisions, field.shape[1]), axis=(1, 2)
    )
    if not np.any(active):
        return
    # Exact g equality only: distinct optical roles and phases are never mixed.
    asymmetry = None if isinstance(component, RayleighComponent) else component.asymmetry[:, point]
    # Measurements favour layer-local dense products for tiny scenes and for
    # HG at <=256 incoming directions: grouping/gathers cost more than they save.
    # This is an execution choice only, never a quadrature or physics change.
    if depth.size <= 2 or (asymmetry is not None and cosines.shape[0] <= 256):
        for layer in np.flatnonzero(active):
            g = None if asymmetry is None else float(asymmetry[layer])
            cells = slice(int(layer) * subdivisions, (int(layer) + 1) * subdivisions)
            phase = _redistribution_operator(cosines, weights, g, delta_m_moments, phase_cache)
            out[cells] += (field[cells] @ phase) * (
                depth[layer] / subdivisions * (1 - _forward_fraction(g, delta_m_moments))
            )
            del phase
        return
    phases = (None,) if asymmetry is None else tuple(dict.fromkeys(asymmetry[active]))
    for parameter in phases:
        g = None if parameter is None else float(parameter)
        layers = np.flatnonzero(active if asymmetry is None else active & (asymmetry == g))
        # Contiguous groups borrow a field view; disjoint layers need one gather.
        # Temporary field/result storage is at most one cell field/source, not
        # the number of layers times a dense angular operator.
        cells = (
            slice(int(layers[0]) * subdivisions, (int(layers[-1]) + 1) * subdivisions)
            if layers[-1] - layers[0] + 1 == layers.size
            else (layers[:, None] * subdivisions + np.arange(subdivisions)).ravel()
        )
        fraction = _forward_fraction(g, delta_m_moments)
        if g is None:
            assert rayleigh_factors is not None
            incoming, outgoing = rayleigh_factors
            redistributed = (field[cells] @ incoming) @ outgoing
            redistributed *= np.repeat(depth[layers] / subdivisions, subdivisions)[:, None]
            out[cells] += redistributed
            del redistributed
            continue
        phase = _redistribution_operator(cosines, weights, g, delta_m_moments, phase_cache)
        redistributed = field[cells] @ phase
        redistributed *= np.repeat(depth[layers] / subdivisions * (1 - fraction), subdivisions)[
            :, None
        ]
        out[cells] += redistributed
        del redistributed
        del phase


def _redistribution_operator(
    cosines: NDArray[np.float64],
    weights: NDArray[np.float64],
    g: float | None,
    moments: int | None,
    cache: dict[PhaseKey, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Borrow a compatible bounded-cache operator or construct a fresh one."""
    key: PhaseKey = (g, moments) if g is not None and g > 0 and moments is not None else g
    phase = cache.get(key)
    if phase is None:
        operator_bytes = cosines.size * np.dtype(np.float64).itemsize
        # Evict before construction, even for an oversized unretained matrix.
        while cache and (
            sum(matrix.nbytes for matrix in cache.values()) + operator_bytes > _PHASE_CACHE_BYTES
        ):
            del cache[next(iter(cache))]
        phase = _prepare_redistribution(cosines, weights, g, moments)
        if operator_bytes <= _PHASE_CACHE_BYTES:
            cache[key] = phase
    return phase


def _prepare_rayleigh_factors(
    directions: NDArray[np.float64],
    outgoing: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fresh read-only factors with shapes (incoming, 7) and (7, outgoing).

    For unit photon directions, expand (u.v)^2 into three squared-coordinate
    products and three cross products. This is the same scalar Rayleigh law,
    not angular/moment truncation; the 4pi quadrature is applied exactly once.
    Factors are call-owned, O(directions) storage, with no global cache.
    """

    def basis(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y, z = vectors.T
        return np.column_stack((np.ones(x.size), x * x, y * y, z * z, x * y, x * z, y * z))

    incoming = basis(directions) * weights[:, None]
    response = (basis(outgoing) * np.array([0.75, 0.75, 0.75, 0.75, 1.5, 1.5, 1.5])).T.copy()
    incoming.flags.writeable = response.flags.writeable = False
    return incoming, response


def _prepare_redistribution(
    cosines: NDArray[np.float64],
    weights: NDArray[np.float64],
    g: float | None,
    delta_m_moments: int | None = None,
) -> NDArray[np.float64]:
    """Fresh incoming x outgoing phase matrix with dOmega/(4*pi) applied once.

    Layer scattering depth is deliberately not folded in: application retains
    the original matmul-then-depth arithmetic and its cell subdivision factor.
    """
    phase = (
        phase_function(cosines, g)
        if delta_m_moments is None
        else _transport_phase(cosines, g, delta_m_moments)
    )
    return phase * weights[:, None]


def _forward_fraction(g: float | None, moments: int | None) -> float:
    """Remove only a forward HG peak; Rayleigh and g<=0 remain physical."""
    return g**moments if g is not None and g > 0 and moments is not None else 0.0


def _transport_phase(
    cosine: NDArray[np.float64], g: float | None, moments: int
) -> NDArray[np.float64]:
    """Fresh phase with a 4pi mean of one, not a fitted HG law.

    Physical g and K have been validated upstream. For forward HG, retain
    degrees 0..K-1 with beta_l*=(g**l-g**K)/(1-g**K); beta_0*=1.
    K<=2*angular_order resolves these polynomials under both the hemisphere
    Gauss rule and periodic azimuth rule. No positivity clipping or discrete
    normalization is applied. Non-forward HG is deliberately not truncated.
    """
    if g is None or g <= 0:
        return phase_function(cosine, g)
    degree = np.arange(moments)
    fraction = g**moments
    coefficients = (2 * degree + 1) * (g**degree - fraction) / (1 - fraction)
    return np.polynomial.legendre.legval(np.clip(cosine, -1, 1), coefficients)


def _reflected(
    exit_radiance: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
    reflectance: float,
) -> float:
    """Lambertian rho/pi times diffuse downward irradiance; excludes the beam."""
    down = mu < 0
    return float(4 * reflectance * np.sum(weights[down] * -mu[down] * exit_radiance[down]))


__all__ = ["angular_grid", "phase_function", "scattering_orders"]
