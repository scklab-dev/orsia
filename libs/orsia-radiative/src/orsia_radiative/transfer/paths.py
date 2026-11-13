"""Calculate direct Lambertian reflection from separate pre-integrated slant depths."""

import numpy as np
from numpy.typing import NDArray


def direct_surface_radiance(
    solar_optical_depth: NDArray[np.float64],
    viewing_optical_depth: NDArray[np.float64],
    reflectance: NDArray[np.float64],
    beam_normal_irradiance: NDArray[np.float64],
    incidence_cosine: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Attenuate sunlight along two paths and reflect it at a Lambertian surface.

    Depths include path geometry and scattering losses already. This operation
    supplies no diffuse radiation or visibility test. Supported shadow is encoded
    by zero incidence; missing or obstructed viewing support must fail upstream.
    Binary scaling retains representable radiance when transmittance underflows.

    Args:
        solar_optical_depth: Finite nonnegative (paths,spectral) slant extinction.
        viewing_optical_depth: Matching outgoing slant extinction.
        reflectance: Matching spectral Lambertian reflectance in [0,1].
        beam_normal_irradiance: Matching nonnegative W m-2 per declared spectral coordinate.
        incidence_cosine: Finite (paths,) illuminated surface cosines in [0,1].

    Returns:
        Fresh (paths,spectral) radiance in W m-2 sr-1 per declared spectral coordinate.

    Raises:
        ValueError: Input shapes, bounds or finite values are invalid.
    """
    _validate(
        solar_optical_depth,
        viewing_optical_depth,
        reflectance,
        beam_normal_irradiance,
        incidence_cosine,
    )
    light, exponent = _scaled_illumination(
        solar_optical_depth, viewing_optical_depth, beam_normal_irradiance, incidence_cosine
    )
    albedo, albedo_exponent = np.frexp(reflectance)
    with np.errstate(under="ignore"):
        return np.ldexp(light * albedo, exponent + albedo_exponent)


def direct_surface_radiance_tangent(
    solar_optical_depth: NDArray[np.float64],
    viewing_optical_depth: NDArray[np.float64],
    reflectance: NDArray[np.float64],
    beam_normal_irradiance: NDArray[np.float64],
    incidence_cosine: NDArray[np.float64],
    *,
    solar_optical_depth_tangent: NDArray[np.float64],
    viewing_optical_depth_tangent: NDArray[np.float64],
    reflectance_tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate extinction and reflectance directions at fixed illumination and visibility.

    The derivative remains defined at zero reflectance without division by albedo.
    Geometry, path support, beam irradiance and incidence remain fixed.
    Binary scaling retains representable signed derivatives even when radiance
    underflows or an unscaled seed sum or product would overflow.

    Args:
        solar_optical_depth: Nonnegative (paths,spectral) incident slant extinction.
        viewing_optical_depth: Matching outgoing slant extinction.
        reflectance: Matching Lambertian fractions in [0,1].
        beam_normal_irradiance: Matching nonnegative W m-2 per declared spectral coordinate.
        incidence_cosine: Illuminated (paths,) surface cosines in [0,1].
        solar_optical_depth_tangent: Finite signed (paths,spectral,k) incident seeds.
        viewing_optical_depth_tangent: Matching outgoing seeds.
        reflectance_tangent: Matching reflectance seeds; parameter units follow callers.

    Returns:
        Fresh radiance (paths,spectral) and signed (paths,spectral,k) derivatives
        in radiance units per parameter. Zero direction count is supported.

    Raises:
        ValueError: Physical inputs or tangent shapes/finite values are invalid,
            or the final tangent exceeds finite float64 support.
    """
    _validate(
        solar_optical_depth,
        viewing_optical_depth,
        reflectance,
        beam_normal_irradiance,
        incidence_cosine,
    )
    shape = solar_optical_depth_tangent.shape
    if (
        len(shape) != 3
        or shape[:2] != solar_optical_depth.shape
        or any(
            seed.shape != shape or not np.isfinite(seed).all()
            for seed in (
                solar_optical_depth_tangent,
                viewing_optical_depth_tangent,
                reflectance_tangent,
            )
        )
    ):
        raise ValueError("tangents require matching finite (paths,spectral,k) arrays")
    light, exponent = _scaled_illumination(
        solar_optical_depth, viewing_optical_depth, beam_normal_irradiance, incidence_cosine
    )
    albedo, albedo_exponent = np.frexp(reflectance)
    depth_seed, depth_exponent = _add_scaled(
        *np.frexp(solar_optical_depth_tangent), *np.frexp(viewing_optical_depth_tangent)
    )
    seed, seed_exponent = _add_scaled(
        *np.frexp(reflectance_tangent),
        -albedo[..., None] * depth_seed,
        albedo_exponent[..., None] + depth_exponent,
    )
    with np.errstate(over="ignore", under="ignore"):
        value = np.ldexp(light * albedo, exponent + albedo_exponent)
        tangent = np.ldexp(light[..., None] * seed, exponent[..., None] + seed_exponent)
    if not np.isfinite(tangent).all():
        raise ValueError("direct radiance tangent arithmetic overflow")
    return value, tangent


def _validate(solar, view, reflectance, irradiance, cosine):
    if (
        solar.ndim != 2
        or solar.shape[1] < 1
        or any(
            array.shape != solar.shape or not np.isfinite(array).all() or np.any(array < 0)
            for array in (solar, view, reflectance, irradiance)
        )
    ):
        raise ValueError("finite nonnegative matching (paths,spectral) physical arrays required")
    if (
        np.any(reflectance > 1)
        or cosine.shape != (solar.shape[0],)
        or not np.isfinite(cosine).all()
        or np.any((cosine < 0) | (cosine > 1))
    ):
        raise ValueError("reflectance and incidence cosines must lie in [0,1]")


def _scaled_illumination(solar, view, irradiance, cosine):
    # Retain binary exponents until the final product, including any tangent seed.
    # At depth >=4096 even the largest finite beam times the sum of three finite
    # seeds underflows. Capping there keeps exp(-depth/8) normal without adding depths.
    sun, sun_exponent = np.frexp(np.exp(-np.minimum(solar, 4096.0) / 8))
    outgoing, outgoing_exponent = np.frexp(np.exp(-np.minimum(view, 4096.0) / 8))
    beam, beam_exponent = np.frexp(irradiance)
    incidence, incidence_exponent = np.frexp(cosine[:, None])
    return (
        beam / np.pi * incidence * sun**8 * outgoing**8,
        beam_exponent + incidence_exponent + 8 * (sun_exponent + outgoing_exponent),
    )


def _add_scaled(left, left_exponent, right, right_exponent):
    # A zero term must not force a tiny nonzero term to underflow during alignment.
    exponent = np.where(
        left == 0,
        right_exponent,
        np.where(right == 0, left_exponent, np.maximum(left_exponent, right_exponent)),
    )
    with np.errstate(under="ignore"):
        mantissa, adjustment = np.frexp(
            np.ldexp(left, left_exponent - exponent) + np.ldexp(right, right_exponent - exponent)
        )
    return mantissa, exponent + adjustment
