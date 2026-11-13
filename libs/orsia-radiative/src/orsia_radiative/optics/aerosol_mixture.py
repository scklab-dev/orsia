"""Mix spherical lognormal aerosol optics with explicit quadrature and Mie provider."""

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

type FloatArray = NDArray[np.float64]


type BoolArray = NDArray[np.bool_]


class LognormalAerosolPopulation(TypedDict):
    """Volume-weighted lognormal radius distribution and spectral refractive index.

    Attributes:
        volume_fraction: Nonnegative external-mixture volume fraction.
        median_radius_um: Positive number-median radius in micrometres.
        geometric_width: Geometric standard deviation greater than one.
        wavelength_um: Increasing refractive-index knot wavelengths, micrometres.
        real_index: Positive real refractive indices on wavelength_um.
        absorption_index: Nonnegative imaginary-index magnitudes on wavelength_um.
            The Mie callback receives negative imaginary refractive index.
    """

    volume_fraction: float
    median_radius_um: float
    geometric_width: float
    wavelength_um: Sequence[float]
    real_index: Sequence[float]
    absorption_index: Sequence[float]


def lognormal_aerosol_optics(
    wavelength: FloatArray,
    populations: Sequence[LognormalAerosolPopulation],
    mie_efficiencies: Callable[
        [complex, FloatArray, float], tuple[FloatArray, FloatArray, FloatArray, FloatArray]
    ],
    *,
    reference_wavelength_nm: float,
    quadrature_order: int = 256,
    calculation_step_nm: float = 0.5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate a volume-normalized external mixture on a supplied nm grid.

    Refractive-index interpolation clamps to the supplied table endpoints.

    Args:
        wavelength: Increasing finite positive target wavelengths, nm.
        populations: Explicit external populations; arrays/sequences are borrowed.
        mie_efficiencies: Pure spherical-Mie callback of complex index, diameter
            vector in um and scalar wavelength in um. Returns extinction,
            scattering, backscatter efficiencies and asymmetry on the diameter axis.
        reference_wavelength_nm: Positive extinction-normalization wavelength, nm.
        quadrature_order: At least 32 Gauss-Legendre nodes over six log-radius sigmas.
        calculation_step_nm: Positive maximum intermediate wavelength step, nm.

    Returns:
        Owned relative extinction, single-scattering albedo and asymmetry vectors
        on wavelength. Scattering weights asymmetry; extinction weights albedo.

    Raises:
        ValueError: Grid, population, quadrature or evaluated optics are invalid.
    """
    if (
        wavelength.ndim != 1
        or len(wavelength) < 2
        or not np.isfinite(wavelength).all()
        or np.any(wavelength <= 0)
        or np.any(np.diff(wavelength) <= 0)
        or quadrature_order < 32
        or not np.isfinite(calculation_step_nm)
        or calculation_step_nm <= 0
        or not np.isfinite(reference_wavelength_nm)
        or reference_wavelength_nm <= 0
        or not populations
    ):
        raise ValueError("ordered positive spectral grid and quadrature controls required")
    for p in populations:
        knots = np.asarray(p["wavelength_um"])
        real = np.asarray(p["real_index"])
        imaginary = np.asarray(p["absorption_index"])
        if (
            knots.ndim != 1
            or len(knots) < 2
            or real.shape != knots.shape
            or imaginary.shape != knots.shape
            or not all(np.isfinite(v).all() for v in (knots, real, imaginary))
            or np.any(np.diff(knots) <= 0)
            or np.any(real <= 0)
            or np.any(imaginary < 0)
            or not np.isfinite(
                [p["volume_fraction"], p["median_radius_um"], p["geometric_width"]]
            ).all()
            or p["volume_fraction"] < 0
            or p["median_radius_um"] <= 0
            or p["geometric_width"] <= 1
        ):
            raise ValueError("valid lognormal population and refractive index knots required")
    nodes, weights = roots_legendre(quadrature_order)
    normal = 6 * nodes
    probability = 6 * weights / np.sqrt(2 * np.pi) * np.exp(-(normal**2) / 2)
    count = max(1, int(np.ceil((wavelength[-1] - wavelength[0]) / calculation_step_nm)))
    calculation = np.linspace(wavelength[0], wavelength[-1], count + 1)
    optics = np.zeros((len(calculation) + 1, 3))
    for population in populations:
        fraction, median, width = (
            population["volume_fraction"],
            population["median_radius_um"],
            population["geometric_width"],
        )
        knots, real, absorption = (
            population["wavelength_um"],
            population["real_index"],
            population["absorption_index"],
        )
        radius = np.exp(np.log(median) + np.log(width) * normal)
        number_weight = fraction / np.sum(probability * radius**3)
        area_weight = probability * np.pi * radius**2 * number_weight
        for index, wave in enumerate(np.r_[reference_wavelength_nm, calculation] / 1000):
            refractive = complex(np.interp(wave, knots, real), -np.interp(wave, knots, absorption))
            extinction, scattering, _, asymmetry = mie_efficiencies(refractive, 2 * radius, wave)
            optics[index] += np.array(
                [
                    np.sum(area_weight * extinction),
                    np.sum(area_weight * scattering),
                    np.sum(area_weight * scattering * asymmetry),
                ]
            )
    if not np.isfinite(optics).all() or np.any(optics[:, :2] <= 0):
        raise ValueError("Mie mixture must have positive finite extinction/scattering")
    relative = optics[1:, 0] / optics[0, 0]
    albedo = optics[1:, 1] / optics[1:, 0]
    asymmetry = optics[1:, 2] / optics[1:, 1]
    if np.any(albedo > 1 + 1e-12) or np.any(abs(asymmetry) > 1):
        raise ValueError("Mie mixture violates albedo/asymmetry domains")
    return (
        np.asarray(np.interp(wavelength, calculation, relative), dtype=float),
        np.asarray(np.interp(wavelength, calculation, albedo), dtype=float),
        np.asarray(np.interp(wavelength, calculation, asymmetry), dtype=float),
    )
