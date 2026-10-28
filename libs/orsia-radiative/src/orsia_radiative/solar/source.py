"""Solar incidence and pure wavelength-to-wavenumber irradiance materialisation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from orsia_radiative.radiance import SpectralIrradiance
from orsia_radiative.solar.spectrum import SolarIrradianceSpectrum
from orsia_radiative.spectral.conversions import density_per_nm_to_per_cm1
from orsia_radiative.spectral.grid import (
    SpectralDomain,
    SpectralGrid,
    convert_spectral_grid,
)
from orsia_radiative.spectral.interpolation import interpolate_point_density


def solar_irradiance(
    spectrum: SolarIrradianceSpectrum,
    target_grid: SpectralGrid,
    *,
    wavelength_domain: SpectralDomain | None = None,
) -> SpectralIrradiance:
    """Convert W m-2 nm-1 wavelength points to target cm-1 point irradiance.

    Uses the density Jacobian then ``interpolate_point_density``. Explicit
    wavelength support is converted with the grid, not inferred. This pure
    operation does no file access, distance scaling or ancillary-field
    propagation. Keep native uncertainty and bandwidth on the supplied spectrum.
    Target grid identity is retained; the values are independent writable copies.

    Args:
        spectrum: Materialised wavelength spectrum in W m-2 nm-1.
        target_grid: Destination cm-1 spectral sampling for radiative transfer.
        wavelength_domain: Optional declared physical support in wavelength units.

    Returns:
        Point-interpolated beam-normal irradiance on the target grid.

    Raises:
        ValueError: Sampling or declared support cannot cover the target grid.
    """
    wavelength_grid = SpectralGrid("wavelength_nm", spectrum.wavelengths_nm, wavelength_domain)
    source_grid = convert_spectral_grid(wavelength_grid)
    source_values = density_per_nm_to_per_cm1(wavelength_grid.points, spectrum.irradiance_per_nm)
    values = interpolate_point_density(source_grid, source_values, target_grid)
    return SpectralIrradiance(target_grid, values)


def solar_source(irradiance: SpectralIrradiance, *, direction: tuple[float, float]) -> SolarSource:
    """Build solar incidence from (zenith_deg, relative_azimuth_deg), in degrees.

    Zenith points toward the Sun; azimuth is the directed horizontal angle from
    incoming photon propagation to the outgoing view, as defined by SolarSource.

    Args:
        irradiance: Beam-normal spectral irradiance on a cm-1 grid.
        direction: (solar zenith, relative azimuth) in local-frame degrees.

    Returns:
        Validated solar source with the supplied irradiance and direction.

    Raises:
        ValueError: Zenith or relative azimuth lies outside the supported interval.
    """
    zenith, azimuth = direction
    return SolarSource(irradiance, zenith, azimuth)


@dataclass(frozen=True, eq=False)
class SolarSource:
    """Solar irradiance incident from an explicit local direction in degrees.

    ``zenith_deg`` is in [0, 90); ``relative_azimuth_deg`` is in [0, 360).
    Zenith is measured toward the Sun from the upward local normal. Incoming
    photons propagate in the opposite direction. Relative azimuth is measured
    from the horizontal projection of that incoming propagation to the horizontal
    projection of the outgoing view, positive about the upward normal (from +x
    toward +y in a right-handed frame). Thus 0 degrees means aligned propagation
    projections, not aligned directions toward the Sun and observer.

    For outward Sun/view azimuths phi_s and phi_v in the same oriented frame,
    supply (phi_v - phi_s - 180) % 360. At solar or view zenith the corresponding
    azimuth is arbitrary; the same input bounds still apply. No geodetic
    transformation is performed. The irradiance record defines spectral sampling;
    safely immutable coordinates may be shared with evaluated optical inputs.
    The Sun is represented by one incident direction, not a resolved solar disc.
    This source selects no transfer method and implies no thermal emission.

    Attributes:
        irradiance: Materialised beam-normal spectrum in W m-2 per cm-1.
        zenith_deg: Sun zenith from the local upward normal, in degrees.
        relative_azimuth_deg: Incoming-to-view azimuthal separation in degrees.
    """

    irradiance: SpectralIrradiance
    zenith_deg: float
    relative_azimuth_deg: float
    __hash__ = None

    def __post_init__(self) -> None:
        if not isfinite(self.zenith_deg) or not 0 <= self.zenith_deg < 90:
            raise ValueError("source zenith must be in [0, 90) degrees")
        if not isfinite(self.relative_azimuth_deg) or not 0 <= self.relative_azimuth_deg < 360:
            raise ValueError("relative azimuth must be in [0, 360) degrees")
