"""Native-wavelength laboratory cross sections and explicit atmospheric evaluation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orsia_radiative.atmosphere import Atmosphere
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid
from orsia_radiative.spectroscopy.components import AbsorptionComponent, _gas


@dataclass(frozen=True, eq=False)
class CrossSectionSpectrum:
    """One laboratory absorption spectrum on its native vacuum-wavelength axis.

    Arrays are independent read-only float64 copies. Support is an ordered tuple
    of disjoint closed intervals; interpolation cannot cross a declared gap.
    Resolution is descriptive because a scalar cannot represent a varying line
    spread function. Unknown pressure stays None.

    Attributes:
        wavelength_nm: Positive increasing (points,) native vacuum wavelengths, nm.
        cross_sections_m2_per_molecule: Matching finite nonnegative absorption samples.
        temperature_k: Positive laboratory temperature in K.
        pressure_pa: Positive laboratory pressure in Pa, or None when unknown.
        support_nm: Exact retained interpolation intervals in vacuum nm.
        resolution: Supplied laboratory resolution description, or None.
    """

    wavelength_nm: NDArray[np.float64]
    cross_sections_m2_per_molecule: NDArray[np.float64]
    temperature_k: float
    pressure_pa: float | None
    support_nm: tuple[tuple[float, float], ...]
    resolution: str | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        axis, sigma = self.wavelength_nm, self.cross_sections_m2_per_molecule
        if (
            axis.ndim != 1
            or axis.size < 2
            or sigma.shape != axis.shape
            or not np.all(np.isfinite(axis))
            or np.any(axis <= 0)
            or np.any(np.diff(axis) <= 0)
            or not np.all(np.isfinite(sigma))
            or np.any(sigma < 0)
        ):
            raise ValueError(
                "spectra require increasing positive axes and nonnegative finite cross sections"
            )
        if not isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("laboratory temperature must be positive and finite")
        if self.pressure_pa is not None and (
            not isfinite(self.pressure_pa) or self.pressure_pa <= 0
        ):
            raise ValueError("known laboratory pressure must be positive and finite")
        previous = -1.0
        mask = np.zeros(axis.size, dtype=np.bool_)
        for lo, hi in self.support_nm:
            selected = (axis >= lo) & (axis <= hi)
            if (
                not all(map(isfinite, (lo, hi)))
                or lo <= previous
                or hi <= lo
                or np.count_nonzero(selected) < 2
                or not np.any(axis == lo)
                or not np.any(axis == hi)
            ):
                raise ValueError(
                    "spectral support needs ordered disjoint intervals with sampled endpoints"
                )
            mask |= selected
            previous = hi
        if not np.all(mask):
            raise ValueError("spectral support must account for every native sample")
        for name in ("wavelength_nm", "cross_sections_m2_per_molecule"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class CrossSectionMeasurements:
    """Temperature-ordered laboratory spectra for one absorber and isotope convention.

    Native axes, pressure knowledge and resolution remain separate per spectrum.
    Only one spectrum per temperature is supported. Arrays belong to immutable
    spectrum snapshots; caller construction buffers are never retained.

    Attributes:
        species: Whole-species absorber name.
        isotopologue: Local isotope label, or 'all' for natural-abundance measurements.
        spectra: Nonempty tuple with strictly increasing laboratory temperatures.
    """

    species: str
    isotopologue: str
    spectra: tuple[CrossSectionSpectrum, ...]
    __hash__ = None

    def __post_init__(self) -> None:
        if not self.species or not self.isotopologue or not self.spectra:
            raise ValueError("measurements require species, isotope convention and spectra")
        if any(
            b.temperature_k <= a.temperature_k
            for a, b in zip(self.spectra, self.spectra[1:], strict=False)
        ):
            raise ValueError("laboratory temperatures must be strictly increasing")


@dataclass(frozen=True, eq=False)
class CrossSectionTemperatures:
    """Requested and effective temperatures for a laboratory-spectrum evaluation.

    All vectors are independent read-only copies, ordered bottom-to-top layer.

    Attributes:
        requested_k: Actual (layers,) atmospheric temperatures in K.
        effective_k: (layers,) interpolation temperatures after the selected policy, K.
        held_layers: (layers,) boolean mask of nearest-endpoint temperature holds.
    """

    requested_k: NDArray[np.float64]
    effective_k: NDArray[np.float64]
    held_layers: NDArray[np.bool_]
    __hash__ = None

    def __post_init__(self) -> None:
        if (
            self.requested_k.ndim != 1
            or self.effective_k.shape != self.requested_k.shape
            or self.held_layers.shape != self.requested_k.shape
        ):
            raise ValueError("temperature diagnostics must align with layers")
        for name in ("requested_k", "effective_k", "held_layers"):
            value = getattr(self, name)
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype))


def measurement_absorption(
    atmosphere: Atmosphere,
    grid: SpectralGrid,
    reference: CrossSectionMeasurements,
    *,
    spectral_interpolation: Literal["linear_native_wavelength"],
    temperature_interpolation: Literal["linear"],
    temperature_outside: Literal["error", "hold_nearest"],
    pressure_model: Literal["independent"],
) -> tuple[AbsorptionComponent, CrossSectionTemperatures]:
    """Evaluate laboratory spectra with explicit interpolation and pressure assumptions.

    Each contributing spectrum is interpolated linearly in its own native vacuum
    wavelength, then linearly in laboratory temperature. Spectral extrapolation
    and interpolation across gaps raise ValueError. Exact T knots require only
    that spectrum's support. Pressure-independent use is an explicit approximation
    even when a laboratory pressure is known. No spectral-density Jacobian applies.

    Args:
        atmosphere: Current temperatures and whole-species molecular columns.
        grid: Increasing target wavenumbers in cm-1.
        reference: Laboratory spectra with native axes, temperatures and pressure knowledge.
        spectral_interpolation: Explicit linear interpolation in native vacuum wavelength.
        temperature_interpolation: Explicit linear laboratory-temperature interpolation.
        temperature_outside: Error or hold the nearest laboratory endpoint.
        pressure_model: Explicit pressure-independent use of the measured spectra.

    Returns:
        Absorption component with writable dimensionless depths and retained
        requested/effective temperature diagnostics, including held-layer mask.

    Raises:
        ValueError: A policy, requested temperature, species or spectral support is invalid.
    """
    require_wavenumber_grid(grid)
    values, diagnostics = interpolate_cross_section_measurements(
        reference,
        atmosphere.temperature_k,
        1e7 / grid.points,
        spectral_interpolation=spectral_interpolation,
        temperature_interpolation=temperature_interpolation,
        temperature_outside=temperature_outside,
        pressure_model=pressure_model,
    )
    columns, _ = _gas(atmosphere, reference.species)
    return AbsorptionComponent(
        grid, reference.species, reference.isotopologue, "cross_section", columns[:, None] * values
    ), diagnostics


def interpolate_cross_section_measurements(
    reference: CrossSectionMeasurements,
    temperatures_k: NDArray[np.float64],
    wavelength_nm: NDArray[np.float64],
    *,
    spectral_interpolation: Literal["linear_native_wavelength"],
    temperature_interpolation: Literal["linear"],
    temperature_outside: Literal["error", "hold_nearest"],
    pressure_model: Literal["independent"],
) -> tuple[NDArray[np.float64], CrossSectionTemperatures]:
    """Interpolate laboratory absorption values while retaining temperature diagnostics.

    Args:
        reference: Temperature-ordered spectra with explicit disconnected support.
        temperatures_k: Positive finite nonempty (layers,) requested temperatures, K.
        wavelength_nm: Positive finite nonempty (points,) target vacuum wavelengths, nm.
        spectral_interpolation: Explicit native-wavelength linear interpolation.
        temperature_interpolation: Explicit linear temperature interpolation.
        temperature_outside: Error or hold_nearest outside laboratory T support.
        pressure_model: Explicit pressure-independent laboratory-spectrum approximation.

    Returns:
        Owned writable float64 (layers,points) cross sections in m2/molecule and
        immutable requested/effective temperature diagnostics.

    Raises:
        ValueError: Policy, input vector or contributing native spectral support is invalid.
    """
    if (
        spectral_interpolation != "linear_native_wavelength"
        or temperature_interpolation != "linear"
        or pressure_model != "independent"
        or temperature_outside not in ("error", "hold_nearest")
    ):
        raise ValueError(
            "unsupported measurement interpolation, temperature policy or pressure model"
        )
    for vector in (temperatures_k, wavelength_nm):
        if (
            vector.ndim != 1
            or vector.size == 0
            or np.any(~np.isfinite(vector))
            or np.any(vector <= 0)
        ):
            raise ValueError(
                "temperatures and wavelengths must be positive finite nonempty vectors"
            )
    knots = np.array([s.temperature_k for s in reference.spectra])
    effective = np.clip(temperatures_k, knots[0], knots[-1])
    held = effective != temperatures_k
    if temperature_outside == "error" and np.any(held):
        raise ValueError("requested temperature outside laboratory support")
    values = np.empty((effective.size, wavelength_nm.size), dtype=np.float64)
    rows: dict[int, NDArray[np.float64]] = {}
    for layer, temperature in enumerate(effective):
        right = min(int(np.searchsorted(knots, temperature)), knots.size - 1)
        left = right if knots[right] == temperature else right - 1
        for index in {left, right}:
            if index not in rows:
                spectrum = reference.spectra[index]
                covered = np.zeros(wavelength_nm.size, dtype=np.bool_)
                for lo, hi in spectrum.support_nm:
                    covered |= (wavelength_nm >= lo) & (wavelength_nm <= hi)
                if not np.all(covered):
                    raise ValueError("target wavelength outside laboratory support or across a gap")
                rows[index] = np.interp(
                    wavelength_nm, spectrum.wavelength_nm, spectrum.cross_sections_m2_per_molecule
                )
        if left == right:
            values[layer] = rows[left]
        else:
            weight = (temperature - knots[left]) / (knots[right] - knots[left])
            values[layer] = (1 - weight) * rows[left] + weight * rows[right]
    return values, CrossSectionTemperatures(temperatures_k, effective, held)


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.frombuffer(values.tobytes(), dtype=np.float64)
