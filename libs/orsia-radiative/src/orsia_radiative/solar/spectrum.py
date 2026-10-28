"""Native solar wavelength-point irradiance, independent of files and providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class SolarIrradianceSpectrum:
    """Wavelength points [nm], irradiance and optional supplied ancillary values.

    Irradiance/uncertainty are W m-2 nm-1; bandwidth is nm, not sample spacing.
    All arrays are independent writable float64 copies of shape (n_points,).
    Uncertainty's statistical meaning is caller-supplied, not inferred here.
    No source identity or reference distance is attached to these physical values.

    Attributes:
        wavelengths_nm: Increasing sampled wavelengths in nanometres.
        irradiance_per_nm: (points,) beam-normal spectral density in W m-2 nm-1.
        uncertainty_per_nm: Optional supplied uncertainty density in the same units.
        bandwidth_nm: Optional supplied (points,) sample bandwidths in nanometres.
    """

    wavelengths_nm: NDArray[np.float64]
    irradiance_per_nm: NDArray[np.float64]
    uncertainty_per_nm: NDArray[np.float64] | None = None
    bandwidth_nm: NDArray[np.float64] | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        wavelengths = self.wavelengths_nm
        irradiance = self.irradiance_per_nm
        if wavelengths.ndim != 1 or wavelengths.size == 0 or not np.all(np.isfinite(wavelengths)):
            raise ValueError("wavelengths must be a nonempty finite vector")
        if np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
            raise ValueError("wavelengths must be positive and strictly increasing")
        if (
            irradiance.shape != wavelengths.shape
            or not np.all(np.isfinite(irradiance))
            or np.any(irradiance < 0)
        ):
            raise ValueError("irradiance must be finite nonnegative and match wavelengths")
        object.__setattr__(self, "wavelengths_nm", np.copy(wavelengths))
        object.__setattr__(self, "irradiance_per_nm", np.copy(irradiance))
        for name in ("uncertainty_per_nm", "bandwidth_nm"):
            values = getattr(self, name)
            if values is None:
                continue
            if (
                values.shape != wavelengths.shape
                or not np.all(np.isfinite(values))
                or np.any(values < 0 if name == "uncertainty_per_nm" else values <= 0)
            ):
                raise ValueError(f"{name} must be valid and match wavelengths")
            object.__setattr__(self, name, np.copy(values))
