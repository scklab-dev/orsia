"""Assemble a finite-slit disperser with entrance-plane rays and detector pixels.

The rectangular slit is an ideal hard field stop in the forward focal plane.
A unit-magnification polynomial map sends accepted entrance coordinates to a
post-slit Gaussian image on the detector. Spatial scanning belongs to geometry.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition
from ..detector import DetectorProperties
from .dispersion import dispersive_imager
from .psfs import SpatialDensity
from .response import ImagingOptics, SpectralParameter


@dataclass(frozen=True, eq=False)
class DispersiveSpectrometer:
    """Physical entrance slit, active detector rectangles and dispersive optics.

    Detector geometry and property maps are read-only snapshots. The factory
    copies the slit into writable storage; direct construction borrows the slit.
    Keep slit arrays and borrowed pure transmission callbacks stable during use.

    Attributes:
        detector: Stable active geometry, pixel QE and electronics.
        pixel_bounds_m: Delegated read-only (row,column,4) active rectangles in metres.
            Columns usually resolve wavelength; rows resolve position along the slit.
        slit_bounds_m: (4,) entrance xmin,xmax,ymin,ymax rectangle in focal-plane metres.
            Its finite x width couples wavelength and across-slit position.
        optics: Unit-magnification polynomial dispersion, full Gaussian finite-pixel
            acceptance and incident pupil cosine; area and throughput apply downstream.
        pixel_bands: Unique achieved-channel identities in row-major pixel order.
            Identities describe native responses, rather than monochromatic wavelengths.
    """

    detector: DetectorProperties
    slit_bounds_m: NDArray[np.float64]
    optics: ImagingOptics
    pixel_bands: tuple[str, ...]

    @property
    def pixel_bounds_m(self) -> NDArray[np.float64]:
        """Return the detector's shared read-only active rectangles in metres."""
        return self.detector.pixel_bounds_m


def dispersive_spectrometer(
    detector: DetectorProperties,
    *,
    slit_bounds_m: NDArray[np.float64],
    focal_length_m: float,
    collecting_area_m2: float,
    spatial: SpatialDensity,
    reference_m: float,
    dispersion: float,
    quadratic_per_m: float = 0,
    smile: float = 0,
    keystone: float = 0,
    transmission: SpectralParameter = 1,
    throughput: float = 1,
) -> DispersiveSpectrometer:
    """Build a slit spectrometer independently of trajectory and scan cadence.

    For entrance (x,y) and d=wavelength-reference, the detector centroid is
    (x+dispersion*d+quadratic_per_m*d**2+smile*y, y+keystone*d).
    The finite slit transmits all admitted entrance rays. The post-slit PSF is
    integrated analytically over each active detector rectangle, including its
    full tails. Detector losses remain losses; responses are never renormalized.
    The pupil-projection cosine multiplies each entrance solid angle once.
    Pre-slit blur, anamorphic magnification, diffraction orders and stray light
    are excluded. Transmission can encode a supplied order-sorting passband.

    Args:
        detector: Explicit detector with stable active rectangles, pixel QE and electronics.
        slit_bounds_m: Borrowed finite ordered (4,) entrance rectangle, metres.
        focal_length_m: Positive entrance focal length, metres.
        collecting_area_m2: Nonnegative entrance pupil area, m^2.
        spatial: Declared Gaussian, prepared tabulated density or pure spatial callback.
        reference_m: Positive reference wavelength with zero dispersion shift, metres.
        dispersion: Finite nonzero linear wavelength-to-detector-x displacement, dimensionless.
        quadratic_per_m: Quadratic wavelength-to-x coefficient, m^-1.
        smile: Entrance-y to detector-x coefficient, dimensionless.
        keystone: Wavelength-to-detector-y coefficient, dimensionless.
        transmission: Scalar or pure wavelength-metre callback returning same-shaped
            physical intensity fractions in [0,1]. Borrowed for the instrument lifetime.
        throughput: Additional physical transmission fraction in [0,1].

    Returns:
        Physical spectrometer retaining the detector, with an owned slit and native channels.

    Raises:
        ValueError: Slit, detector, dispersion, PSF, aperture or throughput is invalid.
    """
    pixels = detector.pixel_bounds_m
    slit = np.array(slit_bounds_m, dtype=np.float64, copy=True)
    if slit.shape != (4,) or not np.isfinite(slit).all() or np.any(slit[[1, 3]] <= slit[[0, 2]]):
        raise ValueError("finite ordered slit (4,) rectangle required")
    if not np.isfinite(dispersion) or dispersion == 0:
        raise ValueError("spectrometer dispersion must be finite and nonzero")
    disperser = dispersive_imager(
        focal_length_m=focal_length_m,
        collecting_area_m2=collecting_area_m2,
        spatial=spatial,
        reference_m=reference_m,
        dispersion=dispersion,
        quadratic_per_m=quadratic_per_m,
        smile=smile,
        keystone=keystone,
        transmission=transmission,
        throughput=throughput,
    )

    def response(
        exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        accepted = disperser.response(exposure, read, wave, acquisition)
        pupil_cosine = np.sqrt(1 - np.sum(acquisition.directions_uv[read] ** 2, axis=1))
        return accepted * pupil_cosine[:, None]

    optics = ImagingOptics(
        focal_length_m, collecting_area_m2, response, disperser.settings, throughput
    )
    bands = tuple(f"r{row}-c{column}" for row, column in np.ndindex(pixels.shape[:2]))
    return DispersiveSpectrometer(detector, slit, optics, bands)
