"""Assemble physical row-strip cameras independently of trajectory and cadence.

A filter covers complete active pixels in selected detector rows. The model
uses a forward virtual focal plane, with columns along x and rows along y.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import j0, j1

from orsia_instruments.acquisition.quadrature import Acquisition
from orsia_instruments.detector import DetectorProperties
from orsia_instruments.optics.pixels import pixel_acceptance
from orsia_instruments.optics.psfs import GaussianPSF, SpatialDensity, airy_psf
from orsia_instruments.optics.response import ImagingOptics

type FilterTransmission = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class ImagerSizing:
    """Reference nadir geometry used to size a physical focal length.

    Attributes:
        altitude_m: Positive reference distance to a locally planar surface, metres.
        ground_sampling_m: Positive reference pixel-centre spacing on that plane, metres.
        pixel_pitch_m: Positive physical detector pixel-centre spacing, metres.
    """

    altitude_m: float
    ground_sampling_m: float
    pixel_pitch_m: float

    def __post_init__(self) -> None:
        values = np.array([self.altitude_m, self.ground_sampling_m, self.pixel_pitch_m])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("reference sizing dimensions must be finite and positive")


@dataclass(frozen=True)
class FilterStrip:
    """A spectral intensity transmission on a half-open range of detector rows.

    Attributes:
        start_row: Inclusive zero-based first row.
        stop_row: Exclusive row stop, greater than start_row.
        band: Nonempty physical band identity; repeated identities require compatible laws.
        transmission: Borrowed pure wavelength-metre callback returning fractions in [0,1].
    """

    start_row: int
    stop_row: int
    band: str
    transmission: FilterTransmission

    def __post_init__(self) -> None:
        if self.start_row < 0 or self.stop_row <= self.start_row or not self.band:
            raise ValueError("strip requires a nonempty identity and ordered nonnegative row range")


@dataclass(frozen=True, eq=False)
class FilterCamera:
    """Bind detector photoelectric properties to one coherent row-strip optical model.

    Construct with filter_camera. Optical declarations are read-only properties
    of the same model that evaluates the response. Rebuild with filter_camera to
    change strips, support or optics. dataclasses.replace may change detector QE
    or electronics when the active pixel rectangles remain exactly unchanged.

    Attributes:
        detector: Stable active geometry, pixel QE and electronics.
        pixel_bounds_m: Shared read-only (row,column,4) active rectangles in metres.
        strips: Row-strip declarations; uncovered rows have zero optical response.
        optics: Finite-pixel response with area and throughput applied downstream once.
        support_radius_m: Positive radial PSF truncation bound, focal-plane metres.
        omitted_psf_mass: Bound on omitted whole-plane PSF mass, or None for
            caller-certified support. Retained tails are not renormalized.
        row_bands: Band identity per row; empty strings identify uncovered rows.

    Detector arrays are read-only snapshots. Borrowed filter and PSF laws must
    remain pure and stable, including during concurrent use.
    """

    detector: DetectorProperties
    _filter_optics: _FilterOptics = field(repr=False)

    def __post_init__(self) -> None:
        if not np.array_equal(self.detector.pixel_bounds_m, self._filter_optics.pixel_bounds_m):
            raise ValueError("replacement detector must retain the camera pixel geometry")

    @property
    def pixel_bounds_m(self) -> NDArray[np.float64]:
        """Return shared read-only active detector rectangles in metres."""
        return self.detector.pixel_bounds_m

    @property
    def strips(self) -> tuple[FilterStrip, ...]:
        """Return the row-strip declarations used by the optical response."""
        return self._filter_optics.strips

    @property
    def optics(self) -> ImagingOptics:
        """Return imaging optics bound to the camera's physical strip model."""
        return self._filter_optics.optics

    @property
    def support_radius_m(self) -> float:
        """Return the response's finite radial PSF support in focal-plane metres."""
        return self._filter_optics.support_radius_m

    @property
    def omitted_psf_mass(self) -> float | None:
        """Return the omitted PSF mass bound, or None for caller-certified support."""
        return self._filter_optics.omitted_psf_mass

    @property
    def row_bands(self) -> tuple[str, ...]:
        """Return physical row identities derived from the strip layout."""
        return self._filter_optics.row_bands


@dataclass(frozen=True, eq=False)
class FocalPlaneImager:
    """Detector-owned local spectral imager with a finite incident-ray domain.

    Attributes:
        detector: Physical active pixels, quantum efficiency and electronics.
        optics: Local spectral/PSF response including finite support and pupil projection.
        support_radius_m: Caller-certified positive PSF centroid expansion, metres.
        pixel_bounds_m: Shared immutable detector rectangles, (row,column,4), metres.

    Optical callbacks are borrowed and must remain pure and stable. Support is
    truncated without renormalization. Declare the omitted mass in the scenario.
    """

    detector: DetectorProperties
    optics: ImagingOptics
    support_radius_m: float

    @property
    def pixel_bounds_m(self) -> NDArray[np.float64]:
        """Return immutable active detector rectangles in metres."""
        return self.detector.pixel_bounds_m


def focal_plane_imager(
    detector: DetectorProperties, optics: ImagingOptics, *, support_radius_m: float
) -> FocalPlaneImager:
    """Attach a detector and certified incident support to local imaging optics.

    Use LVF, two-beam-port or cavity optics. Their finite-pixel integrals are
    preserved, including joint spatial transmission. This boundary applies the
    pupil cosine once and clips incident centroids to expanded pixel rectangles.
    LVF scans use a fixed detector-plane law; supply scalar reference/width for
    that case. Exposure-dependent OPD and cavity spacing remain constant during
    each read. Each interferometer port requires its own detector and instance.

    Args:
        detector: Physical active pixels, QE and electronics.
        optics: Local optical response excluding pupil projection and support clipping.
        support_radius_m: Finite positive caller-certified PSF support in metres.

    Returns:
        Detector-owned imager suitable for camera acquisition and frame simulation.

    Raises:
        ValueError: Support radius is nonfinite or nonpositive.
    """
    if not np.isfinite(support_radius_m) or support_radius_m <= 0:
        raise ValueError("finite positive PSF support radius required")
    return FocalPlaneImager(detector, _supported_optics(optics, support_radius_m), support_radius_m)


def camera_read_bands(
    camera: FilterCamera | FocalPlaneImager, indices: NDArray[np.int64]
) -> tuple[str, ...]:
    """Label physical native channels before grouping their achieved responses.

    Args:
        camera: Detector-owned imager or row-strip camera.
        indices: Original (read,3) exposure,row,column correspondence.

    Returns:
        Strip identities or exposure/pixel identities; compatibility is checked
        separately from these descriptive labels.
    """
    if isinstance(camera, FilterCamera):
        return tuple(camera.row_bands[r] for _, r, _ in indices)
    return tuple(f"{camera.optics.port}:e{e}:r{r}:c{c}" for e, r, c in indices)


def sized_focal_length(sizing: ImagerSizing) -> float:
    """Calculate the paraxial nadir focal length for supplied reference sizing.

    Args:
        sizing: Reference altitude, ground sampling and physical pixel pitch in metres.

    Returns:
        Physical focal length altitude * pitch / ground_sampling in metres.
    """
    return sizing.altitude_m * sizing.pixel_pitch_m / sizing.ground_sampling_m


def filter_strip(
    start_row: int, stop_row: int, *, band: str, transmission: FilterTransmission
) -> FilterStrip:
    """Declare a stable spectral filter covering complete detector rows.

    Args:
        start_row: Inclusive zero-based row start.
        stop_row: Exclusive row stop.
        band: Nonempty physical band identity.
        transmission: Pure borrowed wavelength-metre law returning same-shaped fractions.

    Returns:
        Row-strip physical declaration; assembly checks detector extent and overlap.

    Raises:
        ValueError: Row range or band identity is invalid.
    """
    return FilterStrip(start_row, stop_row, band, transmission)


def filter_camera(
    detector: DetectorProperties,
    strips: Sequence[FilterStrip],
    *,
    focal_length_m: float,
    f_number: float,
    collecting_area_m2: float | None = None,
    throughput: float = 1,
    spatial: SpatialDensity | None = None,
    support_radius_m: float | None = None,
    omitted_psf_mass: float = 1e-6,
    airy_wavelength_max_m: float | None = None,
    pixel_order: int = 8,
    allow_uncovered_rows: bool = False,
) -> FilterCamera:
    """Build a uniform or arbitrarily many-strip camera with explicit PSF support.

    The circular unobscured pupil area is pi*(focal_length/f_number)^2/4.
    A supplied area must agree to 1e-12 relative tolerance. Gaussian support
    uses max(sigma_x,sigma_y)*sqrt(-2*log(omitted_psf_mass)). The radial tail
    budget is exact for isotropic widths and conservative for anisotropic widths.
    Airy support uses a conservative encircled-energy root at the supplied
    maximum wavelength. Custom PSFs require caller-certified radial support.
    Filters lie on active detector pixels, so adjacent strips do not mix within
    an active rectangle. A PSF can move incident light between detector rows.
    Each incident ray receives its pupil-projection cosine once. Together with
    focal-plane solid-angle geometry this gives the local cos^4 field law.

    Args:
        detector: Explicit detector with stable active rectangles, pixel QE and electronics.
        strips: Nonoverlapping half-open row ranges; one strip provides a uniform camera.
        focal_length_m: Positive physical focal length, metres.
        f_number: Positive focal length divided by pupil diameter.
        collecting_area_m2: Optional consistency assertion for circular pupil area, m^2.
        throughput: Post-filter physical fraction in [0,1].
        spatial: Declared Gaussian, prepared tabulated density or pure spatial callback.
        support_radius_m: Required finite positive radial bound for custom PSFs.
        omitted_psf_mass: Gaussian/Airy tail budget strictly between zero and one.
        airy_wavelength_max_m: Select an Airy pupil and bound its positive wavelength support.
        pixel_order: Positive finite-pixel quadrature order for non-Gaussian PSFs.
        allow_uncovered_rows: Permit optically dark rows; overlaps always raise.

    Returns:
        Camera retaining the supplied detector and borrowed pure filter/PSF laws.

    Raises:
        ValueError: Layout, aperture, spatial model or integration support is inconsistent.
    """
    pixels = detector.pixel_bounds_m
    if not np.isfinite([focal_length_m, f_number]).all() or min(focal_length_m, f_number) <= 0:
        raise ValueError("focal length and f-number must be finite and positive")
    area = np.pi * (focal_length_m / f_number) ** 2 / 4
    if collecting_area_m2 is not None and not np.isclose(
        collecting_area_m2, area, rtol=1e-12, atol=0
    ):
        raise ValueError("collecting area disagrees with focal length and f-number")
    if not 0 < omitted_psf_mass < 1 or not np.isfinite(omitted_psf_mass):
        raise ValueError("omitted PSF mass must be strictly between zero and one")
    if pixel_order < 1:
        raise ValueError("pixel order must be positive")
    strips = tuple(strips)
    tail: float | None = omitted_psf_mass
    if airy_wavelength_max_m is not None:
        if spatial is not None or support_radius_m is not None:
            raise ValueError("Airy selection excludes other PSF and support settings")
        if not np.isfinite(airy_wavelength_max_m) or airy_wavelength_max_m <= 0:
            raise ValueError("Airy maximum wavelength must be finite and positive")
        # J0(z)^2 + J1(z)^2 is the remaining radial Airy energy.
        edge = brentq(lambda z: j0(z) ** 2 + j1(z) ** 2 - omitted_psf_mass, 0, 4 / omitted_psf_mass)
        radius = edge * airy_wavelength_max_m * f_number / np.pi
        spatial = partial(airy_psf, f_number=f_number, wavelength_max_m=airy_wavelength_max_m)
    else:
        radius = support_radius_m
    if spatial is None:
        raise ValueError("a spatial density model is required")
    if isinstance(spatial, GaussianPSF):
        if radius is not None:
            raise ValueError("Gaussian support is determined by its omitted-mass budget")
        radius = float(np.max(spatial.sigma_m)) * np.sqrt(-2 * np.log(omitted_psf_mass))
    elif airy_wavelength_max_m is None:
        tail = None
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError("custom PSFs require finite positive radial support")
    support = float(radius)

    model = _FilterOptics(
        pixels,
        strips,
        spatial,
        support,
        tail,
        focal_length_m,
        f_number,
        area,
        throughput,
        pixel_order,
        allow_uncovered_rows,
    )
    return FilterCamera(detector, model)


@dataclass(frozen=True, eq=False)
class _FilterOptics:
    """Own a fixed strip layout, PSF support and the response computed from them.

    Attributes:
        pixel_bounds_m: Borrowed immutable (row,column,4) detector rectangles, metres.
        strips: Stable row ranges, physical band identities and borrowed spectral laws.
        spatial: Pure borrowed spatial density model in inverse square metres.
        support_radius_m: Positive focal-plane centroid expansion, metres.
        omitted_psf_mass: Radial tail bound, or None for caller-certified support.
        focal_length_m: Positive physical projection focal length, metres.
        f_number: Positive focal-length/pupil-diameter ratio.
        collecting_area_m2: Circular pupil collecting area, square metres.
        throughput: Post-response physical intensity fraction.
        pixel_order: Finite-pixel spatial quadrature order.
        allow_uncovered_rows: Permit optically dark detector rows.
        row_bands: Derived band identity per row, empty for uncovered rows.
        optics: Fixed imaging adapter bound to this model's response method.
    """

    pixel_bounds_m: NDArray[np.float64]
    strips: tuple[FilterStrip, ...]
    spatial: SpatialDensity
    support_radius_m: float
    omitted_psf_mass: float | None
    focal_length_m: float
    f_number: float
    collecting_area_m2: float
    throughput: float
    pixel_order: int
    allow_uncovered_rows: bool
    row_bands: tuple[str, ...] = field(init=False)
    optics: ImagingOptics = field(init=False, repr=False)
    _row_laws: tuple[FilterTransmission | None, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows = [""] * self.pixel_bounds_m.shape[0]
        laws: list[FilterTransmission | None] = [None] * len(rows)
        for strip in self.strips:
            if strip.stop_row > len(rows) or any(
                law is not None for law in laws[strip.start_row : strip.stop_row]
            ):
                raise ValueError("strips overlap or exceed detector rows")
            rows[strip.start_row : strip.stop_row] = [strip.band] * (
                strip.stop_row - strip.start_row
            )
            laws[strip.start_row : strip.stop_row] = [strip.transmission] * (
                strip.stop_row - strip.start_row
            )
        if not self.allow_uncovered_rows and any(law is None for law in laws):
            raise ValueError("every detector row must be covered by a strip")
        object.__setattr__(self, "row_bands", tuple(rows))
        object.__setattr__(self, "_row_laws", tuple(laws))
        object.__setattr__(
            self,
            "optics",
            ImagingOptics(
                self.focal_length_m,
                self.collecting_area_m2,
                self.response,
                {"f_number": self.f_number},
                self.throughput,
            ),
        )

    def response(
        self, exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        """Evaluate accepted fractions from this layout, PSF and support.

        Args:
            exposure: Original exposure index for the borrowed spatial density.
            read: Acquisition read index identifying one physical detector rectangle.
            wave: Wavelengths in metres, shape (wavelengths,).
            acquisition: Borrowed incident focal-plane coordinates and directions.

        Returns:
            Fresh (nodes,wavelengths) fractions including filter transmission,
            finite support and pupil projection, excluding area, throughput and QE.

        Raises:
            ValueError: The pixel is ambiguous or the transmission has invalid
                shape, nonfinite values or fractions outside [0,1].
        """
        matches = np.all(self.pixel_bounds_m == acquisition.pixel_bounds_m[read], axis=-1)
        locations = np.argwhere(matches)
        if len(locations) != 1:
            raise ValueError("acquisition pixel must identify one camera detector pixel")
        row = int(locations[0, 0])
        law = self._row_laws[row]
        image = acquisition.image_xy_m[read]
        if law is None:
            return np.zeros((len(image), len(wave)))
        transmission = law(wave)
        if (
            transmission.shape != wave.shape
            or not np.isfinite(transmission).all()
            or np.any((transmission < 0) | (transmission > 1))
        ):
            raise ValueError("strip transmission must be same-shaped finite fractions in [0,1]")
        bounds = acquisition.pixel_bounds_m[read]
        fraction = pixel_acceptance(
            wave,
            bounds,
            image,
            spatial=self.spatial,
            exposure_index=exposure,
            pixel_order=self.pixel_order,
        )
        inside = np.all(
            (image >= bounds[[0, 2]] - self.support_radius_m)
            & (image <= bounds[[1, 3]] + self.support_radius_m),
            axis=1,
        )
        pupil_projection = np.sqrt(1 - np.sum(acquisition.directions_uv[read] ** 2, axis=1))
        return fraction * transmission * (inside * pupil_projection)[:, None]


def _supported_optics(optics: ImagingOptics, support: float) -> ImagingOptics:
    def response(
        exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        image = acquisition.image_xy_m[read]
        bounds = acquisition.pixel_bounds_m[read]
        inside = np.all(
            (image >= bounds[[0, 2]] - support) & (image <= bounds[[1, 3]] + support), axis=1
        )
        pupil_projection = np.sqrt(1 - np.sum(acquisition.directions_uv[read] ** 2, axis=1))
        return (
            optics.response(exposure, read, wave, acquisition)
            * (inside * pupil_projection)[:, None]
        )

    return ImagingOptics(
        optics.focal_length_m,
        optics.collecting_area_m2,
        response,
        optics.settings,
        optics.throughput,
        optics.port,
    )
