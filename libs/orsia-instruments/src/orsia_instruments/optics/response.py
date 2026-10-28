"""Optical response contracts and common assembly of physical imaging descriptions.

Factories snapshot settings; callbacks remain borrowed, pure and pointwise in wavelength.
These bounded paraxial models are not optical design solvers."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition
from ..quadrature import validate_quadrature_order
from .pixels import gaussian_pixel_fraction, modulated_pixel_response, psf_pixel_response
from .psfs import PointSpreadFunction, select_psf

type OpticalResponse = Callable[[int, NDArray[np.float64], Acquisition], NDArray[np.float64]]


type ImageResponse = Callable[[int, int, NDArray[np.float64], Acquisition], NDArray[np.float64]]


type SpectralParameter = float | Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True, eq=False)
class _OpticalModel:
    """A physical image response with its fixed geometry, losses and coordinates."""

    focal_length_m: float
    collecting_area_m2: float
    response: ImageResponse
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ]
    throughput: float = 1
    port: str = "direct"

    def __post_init__(self) -> None:
        if not np.isfinite(self.focal_length_m) or self.focal_length_m <= 0:
            raise ValueError("focal length must be finite and positive")
        if not np.isfinite(self.collecting_area_m2) or self.collecting_area_m2 < 0:
            raise ValueError("collecting area must be finite and in [0, inf]")
        if not np.isfinite(self.throughput) or not 0 <= self.throughput <= 1:
            raise ValueError("throughput must be finite and in [0, 1]")
        settings = {name: exposure_setting(value, name) for name, value in self.settings.items()}
        if not self.port:
            raise ValueError("a physical output port is required")
        object.__setattr__(self, "settings", MappingProxyType(settings))


@dataclass(frozen=True, eq=False, init=False)
class ImagingOptics:
    """Optical model exposing a description derived from its construction.

    The legacy constructor (focal_length_m, collecting_area_m2, response, settings,
    throughput=1, port='direct') remains available for custom responses. The pure
    borrowed response(exposure,read,wavelengths_m,acquisition) returns accepted
    (nodes,wavelengths) fractions excluding area, throughput and QE. Its consistency
    with a custom description is caller-owned; arbitrary closures cannot be proven.

    Factories bind snapshotted settings to the response and this description as one
    model. Public attributes cannot be replaced, but settings arrays are writable;
    callers must not mutate them independently of the response. Rebuild through
    a factory to change physics; dataclasses.replace cannot replace individual
    attributes while silently retaining a stale response. replace(optics) may copy
    the unchanged model. No synchronization or mutable configuration is provided.

    Attributes:
        focal_length_m: Positive focal length in metres.
        collecting_area_m2: Nonnegative entrance-pupil area in square metres.
        response: Borrowed callable returning per-read accepted spectral fractions.
        settings: Named optical values with scalar or exposure-vector shapes.
        throughput: Dimensionless transmission in [0,1] after optical acceptance.
        port: Physical output-port identity for the model.
    """

    _model: _OpticalModel

    def __init__(
        self,
        focal_length_m: float | None = None,
        collecting_area_m2: float | None = None,
        response: ImageResponse | None = None,
        settings: Mapping[
            str,
            float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
        ]
        | None = None,
        throughput: float | None = None,
        port: str | None = None,
        *,
        _model: _OpticalModel | None = None,
    ) -> None:
        """Hold an imaging law and its physical description as one consistent model.

        Args:
            focal_length_m: Positive detector-plane focal length in metres.
            collecting_area_m2: Nonnegative projected collecting area in square metres.
            response: Borrowed optical accepted-fraction callback.
            settings: Named scalar or per-exposure optical setting values.
            throughput: Additional optical transmission fraction, defaulting to one.
            port: Read-port identity, defaulting to "direct".
            _model: Internal already assembled model used instead of public fields.

        Raises:
            TypeError: Required public optical fields are absent.
            ValueError: Public fields conflict with the supplied assembled model.
        """
        if _model is not None:
            if any(
                v is not None
                for v in (focal_length_m, collecting_area_m2, response, settings, throughput, port)
            ):
                raise ValueError(
                    "rebuild the optical model; its description cannot be replaced independently"
                )
            model = _model
        else:
            if (
                focal_length_m is None
                or collecting_area_m2 is None
                or response is None
                or settings is None
            ):
                raise TypeError("focal length, collecting area, response and settings are required")
            model = _OpticalModel(
                focal_length_m,
                collecting_area_m2,
                response,
                settings,
                1 if throughput is None else throughput,
                "direct" if port is None else port,
            )
        object.__setattr__(self, "_model", model)

    @property
    def focal_length_m(self) -> float:
        """Expose the optical projection's focal length.

        Returns:
            Focal length in metres.
        """
        return self._model.focal_length_m

    @property
    def collecting_area_m2(self) -> float:
        """Expose the pupil area used before throughput losses.

        Returns:
            Effective collecting area in square metres.
        """
        return self._model.collecting_area_m2

    @property
    def response(self) -> ImageResponse:
        """Access the optical model's per-read response law.

        Returns:
            Borrowed callable evaluating accepted pixel fractions for each read.
        """
        return self._model.response

    @property
    def settings(
        self,
    ) -> Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ]:
        """Access the optical settings retained from construction.

        Returns:
            Mapping of physical parameter names to recorded settings.
        """
        return self._model.settings

    @property
    def throughput(self) -> float:
        """Access throughput applied after accepted optical fractions.

        Returns:
            Dimensionless throughput factor.
        """
        return self._model.throughput

    @property
    def port(self) -> str:
        """Access the output-port identity carried into detector readings.

        Returns:
            Physical output-port identifier.
        """
        return self._model.port


def imaging_optics(
    focal_length: float,
    area: float,
    psf_sigma: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None,
    throughput: float,
    transmission: Callable[[int, float, float, NDArray[np.float64]], NDArray[np.float64]],
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ],
    *,
    psf: PointSpreadFunction | None = None,
    spatial: bool = False,
    pixel_order: int = 8,
    port: str = "direct",
) -> ImagingOptics:
    """Assemble finite-pixel imaging for a supplied local intensity transmission.

    Focal length and PSF sigma are metres; projected area is m^2. Transmission
    borrows (exposure, x_m, y_m, wavelengths_m) and returns wavelength-shaped
    fractions in [0,1]. With spatial=False it is evaluated at x=y=0; otherwise
    it is integrated jointly with the PSF. Exactly one PSF representation is
    required. Settings and Gaussian widths are snapshotted; callbacks remain
    borrowed and must be pure, nonmutating and pointwise in wavelength. Throughput,
    area and QE are excluded from the returned response's accepted fractions.

    Args:
        focal_length: Projection focal length in metres.
        area: Pupil collecting area in square metres.
        psf_sigma: Scalar or per-exposure Gaussian width in metres; use None
            when ``psf`` supplies the spatial law.
        throughput: Dimensionless post-response transmission factor.
        transmission: Borrowed fraction law of exposure, x, y and wavelength.
        settings: Physical optical settings retained with the model.
        psf: Optional borrowed spatial density law in inverse square metres.
        spatial: Integrate position-dependent transmission jointly with the PSF.
        pixel_order: Positive spatial quadrature order for integrated responses.
        port: Physical output-port identifier for resulting measurements.

    Returns:
        Imaging model with a borrowed per-read response law and recorded settings.

    Raises:
        ValueError: PSF representation or quadrature order is invalid.
    """
    sigma, density = select_psf(psf_sigma, psf)
    validate_quadrature_order(pixel_order)

    def response(
        exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        bounds = acquisition.pixel_bounds_m[read]
        centres = acquisition.image_xy_m[read]
        if sigma is None:
            assert density is not None
            mapped = np.broadcast_to(centres[:, None, :], (len(centres), len(wave), 2))
            # Evaluate a uniform filter once, retaining its shape/range checks.
            uniform = None if spatial else transmission(exposure, 0, 0, wave)
            return psf_pixel_response(
                wave,
                bounds,
                mapped,
                centres,
                exposure,
                psf=density,
                pixel_order=pixel_order,
                transmission=lambda x, y, w: (
                    transmission(exposure, x, y, w) if uniform is None else uniform
                ),
            )
        if not spatial:
            fraction = gaussian_pixel_fraction(bounds, centres, sigma)
            return fraction[:, None] * transmission(exposure, 0, 0, wave)
        return modulated_pixel_response(
            wave,
            bounds,
            centres,
            sigma_m=sigma,
            pixel_order=pixel_order,
            transmission=lambda x, y, wavelength: transmission(exposure, x, y, wavelength),
        )

    return ImagingOptics(focal_length, area, response, settings, throughput, port)


def constant_or_default(value: SpectralParameter, default: float) -> float:
    """Provide a scalar for eager validation while deferring spectral callbacks.

    Args:
        value: Scalar setting or borrowed wavelength-dependent law.
        default: Validation-time scalar used only when a law is supplied.

    Returns:
        Original scalar setting or the supplied fallback value.
    """
    # Validate constants eagerly through the physical kernel, without evaluating
    # user callbacks at arbitrary wavelengths outside their declared support.
    return default if callable(value) else value


def spectral_value(
    value: SpectralParameter,
    wave: NDArray[np.float64],
    name: str,
    *,
    broadcast: bool = False,
) -> NDArray[np.float64]:
    """Evaluate a borrowed scalar or spectral law on wavelengths in metres.

    Scalar inputs yield a zero-dimensional float64 view. A callable must return
    the wavelength shape unless broadcast=True explicitly permits broadcasting.
    The returned array may alias caller storage; no physical range is checked
    here and inputs are never mutated. The family kernel validates its range.

    Args:
        value: Scalar physical setting or borrowed wavelength-dependent callable.
        wave: Wavelength array in metres supplied to a callable.
        name: Setting name used in shape errors.
        broadcast: Permit a callable result to broadcast to the wavelength shape.

    Returns:
        Numeric scalar array for a constant or spectral values with the
        wavelength shape for a callable; output may alias input storage.

    Raises:
        ValueError: Spectral callable returns an unexpected shape when broadcast
            is disabled.
    """
    if not callable(value):
        return np.asarray(value, dtype=np.float64)
    result = value(wave)
    if broadcast:
        return np.broadcast_to(result, wave.shape)
    if result.shape != wave.shape:
        raise ValueError(f"{name} must have the same shape as wavelengths")
    return result


def spatial_offset(
    law: Callable[[float, float], float] | None,
    x: float,
    y: float,
    name: str,
) -> float:
    """Evaluate local optical displacement at an image-plane coordinate.

    Args:
        law: Spatial offset callable in metres, or None for zero displacement.
        x: First image-plane coordinate in metres.
        y: Second image-plane coordinate in metres.
        name: Physical parameter name used in invalid-value diagnostics.

    Returns:
        Finite scalar displacement in metres at the supplied position.

    Raises:
        ValueError: The spatial law returns a nonfinite displacement.
    """
    if law is None:
        return 0.0
    value = law(x, y)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar in metres")
    return value


@overload
def exposure_setting[T: np.floating | np.integer](value: NDArray[T], name: str) -> NDArray[T]: ...


@overload
def exposure_setting[T: np.floating | np.integer](value: T, name: str) -> NDArray[T]: ...


@overload
def exposure_setting(value: int | Sequence[int], name: str) -> NDArray[np.int_]: ...


@overload
def exposure_setting(value: float | Sequence[float], name: str) -> NDArray[np.float64]: ...


def exposure_setting(
    value: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    name: str,
) -> NDArray[np.floating | np.integer]:
    """Freeze a setting's numerical values independently of caller storage.

    Args:
        value: Finite scalar or nonempty one-dimensional exposure vector.
        name: Physical parameter name used in invalid-value diagnostics.

    Returns:
        Independent scalar or exposure array preserving the input dtype.

    Raises:
        ValueError: Values are nonfinite, empty or have more than one axis.
    """
    result = np.copy(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if result.ndim > 1 or result.size == 0:
        raise ValueError(f"{name} must be scalar or a nonempty exposure vector")
    return result


def setting_at_exposure(setting: NDArray[np.floating | np.integer], exposure: int) -> float:
    """Read one exposure's scalar optical setting for kernel evaluation.

    Args:
        setting: Validated scalar or one-dimensional per-exposure array.
        exposure: Zero-based index used for vector settings.

    Returns:
        Selected setting converted to a Python floating value.
    """
    return float(setting if setting.ndim == 0 else setting[exposure])
