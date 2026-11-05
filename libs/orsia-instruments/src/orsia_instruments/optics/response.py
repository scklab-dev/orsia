"""Optical response contracts and common assembly of physical imaging descriptions.

Factories snapshot settings; callbacks remain borrowed, pure and pointwise in wavelength.
These bounded paraxial models are not optical design solvers."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ..acquisition import Acquisition
from ..quadrature import validate_quadrature_order
from .pixels import pixel_acceptance
from .psfs import SpatialDensity

type OpticalResponse = Callable[[int, NDArray[np.float64], Acquisition], NDArray[np.float64]]


type ImageResponse = Callable[[int, int, NDArray[np.float64], Acquisition], NDArray[np.float64]]


type SpectralParameter = float | Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True, eq=False, init=False)
class ImagingOptics:
    """Bind a physical image response to its fixed optical description.

    Factories and custom responses use the same explicit constructor. The pure
    borrowed response(exposure,read,wavelengths_m,acquisition) returns accepted
    (nodes,wavelengths) fractions excluding area, throughput and QE. A custom
    callback's agreement with its description remains caller-owned.

    Public attributes are fixed after construction. Rebuild with the relevant
    factory to change physics. Fields have init=False, so dataclasses.replace
    cannot replace a description independently of its response. Settings are
    independent immutable value snapshots; keep array shape/dtype metadata
    unchanged. Callbacks remain borrowed, pure, stable and pointwise in wavelength.

    Attributes:
        focal_length_m: Positive projection focal length in metres.
        collecting_area_m2: Nonnegative pupil collecting area in square metres.
        response: Borrowed per-read accepted-fraction callback.
        settings: Named immutable scalar or exposure-vector optical setting snapshots.
        throughput: Dimensionless post-response transmission in [0,1].
        port: Nonempty physical output-port identity.
    """

    focal_length_m: float = field(init=False)
    collecting_area_m2: float = field(init=False)
    response: ImageResponse = field(init=False)
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ] = field(init=False)
    throughput: float = field(init=False)
    port: str = field(init=False)

    def __init__(
        self,
        focal_length_m: float,
        collecting_area_m2: float,
        response: ImageResponse,
        settings: Mapping[
            str,
            float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
        ],
        throughput: float = 1,
        port: str = "direct",
    ) -> None:
        """Construct the response and its numerical description together.

        Args:
            focal_length_m: Positive detector-plane focal length in metres.
            collecting_area_m2: Nonnegative projected pupil area in square metres.
            response: Pure borrowed optical accepted-fraction callback.
            settings: Finite scalar or per-exposure optical settings to snapshot.
            throughput: Post-response transmission fraction in [0,1].
            port: Nonempty output-port identity.

        Raises:
            ValueError: Focal length, area, throughput, port or settings are invalid.
        """
        if not np.isfinite(focal_length_m) or focal_length_m <= 0:
            raise ValueError("focal length must be finite and positive")
        if not np.isfinite(collecting_area_m2) or collecting_area_m2 < 0:
            raise ValueError("collecting area must be finite and in [0, inf]")
        if not np.isfinite(throughput) or not 0 <= throughput <= 1:
            raise ValueError("throughput must be finite and in [0, 1]")
        if not port:
            raise ValueError("a physical output port is required")
        copied = {}
        for name, value in settings.items():
            array = exposure_setting(value, name)
            # Immutable bytes prevent writeability from being re-enabled by callers.
            copied[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self, "focal_length_m", focal_length_m)
        object.__setattr__(self, "collecting_area_m2", collecting_area_m2)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "settings", MappingProxyType(copied))
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "port", port)


def imaging_optics(
    focal_length: float,
    area: float,
    spatial: SpatialDensity,
    throughput: float,
    transmission: Callable[[int, float, float, NDArray[np.float64]], NDArray[np.float64]],
    settings: Mapping[
        str, float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer]
    ],
    *,
    spatial_transmission: bool = False,
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
        spatial: Declared Gaussian, prepared tabulated density or pure spatial callback.
        throughput: Dimensionless post-response transmission factor.
        transmission: Borrowed fraction law of exposure, x, y and wavelength.
        settings: Physical optical settings retained with the model.
        spatial_transmission: Integrate a position-dependent transmission jointly with density.
        pixel_order: Positive spatial quadrature order for integrated responses.
        port: Physical output-port identifier for resulting measurements.

    Returns:
        Imaging model with a borrowed per-read response law and recorded settings.

    Raises:
        ValueError: PSF representation or quadrature order is invalid.
    """
    validate_quadrature_order(pixel_order)

    def response(
        exposure: int, read: int, wave: NDArray[np.float64], acquisition: Acquisition
    ) -> NDArray[np.float64]:
        bounds = acquisition.pixel_bounds_m[read]
        centres = acquisition.image_xy_m[read]
        transmission_law = (
            (lambda x, y, w: transmission(exposure, x, y, w))
            if spatial_transmission
            else transmission(exposure, 0, 0, wave)
        )
        return pixel_acceptance(
            wave,
            bounds,
            centres,
            spatial=spatial,
            transmission=transmission_law,
            image_xy_m=centres,
            exposure_index=exposure,
            pixel_order=pixel_order,
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
