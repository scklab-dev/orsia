"""Define detector geometry, photoelectric properties and memoryless physical readout."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .acquisition import ReadSchedule
from .charge import ExpectedElectrons

DETECTOR_MAP_NAMES = (
    "quantum_efficiency",
    "dark_electrons_per_s",
    "read_noise_electrons",
    "full_well_electrons",
    "electrons_per_dn",
    "bias_dn",
    "nonlinearity",
)


@dataclass(frozen=True)
class DetectorElectronics:
    """One homogeneous detector operating point, with electron-domain noise and gain.

    Full well clips collected charge before independent Gaussian read noise.
    electrons_per_dn converts charge to DN, then bias_dn is added, then rounding
    (nearest, ties to even) and ADC clipping to [0,2**adc_bits-1] are applied.
    No persistence, excess noise, correlation or nonlinearity other than clipping.

    Attributes:
        dark_electrons_per_s: Mean thermally generated electrons per second.
        read_noise_electrons: Gaussian read-noise standard deviation in electrons.
        full_well_electrons: Charge cap in electrons before read noise.
        electrons_per_dn: Conversion gain in electrons per digital number.
        bias_dn: Additive digital offset before integer rounding.
        adc_bits: ADC width between one and 32 bits.
    """

    dark_electrons_per_s: float = 0
    read_noise_electrons: float = 0
    full_well_electrons: float = 1e6
    electrons_per_dn: float = 1
    bias_dn: float = 0
    adc_bits: int = 16

    def __post_init__(self) -> None:
        if not np.isfinite(self.dark_electrons_per_s) or self.dark_electrons_per_s < 0:
            raise ValueError("dark current must be finite and in [0, inf]")
        if not np.isfinite(self.read_noise_electrons) or self.read_noise_electrons < 0:
            raise ValueError("read noise must be finite and in [0, inf]")
        if not np.isfinite(self.full_well_electrons) or self.full_well_electrons <= 0:
            raise ValueError("full well must be finite and positive")
        if not np.isfinite(self.electrons_per_dn) or self.electrons_per_dn <= 0:
            raise ValueError("electrons per DN must be finite and positive")
        if not np.isfinite(self.bias_dn) or not 1 <= self.adc_bits <= 32:
            raise ValueError("finite bias and ADC width in [1,32] required")


@dataclass(frozen=True, eq=False)
class DetectorOperatingPoints:
    """Persistent per-read detector parameters for a specified read schedule.

    All arrays are independent writable (reads,) copies preserving input dtype.
    Repeat a pixel's value for each exposure to represent fixed-pattern variation;
    these are not noise draws.
    QE is applied upstream via ``pixel_quantum_efficiency`` to avoid multiplying
    it twice. Nonlinearity c applies ``q*(1+c*q/full_well)`` after full-well
    clipping and before read noise; -0.5 <= c <= 0.5 ensures monotonicity.

    Attributes:
        dark_electrons_per_s: Per-read mean dark-current rate (reads,).
        read_noise_electrons: Per-read Gaussian noise scale in electrons (reads,).
        full_well_electrons: Per-read charge cap in electrons (reads,).
        electrons_per_dn: Per-read conversion gain in electrons per DN (reads,).
        bias_dn: Per-read additive digital offset (reads,).
        nonlinearity: Per-read dimensionless charge curvature in [-0.5, 0.5].
        adc_bits: Shared ADC width between one and 32 bits.
    """

    dark_electrons_per_s: NDArray[np.float64]
    read_noise_electrons: NDArray[np.float64]
    full_well_electrons: NDArray[np.float64]
    electrons_per_dn: NDArray[np.float64]
    bias_dn: NDArray[np.float64]
    nonlinearity: NDArray[np.float64]
    adc_bits: int = 16

    def __post_init__(self) -> None:
        for name in (
            "dark_electrons_per_s",
            "read_noise_electrons",
            "full_well_electrons",
            "electrons_per_dn",
            "bias_dn",
            "nonlinearity",
        ):
            object.__setattr__(self, name, np.copy(getattr(self, name)))
        size = self.dark_electrons_per_s.shape
        if (
            len(size) != 1
            or not size[0]
            or any(
                getattr(self, name).shape != size
                for name in (
                    "read_noise_electrons",
                    "full_well_electrons",
                    "electrons_per_dn",
                    "bias_dn",
                    "nonlinearity",
                )
            )
        ):
            raise ValueError("detector maps must share a nonempty (reads,) shape")
        if not np.isfinite(self.dark_electrons_per_s).all() or np.any(
            self.dark_electrons_per_s < 0
        ):
            raise ValueError("dark current must be finite and in [0, inf]")
        if not np.isfinite(self.read_noise_electrons).all() or np.any(
            self.read_noise_electrons < 0
        ):
            raise ValueError("read noise must be finite and in [0, inf]")
        if not np.isfinite(self.full_well_electrons).all() or np.any(self.full_well_electrons <= 0):
            raise ValueError("full well must be finite and positive")
        if not np.isfinite(self.electrons_per_dn).all() or np.any(self.electrons_per_dn <= 0):
            raise ValueError("electrons per DN must be finite and positive")
        if not np.isfinite(self.nonlinearity).all() or np.any(
            (self.nonlinearity < -0.5) | (self.nonlinearity > 0.5)
        ):
            raise ValueError("nonlinearity must be finite and in [-0.5, 0.5]")
        if not np.isfinite(self.bias_dn).all():
            raise ValueError("bias must be finite")
        if not 1 <= self.adc_bits <= 32:
            raise ValueError("ADC width must be in [1,32]")


@dataclass(frozen=True, eq=False)
class DetectorReadings:
    """Digitised physical reads and separate full-well/ADC flags, not calibrated products.

    Arrays have shape (realisations,reads), immutable owned storage. Read
    identities and exposure support are retained. ADC flags mark analogue values
    outside its range before rounding; reaching full well is flagged at equality.
    Invalid physical support raises upstream rather than becoming saturation.

    Attributes:
        reads: Identity and time support of the simulated physical reads.
        dn: Owned immutable integer counts of shape (realisations,reads).
        full_well_saturated: Boolean flags for reads reaching the full-well cap.
        adc_clipped: Boolean flags for values outside the ADC range before rounding.
    """

    reads: ReadSchedule
    dn: NDArray[np.int64]
    full_well_saturated: NDArray[np.bool_]
    adc_clipped: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = self.dn.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != len(self.reads.sample_ids):
            raise ValueError("readings must have shape (realisations,reads)")
        for name, dtype in (
            ("dn", np.int64),
            ("full_well_saturated", np.bool_),
            ("adc_clipped", np.bool_),
        ):
            array = getattr(self, name)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("readings arrays must match shape and declared dtype")
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=dtype).reshape(shape)
            )
        if np.any(self.dn < 0):
            raise ValueError("DN must be nonnegative")


@dataclass(frozen=True, eq=False)
class DetectorProperties:
    """Stable detector geometry, pixel QE and electronic operating properties.

    All ten fields require explicit values. Geometry and property arrays are
    independent read-only copies; scalar properties are stored as 0-D arrays.
    Keep retained array buffers and shape/dtype metadata stable during use.
    Map shapes must exactly equal the derived (rows,columns) detector shape.
    Columns increase x and rows increase y for regular geometry. Explicit None
    pitch retains arbitrary finite active rectangles for irregular geometry.
    Declared pitch requires centre spacing on each available axis and active
    sizes no larger than pitch, within floating-point coordinate roundoff.
    Single-row and single-column detectors retain both declared pitch components.
    QE is wavelength independent and applied once before photon noise.
    DetectorOperatingPoints defines clipping and nonlinear electronic transfer.

    Attributes:
        pixel_bounds_m: Owned read-only float64 (rows,columns,4) active rectangles
            in metres, ordered xmin,xmax,ymin,ymax. Gaps represent optical loss.
        pitch_m: Positive (x,y) centre spacing in metres, or None for irregular geometry.
        shape: Derived (rows,columns) dimensions of the active bounds.
        quantum_efficiency: Scalar or (rows,columns) photoelectron fraction.
        dark_electrons_per_s: Scalar or map of dark-current rates.
        read_noise_electrons: Scalar or map of Gaussian electron noise scales.
        full_well_electrons: Scalar or map of charge caps in electrons.
        electrons_per_dn: Scalar or map of electron-to-DN conversion gains.
        bias_dn: Scalar or map of digital offsets in DN.
        nonlinearity: Scalar or map of dimensionless charge curvature.
        adc_bits: Shared ADC width between one and 32 bits.
    """

    pixel_bounds_m: NDArray[np.float64]
    pitch_m: tuple[float, float] | None
    quantum_efficiency: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    dark_electrons_per_s: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    read_noise_electrons: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    full_well_electrons: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    electrons_per_dn: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    bias_dn: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    nonlinearity: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    )
    adc_bits: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return detector dimensions as (rows,columns), derived from active bounds."""
        return self.pixel_bounds_m.shape[:2]

    def __post_init__(self) -> None:
        pixels = np.array(self.pixel_bounds_m, dtype=np.float64, copy=True)
        if (
            pixels.ndim != 3
            or pixels.shape[-1] != 4
            or not all(pixels.shape[:2])
            or not np.isfinite(pixels).all()
            or np.any(pixels[..., [1, 3]] <= pixels[..., [0, 2]])
        ):
            raise ValueError("detector pixels must be finite ordered (rows,columns,4) rectangles")
        if self.pitch_m is not None:
            pitch = np.asarray(self.pitch_m, dtype=np.float64)
            if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
                raise ValueError("detector pitch must be a finite positive (x,y) pair or None")
            # Subtraction of focal-plane coordinates can lose low-order bits.
            tolerance = 16 * np.finfo(np.float64).eps * np.max(np.abs(pixels))
            sizes = pixels[..., [1, 3]] - pixels[..., [0, 2]]
            centres = pixels[..., [0, 2]] / 2 + pixels[..., [1, 3]] / 2
            if np.any(sizes > pitch + tolerance):
                raise ValueError("active dimensions cannot exceed pixel pitch")
            for axis, step in ((1, (pitch[0], 0)), (0, (0, pitch[1]))):
                if not np.allclose(np.diff(centres, axis=axis), step, rtol=1e-12, atol=tolerance):
                    raise ValueError("pixel centre spacing must agree with declared pitch")
            object.__setattr__(self, "pitch_m", (float(pitch[0]), float(pitch[1])))
        pixels.flags.writeable = False
        object.__setattr__(self, "pixel_bounds_m", pixels)
        for name in DETECTOR_MAP_NAMES:
            value = np.copy(getattr(self, name))
            if value.ndim not in (0, 2) or not value.size:
                raise ValueError("detector properties must be scalars or nonempty 2-D pixel maps")
            if value.ndim == 2 and value.shape != self.shape:
                raise ValueError(f"{name} map must match (rows,columns) exactly")
            if name == "quantum_efficiency":
                if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                    raise ValueError("QE must be finite and in [0, 1]")
            elif name == "dark_electrons_per_s":
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("dark current must be finite and in [0, inf]")
            elif name == "read_noise_electrons":
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("read noise must be finite and in [0, inf]")
            elif name == "full_well_electrons":
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("full well must be finite and positive")
            elif name == "electrons_per_dn":
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("electrons per DN must be finite and positive")
            elif name == "nonlinearity":
                if not np.isfinite(value).all() or np.any((value < -0.5) | (value > 0.5)):
                    raise ValueError("nonlinearity must be finite and in [-0.5, 0.5]")
            elif not np.isfinite(value).all():
                raise ValueError("bias must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not 1 <= self.adc_bits <= 32:
            raise ValueError("ADC width must be in [1,32]")


def detector_operating_points(
    detector: DetectorProperties, *, rows: NDArray[np.int64], columns: NDArray[np.int64]
) -> DetectorOperatingPoints:
    """Select owned electronic parameters in supplied physical read order.

    QE remains upstream in photon conversion. Selection never consumes RNG.

    Args:
        detector: Physical or independently estimated detector electronic maps.
        rows: Nonempty int64 (reads,) native row indices, including repeats.
        columns: Matching int64 native column indices, including repeats.

    Returns:
        Independent per-read electronic maps and shared ADC width.

    Raises:
        ValueError: Indices have invalid shape, dtype or detector support.
    """
    if (
        rows.ndim != 1
        or not len(rows)
        or columns.shape != rows.shape
        or rows.dtype != np.int64
        or columns.dtype != np.int64
        or np.any(rows < 0)
        or np.any(rows >= detector.shape[0])
        or np.any(columns < 0)
        or np.any(columns >= detector.shape[1])
    ):
        raise ValueError("detector indices must be matching nonempty in-range int64 vectors")
    return DetectorOperatingPoints(
        **{
            name: np.broadcast_to(getattr(detector, name), detector.shape)[rows, columns]
            for name in DETECTOR_MAP_NAMES
            if name != "quantum_efficiency"
        },
        adc_bits=detector.adc_bits,
    )


def regular_pixel_bounds(
    shape: tuple[int, int],
    *,
    pitch_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    active_size_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer]
    | None = None,
    centre_xy_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer] = (0, 0),
) -> NDArray[np.float64]:
    """Fresh writable (rows,columns,4) bounds in xmin,xmax,ymin,ymax order.

    Pitch and active size are scalars or (x,y) pairs, not (row,column) pairs.
    Columns increase x; rows increase y. centre_xy_m is the centre of the whole
    array. Active dimensions default to pitch and cannot exceed it. Gaps are lost
    collecting area, not a second fill-factor multiplier. For irregular instruments,
    supply explicit bounds to DetectorProperties(pixel_bounds_m=..., pitch_m=None,
    ...) before instrument construction. Low-level acquisition kernels can also
    receive explicit bounds.

    Args:
        shape: Positive (rows,columns) detector dimensions.
        pitch_m: Scalar or (x,y) pixel-centre spacing in metres.
        active_size_m: Optional scalar or (x,y) active rectangle dimensions;
            defaults to pitch when omitted.
        centre_xy_m: (x,y) centre of the entire detector array in metres.

    Returns:
        Fresh writable float64 (rows,columns,4) active pixel bounds in metres.

    Raises:
        ValueError: Dimensions, spacing, active size or centre are invalid.
    """
    rows, columns = shape
    if rows < 1 or columns < 1:
        raise ValueError("focal-plane dimensions must be positive")
    pitch = np.asarray(pitch_m, dtype=np.float64)
    if not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pixel pitch must be finite and positive")
    active = pitch if active_size_m is None else np.asarray(active_size_m, dtype=np.float64)
    if active_size_m is not None and (not np.isfinite(active).all() or np.any(active <= 0)):
        raise ValueError("active size must be finite and positive")
    centre = np.copy(centre_xy_m)
    if not np.isfinite(centre).all():
        raise ValueError("pixel centre must be finite")
    if pitch.shape not in ((), (2,)) or active.shape not in ((), (2,)) or centre.shape != (2,):
        raise ValueError("pitch/active size must be scalar or (x,y); centre must be (x,y)")
    pitch = np.broadcast_to(pitch, (2,))
    active = np.broadcast_to(active, (2,))
    if np.any(active > pitch):
        raise ValueError("active dimensions cannot exceed pixel pitch")
    x = centre[0] + (np.arange(columns) - (columns - 1) / 2) * pitch[0]
    y = centre[1] + (np.arange(rows) - (rows - 1) / 2) * pitch[1]
    result = np.empty((rows, columns, 4))
    result[..., 0] = x - active[0] / 2
    result[..., 1] = x + active[0] / 2
    result[..., 2] = y[:, None] - active[1] / 2
    result[..., 3] = y[:, None] + active[1] / 2
    return result


def regular_detector(
    shape: tuple[int, int],
    *,
    pitch_m: float | np.floating | np.integer | Sequence[float] | NDArray[np.floating | np.integer],
    active_size_m: float
    | np.floating
    | np.integer
    | Sequence[float]
    | NDArray[np.floating | np.integer],
    centre_xy_m: Sequence[float] | NDArray[np.floating | np.integer],
    quantum_efficiency: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    dark_electrons_per_s: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    read_noise_electrons: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    full_well_electrons: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    electrons_per_dn: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    bias_dn: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    nonlinearity: float
    | np.floating
    | np.integer
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer],
    adc_bits: int,
) -> DetectorProperties:
    """Build a regular detector with explicit geometry and photoelectric properties.

    All arguments are required. Shape uses (rows,columns); physical dimensions
    use scalar or (x,y) values. Gaps reduce active area directly. Geometry and
    maps become independent read-only snapshots in DetectorProperties.

    Args:
        shape: Positive (rows,columns) detector dimensions.
        pitch_m: Positive scalar or (x,y) pixel-centre spacing in metres.
        active_size_m: Positive scalar or (x,y) active sizes, no larger than pitch.
        centre_xy_m: (x,y) centre of the whole detector in metres.
        quantum_efficiency: Scalar or exact detector-shaped photoelectron fraction.
        dark_electrons_per_s: Scalar or map of nonnegative dark-current rates.
        read_noise_electrons: Scalar or map of nonnegative Gaussian noise scales.
        full_well_electrons: Scalar or map of positive charge caps in electrons.
        electrons_per_dn: Scalar or map of positive electron-to-DN gains.
        bias_dn: Scalar or map of finite digital offsets in DN.
        nonlinearity: Scalar or map of dimensionless charge curvature in [-0.5,0.5].
        adc_bits: Shared ADC width between one and 32 bits.

    Returns:
        Detector with derived shape, retained (x,y) pitch and read-only arrays.

    Raises:
        ValueError: Geometry or physical properties violate detector support.
    """
    pixels = regular_pixel_bounds(
        shape, pitch_m=pitch_m, active_size_m=active_size_m, centre_xy_m=centre_xy_m
    )
    pitch = np.broadcast_to(np.asarray(pitch_m, dtype=np.float64), (2,))
    return DetectorProperties(
        pixel_bounds_m=pixels,
        pitch_m=(float(pitch[0]), float(pitch[1])),
        quantum_efficiency=quantum_efficiency,
        dark_electrons_per_s=dark_electrons_per_s,
        read_noise_electrons=read_noise_electrons,
        full_well_electrons=full_well_electrons,
        electrons_per_dn=electrons_per_dn,
        bias_dn=bias_dn,
        nonlinearity=nonlinearity,
        adc_bits=adc_bits,
    )


def pixel_quantum_efficiency(
    reads: ReadSchedule, efficiencies: Mapping[str, float]
) -> NDArray[np.float64]:
    """Persistent pixel QE map as (reads,1), for ``expected_electrons`` QE input.

    Every read pixel must be supplied; fixed-pattern QE is applied *before*
    photon noise. Spectrally varying QE may instead use the integration API's
    explicit (reads,spectral) input. Never reapply QE at readout.

    Args:
        reads: Ordered physical reads providing pixel identities.
        efficiencies: Exact mapping of every pixel identity to a fraction in [0,1].

    Returns:
        Fresh float64 (reads,1) QE array in physical read order.

    Raises:
        ValueError: Pixel coverage or any efficiency fraction is invalid.
    """
    if set(efficiencies) != set(reads.pixel_ids):
        raise ValueError("QE map must cover exactly the physical pixel identities")
    values = np.array([efficiencies[p] for p in reads.pixel_ids], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("pixel QE must be finite and in [0, 1]")
    return values[:, None]


def electron_moments(
    signal: ExpectedElectrons,
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unclipped analogue-electron mean/variance, including dark Poisson/read Gaussian.

    These describe the hypothetical linear pre-clipping model only; they are not
    DN moments and are not claimed valid after full-well/ADC clipping or rounding.
    Independent reads conditional on the supplied scene and electronics.

    Args:
        signal: Source-resolved mean photoelectrons and exposure durations.
        electronics: Dark-current rates and independent read-noise scales.

    Returns:
        Fresh (reads,) mean and variance arrays in electrons and electrons squared.

    Raises:
        ValueError: Operating points do not align with reads or moments overflow.
    """
    _check_points(signal.reads, electronics)
    duration = np.diff(signal.reads.exposure_bounds_s, axis=1).ravel()
    mean = signal.total + electronics.dark_electrons_per_s * duration
    variance = mean + electronics.read_noise_electrons**2
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("detector moments overflow")
    return mean, variance


def digitize(
    reads: ReadSchedule,
    collected_electrons: NDArray[np.float64],
    read_noise_electrons: NDArray[np.float64],
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> DetectorReadings:
    """Deterministically apply the physical read chain to supplied charge/noise arrays.

    Both inputs must be (realisations,reads); charge is nonnegative. This operation
    on expected charge gives a noiseless transfer, NOT the mean of clipped DN.
    No input mutation; output arrays own immutable storage.

    Args:
        reads: Physical read identities and exposure support.
        collected_electrons: (realisations,reads) nonnegative photo/dark charge.
        read_noise_electrons: Matching signed additive read-noise realizations.
        electronics: Full-well, gain, bias, nonlinear and ADC transfer parameters.

    Returns:
        Owned integer DN readings and separate full-well/ADC clipping flags.

    Raises:
        ValueError: Charge/noise support, operating points or analogue DN is invalid.
    """
    _check_points(reads, electronics)
    charge = collected_electrons
    if not np.isfinite(charge).all() or np.any(charge < 0):
        raise ValueError("collected charge must be finite and in [0, inf]")
    noise = read_noise_electrons
    if (
        charge.ndim != 2
        or charge.shape[0] < 1
        or charge.shape[1] != len(reads.sample_ids)
        or noise.shape != charge.shape
        or not np.all(np.isfinite(noise))
    ):
        raise ValueError("charge/noise must be finite matching (realisations,reads) arrays")
    saturated = charge >= electronics.full_well_electrons
    analogue = np.minimum(charge, electronics.full_well_electrons)
    if isinstance(electronics, DetectorOperatingPoints):
        analogue = analogue * (
            1 + electronics.nonlinearity * analogue / electronics.full_well_electrons
        )
    analogue = analogue + noise
    analogue = analogue / electronics.electrons_per_dn + electronics.bias_dn
    if not np.all(np.isfinite(analogue)):
        raise ValueError("analogue DN overflow")
    maximum = 2**electronics.adc_bits - 1
    clipped = (analogue < 0) | (analogue > maximum)
    dn = np.rint(np.clip(analogue, 0, maximum)).astype(np.int64)
    return DetectorReadings(reads, dn, saturated, clipped)


def correct_digital_readings(
    readings: DetectorReadings,
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> NDArray[np.float64]:
    """Convert unsaturated native DN to pedestal-subtracted electrons.

    Return a fresh float64 (realisations,reads) array using each physical
    exposure duration: electrons_per_dn * (DN - bias_dn) - dark_rate * duration.
    The quantization and read/shot-noise errors remain; this is neither an
    inverse of clipping nor a correction for nonlinear per-read operating points.
    Input arrays are borrowed. Reject flagged reads and nonzero nonlinearity;
    the caller must retain the flags, noise covariance and calibration lineage.

    Args:
        readings: Native DN and saturation flags with read exposure bounds.
        electronics: Linear gain, pedestal and dark-current calibration parameters.

    Returns:
        Fresh float64 (realisations,reads) pedestal/dark-subtracted electron values.

    Raises:
        ValueError: Any read is flagged, electronics are nonlinear or shapes disagree.
    """
    if readings.full_well_saturated.any() or readings.adc_clipped.any():
        raise ValueError("clipped detector reads cannot be linearly corrected")
    _check_points(readings.reads, electronics)
    if isinstance(electronics, DetectorOperatingPoints) and np.any(electronics.nonlinearity != 0):
        raise ValueError("nonlinear detector reads need their own calibration")
    duration = np.diff(readings.reads.exposure_bounds_s, axis=1).ravel()
    return (
        electronics.electrons_per_dn * (readings.dn.astype(np.float64) - electronics.bias_dn)
        - electronics.dark_electrons_per_s * duration
    )


def sample_readout(
    signal: ExpectedElectrons,
    electronics: DetectorElectronics | DetectorOperatingPoints,
    rng: np.random.Generator,
    *,
    realisations: int = 1,
    row_ids: Sequence[str] | None = None,
    column_ids: Sequence[str] | None = None,
    exposure_ids: Sequence[str] | None = None,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
    extra_read_noise_electrons: NDArray[np.float64] | None = None,
) -> DetectorReadings:
    """Draw conditional Poisson charge and independent/group-common Gaussian noise.

    The explicit caller-owned RNG is advanced; do not share it concurrently.
    NumPy's Poisson count range is checked by its generator; no Gaussian fallback.
    Summing source means before Poisson preserves the conditional law. Each
    supplied row/column/exposure group has one common zero-mean Gaussian draw
    per realisation, renewed between realisations; IDs control correlation, not
    adjacency or ordering. Zero-amplitude components consume no extra RNG draws.
    Persistent pixel response variation comes from supplied per-read points/QE.
    The RNG stream depends on read ordering and active groups; no persistence.
    Optional extra_read_noise_electrons is an explicit finite (realisations,reads)
    additive read-noise realization, borrowed without mutation. This permits a
    caller to add an explicitly generated correlated noise realization. It is
    applied after full-well clipping, along with the other read noise.

    Args:
        signal: Per-read expected photoelectrons and physical exposure durations.
        electronics: Dark rate, read-noise scale, gain and clipping configuration.
        rng: Explicit generator advanced for charge and Gaussian noise draws.
        realisations: Positive number of independent readout realizations.
        row_ids: Optional per-read group names for shared row noise.
        column_ids: Optional per-read group names for shared column noise.
        exposure_ids: Optional per-read group names for shared exposure noise.
        row_noise_electrons: Shared row-noise standard deviation in electrons.
        column_noise_electrons: Shared column-noise standard deviation in electrons.
        exposure_noise_electrons: Shared exposure-noise standard deviation in electrons.
        extra_read_noise_electrons: Optional (realisations,reads) additive samples.

    Returns:
        Digitised readings and clipping flags in (realisations,reads) arrays.

    Raises:
        ValueError: Counts, noise scales or group identities are invalid.
    """
    if realisations < 1:
        raise ValueError("realisations must be positive")
    mean, _ = electron_moments(signal, electronics)
    shape = (realisations, len(mean))
    if extra_read_noise_electrons is not None and (
        extra_read_noise_electrons.shape != shape
        or not np.all(np.isfinite(extra_read_noise_electrons))
    ):
        raise ValueError("extra read noise must be finite (realisations,reads)")
    charge = rng.poisson(mean, size=shape).astype(np.float64)
    noise = rng.normal(0, electronics.read_noise_electrons, size=shape)
    if extra_read_noise_electrons is not None:
        noise += extra_read_noise_electrons
    for ids, amplitude, name in (
        (row_ids, row_noise_electrons, "row"),
        (column_ids, column_noise_electrons, "column"),
        (exposure_ids, exposure_noise_electrons, "exposure"),
    ):
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError(f"{name} common noise must be finite and in [0, inf]")
        if amplitude:
            groups, indices = _group_indices(ids, len(mean), name)
            noise += rng.normal(0, amplitude, size=(realisations, len(groups)))[:, indices]
        elif ids is not None:
            _group_indices(ids, len(mean), name)
    return digitize(signal.reads, charge, noise, electronics)


def electron_covariance(
    signal: ExpectedElectrons,
    electronics: DetectorElectronics | DetectorOperatingPoints,
    *,
    row_ids: Sequence[str] | None = None,
    column_ids: Sequence[str] | None = None,
    exposure_ids: Sequence[str] | None = None,
    row_noise_electrons: float = 0,
    column_noise_electrons: float = 0,
    exposure_noise_electrons: float = 0,
) -> NDArray[np.float64]:
    """Unclipped charge-plus-read covariance (reads,reads), in electrons squared.

    No nonlinear/full-well/ADC/quantised moments are implied. A Gaussian common
    component contributes sigma² where physical reads share its group identity.

    Args:
        signal: Mean photoelectrons and physical exposure durations per read.
        electronics: Dark rate and independent Gaussian read-noise scales.
        row_ids: Optional grouping identity per read for shared row noise.
        column_ids: Optional grouping identity per read for shared column noise.
        exposure_ids: Optional grouping identity per read for shared exposure noise.
        row_noise_electrons: Shared row-noise standard deviation in electrons.
        column_noise_electrons: Shared column-noise standard deviation in electrons.
        exposure_noise_electrons: Shared exposure-noise standard deviation in electrons.

    Returns:
        Fresh float64 (reads,reads) conditional covariance in electrons squared.

    Raises:
        ValueError: Noise scales, required group identities or moments are invalid.
    """
    _, variance = electron_moments(signal, electronics)
    result = np.diag(variance)
    for ids, amplitude, name in (
        (row_ids, row_noise_electrons, "row"),
        (column_ids, column_noise_electrons, "column"),
        (exposure_ids, exposure_noise_electrons, "exposure"),
    ):
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError(f"{name} common noise must be finite and in [0, inf]")
        if ids is not None:
            _, indices = _group_indices(ids, len(variance), name)
            result += amplitude**2 * (indices[:, None] == indices[None, :])
        elif amplitude:
            raise ValueError(f"{name} noise requires group identities")
    return result


def check_detector_maps(
    detector: DetectorProperties, shape: tuple[int, int], *, include_qe: bool = False
) -> None:
    """Check scalar/exact (rows,columns) map support without copying or applying it.

    Detector geometry must match the supplied raster shape. Property maps are
    validated against that geometry at construction. QE is checked here only
    when include_qe=True; a supplied spectral QE callback
    replaces that map. Electronics maps are always checked. No broadcasting
    across a single image axis is accepted.

    Args:
        detector: Scalar or spatial detector-electronics and QE maps to check.
        shape: Required (rows,columns) map shape in detector pixel order.
        include_qe: Check QE map dimensions when a spectral callback does not replace it.

    Raises:
        ValueError: Detector geometry or a required map differs from the raster shape.
    """
    if detector.shape != shape:
        raise ValueError("detector shape must match (rows,columns) exactly")
    for name in DETECTOR_MAP_NAMES:
        if name == "quantum_efficiency" and not include_qe:
            continue
        value = np.asarray(getattr(detector, name))
        if value.ndim == 2 and value.shape != shape:
            raise ValueError(f"{name} map must match (rows,columns) exactly")


def _check_points(
    reads: ReadSchedule,
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> None:
    if isinstance(electronics, DetectorOperatingPoints) and len(
        electronics.dark_electrons_per_s
    ) != len(reads.sample_ids):
        raise ValueError("detector operating points must match physical reads")


def _group_indices(
    ids: Sequence[str] | None, n: int, name: str
) -> tuple[dict[str, int], NDArray[np.intp]]:
    if ids is None or len(ids) != n or any(not identity for identity in ids):
        raise ValueError(f"{name} noise requires one nonempty group ID per physical read")
    groups: dict[str, int] = {}
    indices = np.empty(n, dtype=np.intp)
    for i, identity in enumerate(ids):
        indices[i] = groups.setdefault(identity, len(groups))
    return groups, indices
