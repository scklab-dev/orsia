"""Independent conditional counting and memoryless physical readout, before processing."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from .acquisition import Acquisition, ReadSchedule, read_schedule
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
    """Persistent per-read detector parameters for a specified acquisition.

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

    Arrays have shape (realisations,reads), immutable owned storage. Acquisition
    identities and exposure support are retained. ADC flags mark analogue values
    outside its range before rounding; reaching full well is flagged at equality.
    Invalid physical support raises upstream rather than becoming saturation.

    Attributes:
        acquisition: Identity and time support of the simulated physical reads.
        dn: Owned immutable integer counts of shape (realisations,reads).
        full_well_saturated: Boolean flags for reads reaching the full-well cap.
        adc_clipped: Boolean flags for values outside the ADC range before rounding.
    """

    acquisition: Acquisition | ReadSchedule
    dn: NDArray[np.int64]
    full_well_saturated: NDArray[np.bool_]
    adc_clipped: NDArray[np.bool_]

    @property
    def reads(self) -> ReadSchedule:
        """Expose the readings' geometry-free physical read identities and timing.

        Returns:
            Existing physical schedule or a fresh timing snapshot from Acquisition.
        """
        return read_schedule(self.acquisition)

    def __post_init__(self) -> None:
        shape = self.dn.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != len(self.acquisition.sample_ids):
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
class DetectorMaps:
    """Fixed pixel QE and electronics, supplied as scalars or (rows,columns) maps.

    Arrays are independent writable copies. Exact map shape is checked against the
    focal plane on simulation, with no implicit row/column broadcasting. QE is
    wavelength independent and applied once, before photon noise. Electronics use
    electrons, electrons/s, electrons/DN and DN; nonlinearity is dimensionless.
    The existing DetectorOperatingPoints transfer defines clipping/nonlinearity.

    Attributes:
        quantum_efficiency: Scalar or (rows,columns) photoelectron fraction.
        dark_electrons_per_s: Scalar or map of dark-current rates.
        read_noise_electrons: Scalar or map of Gaussian electron noise scales.
        full_well_electrons: Scalar or map of charge caps in electrons.
        electrons_per_dn: Scalar or map of electron-to-DN conversion gains.
        bias_dn: Scalar or map of digital offsets in DN.
        nonlinearity: Scalar or map of dimensionless charge curvature.
        adc_bits: Shared ADC width between one and 32 bits.
    """

    quantum_efficiency: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 1
    dark_electrons_per_s: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 0
    read_noise_electrons: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 0
    full_well_electrons: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 1e6
    electrons_per_dn: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 1
    bias_dn: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 0
    nonlinearity: (
        float
        | np.floating
        | np.integer
        | Sequence[Sequence[float]]
        | NDArray[np.floating | np.integer]
    ) = 0
    adc_bits: int = 16

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "adc_bits":
                continue
            value = np.copy(getattr(self, field.name))
            if value.ndim not in (0, 2) or not value.size:
                raise ValueError("detector properties must be scalars or nonempty 2-D pixel maps")
            if field.name == "quantum_efficiency":
                if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                    raise ValueError("QE must be finite and in [0, 1]")
            elif field.name == "dark_electrons_per_s":
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("dark current must be finite and in [0, inf]")
            elif field.name == "read_noise_electrons":
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError("read noise must be finite and in [0, inf]")
            elif field.name == "full_well_electrons":
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("full well must be finite and positive")
            elif field.name == "electrons_per_dn":
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("electrons per DN must be finite and positive")
            elif field.name == "nonlinearity":
                if not np.isfinite(value).all() or np.any((value < -0.5) | (value > 0.5)):
                    raise ValueError("nonlinearity must be finite and in [-0.5, 0.5]")
            elif not np.isfinite(value).all():
                raise ValueError("bias must be finite")
            object.__setattr__(self, field.name, value)
        if not 1 <= self.adc_bits <= 32:
            raise ValueError("ADC width must be in [1,32]")


def pixel_quantum_efficiency(
    acquisition: Acquisition | ReadSchedule, efficiencies: Mapping[str, float]
) -> NDArray[np.float64]:
    """Persistent pixel QE map as (reads,1), for ``expected_electrons`` QE input.

    Every read pixel must be supplied; fixed-pattern QE is applied *before*
    photon noise. Spectrally varying QE may instead use the integration API's
    explicit (reads,spectral) input. Never reapply QE at readout.

    Args:
        acquisition: Ordered physical reads providing pixel identities.
        efficiencies: Exact mapping of every pixel identity to a fraction in [0,1].

    Returns:
        Fresh float64 (reads,1) QE array in physical read order.

    Raises:
        ValueError: Pixel coverage or any efficiency fraction is invalid.
    """
    if set(efficiencies) != set(acquisition.pixel_ids):
        raise ValueError("QE map must cover exactly the physical pixel identities")
    values = np.array([efficiencies[p] for p in acquisition.pixel_ids], dtype=np.float64)
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
    _check_points(signal.acquisition, electronics)
    duration = np.diff(signal.acquisition.exposure_bounds_s, axis=1).ravel()
    mean = signal.total + electronics.dark_electrons_per_s * duration
    variance = mean + electronics.read_noise_electrons**2
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("detector moments overflow")
    return mean, variance


def digitize(
    acquisition: Acquisition | ReadSchedule,
    collected_electrons: NDArray[np.float64],
    read_noise_electrons: NDArray[np.float64],
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> DetectorReadings:
    """Deterministically apply the physical read chain to supplied charge/noise arrays.

    Both inputs must be (realisations,reads); charge is nonnegative. This operation
    on expected charge gives a noiseless transfer, NOT the mean of clipped DN.
    No input mutation; output arrays own immutable storage.

    Args:
        acquisition: Physical read identities and exposure support.
        collected_electrons: (realisations,reads) nonnegative photo/dark charge.
        read_noise_electrons: Matching signed additive read-noise realizations.
        electronics: Full-well, gain, bias, nonlinear and ADC transfer parameters.

    Returns:
        Owned integer DN readings and separate full-well/ADC clipping flags.

    Raises:
        ValueError: Charge/noise support, operating points or analogue DN is invalid.
    """
    _check_points(acquisition, electronics)
    charge = collected_electrons
    if not np.isfinite(charge).all() or np.any(charge < 0):
        raise ValueError("collected charge must be finite and in [0, inf]")
    noise = read_noise_electrons
    if (
        charge.ndim != 2
        or charge.shape[0] < 1
        or charge.shape[1] != len(acquisition.sample_ids)
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
    return DetectorReadings(acquisition, dn, saturated, clipped)


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
    _check_points(readings.acquisition, electronics)
    if isinstance(electronics, DetectorOperatingPoints) and np.any(electronics.nonlinearity != 0):
        raise ValueError("nonlinear detector reads need their own calibration")
    duration = np.diff(readings.acquisition.exposure_bounds_s, axis=1).ravel()
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
    return digitize(signal.acquisition, charge, noise, electronics)


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
    detector: DetectorMaps, shape: tuple[int, int], *, include_qe: bool = False
) -> None:
    """Check scalar/exact (rows,columns) map support without copying or applying it.

    QE is checked only when include_qe=True; a supplied spectral QE callback
    replaces that map. Electronics maps are always checked. No broadcasting
    across a single image axis is accepted.

    Args:
        detector: Scalar or spatial detector-electronics and QE maps to check.
        shape: Required (rows,columns) map shape in detector pixel order.
        include_qe: Check QE map dimensions when a spectral callback does not replace it.

    Raises:
        ValueError: A required two-dimensional detector map differs from shape.
    """
    for name in DETECTOR_MAP_NAMES:
        if name == "quantum_efficiency" and not include_qe:
            continue
        value = np.asarray(getattr(detector, name))
        if value.ndim == 2 and value.shape != shape:
            raise ValueError(f"{name} map must match (rows,columns) exactly")


def _check_points(
    acquisition: Acquisition | ReadSchedule,
    electronics: DetectorElectronics | DetectorOperatingPoints,
) -> None:
    if isinstance(electronics, DetectorOperatingPoints) and len(
        electronics.dark_electrons_per_s
    ) != len(acquisition.sample_ids):
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
