"""Dense radiance integration into source-resolved expected photoelectrons."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid
from orsia_radiative.spectral.grid import spectral_trapezoid_weights
from scipy.constants import c, h

from orsia_instruments.acquisition.los import (
    LineOfSightRadiance,
    LineOfSightSamples,
    RadianceCoverage,
    sample_line_of_sight_radiance,
)
from orsia_instruments.acquisition.quadrature import Acquisition, read_schedule
from orsia_instruments.optics.response import OpticalResponse
from orsia_instruments.photodetection.charge import ExpectedElectrons
from orsia_instruments.sampling.angular import AngularRadianceSamples, sample_radiance

type QuantumEfficiency = Callable[[NDArray[np.int64], NDArray[np.float64]], NDArray[np.float64]]


type Efficiency = (
    float
    | np.floating
    | np.integer
    | Sequence[float]
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer]
)


def efficiency_samples(value: Efficiency, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    """Broadcast supplied physical fractions onto explicit read/spectral axes.

    Args:
        value: Scalar or array of finite accepted fractions in [0,1].
        shape: Requested (read,spectral) dimensions under NumPy broadcasting.
        name: Physical factor name used in invalid-domain diagnostics.

    Returns:
        Borrowed read-only float64 broadcast view, potentially aliasing caller storage.
        Keep supplied values stable until the consuming calculation completes.

    Raises:
        ValueError: Fractions or broadcasting are invalid.
    """
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return np.broadcast_to(array, shape)


@dataclass(frozen=True, eq=False)
class OpticalAcceptance:
    """Physical accepted fractions on one immutable acquisition and spectral support.

    This snapshot excludes scene radiance, QE, area, throughput and quadrature.
    Reprepare after changing optics or geometry. Callback freshness belongs to the
    caller; the record retains no callback or mutable cache.

    Attributes:
        acquisition: Shared immutable acquisition defining read/node correspondence.
        wavelengths_m: Owned immutable distinct positive wavelengths (wavelengths,), metres.
        fractions: Owned immutable finite (reads,nodes,wavelengths) fractions in [0,1].
    """

    acquisition: Acquisition
    wavelengths_m: NDArray[np.float64]
    fractions: NDArray[np.float64]

    def __post_init__(self) -> None:
        wave = np.array(self.wavelengths_m, dtype=np.float64, copy=True)
        fractions = np.array(self.fractions, dtype=np.float64, copy=True)
        if (
            wave.ndim != 1
            or not len(wave)
            or not np.isfinite(wave).all()
            or np.any(wave <= 0)
            or len(np.unique(wave)) != len(wave)
        ):
            raise ValueError("wavelengths must be distinct positive finite vector")
        if (
            fractions.shape != (*self.acquisition.times_s.shape, len(wave))
            or not np.isfinite(fractions).all()
            or np.any((fractions < 0) | (fractions > 1))
        ):
            raise ValueError("optical fractions must be finite (reads,nodes,wavelengths) in [0, 1]")
        for name, array in (("wavelengths_m", wave), ("fractions", fractions)):
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
            )


def sample_optical_acceptance(
    acquisition: Acquisition,
    response: OpticalResponse,
    wavelengths_m: Sequence[float] | NDArray[np.floating | np.integer],
) -> OpticalAcceptance:
    """Sample each optical callback once on full spectral support for explicit reuse.

    Sampling copies each callback result immediately, permitting reused scratch
    buffers. Retention is explicit; batch orchestrators consume and release these
    snapshots to bound memory. The acquisition is immutable and shared by identity.

    Args:
        acquisition: Immutable physical read and quadrature geometry.
        response: Pure callback returning finite (nodes,wavelengths) fractions per read.
        wavelengths_m: Distinct positive finite wavelengths in metres, in any order.

    Returns:
        Independent immutable accepted fractions, excluding radiance and electronics.

    Raises:
        ValueError: Wavelengths or callback shapes/physical fractions are invalid.
    """
    wave = np.array(wavelengths_m, dtype=np.float64, copy=True)
    if (
        wave.ndim != 1
        or not len(wave)
        or not np.isfinite(wave).all()
        or np.any(wave <= 0)
        or len(np.unique(wave)) != len(wave)
    ):
        raise ValueError("wavelengths must be distinct positive finite vector")
    fractions = np.empty((*acquisition.times_s.shape, len(wave)))
    for read in range(len(fractions)):
        local = response(read, wave, acquisition)
        if local.shape != fractions.shape[1:]:
            raise ValueError("response must have shape (nodes,wavenumbers)")
        if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
            raise ValueError("optical accepted fraction must be finite and in [0, 1]")
        fractions[read] = local
    return OpticalAcceptance(acquisition, wave, fractions)


def photoelectron_response(
    grid: SpectralGrid,
    acceptance: OpticalAcceptance,
    *,
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
) -> NDArray[np.float64]:
    """Build photoelectron coefficients from explicitly sampled optical fractions.

    The coefficients multiply spectral radiance in the declared coordinate and apply trapezoid
    weights, inverse photon energy, projected area, throughput, QE and dOmega dt once.
    Only exact prepared wavelengths are selected, including subsets/permutations;
    no optical callback is evaluated or retained. Inputs remain unchanged.

    Args:
        grid: Positive nm or cm-1 samples with complete endpoint-sampled coverage intervals.
        acceptance: Immutable physical fractions and their original acquisition/spectral support.
        collecting_area_m2: Finite nonnegative projected collecting area in square metres.
        throughput: Finite physical fractions broadcastable to (reads,wavelengths).
        quantum_efficiency: Physical fractions broadcastable to (reads,wavelengths), or
            a pure callback (read_indices,wavelengths_m) returning that exact shape.

    Returns:
        Fresh writable float64 (reads,nodes,spectral) coefficients in electrons
        per unit spectral radiance; the acquisition defines angular-measure convention.

    Raises:
        ValueError: Grid, exact support, area, fractions or QE callback shape is invalid.
    """
    weights = photon_conversion_weights(grid)
    wavelengths = 0.01 / grid.points if grid.coordinate == "wavenumber_cm1" else grid.points * 1e-9
    order = np.argsort(acceptance.wavelengths_m)
    support = acceptance.wavelengths_m[order]
    indices = np.searchsorted(support, wavelengths)
    if np.any(indices == len(support)) or np.any(
        support[np.minimum(indices, len(support) - 1)] != wavelengths
    ):
        raise ValueError("wavelength outside exact optical acceptance support")
    aligned = OpticalAcceptance(
        acceptance.acquisition, wavelengths, acceptance.fractions[:, :, order[indices]]
    )
    return weighted_photoelectron_response(
        aligned,
        weights,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )


def photon_conversion_weights(grid: SpectralGrid) -> NDArray[np.float64]:
    """Prepare full-domain spectral trapezoid weights divided by photon energy.

    Args:
        grid: Positive nm or cm-1 samples with explicit coverage. Every interval
            requires at least two points, including both declared endpoints.
            Disconnected intervals are integrated separately without bridging gaps.

    Returns:
        Fresh (spectral,) coordinate-measure/J weights. Compute once before
        partitioning a spectral integral; slices retain the complete-domain rule.

    Raises:
        ValueError: Coverage, endpoint sampling or finite conversion is invalid.
    """
    points = grid.points
    weights = spectral_trapezoid_weights(grid)
    energy = (
        h * c * 100 * points if grid.coordinate == "wavenumber_cm1" else h * c / (points * 1e-9)
    )
    weights /= energy
    if not np.isfinite(weights).all():
        raise ValueError("photon conversion exceeds finite support")
    return weights


def weighted_photoelectron_response(
    acceptance: OpticalAcceptance,
    photon_weights: NDArray[np.float64],
    *,
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
) -> NDArray[np.float64]:
    """Apply supplied spectral photon weights and physical acceptance factors once.

    Args:
        acceptance: Immutable read/node fractions in the supplied wavelength order.
        photon_weights: Finite nonnegative (spectral,) coordinate-measure/J factors,
            aligned with acceptance wavelengths. Use full-domain photon_conversion_weights
            or its explicit slice; the caller owns the chosen quadrature measure.
        collecting_area_m2: Finite nonnegative projected collecting area, m2.
        throughput: Physical fractions broadcastable to (reads,spectral).
        quantum_efficiency: Physical fractions with that shape, or a callback receiving
            local read indices and acceptance wavelengths in metres.

    Returns:
        Fresh (reads,nodes,spectral) electron coefficients per declared radiance-density
        unit. Time/angular quadrature, area, throughput and QE enter exactly once.
        A one-point slice is valid when its weight came from the full-domain rule.

    Raises:
        ValueError: Weights, area, fractions or QE callback shapes are invalid.
    """
    area = collecting_area_m2
    if not np.isfinite(area) or area < 0:
        raise ValueError("collecting area must be finite and in [0, inf]")
    n, _ = acceptance.acquisition.times_s.shape
    k = acceptance.wavelengths_m.size
    if (
        photon_weights.shape != (k,)
        or not np.isfinite(photon_weights).all()
        or np.any(photon_weights < 0)
    ):
        raise ValueError("photon weights must be finite nonnegative on acceptance samples")
    tau = efficiency_samples(throughput, (n, k), "throughput")
    if callable(quantum_efficiency):
        qe_values = quantum_efficiency(
            np.arange(n, dtype=np.int64), acceptance.wavelengths_m.copy()
        )
        if qe_values.shape != (n, k):
            raise ValueError("QE callback must return (reads,wavenumbers)")
    else:
        qe_values = quantum_efficiency
    qe = efficiency_samples(qe_values, (n, k), "QE")
    # Keep the full-grid path's spectral-major storage for consistent contraction
    # rounding; a spectral block uses the same layout and arithmetic ordering.
    accepted = np.moveaxis(acceptance.fractions, -1, 0).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = (
            accepted
            * acceptance.acquisition.weights_sr_s[None, ...]
            * (area * tau * qe * photon_weights).T[:, :, None]
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("photoelectron response arithmetic overflow")
    return np.moveaxis(coefficients, 0, -1)


def expected_electrons(
    field: AngularRadianceSamples,
    acquisition: Acquisition,
    response: OpticalResponse | OpticalAcceptance,
    *,
    collecting_area_m2: float,
    throughput: float
    | np.floating
    | np.integer
    | Sequence[float]
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer] = 1,
    quantum_efficiency: float
    | np.floating
    | np.integer
    | Sequence[float]
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer]
    | QuantumEfficiency = 1,
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Integrate one source path into electrons, without hidden extrapolation or RNG.

    response(read,wavelengths_m,acquisition) returns dimensionless accepted
    fractions (nodes,wavenumbers). Throughput and QE are scalars or (reads,spectral)
    broadcastable arrays in [0,1]; area is already projected. Trapezoidal cm-1
    weights multiply per-cm-1 radiance and 1/(h*c*100*wavenumber). Pixel acceptance
    and dOmega dt enter once. No extra exposure/pixel-area multiplier is applied.
    Interpolated radiance and response weights cover all reads, nodes and wavelengths.
    The supplied response must be pure; inputs are borrowed.
    Alternatively QE(read_indices,wavelengths_m) returns exactly (reads,wavenumbers)
    finite fractions in [0,1] for all local reads and full spectral support. Wavelengths
    need not increase. Callbacks are borrowed, pure, nonmutating and responsible
    for rejecting unsupported wavelengths; no interpolation/extrapolation is added.

    Args:
        field: Scene radiance field with temporal, angular and spectral support.
        acquisition: Read and quadrature-node geometry within the field coverage.
        response: Accepted-fraction callback for each read on the full grid, or
            OpticalAcceptance retaining this acquisition and exact wavelength support.
        collecting_area_m2: Projected collecting area in square metres.
        throughput: Physical fractions broadcastable to (reads, wavelengths).
        quantum_efficiency: Fractions broadcastable to (reads, wavelengths), or
            an all-read callback over wavelengths in metres.
        source_name: Label for this source's electron-count contribution.

    Returns:
        Source-labelled electrons retaining the acquisition's immutable physical read schedule.

    Raises:
        ValueError: If scene coverage, spectral support or physical response
            coefficients have invalid support or callback output shapes.
    """
    exposures = acquisition.exposure_bounds_s
    directions = acquisition.direction_bounds_uv
    if (
        np.any(exposures[:, 0] < field.times_s[0])
        or np.any(exposures[:, 1] > field.times_s[-1])
        or np.any(directions[:, 0] < field.u[0])
        or np.any(directions[:, 1] > field.u[-1])
        or np.any(directions[:, 2] < field.v[0])
        or np.any(directions[:, 3] > field.v[-1])
    ):
        raise ValueError("entire acquisition support must lie within scene coverage")
    n, k = len(acquisition.sample_ids), len(field.grid.points)
    nodes = acquisition.times_s.shape[1]
    times = np.asarray(acquisition.times_s, dtype=np.float64).ravel()
    directions_uv = np.asarray(acquisition.directions_uv, dtype=np.float64).reshape(-1, 2)
    radiance_all = sample_radiance(field, times, directions_uv, slice(None)).reshape(n, nodes, k)
    coefficients = photoelectron_response(
        field.grid,
        _acceptance(acquisition, response, 0.01 / field.grid.points),
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    counts = np.einsum("nqk,nqk->n", radiance_all, coefficients)
    return ExpectedElectrons(read_schedule(acquisition), {source_name: counts})


def line_of_sight_expected_electrons(
    rays: LineOfSightSamples,
    grid: SpectralGrid,
    scene: LineOfSightRadiance,
    coverage: RadianceCoverage,
    response: OpticalResponse | OpticalAcceptance,
    *,
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Integrate GCRS sight-line radiance using the optical charge kernel.

    The scene and coverage callbacks use GCRS rays and their complete support.
    The optical callback uses rays.acquisition, where directions_uv are *sensor*
    direction cosines and image_xy_m are virtual forward image coordinates. The
    shared photon kernel applies projected pupil area, physical throughput,
    accepted fractions and QE once to dOmega dt and cm^-1 spectral weights.
    The result retains the acquisition's physical read identities for subsequent
    source combination and detector readout. Inputs are borrowed; the electron
    contribution is owned. This mean-charge calculation is deterministic;
    detector sampling uses an explicit caller-owned random generator.

    Args:
        rays: GCRS sight lines with sensor-frame optical acquisition.
        grid: Fully sampled connected positive wavenumber interval in cm^-1.
        scene: Pure ray/time/spectral radiance callback with complete-grid output.
        coverage: Full continuous support and spectral-interval assertion per read.
        response: Callback returning (nodes,wavenumbers) accepted fractions in [0,1], or
            OpticalAcceptance retaining rays.acquisition and exact wavelength support.
        collecting_area_m2: Projected pupil area in square metres.
        throughput: Physical fractional throughput broadcastable to (reads,spectral).
        quantum_efficiency: Fractional QE or all-read spectral QE callback.
        source_name: Nonempty name for the independent optical source contribution.

    Returns:
        Named mean photoelectrons per physical read, before detector noise.

    Raises:
        ValueError: Coverage, scene, optical or spectral inputs fail validation.
    """
    radiance = sample_line_of_sight_radiance(rays, grid, scene, coverage)
    coefficients = photoelectron_response(
        grid,
        _acceptance(rays.acquisition, response, 0.01 / grid.points),
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    counts = np.einsum("nqk,nqk->n", radiance, coefficients)
    return ExpectedElectrons(read_schedule(rays.acquisition), {source_name: counts})


def _acceptance(
    acquisition: Acquisition,
    response: OpticalResponse | OpticalAcceptance,
    wave: NDArray[np.float64],
) -> OpticalAcceptance:
    if isinstance(response, OpticalAcceptance):
        if response.acquisition is not acquisition:
            raise ValueError("optical acceptance requires its original acquisition")
        return response
    return sample_optical_acceptance(acquisition, response, wave)
