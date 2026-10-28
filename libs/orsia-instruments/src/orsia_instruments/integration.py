"""Dense radiance integration into source-resolved expected photoelectrons."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid
from scipy.constants import c, h

from .acquisition import Acquisition
from .charge import ExpectedElectrons
from .line_of_sight import LineOfSightSamples
from .line_of_sight_radiance import (
    LineOfSightRadiance,
    RadianceCoverage,
    sample_line_of_sight_radiance,
)
from .optics.response import OpticalResponse
from .scene import RadianceField, sample_radiance

type QuantumEfficiency = Callable[[NDArray[np.int64], NDArray[np.float64]], NDArray[np.float64]]
type Efficiency = (
    float
    | np.floating
    | np.integer
    | Sequence[float]
    | Sequence[Sequence[float]]
    | NDArray[np.floating | np.integer]
)


def photoelectron_response(
    grid: SpectralGrid,
    acquisition: Acquisition,
    response: OpticalResponse,
    *,
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
) -> NDArray[np.float64]:
    """Calculate positive coefficients[read,node,wavenumber] for a complete grid.

    Coefficients multiply per-cm-1 spectral radiance to give photoelectrons and
    include trapezoid weights, 1/(h*c*100*wavenumber), projected collecting area,
    throughput, QE, optical accepted fraction and Acquisition's dOmega dt once.
    The grid must use increasing positive cm-1 points in one connected interval.
    Throughput/QE broadcast to (reads,spectral), with fractions in [0,1]. A pure
    QE callback receives all read indices and all wavelengths in metres and
    returns (reads,wavenumbers). Optical response retains its per-read callback
    contract and is called once per read with the complete wavelength grid.
    The result is fresh writable float64 storage; inputs are borrowed.
    Acquisition determines the angular-measure convention, independently of this
    spectral accounting. No scene, provider, detector or random state is retained.

    Args:
        grid: Connected, increasing positive wavenumber grid in cm^-1.
        acquisition: Read and quadrature-node geometry with solid-angle time weights.
        response: Borrowed callback yielding (nodes,wavenumbers) accepted fractions.
        collecting_area_m2: Projected collecting area in square metres.
        throughput: Physical fractions broadcastable to (reads, wavelengths).
        quantum_efficiency: Fractions broadcastable to (reads, wavelengths), or
            a callback accepting read indices and wavelengths in metres.
    Returns:
        Owned (read,node,wavenumber) coefficients in photoelectrons per unit
        per-cm^-1 radiance.

    Raises:
        ValueError: Grid, area, fractions or callback output violate physical support.
    """
    area = collecting_area_m2
    if not np.isfinite(area) or area < 0:
        raise ValueError("collecting area must be finite and in [0, inf]")
    points = grid.points
    if (
        grid.coordinate != "wavenumber_cm1"
        or grid.domain is None
        or grid.domain.intervals != ((points[0], points[-1]),)
    ):
        raise ValueError("photoelectron response requires one connected wavenumber_cm1 grid")
    if len(points) < 2 or np.any(points <= 0) or np.any(np.diff(points) <= 0):
        raise ValueError("wavenumbers must be positive and strictly increasing")
    n, nodes = acquisition.times_s.shape
    k = len(points)
    tau = _fractions(throughput, (n, k), "throughput")
    indices = np.arange(n, dtype=np.int64)
    if callable(quantum_efficiency):
        qe_values = quantum_efficiency(indices, 0.01 / points)
        if qe_values.shape != (n, k):
            raise ValueError("QE callback must return (reads,wavenumbers)")
    else:
        qe_values = quantum_efficiency
    qe = _fractions(qe_values, (n, k), "QE")
    ds = np.diff(points)
    weights = np.zeros(k)
    weights[:-1] += ds / 2
    weights[1:] += ds / 2
    photon_weights = weights / (h * c * 100 * points)
    accepted = np.empty((n, nodes, k))
    wavelengths = 0.01 / points
    for read in range(n):
        local = response(read, wavelengths, acquisition)
        if local.shape != (nodes, k):
            raise ValueError("response must have shape (nodes,wavenumbers)")
        if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
            raise ValueError("optical accepted fraction must be finite and in [0, 1]")
        accepted[read] = local
    return (
        accepted
        * acquisition.weights_sr_s[..., None]
        * (area * tau * qe * photon_weights)[:, None, :]
    )


def sampled_optical_response(
    acquisition: Acquisition,
    response: OpticalResponse,
    wavelengths_m: Sequence[float] | NDArray[np.floating | np.integer],
) -> OpticalResponse:
    """Snapshot accepted fractions for reuse with different radiance fields.

    Preparation invokes the pure response per read on the full grid exactly once.
    Storage is float64 (reads,nodes,wavelengths), independent of radiance, throughput,
    QE and detector physics. The returned callback requires the identical acquisition
    object and exact supplied wavelengths (in any order); it never interpolates.
    Reprepare after changing any optical parameter. No mutable cache, provider or
    original response is retained. Returned fractions are owned writable copies;
    callers must not mutate them while reusing the prepared response.
    This is an explicit memory-for-execution trade, not automatic physics deduplication.

    Args:
        acquisition: Original acquisition object retained for identity checks.
        response: Borrowed optical accepted-fraction callback to evaluate once.
        wavelengths_m: Distinct positive wavelengths in metres, in any order.

    Returns:
        Callback returning fresh (nodes, wavelengths) fractions for a read and
        exact prepared wavelengths from the identical acquisition object.

    Raises:
        ValueError: If preparation inputs or sampled response fractions are invalid;
            the returned callback also rejects unknown wavelengths and acquisitions.
    """
    wave = np.copy(wavelengths_m)
    if not np.isfinite(wave).all():
        raise ValueError("wavelengths must be finite")
    if wave.ndim != 1 or not len(wave) or np.any(wave <= 0):
        raise ValueError("wavelengths must be a nonempty positive vector")
    order = np.argsort(wave)
    sorted_wave = np.copy(wave[order])
    if np.any(np.diff(sorted_wave) == 0):
        raise ValueError("prepared wavelengths must be distinct")
    n, nodes = acquisition.times_s.shape
    fractions = np.empty((n, nodes, len(wave)))
    for read in range(n):
        local = response(read, wave, acquisition)
        if local.shape != (nodes, len(wave)):
            raise ValueError("response must have shape (nodes,wavenumbers)")
        if not np.isfinite(local).all() or np.any((local < 0) | (local > 1)):
            raise ValueError("optical accepted fraction must be finite and in [0, 1]")
        fractions[read] = local
    retained = np.copy(fractions[:, :, order])

    def prepared(
        read: int, wavelengths: NDArray[np.float64], current: Acquisition
    ) -> NDArray[np.float64]:
        if current is not acquisition:
            raise ValueError("prepared response requires its original acquisition")
        if read < 0 or read >= n:
            raise ValueError("read outside prepared acquisition")
        if wavelengths.ndim != 1 or not np.all(np.isfinite(wavelengths)):
            raise ValueError("requested wavelengths must be a finite vector")
        indices = np.searchsorted(sorted_wave, wavelengths)
        if np.any(indices == len(sorted_wave)) or np.any(
            sorted_wave[np.minimum(indices, len(sorted_wave) - 1)] != wavelengths
        ):
            raise ValueError("wavelength outside prepared support")
        return np.copy(retained[read][:, indices])

    return prepared


def expected_electrons(
    field: RadianceField,
    acquisition: Acquisition,
    response: OpticalResponse,
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
        response: Accepted-fraction callback for each read on the full grid.
        collecting_area_m2: Projected collecting area in square metres.
        throughput: Physical fractions broadcastable to (reads, wavelengths).
        quantum_efficiency: Fractions broadcastable to (reads, wavelengths), or
            an all-read callback over wavelengths in metres.
        source_name: Label for this source's electron-count contribution.

    Returns:
        Expected electrons per read, attributed to source_name on the acquisition.

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
        acquisition,
        response,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    counts = np.einsum("nqk,nqk->n", radiance_all, coefficients)
    return ExpectedElectrons(acquisition, {source_name: counts})


def line_of_sight_expected_electrons(
    rays: LineOfSightSamples,
    grid: SpectralGrid,
    scene: LineOfSightRadiance,
    coverage: RadianceCoverage,
    response: OpticalResponse,
    *,
    collecting_area_m2: float,
    throughput: Efficiency = 1,
    quantum_efficiency: Efficiency | QuantumEfficiency = 1,
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Integrate Earth-frame sight-line radiance using the optical charge kernel.

    The scene and coverage callbacks use the Earth rays and their complete support.
    The optical callback uses rays.acquisition, where directions_uv are *sensor*
    direction cosines and image_xy_m are virtual forward image coordinates. The
    shared photon kernel applies projected pupil area, physical throughput,
    accepted fractions and QE once to dOmega dt and cm^-1 spectral weights.
    The result retains the acquisition's physical read identities for subsequent
    source combination and detector readout. Inputs are borrowed; the electron
    contribution is owned. This mean-charge calculation is deterministic;
    detector sampling uses an explicit caller-owned random generator.

    Args:
        rays: Earth-frame sight lines with sensor-frame optical acquisition.
        grid: Fully sampled connected positive wavenumber interval in cm^-1.
        scene: Pure ray/time/spectral radiance callback with complete-grid output.
        coverage: Full continuous support and spectral-interval assertion per read.
        response: Accepted optical intensity fractions (nodes,wavenumbers), in [0,1].
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
        rays.acquisition,
        response,
        collecting_area_m2=collecting_area_m2,
        throughput=throughput,
        quantum_efficiency=quantum_efficiency,
    )
    counts = np.einsum("nqk,nqk->n", radiance, coefficients)
    return ExpectedElectrons(rays.acquisition, {source_name: counts})


def _fractions(value: Efficiency, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return np.broadcast_to(array, shape)
