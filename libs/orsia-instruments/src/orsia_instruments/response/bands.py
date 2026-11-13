"""Retain native spectral sensitivity and normalized spatial support for calibration."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.spectral.bands import SpectralBand
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid

from orsia_instruments.acquisition.schedule import RasterReadSchedule, ReadSchedule


@dataclass(frozen=True, eq=False)
class NativeBandResponse:
    """Physical per-read sensitivity and sampled flat-spectrum spatial response.

    Physical read coefficients include photon conversion, quadrature, area,
    throughput, QE and time/angle integration once. Shared table rows retain
    spectral shape; optional read sensitivity sets their absolute normalization.
    Each read produces the dot product of its physical coefficients and incident
    spectral radiance. Normalization recovers the achieved photon-weighted mean.
    Spatial weights marginalize the same response under a
    spectrally flat unit-radiance assumption. Coefficient and node arrays are
    independent writable copies; canonical spectral coordinates remain immutable.
    These two marginals retain achieved bands and flat-spectrum registration
    weights. They do not retain the joint spectral/spatial kernel for inversion;
    heterogeneous spectra can violate a separable reconstruction interpretation.

    Attributes:
        schedule: Shared immutable physical reads and original native raster correspondence.
        reads: Derived identity and exposure schedule from the raster correspondence.
        grid: Shared canonical cm^-1 grid with immutable spectral coordinates.
        read_bands: Physical filter or dispersed-channel identities, with empty
            strings for dark pixels. Dispersed identities can be specific to each
            detector row and column to retain smile/keystone response differences.
        spectral_coefficients: Owned nonnegative (reads,spectral) electrons per
            unit spectral radiance for legacy dense construction, or
            (responses,spectral) shared table when response_indices is supplied.
            Includes spectral quadrature. Table rows need not be normalized.
        spatial_weights: Owned nonnegative (reads,nodes) weights, summing to one
            for responsive reads and zero for zero-response reads.
        node_times_s: Owned (reads,nodes) elapsed-SI-second quadrature times.
        node_directions_uv: Owned (reads,nodes,2) sensor direction cosines.
        response_indices: Optional owned int64 (reads,) table-row mapping. Omission
            retains the legacy one-row-per-read representation without deduplication.
        read_sensitivity: Optional owned finite nonnegative (reads,) absolute
            electrons per unit radiance. Overrides mapped table-row sums while
            preserving normalized spectral shape. Zero marks a dark read.
    """

    schedule: RasterReadSchedule
    grid: SpectralGrid
    read_bands: tuple[str, ...]
    spectral_coefficients: NDArray[np.float64]
    spatial_weights: NDArray[np.float64]
    node_times_s: NDArray[np.float64]
    node_directions_uv: NDArray[np.float64]
    response_indices: NDArray[np.int64] | None = None
    read_sensitivity: NDArray[np.float64] | None = None

    @property
    def reads(self) -> ReadSchedule:
        """Return physical identity and timing from the retained raster schedule."""
        return self.schedule.reads

    @property
    def sensitivity(self) -> NDArray[np.float64]:
        """Return owned (reads,) absolute electron/radiance sensitivity."""
        if self.read_sensitivity is not None:
            return self.read_sensitivity.copy()
        mass = self.spectral_coefficients.sum(axis=1)
        return mass if self.response_indices is None else mass[self.response_indices]

    @property
    def normalized_responses(self) -> NDArray[np.float64]:
        """Return owned normalized table rows, leaving zero-response rows zero."""
        mass = self.spectral_coefficients.sum(axis=1, keepdims=True)
        return np.divide(
            self.spectral_coefficients,
            mass,
            out=np.zeros_like(self.spectral_coefficients),
            where=mass > 0,
        )

    def response_rows(self, indices: NDArray[np.int64]) -> NDArray[np.int64]:
        """Map selected native read indices to owned spectral table-row indices.

        Args:
            indices: Int64 vector of native read indices; may be empty or repeated.

        Returns:
            Owned vector in supplied read order, without spectral expansion.

        Raises:
            ValueError: Indices are non-vector, non-int64 or outside native support.
        """
        if (
            indices.dtype != np.int64
            or indices.ndim != 1
            or np.any((indices < 0) | (indices >= len(self.reads.sample_ids)))
        ):
            raise ValueError("response selection requires int64 native read indices in range")
        return indices.copy() if self.response_indices is None else self.response_indices[indices]

    def coefficients_for_reads(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Explicitly allocate physical coefficients for selected reads.

        Args:
            indices: Int64 native read indices in desired order, including repeats.

        Returns:
            Owned (selected reads,spectral) electron/radiance coefficients, scaled
            by absolute read sensitivity when supplied. This is an explicit dense
            expansion; calibration and grouping operate on the table directly.

        Raises:
            ValueError: Native read indices are invalid.
        """
        rows = self.response_rows(indices)
        if self.read_sensitivity is None:
            return self.spectral_coefficients[rows]
        return self.normalized_responses[rows] * self.read_sensitivity[indices, None]

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        points = self.grid.points
        spectral = np.array(self.spectral_coefficients, dtype=np.float64, copy=True)
        spatial = np.array(self.spatial_weights, dtype=np.float64, copy=True)
        times = np.array(self.node_times_s, dtype=np.float64, copy=True)
        directions = np.array(self.node_directions_uv, dtype=np.float64, copy=True)
        n = len(self.reads.sample_ids)
        indices = self.response_indices
        if indices is not None:
            if (
                indices.dtype != np.int64
                or indices.shape != (n,)
                or np.any(indices < 0)
                or spectral.ndim != 2
                or np.any(indices >= spectral.shape[0])
            ):
                raise ValueError("response indices must map every read to a spectral table row")
            indices = indices.copy()
        if (
            len(points) < 2
            or len(self.read_bands) != n
            or spectral.ndim != 2
            or spectral.shape[1] != len(points)
            or spectral.shape[0] < 1
            or (indices is None and spectral.shape[0] != n)
            or spatial.ndim != 2
            or spatial.shape[0] != n
            or not spatial.shape[1]
            or not np.isfinite(spectral).all()
            or np.any(spectral < 0)
            or not np.isfinite(spatial).all()
            or np.any(spatial < 0)
            or times.shape != spatial.shape
            or directions.shape != (*spatial.shape, 2)
            or not np.isfinite(times).all()
            or not np.isfinite(directions).all()
            or np.any(times < self.reads.exposure_bounds_s[:, :1])
            or np.any(times > self.reads.exposure_bounds_s[:, 1:])
            or np.any(np.sum(directions**2, axis=-1) >= 1)
        ):
            raise ValueError("native response nodes, shapes or positive coefficients are invalid")
        mass = spectral.sum(axis=1)
        if not np.isfinite(mass).all():
            raise ValueError("summed spectral sensitivity must be finite")
        sensitivity = mass if indices is None else mass[indices]
        if self.read_sensitivity is not None:
            supplied = np.array(self.read_sensitivity, dtype=np.float64, copy=True)
            if (
                supplied.shape != (n,)
                or not np.isfinite(supplied).all()
                or np.any(supplied < 0)
                or np.any((sensitivity == 0) & (supplied > 0))
            ):
                raise ValueError(
                    "read sensitivity must be finite nonnegative with spectral support"
                )
            sensitivity = supplied
            object.__setattr__(self, "read_sensitivity", supplied)
        responsive = sensitivity > 0
        if not np.allclose(spatial.sum(axis=1), responsive.astype(float), rtol=1e-12, atol=1e-14):
            raise ValueError("native spatial weights must normalize for each responsive read")
        object.__setattr__(self, "spectral_coefficients", spectral)
        object.__setattr__(self, "response_indices", indices)
        object.__setattr__(self, "spatial_weights", spatial)
        object.__setattr__(self, "node_times_s", times)
        object.__setattr__(self, "node_directions_uv", directions)


def group_native_bands(
    response: NativeBandResponse, *, response_tolerance: float = 1e-8
) -> tuple[NativeBandResponse, tuple[SpectralBand, ...]]:
    """Group native reads only when their achieved photon responses agree.

    Stable first-occurrence grouping uses L1 distance to a representative and
    every member of its group. All spectral peaks remain in each positive band.
    Dark reads receive an empty label. Grouping changes descriptive band labels;
    physical read, pixel, exposure and port identities remain unchanged.

    Args:
        response: Supplied characterized spectral coefficients and spatial marginal.
        response_tolerance: Finite nonnegative pairwise normalized-response L1 tolerance.

    Returns:
        Owned response with compatible group labels, and corresponding physical bands.
        An entirely dark response returns no bands.

    Raises:
        ValueError: The compatibility tolerance is negative or nonfinite.
    """
    if not np.isfinite(response_tolerance) or response_tolerance < 0:
        raise ValueError("response tolerance must be finite and nonnegative")
    groups: list[list[NDArray[np.float64]]] = []
    bands: list[SpectralBand] = []
    labels: list[str] = []
    normalized = response.normalized_responses
    rows = response.response_rows(np.arange(len(response.reads.sample_ids), dtype=np.int64))
    known: dict[bytes, str] = {}
    row_labels: dict[int, str] = {}
    for row, mass in zip(rows, response.sensitivity, strict=True):
        if mass == 0:
            labels.append("")
            continue
        if int(row) in row_labels:
            labels.append(row_labels[int(row)])
            continue
        weights = normalized[row]
        key = weights.tobytes()
        if key in known:
            labels.append(known[key])
            row_labels[int(row)] = labels[-1]
            continue
        for index, members in enumerate(groups):
            if all(np.sum(np.abs(weights - member)) <= response_tolerance for member in members):
                members.append(weights)
                labels.append(bands[index].name)
                break
        else:
            name = f"native-{len(bands)}"
            groups.append([weights])
            bands.append(SpectralBand(name, response.grid, weights))
            labels.append(name)
        known[key] = labels[-1]
        row_labels[int(row)] = labels[-1]
    return replace(response, read_bands=tuple(labels)), tuple(bands)
