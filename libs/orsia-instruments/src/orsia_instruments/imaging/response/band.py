"""Retain native spectral sensitivity and normalized spatial support for calibration."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.bands import SpectralBand
from orsia_radiative.spectral.grid import SpectralGrid, require_wavenumber_grid

from ...acquisition import RasterReadSchedule, ReadSchedule


@dataclass(frozen=True, eq=False)
class NativeBandResponse:
    """Physical per-read sensitivity and sampled flat-spectrum spatial response.

    Spectral coefficients include photon conversion, quadrature, area, throughput,
    QE and time/angle integration once. A uniform spectral radiance L produces
    electrons=sum(spectral_coefficients * L). Normalization recovers the achieved
    photon-weighted mean. Spatial weights marginalize the same response under a
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
            unit spectral radiance, including spectral quadrature.
        spatial_weights: Owned nonnegative (reads,nodes) weights, summing to one
            for responsive reads and zero for zero-response reads.
        node_times_s: Owned (reads,nodes) elapsed-SI-second quadrature times.
        node_directions_uv: Owned (reads,nodes,2) sensor direction cosines.
    """

    schedule: RasterReadSchedule
    grid: SpectralGrid
    read_bands: tuple[str, ...]
    spectral_coefficients: NDArray[np.float64]
    spatial_weights: NDArray[np.float64]
    node_times_s: NDArray[np.float64]
    node_directions_uv: NDArray[np.float64]

    @property
    def reads(self) -> ReadSchedule:
        """Return physical identity and timing from the retained raster schedule."""
        return self.schedule.reads

    def __post_init__(self) -> None:
        require_wavenumber_grid(self.grid)
        points = self.grid.points
        spectral = np.array(self.spectral_coefficients, dtype=np.float64, copy=True)
        spatial = np.array(self.spatial_weights, dtype=np.float64, copy=True)
        times = np.array(self.node_times_s, dtype=np.float64, copy=True)
        directions = np.array(self.node_directions_uv, dtype=np.float64, copy=True)
        n = len(self.reads.sample_ids)
        if (
            len(points) < 2
            or len(self.read_bands) != n
            or spectral.shape != (n, len(points))
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
        responsive = spectral.sum(axis=1) > 0
        if not np.allclose(spatial.sum(axis=1), responsive.astype(float), rtol=1e-12, atol=1e-14):
            raise ValueError("native spatial weights must normalize for each responsive read")
        object.__setattr__(self, "spectral_coefficients", spectral)
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
    for coefficients in response.spectral_coefficients:
        mass = coefficients.sum()
        if mass == 0:
            labels.append("")
            continue
        weights = coefficients / mass
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
    return replace(response, read_bands=tuple(labels)), tuple(bands)
