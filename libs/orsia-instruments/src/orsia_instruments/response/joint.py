"""Project actual joint photon footprints onto bounded spatial and spectral cells."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray
from orsia_geometry.grids import cell_edges, cell_indices
from orsia_numerics.operators import linear_operator
from orsia_radiative.spectral.grid import SpectralGrid
from scipy.sparse.linalg import LinearOperator

from orsia_instruments.acquisition.schedule import RasterReadSchedule, check_raster_alignment
from orsia_instruments.acquisition.surface import SurfaceAcquisitionGrid
from orsia_instruments.calibration.frames import CalibratedElectrons
from orsia_instruments.optics.camera import FilterCamera, FocalPlaneImager
from orsia_instruments.optics.spectrometer import DispersiveSpectrometer
from orsia_instruments.photodetection.frames import frame_response_coefficients


@dataclass(frozen=True, eq=False)
class SpatialSpectralResponse:
    """Joint electron response of native reads to a static piecewise-constant scene.

    Factory outputs own writable storage; schedules retain immutable original
    identities. Keep these arrays stable during use. Values have C-order
    (y,x,spectral_bin) coordinates. A state entry is spectral radiance density,
    W m^-2 sr^-1 per (cm^-1), constant inside its cell and bin at every exposure.

    Attributes:
        matrix: Nonnegative (stacked_read,state) electrons per unit spectral radiance.
        schedules: Separate native schedules in matrix stacking order; sample IDs
            may coincide across ports, so the schedule index is part of identity.
        x_edges_m: Owned increasing surface-cell edges in metres.
        y_edges_m: Owned increasing surface-cell edges in metres.
        spectral_edges_cm1: Owned increasing spectral-bin edges in cm^-1.
        state_shape: Derived (y,x,spectral_bin) state dimensions.
        coverage: Derived positive sensitivity per state coordinate; it indicates
            observation support and is independent of inverse identifiability.
    """

    matrix: NDArray[np.float64]
    schedules: tuple[RasterReadSchedule, ...]
    x_edges_m: NDArray[np.float64]
    y_edges_m: NDArray[np.float64]
    spectral_edges_cm1: NDArray[np.float64]

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Return the C-order spatial/spectral scene shape."""
        return (len(self.y_edges_m) - 1, len(self.x_edges_m) - 1, len(self.spectral_edges_cm1) - 1)

    @property
    def coverage(self) -> NDArray[np.float64]:
        """Return owned total electron sensitivity per spatial/spectral cell."""
        return self.matrix.sum(axis=0).reshape(self.state_shape)


def stack_frame_electrons(
    response: SpatialSpectralResponse,
    charges: Sequence[CalibratedElectrons],
    *,
    realisation: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Align corrected port observations with a joint physical response.

    Each supplied charge must preserve its corresponding response schedule,
    including native order, pixel, exposure and physical port identity. Both
    missing-read masks and corrected negative observations are preserved.
    Any combined electron covariance supplied to the numerical solver must use
    this same response stacking order. Individual charge covariance remains in
    each charge record; cross-port covariance is a separately supplied model.

    Args:
        response: Joint operator with explicit ordered native schedules.
        charges: Separately electronically corrected observations, one per schedule.
        realisation: Zero-based realization available in every supplied charge.

    Returns:
        Owned (stacked_read,) electron observations and Boolean validity.

    Raises:
        ValueError: Schedule count, identity, timing, shape or realization disagrees.
    """
    charges = tuple(charges)
    if len(charges) != len(response.schedules) or not charges:
        raise ValueError("one calibrated charge record per response schedule required")
    for schedule, charge in zip(response.schedules, charges, strict=True):
        check_raster_alignment(schedule, charge.schedule)
        n = len(schedule.reads.sample_ids)
        if (
            charge.values.ndim != 2
            or charge.values.shape[1] != n
            or charge.valid.shape != charge.values.shape
            or charge.valid.dtype != np.bool_
            or not 0 <= realisation < charge.values.shape[0]
            or not np.isfinite(charge.values[charge.valid]).all()
        ):
            raise ValueError("calibrated charge shape, validity or realization disagrees")
    if (
        sum(len(schedule.reads.sample_ids) for schedule in response.schedules)
        != response.matrix.shape[0]
    ):
        raise ValueError("joint matrix row count differs from retained schedules")
    return (
        np.concatenate([charge.values[realisation] for charge in charges]),
        np.concatenate([charge.valid[realisation] for charge in charges]),
    )


def spatial_spectral_operator(
    instruments: Sequence[FilterCamera | FocalPlaneImager | DispersiveSpectrometer],
    estimated_grids: Sequence[SurfaceAcquisitionGrid],
    spectral_grid: SpectralGrid,
    *,
    x_edges_m: NDArray[np.float64],
    y_edges_m: NDArray[np.float64],
    spectral_edges_cm1: NDArray[np.float64],
    max_bytes: int,
    read_batch_size: int = 16,
    spectral_block_size: int | None = None,
    valid: NDArray[np.bool_] | None = None,
) -> LinearOperator:
    """Apply a joint electron response and its Euclidean transpose in bounded batches.

    Uses the same physical photon coefficients and cell basis as
    ``spatial_spectral_response``. The transpose is an unnormalized accumulation;
    normalized reconstruction has a separate contract. Each call recomputes
    coefficient batches and owns its writable buffers. Instruments, grids and
    spectral support are borrowed and MUST remain stable for the operator lifetime,
    including concurrent calls. Cell edges and the validity mask are snapshotted.

    Args:
        instruments: Detector models in all-port stacking order.
        estimated_grids: Corresponding native read schedules and footprint quadrature.
        spectral_grid: Connected positive cm-1 photon integration nodes.
        x_edges_m: Increasing surface x cell edges covering complete footprints.
        y_edges_m: Increasing surface y cell edges covering complete footprints.
        spectral_edges_cm1: Positive increasing bins covering spectral support.
        max_bytes: Positive budget for estimated coefficient workspace plus one
            result and gathered direction block. Adapter copies, inputs and
            coefficient-provider internal allocations are outside this estimate.
        read_batch_size: Positive maximum reads per coefficient batch.
        spectral_block_size: Optional positive spectral slice size with full-domain quadrature.
        valid: Optional fixed (all_reads,) mask. Selected rows preserve port and
            native read order; transpose inserts zero cotangents for omitted rows.

    Returns:
        Float64 operator with shape (selected_reads, y_cells*x_cells*bins), C-order
        state coordinates and direction-last block products. Results own storage.

    Raises:
        ValueError: Support, budgets, mask or physical coefficient inputs disagree.
        MemoryError: Explicit per-call workspace estimate exceeds max_bytes.
    """
    instruments, grids = tuple(instruments), tuple(estimated_grids)
    x, y, edges, shape, rows, nodes = _joint_support(
        instruments,
        grids,
        spectral_grid,
        x_edges_m,
        y_edges_m,
        spectral_edges_cm1,
        max_bytes,
        read_batch_size,
    )
    columns = prod(shape)
    mask = np.ones(rows, dtype=np.bool_) if valid is None else valid.copy()
    if mask.dtype != np.bool_ or mask.shape != (rows,) or not mask.any():
        raise ValueError("valid must select at least one native read")
    selected_rows = np.flatnonzero(mask)
    spectral_index = cell_indices(edges, spectral_grid.points)
    batch = min(read_batch_size, rows)
    spectral_count = (
        len(spectral_grid.points) if spectral_block_size is None else spectral_block_size
    )
    if spectral_count < 1:
        raise ValueError("spectral_block_size must be positive")
    samples = batch * nodes * min(spectral_count, len(spectral_grid.points))

    def apply(values: NDArray[np.float64], reverse: bool) -> NDArray[np.float64]:
        k = values.shape[1]
        size = columns if reverse else len(selected_rows)
        required = 8 * (8 * samples + (size + samples + rows) * k)
        if required > max_bytes:
            raise MemoryError(f"response products require estimated {required} bytes")
        out = np.zeros((columns if reverse else rows, k), dtype=np.float64)
        cotangent = np.zeros((rows, k), dtype=np.float64) if reverse else None
        if cotangent is not None:
            cotangent[selected_rows] = values
        offset = 0
        for instrument, grid in zip(instruments, grids, strict=True):
            for selected, spectral_slice, coefficients in frame_response_coefficients(
                instrument,
                grid,
                spectral_grid,
                batch_size=read_batch_size,
                spectral_block_size=spectral_block_size,
            ):
                xy = grid.xy_m[selected]
                ix, iy = cell_indices(x, xy[..., 0]), cell_indices(y, xy[..., 1])
                cell = (iy * shape[1] + ix) * shape[2]
                for local in range(len(coefficients)):
                    row = offset + selected.start + local
                    index = (cell[local, :, None] + spectral_index[spectral_slice]).ravel()
                    weights = coefficients[local].ravel()
                    if cotangent is not None:
                        np.add.at(out, index, weights[:, None] * cotangent[row])
                    else:
                        out[row] += weights @ values[index]
            offset += len(grid.schedule.reads.sample_ids)
        return out if reverse else out[selected_rows]

    return linear_operator(
        (len(selected_rows), columns), lambda v: apply(v, False), lambda w: apply(w, True)
    )


def spatial_spectral_response(
    instruments: Sequence[FilterCamera | FocalPlaneImager | DispersiveSpectrometer],
    estimated_grids: Sequence[SurfaceAcquisitionGrid],
    spectral_grid: SpectralGrid,
    *,
    x_edges_m: NDArray[np.float64],
    y_edges_m: NDArray[np.float64],
    spectral_edges_cm1: NDArray[np.float64],
    max_bytes: int,
    read_batch_size: int = 16,
    spectral_block_size: int | None = None,
) -> SpatialSpectralResponse:
    """Build a bounded linear operator using actual footprints and all physical ports.

    Estimated instruments supply spectral/QE characterization; separately supplied
    geometry locates each quadrature node. Both use the existing frame photon
    coefficient generator. No separability approximation is made. The scene is
    static, direction-independent and piecewise constant in the selected cells
    and bins. Every continuous declared footprint must lie inside the spatial
    domain, and every spectral node inside the bin domain. Exterior illumination
    is therefore included explicitly in scene cells rather than discarded.
    Quadrature samples represent the bin basis with half-open cells and a final
    inclusive upper edge; refine the spectral and spatial grids at discontinuities.
    No spectral content outside the supplied connected grid is inferred.

    Args:
        instruments: Detector-owned models in explicit stacking order, one per port/grid.
        estimated_grids: Matching estimated sensor/ground quadrature, one per instrument.
        spectral_grid: Positive connected cm^-1 integration nodes for all ports.
        x_edges_m: Increasing surface x cell edges in metres.
        y_edges_m: Increasing surface y cell edges in metres.
        spectral_edges_cm1: Increasing positive spectral-bin edges in cm^-1.
        max_bytes: Explicit positive budget for estimated dense construction storage.
        read_batch_size: Positive maximum reads per photon coefficient batch.
        spectral_block_size: Optional positive spectral slice size with full-domain quadrature.

    Returns:
        Positive physical response with separate native schedules and spatial/spectral axes.

    Raises:
        ValueError: Inputs disagree or full spatial/spectral support is not covered.
        MemoryError: Estimated dense storage exceeds the caller's resource budget.
    """
    instruments, grids = tuple(instruments), tuple(estimated_grids)
    x, y, spectral_edges, shape, rows, nodes = _joint_support(
        instruments,
        grids,
        spectral_grid,
        x_edges_m,
        y_edges_m,
        spectral_edges_cm1,
        max_bytes,
        read_batch_size,
    )
    columns = prod(shape)
    batch = min(read_batch_size, rows)
    spectral_count = (
        len(spectral_grid.points) if spectral_block_size is None else spectral_block_size
    )
    if spectral_count < 1:
        raise ValueError("spectral_block_size must be positive")
    required = 8 * (
        rows * columns + 8 * batch * nodes * min(spectral_count, len(spectral_grid.points))
    )
    if required > max_bytes:
        raise MemoryError(
            f"joint response requires estimated {required} bytes; budget is {max_bytes}"
        )
    matrix = np.zeros((rows, columns))
    spectral_index = cell_indices(spectral_edges, spectral_grid.points)
    offset = 0
    for instrument, grid in zip(instruments, grids, strict=True):
        for selected, spectral_slice, coefficients in frame_response_coefficients(
            instrument,
            grid,
            spectral_grid,
            batch_size=read_batch_size,
            spectral_block_size=spectral_block_size,
        ):
            xy = grid.xy_m[selected]
            ix = cell_indices(x, xy[..., 0])
            iy = cell_indices(y, xy[..., 1])
            cell = (iy * shape[1] + ix) * shape[2]
            for local in range(len(coefficients)):
                np.add.at(
                    matrix[offset + selected.start + local],
                    (cell[local, :, None] + spectral_index[spectral_slice]).ravel(),
                    coefficients[local].ravel(),
                )
        offset += len(grid.schedule.reads.sample_ids)
    return SpatialSpectralResponse(
        matrix, tuple(grid.schedule for grid in grids), x, y, spectral_edges
    )


def _joint_support(
    instruments: tuple[FilterCamera | FocalPlaneImager | DispersiveSpectrometer, ...],
    grids: tuple[SurfaceAcquisitionGrid, ...],
    spectral_grid: SpectralGrid,
    x_edges_m: NDArray[np.float64],
    y_edges_m: NDArray[np.float64],
    spectral_edges_cm1: NDArray[np.float64],
    max_bytes: int,
    read_batch_size: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[int, int, int], int, int
]:
    """Validate the shared physical cell basis and native read support.

    Returns:
        Owned edges, state shape, native row count and largest quadrature node count.

    Raises:
        ValueError: Support or budgets are inconsistent.
    """
    if not instruments or len(instruments) != len(grids) or read_batch_size < 1 or max_bytes < 1:
        raise ValueError("matching nonempty instruments/grids and positive budgets required")
    x, y, edges = (cell_edges(a) for a in (x_edges_m, y_edges_m, spectral_edges_cm1))
    if edges[0] <= 0 or edges[0] > spectral_grid.points[0] or edges[-1] < spectral_grid.points[-1]:
        raise ValueError("spectral cells must contain every declared integration node")
    for grid in grids:
        bounds = grid.bounds_m
        if (
            not np.isfinite(bounds).all()
            or np.any(bounds[:, [0, 2]] < [x[0], y[0]])
            or np.any(bounds[:, [1, 3]] > [x[-1], y[-1]])
            or not np.isfinite(grid.xy_m).all()
            or np.any(grid.xy_m < [x[0], y[0]])
            or np.any(grid.xy_m > [x[-1], y[-1]])
        ):
            raise ValueError("spatial cells must contain the complete continuous footprint")
    shape = (len(y) - 1, len(x) - 1, len(edges) - 1)
    rows = sum(len(grid.schedule.reads.sample_ids) for grid in grids)
    nodes = max(grid.rays.acquisition.times_s.shape[1] for grid in grids)
    return x, y, edges, shape, rows, nodes
