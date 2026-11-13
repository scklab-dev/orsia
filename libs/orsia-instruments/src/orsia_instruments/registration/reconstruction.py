"""Register calibrated native band means by normalized spatial accumulation.

This bounded adjoint accumulation smooths heterogeneous scenes. It assumes a
static common-view band radiance over the combined exposures. Spectral/spatial
coupling uses the supplied flat-spectrum marginal kernel and remains an explicit
approximation for wavelength-dependent PSFs and finite-slit dispersion on
heterogeneous spectra. Slit width, smile and keystone can make the joint response
nonseparable even with a constant PSF. Row-specific channel identities retain
these distinct achieved bands; scalar calibration does not unmix wavelengths."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from orsia_geometry.grids import cell_edges, cell_indices
from orsia_radiative.spectral.bands import SpectralBand

from orsia_instruments.acquisition.surface import SurfaceAcquisitionGrid
from orsia_instruments.calibration.frames import CalibratedFrames
from orsia_instruments.observation.measurements import BandRadianceCube


def reconstruct_radiance_cube(
    calibrated: CalibratedFrames,
    estimated_grid: SurfaceAcquisitionGrid,
    *,
    x_edges_m: NDArray[np.float64],
    y_edges_m: NDArray[np.float64],
    bands: Sequence[SpectralBand],
    realisation: int = 0,
    read_valid: NDArray[np.bool_] | None = None,
    response_tolerance: float = 1e-8,
    minimum_coverage: float = 0,
) -> BandRadianceCube:
    """Accumulate compatible native band means on a separate requested ground grid.

    Each native value contributes through its supplied normalized spatial kernel
    at separately estimated ground coordinates. Division by accumulated per-band
    weight preserves uniform radiance wherever coverage is positive. Cell edges
    are half-open except for the final inclusive upper edge. Samples outside the
    output grid are discarded; missing bands and invalid cells remain NaN.
    The method performs registration and smoothing without spatial deconvolution
    or spectral inversion. Native normalized spectral weights must agree with the
    requested band and with each other within the supplied L1 tolerance. The cube
    retains the first compatible achieved response for each supported band; wholly
    missing bands retain the requested response and have false validity everywhere.

    Args:
        calibrated: Supplied-calibration native band means with read validity.
            Slit channels retain row/column-specific achieved responses. Exact
            band interpretation requires spatially uniform spectra within each
            read, or a separable response; heterogeneous cases retain smoothing
            and nonseparable spectral/spatial mixing errors.
        estimated_grid: Separately supplied estimated geometry, with identical native
            identities, exposure timing and sensor quadrature as the calibration.
        x_edges_m: Finite increasing (x+1,) cell edges in the surface metre chart.
        y_edges_m: Finite increasing (y+1,) cell edges in the same chart.
        bands: Ordered requested physical responses with unique names.
        realisation: Zero-based calibrated realization to accumulate.
        read_valid: Optional Boolean (reads,) extra quality mask, independent of saturation.
        response_tolerance: Nonnegative finite maximum L1 distance between normalized weights.
        minimum_coverage: Nonnegative finite accumulated-weight threshold for validity.

    Returns:
        Owned (y,x,band) cube with achieved responses, coverage, validity and NaN gaps.

    Raises:
        ValueError: Incompatible read geometry, output grid, quality mask or spectral responses.
    """
    x, y = cell_edges(x_edges_m), cell_edges(y_edges_m)
    bands = tuple(bands)
    if not bands or len({band.name for band in bands}) != len(bands):
        raise ValueError("requested cube bands must have unique nonempty identities")
    if (
        not np.isfinite([response_tolerance, minimum_coverage]).all()
        or min(response_tolerance, minimum_coverage) < 0
        or not 0 <= realisation < calibrated.values.shape[0]
    ):
        raise ValueError("invalid response tolerance, coverage threshold or realisation")
    response = calibrated.response
    rays = estimated_grid.rays.acquisition
    reads = response.reads
    if (
        rays.sample_ids != reads.sample_ids
        or rays.pixel_ids != reads.pixel_ids
        or rays.setting_ids != reads.setting_ids
        or rays.port_ids != reads.port_ids
        or not np.array_equal(rays.exposure_bounds_s, reads.exposure_bounds_s)
        or not np.array_equal(rays.times_s, response.node_times_s)
        or not np.array_equal(rays.directions_uv, response.node_directions_uv)
        or estimated_grid.xy_m.shape != (*response.spatial_weights.shape, 2)
    ):
        raise ValueError(
            "estimated geometry must preserve calibrated native identities and sensor quadrature"
        )
    good = calibrated.valid[realisation].copy()
    if read_valid is not None:
        if read_valid.shape != good.shape or read_valid.dtype != np.bool_:
            raise ValueError("quality mask must be Boolean (reads,)")
        good &= read_valid
    shape = (len(y) - 1, len(x) - 1, len(bands))
    numerator, coverage = np.zeros(shape), np.zeros(shape)
    achieved: list[SpectralBand] = []
    names = np.array(response.read_bands)
    normalized = response.normalized_responses
    for band_index, band in enumerate(bands):
        selected = np.flatnonzero(good & (names == band.name))
        if not len(selected):
            achieved.append(band)
            continue
        if not band.grid.same_points(response.grid):
            raise ValueError("requested band nodes differ from calibrated spectral support")
        rows = np.unique(response.response_rows(selected))
        weights = normalized[rows]
        reference = normalized[response.response_rows(selected[:1])[0]]
        if np.any(np.sum(np.abs(weights - band.weights), axis=1) > response_tolerance) or np.any(
            np.sum(np.abs(weights - reference), axis=1) > response_tolerance
        ):
            raise ValueError("requested band is incompatible with achieved photon response")
        achieved.append(SpectralBand(band.name, response.grid, reference))
        xy = estimated_grid.xy_m[selected]
        ix = cell_indices(x, xy[..., 0])
        iy = cell_indices(y, xy[..., 1])
        inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
        spatial = response.spatial_weights[selected]
        native_values = calibrated.values[realisation, selected, None]
        np.add.at(coverage[..., band_index], (iy[inside], ix[inside]), spatial[inside])
        np.add.at(
            numerator[..., band_index], (iy[inside], ix[inside]), (spatial * native_values)[inside]
        )
    valid = coverage > minimum_coverage
    values = np.divide(numerator, coverage, out=np.full(shape, np.nan), where=valid)
    return BandRadianceCube(
        (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, tuple(achieved), values, coverage, valid
    )
