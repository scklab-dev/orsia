"""Sample caller-supplied spectral radiance along Earth-frame sight lines."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid

from .line_of_sight import LineOfSightSamples

# The scene receives borrowed (read,node,3) origins and unit directions,
# (read,node) times and (spectral,) cm^-1 points, and returns (read,node,spectral).
type LineOfSightRadiance = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
# The provider attests continuous full-support coverage, including spectral edges,
# exposure endpoints and transformed FOV boundaries, per physical read.
type RadianceCoverage = Callable[[LineOfSightSamples, SpectralGrid], NDArray[np.bool_]]


def sample_line_of_sight_radiance(
    rays: LineOfSightSamples,
    grid: SpectralGrid,
    scene: LineOfSightRadiance,
    coverage: RadianceCoverage,
) -> NDArray[np.float64]:
    """Sample fully supported sight-line radiance per cm^-1.

    The independent coverage callback receives the complete read intervals and
    sensor FOV bounds through rays.acquisition, plus transformed sampled FOV
    corners and their times. It must assert coverage for *every point* of each
    continuous time/FOV support and the entire spectral interval, including
    points between quadrature nodes and sampled corners. Continuous motion and scene domain
    certification belong to the provider. Returned radiance is owned writable
    float64 storage; callbacks borrow their inputs and must be pure.
    Radiance uses the observation-ray travel direction (instrument toward scene);
    provider adapters convert incoming-photon or local RadianceField conventions.

    Args:
        rays: Read-level Earth rays and complete input time/FOV supports.
        grid: Caller-selected one-interval wavenumber grid in cm^-1.
        scene: Provider returning unpolarized W m^-2 sr^-1 per (cm^-1) radiance
            on (read,node,spectral) from origins, directions, times and points.
        coverage: Provider attestation of full continuous time/FOV and spectral
            support as a boolean (reads,) mask, using its own scene domain policy.

    Returns:
        Owned (reads,nodes,spectral) finite nonnegative spectral radiance.

    Raises:
        ValueError: Incomplete declared support, invalid grid or radiance values.
    """
    points = grid.points
    if (
        grid.coordinate != "wavenumber_cm1"
        or grid.domain is None
        or len(points) < 2
        or grid.domain.intervals != ((points[0], points[-1]),)
        or np.any(points <= 0)
        or np.any(np.diff(points) <= 0)
    ):
        raise ValueError("sight-line radiance requires one fully sampled positive cm^-1 interval")
    supported = np.asarray(coverage(rays, grid))
    if supported.shape != (len(rays.acquisition.sample_ids),) or supported.dtype != np.bool_:
        raise ValueError("scene coverage must return a boolean (reads,) mask")
    if not np.all(supported):
        raise ValueError("entire line-of-sight support must lie within scene coverage")
    values = np.array(
        scene(rays.origins_e_m, rays.directions_e, rays.acquisition.times_s, points),
        dtype=np.float64,
        copy=True,
    )
    if (
        values.shape != (*rays.acquisition.times_s.shape, len(points))
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("scene radiance must be finite nonnegative (reads,nodes,spectral)")
    return values
