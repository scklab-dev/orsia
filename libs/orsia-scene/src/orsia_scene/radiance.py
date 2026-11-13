"""Evaluate physical direct-solar scenes through cell or path-resolved methods."""

from collections.abc import Callable
from copy import copy
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import LayeredCells, affine_surface_points, segment_layer_paths, surface_points
from orsia_geometry.frames import require_time_support, require_utc_epoch
from orsia_geometry.rays import validate_rays
from orsia_radiative import (
    SpectralGrid,
    integrate_path_optical_depth,
    integrate_path_optical_depth_tangent,
)
from orsia_radiative.transfer.paths import direct_surface_radiance, direct_surface_radiance_tangent

from orsia_scene.gas import GasLayerColumns, validate_layer_interfaces
from orsia_scene.optical_depth import SpatialOpticalDepth
from orsia_scene.projection import project_layer_columns_by_centroid
from orsia_scene.surface import affine_reflectance_weights


def surface_radiance(
    xy_m: NDArray[np.float64],
    viewing_directions: NDArray[np.float64],
    *,
    height_m: float | None = None,
    slopes: NDArray[np.float64] | None = None,
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    visibility: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]
    | Literal["assume_clear"]
    | None = None,
    height_edges_m: NDArray[np.float64],
    solar_direction: NDArray[np.float64],
    columns: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]], NDArray[np.float64]
    ],
    cross_sections_m2: NDArray[np.float64],
    reflectance: NDArray[np.float64],
    beam_normal_irradiance: NDArray[np.float64],
    mode: Literal["independent_column", "heterogeneous_direct_solar"],
    minimum_upward_cosine: float,
    horizontal_bounds_m: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    """Evaluate direct Lambertian radiance from supplied molecular columns.

    Independent columns sample vertical gas paths and apply solar/view secants;
    they require horizontal affine terrain. Heterogeneous direct solar transfer samples
    separate displaced solar and viewing paths. Both modes use absorption-only
    transfer with layer-constant spectroscopy and sensors above the model top.
    The surface must be illuminated and visible throughout each queried ray.

    Args:
        xy_m: Finite (queries,2) horizontal surface coordinates in metres.
        viewing_directions: (queries,3) local unit vectors from surface to sensor.
        height_m: Affine terrain height at x=y=0, in metres. Supply with slopes,
            or omit both when supplying height and gradient.
        slopes: Affine (2,) dimensionless height derivatives in x,y order.
        height: Optional callable, (queries,2) coordinates to (queries,) metre
            heights. Supply with gradient instead of the affine parameters.
        gradient: Matching callable returning (queries,2) height derivatives
            (dz/dx,dz/dy). Callers ensure consistency with height.
        visibility: Required for callable terrain. A callback receives borrowed
            (queries,3) surface points and outward unit directions, once for solar
            rays and once for viewing rays. Return a Boolean (queries,) mask of
            clear full rays toward Sun/sensor, including terrain beyond the
            atmospheric model. Any false entry rejects the batch. Alternatively,
            "assume_clear" explicitly delegates this check to the caller. None
            uses analytic affine visibility only. Local incidence alone does not
            establish terrain visibility. Callback state must remain stable.
        height_edges_m: Increasing finite (layers+1,) atmospheric heights in metres.
        solar_direction: (3,) local unit vector from surface toward the Sun.
        columns: Called with segment starts/ends (segments,3), in metres, and
            source-layer indices (segments,). Return only finite nonnegative
            molecular columns (segments,species) in molecules/m2. The callable
            owns the physical field and its support; inputs are borrowed and must
            not be mutated. Segment amounts already include path length.
        cross_sections_m2: Finite nonnegative (layers,species,spectral) m2/molecule,
            constant within each layer.
        reflectance: (queries,spectral) Lambertian fractions in [0,1].
        beam_normal_irradiance: Nonnegative (queries,spectral) W m-2 per (cm-1).
        mode: Independent vertical columns or heterogeneous direct solar paths.
        minimum_upward_cosine: Positive vertical direction bound <=1 for both
            solar and viewing rays. Grazing or descending paths are rejected.
        horizontal_bounds_m: Closed (xmin,xmax,ymin,ymax) metre rectangle for
            surface points and entire sampled paths. Caller certifies continuous
            observation coverage separately, including unsampled directions.

    Returns:
        Fresh writable float64 radiance (queries,spectral), W m-2 sr-1 per (cm-1).
        Empty queries are supported. Inputs and callable state remain unchanged.

    Raises:
        ValueError: Shapes, physical values, mode, visibility or support are
            invalid, or optical arithmetic exceeds finite support. Supplied
            callback errors propagate.
    """
    incidence, paths = _prepare_surface_paths(
        xy_m,
        viewing_directions,
        height_m=height_m,
        slopes=slopes,
        height=height,
        gradient=gradient,
        visibility=visibility,
        height_edges_m=height_edges_m,
        solar_direction=solar_direction,
        cross_sections_m2=cross_sections_m2,
        mode=mode,
        minimum_upward_cosine=minimum_upward_cosine,
        horizontal_bounds_m=horizontal_bounds_m,
    )
    depths = []
    for starts, ends, offsets, layers in paths:
        amounts = columns(starts, ends, layers)
        _require_column_axes(amounts, len(starts), cross_sections_m2.shape[1])
        depths.append(
            integrate_path_optical_depth(amounts, cross_sections_m2[layers], path_offsets=offsets)
        )
    solar_depth, viewing_depth = _slant_depths(depths, solar_direction, viewing_directions)
    return direct_surface_radiance(
        solar_depth, viewing_depth, reflectance, beam_normal_irradiance, incidence
    )


def surface_radiance_tangent(
    xy_m: NDArray[np.float64],
    viewing_directions: NDArray[np.float64],
    *,
    height_m: float | None = None,
    slopes: NDArray[np.float64] | None = None,
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    visibility: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]
    | Literal["assume_clear"]
    | None = None,
    height_edges_m: NDArray[np.float64],
    solar_direction: NDArray[np.float64],
    columns: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
    cross_sections_m2: NDArray[np.float64],
    reflectance: NDArray[np.float64],
    reflectance_tangent: NDArray[np.float64],
    beam_normal_irradiance: NDArray[np.float64],
    mode: Literal["independent_column", "heterogeneous_direct_solar"],
    minimum_upward_cosine: float,
    horizontal_bounds_m: tuple[float, float, float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate direct Lambertian radiance and explicit gas/reflectance directions.

    Independent columns sample vertical gas paths at each surface point and apply
    solar/view secants; they require horizontal affine terrain. Heterogeneous direct solar
    transfer samples separate displaced solar and viewing paths. Both modes are
    absorption-only with fixed layer spectroscopy and sensors above the model top.
    The surface must be illuminated and visible throughout each queried ray.
    Terrain and visibility remain fixed under differentiation.

    Args:
        xy_m: Finite (queries,2) horizontal surface coordinates in metres.
        viewing_directions: (queries,3) local unit vectors from surface to sensor.
        height_m: Affine terrain height at x=y=0, in metres. Supply with slopes,
            or omit both when supplying height and gradient.
        slopes: Affine (2,) dimensionless height derivatives in x,y order.
        height: Optional callable, (queries,2) coordinates to (queries,) metre
            heights. Supply with gradient instead of the affine parameters.
        gradient: Matching callable returning (queries,2) height derivatives
            (dz/dx,dz/dy). Callers ensure consistency with height.
        visibility: Required for callable terrain. Same full-ray Boolean callback
            or explicit "assume_clear" policy as surface_radiance. The callback
            receives (queries,3) points and outward directions separately for
            solar and viewing rays. Any false entry rejects the batch. None uses
            analytic affine visibility only. Keep callback state stable.
        height_edges_m: Increasing finite (layers+1,) atmospheric heights in metres.
        solar_direction: (3,) local unit vector from surface toward the Sun.
        columns: Called with segment starts/ends (segments,3), in metres, and
            source-layer indices (segments,). Return molecular columns
            (segments,species) and signed seeds (segments,species,k). The callable
            owns the physical field and its support; inputs are borrowed and must
            not be mutated. ``puff_molecular_columns`` can supply this calculation.
        cross_sections_m2: Nonnegative (layers,species,spectral) m2/molecule,
            constant within each layer and held fixed under differentiation.
        reflectance: (queries,spectral) Lambertian fractions in [0,1].
        reflectance_tangent: Signed (queries,spectral,k) reflectance seeds.
        beam_normal_irradiance: (queries,spectral) W m-2 per (cm-1).
        mode: Independent vertical columns or heterogeneous direct solar paths.
        minimum_upward_cosine: Positive vertical direction bound <=1 for both
            solar and viewing rays. Grazing or descending paths are rejected.
        horizontal_bounds_m: Closed (xmin,xmax,ymin,ymax) metre rectangle for
            surface points and entire sampled paths. Caller certifies continuous
            observation coverage separately, including unsampled directions.

    Returns:
        Fresh radiance (queries,spectral), W m-2 sr-1 per (cm-1), and signed
        tangents (queries,spectral,k) per caller parameter; k may be zero.

    Raises:
        ValueError: Shapes, physical values, mode, visibility or support are
            invalid, or optical/transfer arithmetic exceeds finite support.
    """
    incidence, paths = _prepare_surface_paths(
        xy_m,
        viewing_directions,
        height_m=height_m,
        slopes=slopes,
        height=height,
        gradient=gradient,
        visibility=visibility,
        height_edges_m=height_edges_m,
        solar_direction=solar_direction,
        cross_sections_m2=cross_sections_m2,
        mode=mode,
        minimum_upward_cosine=minimum_upward_cosine,
        horizontal_bounds_m=horizontal_bounds_m,
    )
    depths, seeds = [], []
    for starts, ends, offsets, layers in paths:
        amounts, damounts = columns(starts, ends, layers)
        _require_column_axes(amounts, len(starts), cross_sections_m2.shape[1])
        depth, seed = integrate_path_optical_depth_tangent(
            amounts, cross_sections_m2[layers], damounts, path_offsets=offsets
        )
        depths.append(depth)
        seeds.append(seed)
    solar_depth, viewing_depth = _slant_depths(depths, solar_direction, viewing_directions)
    solar_seed, viewing_seed = _slant_depths(seeds, solar_direction, viewing_directions)
    return direct_surface_radiance_tangent(
        solar_depth,
        viewing_depth,
        reflectance,
        beam_normal_irradiance,
        incidence,
        solar_optical_depth_tangent=solar_seed,
        viewing_optical_depth_tangent=viewing_seed,
        reflectance_tangent=reflectance_tangent,
    )


def _prepare_surface_paths(
    xy_m: NDArray[np.float64],
    viewing_directions: NDArray[np.float64],
    *,
    height_m: float | None,
    slopes: NDArray[np.float64] | None,
    height: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    visibility: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]
    | Literal["assume_clear"]
    | None,
    height_edges_m: NDArray[np.float64],
    solar_direction: NDArray[np.float64],
    cross_sections_m2: NDArray[np.float64],
    mode: Literal["independent_column", "heterogeneous_direct_solar"],
    minimum_upward_cosine: float,
    horizontal_bounds_m: tuple[float, float, float, float],
) -> tuple[
    NDArray[np.float64],
    list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]],
]:
    """Validate physical support and prepare incidence and segmented scene paths.

    Returns:
        Surface incidence cosines and one vertical or two solar/view segment
        tuples, each containing starts, ends, path offsets and layer indices.

    Raises:
        ValueError: Physical shapes, values, mode, visibility or support are invalid.
    """
    if mode not in ("independent_column", "heterogeneous_direct_solar"):
        raise ValueError("unsupported scene transfer mode")
    validate_layer_interfaces(height_edges_m)
    if (
        cross_sections_m2.ndim != 3
        or cross_sections_m2.shape[0] != len(height_edges_m) - 1
        or cross_sections_m2.shape[1] < 1
        or cross_sections_m2.shape[2] < 1
        or not np.isfinite(cross_sections_m2).all()
        or np.any(cross_sections_m2 < 0)
    ):
        raise ValueError("cross sections require finite nonnegative (layers,species,spectral)")
    if xy_m.ndim != 2 or xy_m.shape[1] != 2:
        raise ValueError("surface queries require (queries,2) coordinates")
    _require_horizontal_support(xy_m, horizontal_bounds_m)
    if height is not None or gradient is not None:
        if height is None or gradient is None or height_m is not None or slopes is not None:
            raise ValueError("supply height and gradient together, without affine parameters")
        if mode == "independent_column":
            raise ValueError("independent columns require horizontal affine terrain")
        if visibility is None:
            raise ValueError("callable terrain requires an explicit visibility policy")
        surface, normal = surface_points(xy_m, height=height, gradient=gradient)
    else:
        if height_m is None or slopes is None:
            raise ValueError("supply affine height_m and slopes, or height and gradient")
        if mode == "independent_column" and np.any(slopes != 0):
            raise ValueError("independent columns require horizontal affine terrain")
        surface, affine_normal = affine_surface_points(xy_m, height_m=height_m, slopes=slopes)
        normal = np.broadcast_to(affine_normal, surface.shape)
    validate_rays(surface, viewing_directions)
    if solar_direction.shape != (3,):
        raise ValueError("solar direction requires a unit (3,) vector")
    validate_rays(np.zeros(3), solar_direction)
    if (
        not np.isfinite(minimum_upward_cosine)
        or not 0 < minimum_upward_cosine <= 1
        or np.any(viewing_directions[:, 2] < minimum_upward_cosine)
        or solar_direction[2] < minimum_upward_cosine
        or np.any(np.sum(viewing_directions * normal, axis=1) <= 0)
        or np.any(normal @ solar_direction <= 0)
    ):
        raise ValueError("unsupported upward direction, visibility transition or shadow")
    if callable(visibility):
        for outward in (np.broadcast_to(solar_direction, surface.shape), viewing_directions):
            clear = visibility(surface, outward)
            if clear.shape != (len(surface),) or clear.dtype != np.bool_ or not clear.all():
                raise ValueError(
                    "visibility callback must return all-clear Boolean (queries,) mask"
                )
    elif visibility is not None and visibility != "assume_clear":
        raise ValueError("unsupported visibility policy")
    directions = (
        (np.broadcast_to([0.0, 0.0, 1.0], surface.shape),)
        if mode == "independent_column"
        else (np.broadcast_to(solar_direction, surface.shape), viewing_directions)
    )
    paths = []
    for direction in directions:
        starts, ends, offsets, layers = segment_layer_paths(
            surface, direction, height_edges_m, minimum_upward_cosine=minimum_upward_cosine
        )
        _require_horizontal_support(ends[:, :2], horizontal_bounds_m)
        paths.append((starts, ends, offsets, layers))
    return normal @ solar_direction, paths


def _slant_depths(
    depths: list[NDArray[np.float64]],
    solar_direction: NDArray[np.float64],
    viewing_directions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply secants to one vertical depth array or retain two sampled slant arrays.

    Returns:
        Solar and viewing slant depths or seeds, preserving trailing axes.
    """
    if len(depths) == 2:
        return depths[0], depths[1]
    vertical = depths[0]
    viewing_cosine = viewing_directions[:, 2].reshape(
        (len(viewing_directions),) + (1,) * (vertical.ndim - 1)
    )
    return vertical / solar_direction[2], vertical / viewing_cosine


def _require_column_axes(amounts: NDArray[np.float64], segments: int, species: int) -> None:
    if amounts.shape != (segments, species):
        raise ValueError("column callable must preserve segment and species axes")


def _require_horizontal_support(
    xy: NDArray[np.float64], bounds: tuple[float, float, float, float]
) -> None:
    xmin, xmax, ymin, ymax = bounds
    if (
        not np.isfinite(bounds).all()
        or xmin >= xmax
        or ymin >= ymax
        or not np.isfinite(xy).all()
        or np.any((xy[..., 0] < xmin) | (xy[..., 0] > xmax))
        or np.any((xy[..., 1] < ymin) | (xy[..., 1] > ymax))
    ):
        raise ValueError("surface or atmospheric path outside declared chart")


type FloatArray = NDArray[np.float64]


class DirectSolarScene:
    """Own stable physical cell-scene preparation and independent gas/tangent bindings.

    Geometry and optical preparation remain shared stable owners. Surface values,
    illumination and tangent seeds are immutable snapshots. Rebinding returns an
    independent scene sharing unchanged preparation. Each evaluation owns scratch
    and outputs. No instrument, band, read count, processing mask or RNG is retained.
    """

    def __init__(
        self,
        optical_depth: SpatialOpticalDepth,
        *,
        anchor_wavelengths_nm: tuple[float, float],
        surface_reflectance: FloatArray,
        beam_normal_irradiance: FloatArray,
        solar_direction: FloatArray,
        epoch: datetime,
        validity_s: tuple[float, float],
        method: Literal["layer_centroid", "independent_column"],
        continuation: Literal["finite", "infinite_uniform"],
        visibility: Literal["assume_clear"],
        minimum_upward_cosine: float = 0.1,
    ) -> None:
        """Prepare a frozen Lambertian scene on regular finite cells.

        Args:
            optical_depth: Stable background/extinction preparation with complete cells.
                Cell facets are horizontal; background transfer uses local vertical columns.
            anchor_wavelengths_nm: Two ordered wavelengths spanning the prepared samples.
            surface_reflectance: Fractions (y,x,anchor=2), in [0,1].
            beam_normal_irradiance: Nonnegative (spectral,) W/m2 per prepared coordinate.
            solar_direction: Unit (x,y,z) outward surface-to-Sun direction in the cell frame.
            epoch: Timezone-aware UTC origin for query elapsed seconds.
            validity_s: Closed finite elapsed-time interval asserting the entire scene is frozen.
            method: Two-path centroid projection or local vertical excess columns with secants.
            continuation: Explicit finite or infinite-uniform centroid boundary law.
            visibility: Caller assertion of clear solar/view paths. Stepped terrain
                occlusion and continuous acquisition coverage require external checks.
            minimum_upward_cosine: Positive vertical cosine bound <=1 for Sun and views.

        Raises:
            ValueError: Geometry, spectra, physical values or selected methods are invalid.
        """
        require_utc_epoch(epoch)
        require_time_support(np.empty(0), validity_s)
        if (
            method not in ("layer_centroid", "independent_column")
            or continuation not in ("finite", "infinite_uniform")
            or visibility != "assume_clear"
            or not np.isfinite(minimum_upward_cosine)
            or not 0 < minimum_upward_cosine <= 1
        ):
            raise ValueError(
                "explicit supported scene method, continuation and visibility required"
            )
        cells = optical_depth.support
        lo, hi = cells.horizontal.lower_xy_m, cells.horizontal.upper_xy_m
        if len(cells.horizontal.shape) != 2 or not all(cells.horizontal.shape):
            raise ValueError("scene requires nonempty regular (y,x) cells")
        width = hi[0, 0] - lo[0, 0]
        y, x = np.indices(cells.horizontal.shape)
        centres = lo[0, 0] + width / 2 + np.stack((x * width[0], y * width[1]), axis=-1)
        if not np.array_equal(lo, centres - width / 2) or not np.array_equal(
            hi, centres + width / 2
        ):
            raise ValueError("scene requires regular tiled rectangular cells")
        if (
            surface_reflectance.shape != (*cells.horizontal.shape, 2)
            or not np.isfinite(surface_reflectance).all()
            or np.any((surface_reflectance < 0) | (surface_reflectance > 1))
        ):
            raise ValueError("surface reflectance must contain (y,x,2) fractions in [0,1]")
        affine_reflectance_weights(optical_depth.spectral, anchor_wavelengths_nm)
        if (
            beam_normal_irradiance.shape != optical_depth.spectral.points.shape
            or not np.isfinite(beam_normal_irradiance).all()
            or np.any(beam_normal_irradiance < 0)
        ):
            raise ValueError("irradiance must be finite nonnegative on prepared spectral samples")
        if solar_direction.shape != (3,):
            raise ValueError("solar direction must be a unit (3,) vector")
        validate_rays(np.zeros(3), solar_direction)
        if solar_direction[2] < minimum_upward_cosine:
            raise ValueError("solar direction is outside upward support")
        self._optical_depth = optical_depth
        self._anchors = tuple(anchor_wavelengths_nm)
        self._reflectance = _snapshot(surface_reflectance)
        self._irradiance = _snapshot(beam_normal_irradiance)
        self._sun = _snapshot(solar_direction)
        self._epoch, self._validity = epoch, tuple(validity_s)
        self._method, self._continuation = method, continuation
        self._minimum_cosine = minimum_upward_cosine
        self._spacing = (float(width[1]), float(width[0]))
        self._x_edges = np.concatenate((lo[0, :, 0], hi[0, -1:, 0]))
        self._y_edges = np.concatenate((lo[:, 0, 1], hi[-1:, 0, 1]))
        self._gas: tuple[GasLayerColumns, ...] = ()
        self._gas_scales = _snapshot(np.empty(0))
        self._species_sections = {
            name: _snapshot(optical_depth.species_cross_sections(name))
            for name in optical_depth.atmosphere.species
        }
        self._gas_seeds: FloatArray | None = None
        self._surface_seeds: FloatArray | None = None

    @property
    def support(self) -> LayeredCells:
        """Return the shared complete immutable cell support."""
        return self._optical_depth.support

    @property
    def spectral(self) -> SpectralGrid:
        """Return the shared prepared spectral coordinate and samples."""
        return self._optical_depth.spectral

    @property
    def epoch(self) -> datetime:
        """Return the UTC origin shared by all query times."""
        return self._epoch

    @property
    def validity_s(self) -> tuple[float, float]:
        """Return the closed frozen-scene interval in elapsed seconds."""
        return self._validity

    def with_gas(
        self,
        gas_layers: tuple[GasLayerColumns, ...],
        *,
        scales: FloatArray | None = None,
    ) -> DirectSolarScene:
        """Bind excess gas without rebuilding unchanged physical preparation.

        Args:
            gas_layers: Immutable amount/centroid records on the exact scene support.
                Multiple contributions may share a species. Empty tuples remove excess.
                Nonzero water excess with prepared continuum requires a separate state update.
            scales: Optional finite nonnegative (gas_contribution,) amplitudes; default
                is one. Zero amplitude retains the supplied reference field/centroids
                for a nonzero absolute-amplitude tangent at the null scene.

        Returns:
            Independent scene sharing unchanged preparation. Tangent bindings are cleared
            because their contribution order belongs to the previous gas binding.

        Raises:
            ValueError: Gas support, species or selected physical continuation is invalid.
        """
        amplitudes = np.ones(len(gas_layers)) if scales is None else scales
        if (
            amplitudes.shape != (len(gas_layers),)
            or not np.isfinite(amplitudes).all()
            or np.any(amplitudes < 0)
        ):
            raise ValueError("gas scales must be finite nonnegative per contribution")
        for gas in gas_layers:
            if gas.species not in self._species_sections:
                raise ValueError("gas species is absent from the physical preparation")
            if not gas.support.matches(self.support):
                raise ValueError("gas must match scene cells, frame, terrain and interfaces")
            if (
                gas.species == "H2O"
                and self._optical_depth.continuum is not None
                and np.any(gas.columns_molecules_m2)
            ):
                raise ValueError("excess water requires updated continuum density products")
            if self._continuation == "infinite_uniform":
                project_layer_columns_by_centroid(
                    gas.columns_molecules_m2,
                    self._heights(gas),
                    (0.0, 0.0),
                    self._spacing,
                    continuation=self._continuation,
                )
        bound = copy(self)
        bound._gas = tuple(gas_layers)
        bound._gas_scales = _snapshot(amplitudes)
        bound._gas_seeds = bound._surface_seeds = None
        return bound

    def with_tangents(
        self,
        gas_scale_tangent: FloatArray,
        surface_anchor_tangent: FloatArray,
    ) -> DirectSolarScene:
        """Bind a fixed block of physical gas-scale and surface-reflectance directions.

        Args:
            gas_scale_tangent: Signed (gas_contribution,direction) amplitude seeds
                multiplying each reference column field, including at zero nominal scale.
                Centroids, support and spectroscopy stay fixed.
            surface_anchor_tangent: Signed (y,x,anchor=2,direction) reflectance seeds
                in the same direction order. Zero directions are valid.

        Returns:
            Independent scene with immutable direction snapshots. Its radiance method
            still evaluates values only; radiance_tangent evaluates the bound directions.

        Raises:
            ValueError: Seed shapes, direction counts or finite values disagree.
        """
        if (
            gas_scale_tangent.ndim != 2
            or gas_scale_tangent.shape[0] != len(self._gas)
            or surface_anchor_tangent.shape
            != (*self._reflectance.shape, gas_scale_tangent.shape[1])
            or not np.isfinite(gas_scale_tangent).all()
            or not np.isfinite(surface_anchor_tangent).all()
        ):
            raise ValueError("matching finite gas-scale and surface-anchor directions required")
        bound = copy(self)
        bound._gas_seeds, bound._surface_seeds = map(
            _snapshot, (gas_scale_tangent, surface_anchor_tangent)
        )
        return bound

    def radiance(
        self,
        xy_m: FloatArray,
        view_local: FloatArray,
        times_s: FloatArray,
        spectral: SpectralGrid,
    ) -> FloatArray:
        """Evaluate physical spectral radiance on piecewise-constant cell reconstruction.

        Args:
            xy_m: Finite (query,2) metre coordinates in the support's local (x,y) frame.
                Cells use half-open bounds; exact outer upper edges are excluded.
            view_local: Unit outward surface-to-sensor directions (query,3).
            times_s: Elapsed seconds (query,) inside the declared frozen interval.
            spectral: Exact subset of prepared samples, in the same density coordinate.

        Returns:
            Fresh (query,spectral) W/m2/sr per declared coordinate, including empty
            queries. No spectral integration, MTF, detector conversion or derivatives occur.

        Raises:
            ValueError: Physical queries, masks, spectrum or radiance arithmetic fail.
        """
        return self._evaluate(xy_m, view_local, times_s, spectral, tangent=False)[0]

    def radiance_tangent(
        self,
        xy_m: FloatArray,
        view_local: FloatArray,
        times_s: FloatArray,
        spectral: SpectralGrid,
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate radiance and the explicitly bound physical direction block.

        Args:
            xy_m: Supported (query,2) local metre coordinates.
            view_local: Supported outward unit (query,3) directions, fixed under differentiation.
            times_s: Supported frozen-scene elapsed seconds (query,).
            spectral: Exact same-coordinate subset of prepared spectral samples.

        Returns:
            Fresh radiance (query,spectral) and signed (query,spectral,direction)
            derivatives in radiance units per parameter. Geometry, time, centroids,
            illumination and spectroscopy derivatives are outside this contract.

        Raises:
            ValueError: Directions are unbound or queries/seeds/arithmetic are invalid.
        """
        if self._gas_seeds is None or self._surface_seeds is None:
            raise ValueError("bind physical tangent directions before tangent evaluation")
        values, derivative = self._evaluate(xy_m, view_local, times_s, spectral, tangent=True)
        assert derivative is not None
        return values, derivative

    def _evaluate(self, xy, views, times, spectral, *, tangent):
        require_time_support(times, self.validity_s)
        if (
            xy.shape != (times.size, 2)
            or views.shape != (times.size, 3)
            or not np.isfinite(xy).all()
        ):
            raise ValueError("flat (query,2), (query,3) and time axes must agree")
        validate_rays(np.zeros_like(views), views)
        if np.any(views[:, 2] < self._minimum_cosine):
            raise ValueError("view directions are outside upward support")
        x = np.searchsorted(self._x_edges, xy[:, 0], side="right") - 1
        y = np.searchsorted(self._y_edges, xy[:, 1], side="right") - 1
        pixels = np.stack((y, x), axis=-1)
        indices = self._optical_depth.spectral_indices(spectral)
        background = self._optical_depth.at_pixels(pixels, spectral)
        basis = affine_reflectance_weights(spectral, self._anchors)
        reflectance = self._reflectance[y, x] @ basis.T
        solar = background / self._sun[2]
        viewing = background / views[:, 2, None]
        ds = dv = dr = None
        if tangent:
            assert self._gas_seeds is not None and self._surface_seeds is not None
            ds = np.zeros((*background.shape, self._gas_seeds.shape[1]))
            dv = np.zeros_like(ds)
            dr = np.einsum("qap,ka->qkp", self._surface_seeds[y, x], basis)
        unique_views, groups = np.unique(views, axis=0, return_inverse=True)
        for number, gas in enumerate(self._gas):
            sigma = self._species_sections[gas.species][:, indices]
            if self._method == "independent_column":
                vertical = gas.columns_molecules_m2[y, x] @ sigma
                incoming = vertical / self._sun[2]
                outgoing = vertical / views[:, 2, None]
            else:
                incoming = self._project(gas, self._sun)[y, x] @ sigma / self._sun[2]
                outgoing = np.empty_like(background)
                for group, view in enumerate(unique_views):
                    selected = groups == group
                    outgoing[selected] = (
                        self._project(gas, view)[y[selected], x[selected]] @ sigma / view[2]
                    )
            solar += incoming * self._gas_scales[number]
            viewing += outgoing * self._gas_scales[number]
            if tangent:
                assert ds is not None and dv is not None and self._gas_seeds is not None
                ds += incoming[..., None] * self._gas_seeds[number]
                dv += outgoing[..., None] * self._gas_seeds[number]
        irradiance = np.broadcast_to(self._irradiance[indices], background.shape)
        incidence = np.full(times.size, self._sun[2])
        if tangent:
            return direct_surface_radiance_tangent(
                solar,
                viewing,
                reflectance,
                irradiance,
                incidence,
                solar_optical_depth_tangent=ds,
                viewing_optical_depth_tangent=dv,
                reflectance_tangent=dr,
            )
        return direct_surface_radiance(solar, viewing, reflectance, irradiance, incidence), None

    def _project(self, gas, outward):
        # A ray from a surface cell reaches gas at x_surface + H * outward_xy/outward_z.
        # Forward deposition therefore moves the gas footprint by the negative slope.
        return project_layer_columns_by_centroid(
            gas.columns_molecules_m2,
            self._heights(gas),
            -outward[[1, 0]] / outward[2],
            self._spacing,
            continuation=self._continuation,
        )

    def _heights(self, gas):
        return np.where(
            gas.columns_molecules_m2 > 0,
            np.maximum(gas.centroid_altitude_m - self.support.surface_altitude_m[..., None], 0),
            0,
        )


def _snapshot(values: FloatArray) -> FloatArray:
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
