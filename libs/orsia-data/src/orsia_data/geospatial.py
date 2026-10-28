"""Local raster IO and explicit spatial preparation, independent of providers.

Optional Rasterio/PROJ imports happen only at operations needing them. No remote
GDAL paths are accepted. Arrays returned by operations are fresh writable buffers;
records retain caller arrays without copying. Do not mutate inputs during a call.
Native execution is synchronous; resource and thread scheduling is caller-owned.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyproj import CRS

type Window = tuple[int, int, int, int]
type Resampling = Literal["nearest", "bilinear"]


@dataclass(frozen=True)
class RasterGrid:
    """Locate a raster's pixel edges and dimensions in a declared horizontal CRS.

    x=a*column+b*row+c; y=d*column+e*row+f. Pixel centres use half-integers.
    CRS is a pyproj.CRS, not an inferred EPSG string. Rotation is supported.

    Attributes:
        crs: Two-dimensional projected or geographic pyproj coordinate system.
        transform: Six affine pixel-edge coefficients (a, b, c, d, e, f), in CRS units.
        shape: Positive (rows, columns) pixel counts.
    """

    crs: CRS
    transform: tuple[float, float, float, float, float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.crs.axis_info) != 2 or not (self.crs.is_geographic or self.crs.is_projected):
            raise ValueError("raster grids require a two-dimensional horizontal CRS")
        a, b, _, d, e, _ = self.transform
        if not np.all(np.isfinite(self.transform)) or a * e - b * d == 0:
            raise ValueError("raster transform must be finite and invertible")
        if min(self.shape) <= 0:
            raise ValueError("raster dimensions must be positive")


@dataclass(frozen=True)
class RasterBand:
    """A scalar field, its decoded-validity mask, spatial support and units.

    values and valid have shape grid.shape, dtype float64 and bool. False validity
    means absent/undecodable, not scientifically rejected. Categorical fields
    contain exact integer labels/bit flags and cannot be interpolated bilinearly.
    Buffers are borrowed and mutable; frozen prevents rebinding, not mutation.

    Attributes:
        values: Borrowed float64 pixel values with shape ``grid.shape``.
        valid: Borrowed Boolean decoded-validity mask with shape ``grid.shape``.
        grid: Coordinate and pixel layout for both arrays.
        units: Declared physical units of valid pixel values.
        categorical: Whether valid values encode integer labels instead of continuous data.
    """

    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    grid: RasterGrid
    units: str
    categorical: bool = False

    def __post_init__(self) -> None:
        if self.values.shape != self.grid.shape or self.valid.shape != self.grid.shape:
            raise ValueError("raster values/mask must match grid shape")
        if self.valid.dtype != np.bool_:
            raise ValueError("raster validity requires a bool mask, not integer indices")
        values = self.values[self.valid]
        if not self.units or not np.all(np.isfinite(values)):
            raise ValueError("valid raster values must be finite and units declared")
        if self.categorical and np.any(values != np.floor(values)):
            raise ValueError("categorical raster values must be integer codes")


def window_grid(grid: RasterGrid, window: Window) -> RasterGrid:
    """Derive raster coordinates for a bounded integer crop.

    Args:
        grid: Spatial support of the original raster.
        window: Integer (row_start,column_start,height,width) within that raster.

    Returns:
        Pixel-edge transform and (height,width) shape of the cropped raster.

    Raises:
        ValueError: Window has invalid dimensions or exceeds the original support.
    """
    row, col, height, width = window
    if min(row, col) < 0 or min(height, width) <= 0:
        raise ValueError("window must have nonnegative origin and positive size")
    if row + height > grid.shape[0] or col + width > grid.shape[1]:
        raise ValueError("window exceeds raster support")
    a, b, c, d, e, f = grid.transform
    return RasterGrid(
        grid.crs, (a, b, c + a * col + b * row, d, e, f + d * col + e * row), (height, width)
    )


def window_from_bounds(
    grid: RasterGrid,
    bounds: tuple[float, float, float, float],
    *,
    clip: bool,
) -> Window:
    """Enclosing pixel window for (xmin,ymin,xmax,ymax) in the raster CRS.

    All four corners are inverse-transformed, including rotated grids. This is
    an enclosing rectangle, not polygon clipping. clip=False rejects outside
    support; clip=True intersects with the source but still rejects no overlap.
    No CRS transformation, dateline wrapping or coordinate extrapolation.

    Args:
        grid: Source raster grid, including its pixel-edge affine transform.
        bounds: Increasing (xmin,ymin,xmax,ymax) coordinates in the grid CRS.
        clip: Intersect the enclosing window with raster support when true.

    Returns:
        (row_start,column_start,height,width) integer pixel window.

    Raises:
        ValueError: Bounds are nonfinite, reversed or outside supported pixels.
    """
    xmin, ymin, xmax, ymax = bounds
    if not np.all(np.isfinite(bounds)) or xmin >= xmax or ymin >= ymax:
        raise ValueError("bounds must be finite and increasing")
    a, b, c, d, e, f = grid.transform
    x, y = np.array([xmin, xmax, xmax, xmin]), np.array([ymin, ymin, ymax, ymax])
    col = (e * (x - c) - b * (y - f)) / (a * e - b * d)
    row = (-d * (x - c) + a * (y - f)) / (a * e - b * d)
    left, right = int(np.floor(col.min())), int(np.ceil(col.max()))
    top, bottom = int(np.floor(row.min())), int(np.ceil(row.max()))
    if clip:
        left, right = max(0, left), min(grid.shape[1], right)
        top, bottom = max(0, top), min(grid.shape[0], bottom)
    window = (top, left, bottom - top, right - left)
    window_grid(grid, window)
    return window


def iter_raster_windows(grid: RasterGrid, block_shape: tuple[int, int]) -> Iterator[Window]:
    """Partition a raster into bounded windows for incremental local processing.

    Args:
        grid: Raster support to cover completely.
        block_shape: Positive (row_count,column_count) block extents in pixels.

    Yields:
        Row-major disjoint (row,column,height,width) windows, including short edges.

    Raises:
        ValueError: Requested block height or width is nonpositive.
    """
    height, width = block_shape
    if min(height, width) <= 0:
        raise ValueError("block dimensions must be positive")
    for row in range(0, grid.shape[0], height):
        for col in range(0, grid.shape[1], width):
            yield row, col, min(height, grid.shape[0] - row), min(width, grid.shape[1] - col)


def read_raster_grid(path: Path) -> RasterGrid:
    """Inspect local raster spatial support before reading its pixel samples.

    Args:
        path: Existing local GeoTIFF or JP2 path with affine horizontal CRS.

    Returns:
        Coordinate system, pixel-edge transform and raster dimensions.

    Raises:
        ValueError: Raster lacks supported affine two-dimensional georeferencing.
    """
    with _open_raster(path) as source:
        return _raster_grid(source)


def read_raster_band(
    path: Path,
    *,
    units: str,
    band: int = 1,
    window: Window | None = None,
    categorical: bool = False,
) -> RasterBand:
    """Read raw GeoTIFF/JP2 samples and nodata mask from an existing LOCAL file.

    Does not apply scale/offset tags: product readers own decoding. Single band
    selection is one-based. Windows avoid reading the whole raster. GCP/RPC-only
    geolocation is unsupported; an affine CRS is required. No resampling.

    Args:
        path: Existing local GeoTIFF or JP2 containing georeferenced samples.
        units: Caller-declared physical units of the stored raw values.
        band: One-based band index, defaulting to the first raster band.
        window: Optional (row,column,height,width) crop in original raster pixels.
        categorical: Mark values that require nearest-neighbour spatial resampling.

    Returns:
        Float64 values and validity on the full or cropped grid, with fresh arrays;
        invalid samples contain NaN and the provided units are retained.

    Raises:
        ValueError: Grid georeferencing, crop or selected band is unsupported.
    """
    from rasterio.windows import Window as RioWindow

    with _open_raster(path) as source:
        grid = _raster_grid(source)
        selection = None
        if window is not None:
            grid = window_grid(grid, window)
            row, col, height, width = window
            selection = RioWindow(col, row, width, height)
        if not 1 <= band <= source.count:
            raise ValueError("raster band is outside available support")
        raw = source.read(band, window=selection, masked=True)
        values = np.array(raw.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(raw) & np.isfinite(values)
    values[~valid] = np.nan
    return RasterBand(values, valid, grid, units, categorical)


def crop_raster(band: RasterBand, window: Window) -> RasterBand:
    """Extract an independently writable spatial subset of a raster band.

    Args:
        band: Source values, validity, units and raster grid.
        window: Integer (row_start,column_start,height,width) in source pixels.

    Returns:
        Fresh values and validity arrays on the cropped pixel-edge grid.

    Raises:
        ValueError: Window falls outside the source grid or has invalid size.
    """
    grid = window_grid(band.grid, window)
    row, col, height, width = window
    selection = np.s_[row : row + height, col : col + width]
    return RasterBand(
        band.values[selection].copy(),
        band.valid[selection].copy(),
        grid,
        band.units,
        band.categorical,
    )


def reproject_raster(
    band: RasterBand,
    target: RasterGrid,
    *,
    method: Resampling,
    minimum_coverage: float,
) -> tuple[RasterBand, NDArray[np.float64]]:
    """Warp onto an explicit grid, returning data and kernel-valid weight [0,1].

    Coverage is interpolation support, NOT geometric area overlap or uncertainty.
    Missing samples contribute zero numerator and weight; retained values are
    normalized by valid weight. minimum_coverage in (0,1] explicitly controls
    acceptance of partially valid kernels. No extrapolation beyond source extent.
    Categorical data requires nearest. No vertical datum transformation occurs.
    Memory scales with source and target windows; no global executor is created.

    Args:
        band: Source samples, validity mask, raster grid and units.
        target: Destination CRS, pixel-edge affine transform and output shape.
        method: Nearest or bilinear resampling; categorical bands require nearest.
        minimum_coverage: Minimum valid kernel fraction in (0,1] for output validity.

    Returns:
        A fresh warped band and a float64 coverage array on the target grid.
        Invalid band samples are NaN; coverage describes interpolation support.

    Raises:
        ValueError: Coverage is outside (0,1] or the method is incompatible with the band.
    """
    import rasterio
    from rasterio.enums import Resampling as RioResampling
    from rasterio.warp import reproject

    _method(band, method)
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum coverage must be in (0,1]")
    numerator = np.zeros(target.shape, dtype=np.float64)
    coverage = np.zeros(target.shape, dtype=np.float64)
    for source, dest in (
        (np.where(band.valid, band.values, 0.0), numerator),
        (band.valid.astype(np.float64), coverage),
    ):
        reproject(
            source,
            dest,
            src_transform=rasterio.Affine(*band.grid.transform),
            src_crs=band.grid.crs.to_wkt(),
            dst_transform=rasterio.Affine(*target.transform),
            dst_crs=target.crs.to_wkt(),
            resampling=getattr(RioResampling, method),
            dst_nodata=float("nan"),
            init_dest_nodata=True,
            num_threads=1,
        )
    coverage = np.clip(np.nan_to_num(coverage, nan=0.0), 0.0, 1.0)
    valid = (coverage > 0) & (coverage >= minimum_coverage - 1e-12)
    values = np.full(target.shape, np.nan)
    np.divide(numerator, coverage, out=values, where=valid)
    return RasterBand(values, valid, target, band.units, band.categorical), coverage


def mosaic_rasters(
    bands: tuple[RasterBand, ...],
    target: RasterGrid,
    *,
    method: Resampling,
    overlap: Literal["first", "last"],
    minimum_coverage: float,
) -> RasterBand:
    """Warp and combine explicit bands in caller order; no automatic hole filling.

    First/last refers to the first/last valid warped value. All sources must share
    units and categorical semantics. Only one warped source is retained at a time.

    Args:
        bands: Ordered source bands with matching units and categorical interpretation.
        target: Destination raster grid for every source.
        method: Nearest or bilinear resampling of each source.
        overlap: Choose the first or last valid warped value at each destination pixel.
        minimum_coverage: Required valid resampling-kernel fraction in (0,1].

    Returns:
        New target-grid band with NaN samples where every source is invalid.

    Raises:
        ValueError: Sources are empty or incompatible, or overlap/coverage is invalid.
    """
    if not bands or overlap not in ("first", "last"):
        raise ValueError("mosaic needs bands and explicit first/last overlap policy")
    values = np.full(target.shape, np.nan)
    valid = np.zeros(target.shape, dtype=bool)
    for band in bands:
        if (band.units, band.categorical) != (bands[0].units, bands[0].categorical):
            raise ValueError("mosaic units and categorical semantics must agree")
        warped, _ = reproject_raster(band, target, method=method, minimum_coverage=minimum_coverage)
        use = warped.valid & (~valid if overlap == "first" else True)
        values[use] = warped.values[use]
        valid |= use
    return RasterBand(values, valid, target, bands[0].units, bands[0].categorical)


def sample_raster(
    band: RasterBand,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    method: Resampling,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Sample coordinates in the raster CRS; return fresh values/validity.

    No reprojection, missing-data filling or extrapolation. Nearest uses pixel
    footprints; bilinear requires support from every nonzero centre weight.
    Arrays x/y must have identical shape. Nonfinite positions are invalid.

    Args:
        band: Source band including its grid, samples and validity mask.
        x: Sampling x coordinates in the band's CRS, with the output shape.
        y: Sampling y coordinates in the band's CRS, matching x in shape.
        method: Nearest-pixel or bilinear interpolation of valid samples.

    Returns:
        Fresh float64 values and Boolean validity arrays shaped like x;
        samples outside support are NaN and marked invalid.

    Raises:
        ValueError: Coordinate shapes differ or method conflicts with categorical samples.
    """
    _method(band, method)
    if x.shape != y.shape:
        raise ValueError("sampling coordinates must have matching shapes")
    a, b, c, d, e, f = band.grid.transform
    det = a * e - b * d
    col = (e * (x - c) - b * (y - f)) / det
    row = (-d * (x - c) + a * (y - f)) / det
    finite = np.isfinite(col) & np.isfinite(row)
    col = np.where(finite, col, -2.0)
    row = np.where(finite, row, -2.0)
    # Clip before integer conversion to avoid overflow at huge outside positions.
    col = np.clip(col, -2, band.grid.shape[1] + 1)
    row = np.clip(row, -2, band.grid.shape[0] + 1)
    if method == "bilinear":
        col, row = col - 0.5, row - 0.5
    ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
    values = np.zeros(x.shape, dtype=np.float64)
    valid = finite.copy()
    offsets = (
        ((0, 0, np.ones(x.shape)),)
        if method == "nearest"
        else (
            (0, 0, (1 - (col - ci)) * (1 - (row - ri))),
            (1, 0, (col - ci) * (1 - (row - ri))),
            (0, 1, (1 - (col - ci)) * (row - ri)),
            (1, 1, (col - ci) * (row - ri)),
        )
    )
    for dx, dy, weight in offsets:
        xx, yy = ci + dx, ri + dy
        inside = (xx >= 0) & (xx < band.grid.shape[1]) & (yy >= 0) & (yy < band.grid.shape[0])
        xx, yy = np.clip(xx, 0, band.grid.shape[1] - 1), np.clip(yy, 0, band.grid.shape[0] - 1)
        supported = inside & band.valid[yy, xx]
        valid &= supported | (weight == 0)
        values += weight * np.where(supported, band.values[yy, xx], 0)
    values[~valid] = np.nan
    return values, valid


def _method(band: RasterBand, method: Resampling) -> None:
    if method not in ("nearest", "bilinear"):
        raise ValueError("unsupported raster resampling method")
    if band.categorical and method != "nearest":
        raise ValueError("categorical rasters require nearest resampling")


def _open_raster(path: Path) -> Any:
    import rasterio

    local = path.resolve(strict=True)
    if not local.is_file():
        raise ValueError("raster path must be a local regular file")
    with local.open("rb") as stream:
        header = stream.read(12)
    if header[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        driver = "GTiff"
    elif header == b"\x00\x00\x00\x0cjP  \r\n\x87\n":
        driver = "JP2OpenJPEG"
    else:
        raise ValueError("only local GeoTIFF and boxed JPEG2000 rasters are supported")
    return rasterio.open(local, driver=driver, sharing=False)


def _raster_grid(source: Any) -> RasterGrid:
    from pyproj import CRS

    if source.crs is None:
        raise ValueError("raster has no affine CRS")
    return RasterGrid(
        CRS.from_wkt(source.crs.to_wkt()),
        tuple(source.transform)[:6],
        (source.height, source.width),
    )
