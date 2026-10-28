"""Copernicus DEM public COG distribution, without scientific decoding."""

from math import ceil, floor, isfinite
from pathlib import Path
from typing import Literal, overload

from ..acquisition import download_file


@overload
def download_copernicus_dem(
    *,
    south_deg: int,
    west_deg: int,
    resolution_m: Literal[30, 90],
    destination: Path,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> Path: ...


@overload
def download_copernicus_dem(
    *,
    bbox: tuple[float, float, float, float],
    resolution_m: Literal[30, 90],
    directory: Path,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    max_tiles: int = 100,
) -> list[Path]: ...


def download_copernicus_dem(
    *,
    resolution_m: Literal[30, 90],
    bbox: tuple[float, float, float, float] | None = None,
    directory: Path | None = None,
    south_deg: int | None = None,
    west_deg: int | None = None,
    destination: Path | None = None,
    timeout_s: float = 30.0,
    max_bytes: int = 128 * 1024 * 1024,
    max_tiles: int = 100,
    expected_sha256: str | None = None,
) -> Path | list[Path]:
    """Download public Copernicus DSM tiles by WGS84 rectangle or tile coordinates.

    With bbox=(west,south,east,north) in degrees and directory, return a list of
    paths named after the native tiles, even for a single tile. Select nominal
    one-degree cells with positive-area intersection; exact north/east integer
    edges do not add neighbours. west > east crosses the antimeridian. Order is
    south to north, then west to east through the rectangle, wrapping at 180.
    max_tiles bounds the selection before any filesystem/network effects.

    Alternatively, south_deg/west_deg and destination retain the single-tile
    Path return and optional expected_sha256. The two modes cannot be mixed.
    Bbox mode does not accept a shared digest for different files.

    Integer south/west identify the provider tile, not exact pixel-edge bounds:
    use the stored affine transform (COGs remove shared south/east samples).
    Bbox selection does not certify pixel-level support or valid elevations.
    There is no crop, mosaic, resampling, availability filtering or implicit cache.
    This is the mutable public GLO-30/GLO-90 distribution, not a release selector.
    Missing/ocean/restricted tiles raise HTTP errors; no sea-level substitution.

    Transfers are sequential, with timeout_s and max_bytes applied per tile.
    Existing target files (including symlinks) fail before any transfer. Each
    file uses download_file's atomic no-clobber publication, but the batch is not
    atomic: earlier completed tiles remain on failure, and no partial list is
    returned. Parent directories may remain. Do not mutate targets concurrently.

    Args:
        resolution_m: Nominal 30 or 90 metre public DEM product selection.
        bbox: Optional WGS84 (west,south,east,north) rectangle in degrees for tiles.
        directory: Output directory for rectangle-mode tiles.
        south_deg: Integer latitude of the selected single tile's southwest corner.
        west_deg: Integer longitude of the selected single tile's southwest corner.
        destination: Explicit path for the single selected tile.
        timeout_s: Per-tile network-operation timeout in seconds.
        max_bytes: Per-tile response size bound in bytes.
        max_tiles: Positive cap on rectangle-mode tile selection.
        expected_sha256: Optional single-tile hexadecimal content digest.

    Returns:
        Destination Path for a single tile, or ordered list of native tile paths
        for rectangle selection, after all requested transfers succeed.

    Raises:
        ValueError: Modes are mixed or selection, resolution or tile cap is invalid.
        FileExistsError: Any selected target path already exists.
    """
    if resolution_m not in (30, 90):
        raise ValueError("Copernicus DEM resolution must be 30 or 90 metres")
    if bbox is None:
        if directory is not None or south_deg is None or west_deg is None or destination is None:
            raise ValueError("provide bbox and directory, or south_deg/west_deg and destination")
        if not -90 <= south_deg < 90 or not -180 <= west_deg < 180:
            raise ValueError("invalid integer DEM tile coordinates")
        tile = _tile_name(south_deg, west_deg, resolution_m)
        return download_file(
            _tile_url(tile, resolution_m),
            destination,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            expected_sha256=expected_sha256,
        )
    if directory is None or any(
        value is not None for value in (south_deg, west_deg, destination, expected_sha256)
    ):
        raise ValueError("bbox requires directory and cannot be mixed with single-tile arguments")
    tiles = [_tile_name(s, w, resolution_m) for s, w in _tile_coordinates(bbox, max_tiles)]
    paths = [directory / f"{tile}.tif" for tile in tiles]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for tile, path in zip(tiles, paths, strict=True):
        download_file(_tile_url(tile, resolution_m), path, timeout_s=timeout_s, max_bytes=max_bytes)
    return paths


def _tile_coordinates(
    bbox: tuple[float, float, float, float], max_tiles: int
) -> list[tuple[int, int]]:
    west, south, east, north = bbox
    if not all(isfinite(value) for value in bbox) or not (
        -180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90
    ):
        raise ValueError("bbox must contain finite WGS84 bounds with south < north")
    if west == east or (west == 180 and east == -180):
        raise ValueError("bbox must have positive longitude extent")
    intervals = ((west, east),) if west < east else ((west, 180.0), (-180.0, east))
    # Near-global wrapping rectangles can intersect the same cell at both ends.
    longitudes = list(
        dict.fromkeys(w for left, right in intervals for w in range(floor(left), ceil(right)))
    )
    latitudes = range(floor(south), ceil(north))
    count = len(latitudes) * len(longitudes)
    if max_tiles <= 0 or count > max_tiles:
        raise ValueError(
            f"bbox selects {count} tiles; max_tiles must be positive and at least {count}"
        )
    return [(s, w) for s in latitudes for w in longitudes]


def _tile_name(south_deg: int, west_deg: int, resolution_m: Literal[30, 90]) -> str:
    lat = f"{'N' if south_deg >= 0 else 'S'}{abs(south_deg):02d}_00"
    lon = f"{'E' if west_deg >= 0 else 'W'}{abs(west_deg):03d}_00"
    return f"Copernicus_DSM_COG_{10 if resolution_m == 30 else 30}_{lat}_{lon}_DEM"


def _tile_url(tile: str, resolution_m: Literal[30, 90]) -> str:
    return f"https://copernicus-dem-{resolution_m}m.s3.amazonaws.com/{tile}/{tile}.tif"
