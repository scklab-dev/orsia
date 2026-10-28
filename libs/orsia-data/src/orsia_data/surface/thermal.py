"""Independent ECOSTRESS tiled layers and ASTER GED climatology readers.

No ECOSTRESS/ASTER combination, cloud acceptance, emissivity interpolation or
temperature substitution. Source files/metadata remain application-owned.
"""

import re
from pathlib import Path

import numpy as np

from ..geospatial import RasterBand, RasterGrid, Window, read_raster_band, window_grid


def read_ecostress_l2t(
    path: Path,
    *,
    layer: str,
    window: Window | None = None,
) -> RasterBand:
    """Read one ECOv002_L2T_LSTE native GeoTIFF layer, retaining raw quality flags.

    Native L2T fields are physical float values (K, dimensionless emissivity,
    metres or degrees), NOT the packed integers of the swath product. No scale
    factors are applied a second time. QC/cloud/water remain integer codes with
    no implicit acceptance filtering. EmisWB is broadband, not five-band emissivity.
    Requires the native filename product/version and selected layer suffix.
    Other editions, swath geolocation and L2G HDF5 are not silently reinterpreted.

    Args:
        path: Local native v002 L2T GeoTIFF with a suffix matching ``layer``.
        layer: Native LST, LST_err, EmisWB, height, view_zenith, QC, cloud or water layer.
        window: Optional (row, column, height, width) pixel window.

    Returns:
        Raster band with native physical units or categorical codes, a validity
        mask and a grid aligned to the selected raster window.

    Raises:
        ValueError: If filename, layer or window support is invalid.
    """
    if not path.name.startswith("ECOv002_L2T_LSTE_") or not path.name.endswith(f"_{layer}.tif"):
        raise ValueError("expected native ECOSTRESS v002 L2T filename and layer")
    units, categorical = _ecostress_layer(layer)
    raw = read_raster_band(path, units=units, window=window, categorical=categorical)
    valid = raw.valid.copy()
    values = raw.values.copy()
    if not categorical:
        if units == "K":
            valid &= values >= 0 if layer.endswith("err") else values > 0
        elif units == "1":
            valid &= (values >= 0) & (values <= 1)
    values[~valid] = np.nan
    return RasterBand(values, valid, raw.grid, units, categorical)


def read_aster_ged(
    path: Path,
    *,
    fields: tuple[str, ...],
    window: Window | None = None,
) -> dict[str, RasterBand]:
    """Read AG100.v003 HDF5 fields on their 0.001-degree WGS84 grid.

    Supported fields: emissivity_10..14, emissivity_sdev_10..14, temperature,
    temperature_sdev. The first two are native band
    means and climatological standard deviations, NOT a high-resolution spectrum
    or per-observation retrieval uncertainty. Temperature is the product's mean
    kinetic temperature [K], not synchronous ECOSTRESS temperature.
    Filename supplies integer NW corner, verified against selected native
    Geolocation/Latitude and Longitude centres (float32 tolerance 2e-5 degrees).
    Native arrays must be (5,1000,1000) or (1000,1000), row-major; no axis guessing.
    Missing -9999 is masked before scaling. Arrays are new writable buffers.

    Args:
        path: Local AG100.v003 HDF5 tile with its NW corner encoded in the filename.
        fields: Distinct supported native band means, standard deviations or temperature.
        window: Optional (row, column, height, width) pixel window in the tile.

    Returns:
        Requested fields as raster bands with validity masks and aligned grid;
        emissivities are dimensionless and temperatures are in kelvin.

    Raises:
        ValueError: If filename, selection, grid, field dimensions or values are invalid.
    """
    import h5py
    from pyproj import CRS

    match = re.fullmatch(r"AG100\.v003\.(-?\d+)\.(-?\d+)\.0001\.h5", path.name)
    if match is None or not fields or len(set(fields)) != len(fields):
        raise ValueError("explicit distinct fields and AG100.v003 native filename required")
    north, west = map(int, match.groups())
    if not -89 <= north <= 90 or not -180 <= west <= 179:
        raise ValueError("invalid ASTER tile corner")
    full = RasterGrid(CRS.from_epsg(4326), (0.001, 0, west, 0, -0.001, north), (1000, 1000))
    selected = window if window is not None else (0, 0, 1000, 1000)
    grid = window_grid(full, selected)
    row, col, height, width = selected
    selection = np.s_[row : row + height, col : col + width]
    result = {}
    with h5py.File(path, "r") as source:
        for name, expected in (
            ("Latitude", north - (np.arange(row, row + height)[:, None] + 0.5) * 0.001),
            ("Longitude", west + (np.arange(col, col + width)[None, :] + 0.5) * 0.001),
        ):
            native = source[f"Geolocation/{name}"]
            if native.shape != full.shape or not np.allclose(
                native[selection], expected, rtol=0, atol=2e-5
            ):
                raise ValueError("ASTER geolocation does not match native centre grid")
        for field in fields:
            key, index, scale, units = _aster_field(field)
            dataset = source[key]
            if dataset.shape != ((5, *full.shape) if index is not None else full.shape):
                raise ValueError("unsupported ASTER field dimensions")
            raw = np.array(
                dataset[(index, *selection) if index is not None else selection], dtype=np.float64
            )
            valid = np.isfinite(raw) & (raw != -9999)
            values = raw * scale
            valid &= values >= 0
            if field.startswith("emissivity"):
                valid &= values <= 1
            values[~valid] = np.nan
            result[field] = RasterBand(values, valid, grid, units)
    return result


def _ecostress_layer(layer: str) -> tuple[str, bool]:
    if layer in ("QC", "cloud", "water"):
        return "1", True
    if layer in ("LST", "LST_err"):
        return "K", False
    if layer == "EmisWB":
        return "1", False
    if layer == "height":
        return "m", False
    if layer == "view_zenith":
        return "degree", False
    raise ValueError("unsupported ECOSTRESS L2T layer")


def _aster_field(field: str) -> tuple[str, int | None, float, str]:
    match = re.fullmatch(r"emissivity_(sdev_)?(1[0-4])", field)
    if match:
        sdev = bool(match[1])
        return (
            "Emissivity/SDev" if sdev else "Emissivity/Mean",
            int(match[2]) - 10,
            0.0001 if sdev else 0.001,
            "1",
        )
    fields = {
        "temperature": ("Temperature/Mean", None, 0.01, "K"),
        "temperature_sdev": ("Temperature/SDev", None, 0.01, "K"),
    }
    if field not in fields:
        raise ValueError("unsupported ASTER GED field")
    return fields[field]
