"""Sentinel-2 SAFE L2A native-band reflectance; no harmonisation or BRDF model."""

from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import numpy as np

from ..geospatial import RasterBand, Window, read_raster_band

_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def read_sentinel2_metadata(safe: Path) -> dict[str, str]:
    """Read product identity, baseline and actual product time bounds from SAFE XML.

    Returns PRODUCT_URI, PROCESSING_BASELINE, PRODUCT_START_TIME, PRODUCT_STOP_TIME
    and PRODUCT_TYPE without rewriting their native strings. This is product
    support, not a guarantee of one acquisition time for all pixels/granules.

    Args:
        safe: Local SAFE product root containing MTD_MSIL2A.xml.

    Returns:
        Native product URI, baseline, start/stop time and product type strings,
        keyed by their original XML field names.

    Raises:
        ValueError: If metadata exceeds the size bound or required fields are
            missing, ambiguous or outside the selected L2A product type.
    """
    root = _metadata(safe)
    return {
        name: _one_text(root, name)
        for name in (
            "PRODUCT_URI",
            "PROCESSING_BASELINE",
            "PRODUCT_START_TIME",
            "PRODUCT_STOP_TIME",
            "PRODUCT_TYPE",
        )
    }


def read_sentinel2_l2a(
    safe: Path,
    *,
    granule: str,
    bands: tuple[str, ...],
    resolution_m: Literal[10, 20, 60],
    window: Window | None = None,
) -> dict[str, RasterBand]:
    """Read selected SAFE JP2 bands at one explicit native product resolution.

    granule is the exact GRANULE directory name. bands selects B01..B12 (not B10)
    and/or SCL; unavailable combinations fail, never resample. BOA reflectance is
    (DN+metadata band offset)/metadata quantification. DN=0 is absent; negative
    reflectance is retained, not clipped. Pre-04.00 products have zero offset only
    when the offset list is absent; later products must supply requested offsets.
    SCL is categorical with class 0 absent; cloud/snow classes remain valid codes.
    No quality acceptance rule, band alignment, gap fill or spectral synthesis.
    Windows use each selected raster's pixel indices. Metadata files are bounded
    local XML without DTD/entities, and image paths cannot escape the SAFE root.

    Args:
        safe: Local Sentinel-2 L2A SAFE product root.
        granule: Exact name of one GRANULE directory within the product.
        bands: Distinct native BOA reflectance band identifiers or SCL.
        resolution_m: Selected native raster resolution in metres.
        window: Optional (row, column, height, width) window in each raster's pixels.

    Returns:
        Raster bands by requested identifier, with dimensionless reflectance or
        categorical SCL values and per-band validity masks and grids.

    Raises:
        ValueError: If selection, SAFE metadata, image location, quantification
            or decoded SCL support is invalid.
    """
    if resolution_m not in (10, 20, 60) or not bands or len(set(bands)) != len(bands):
        raise ValueError("explicit distinct bands and supported native resolution required")
    if any(band not in (*_BANDS, "SCL") or band == "B10" for band in bands):
        raise ValueError("unsupported L2A band")
    if not granule or Path(granule).name != granule or granule in (".", ".."):
        raise ValueError("granule must be one explicit directory name")
    root = _metadata(safe)
    quantification = float(_one_text(root, "BOA_QUANTIFICATION_VALUE"))
    baseline = float(_one_text(root, "PROCESSING_BASELINE"))
    if not np.isfinite(quantification) or quantification <= 0 or not np.isfinite(baseline):
        raise ValueError("invalid L2A quantification/baseline")
    offsets: dict[str, float] = {}
    for element in root.iter("BOA_ADD_OFFSET"):
        index = int(element.attrib["band_id"])
        if not 0 <= index < len(_BANDS) or _BANDS[index] in offsets:
            raise ValueError("invalid/duplicate L2A offset band")
        offsets[_BANDS[index]] = float(element.text or "nan")
    images = [node.text for node in root.iter("IMAGE_FILE") if node.text]
    base = safe.resolve(strict=True)
    result = {}
    for name in bands:
        suffix = f"_{name}_{resolution_m}m"
        candidates = [
            image
            for image in images
            if image.startswith(f"GRANULE/{granule}/IMG_DATA/R{resolution_m}m/")
            and Path(image).stem.endswith(suffix)
        ]
        if len(candidates) != 1:
            raise ValueError("requested band/granule/resolution missing or ambiguous in SAFE XML")
        relative = candidates[0]
        path = (base / (relative if relative.endswith(".jp2") else relative + ".jp2")).resolve()
        if not path.is_relative_to(base):
            raise ValueError("SAFE image path escapes product directory")
        raw = read_raster_band(path, units="1", window=window, categorical=name == "SCL")
        valid = raw.valid & (raw.values != 0)
        if name == "SCL":
            if np.any(valid & ((raw.values < 1) | (raw.values > 11))):
                raise ValueError("unsupported SCL class")
            values = raw.values.copy()
        else:
            offset = offsets.get(name, 0.0 if baseline < 4 and not offsets else np.nan)
            if not np.isfinite(offset):
                raise ValueError("requested BOA band offset missing/nonfinite")
            values = (raw.values + offset) / quantification
        values[~valid] = np.nan
        result[name] = RasterBand(values, valid, raw.grid, "1", name == "SCL")
    return result


def _metadata(safe: Path) -> ET.Element:
    with (safe / "MTD_MSIL2A.xml").open("rb") as stream:
        text = stream.read(8 * 1024 * 1024 + 1)
    if len(text) > 8 * 1024 * 1024 or b"<!DOCTYPE" in text or b"<!ENTITY" in text:
        raise ValueError("SAFE metadata exceeds bound or declares XML entities")
    root = ET.fromstring(text)
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    if _one_text(root, "PRODUCT_TYPE") != "S2MSI2A":
        raise ValueError("SAFE product is not Sentinel-2 L2A")
    return root


def _one_text(root: ET.Element, tag: str) -> str:
    values = [node.text for node in root.iter(tag)]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"missing/ambiguous SAFE metadata: {tag}")
    return values[0]
