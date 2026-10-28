"""Compatibility imports for named services and common file publication."""

from .acquisition import retrieve_file
from .services.cdse.odata import download_cdse, search_sentinel2_l2a
from .services.earthdata.cmr import (
    EarthdataProduct,
    download_earthdata,
    earthdata_links,
    search_earthdata,
)

__all__ = [
    "EarthdataProduct",
    "download_cdse",
    "download_earthdata",
    "earthdata_links",
    "retrieve_file",
    "search_earthdata",
    "search_sentinel2_l2a",
]
