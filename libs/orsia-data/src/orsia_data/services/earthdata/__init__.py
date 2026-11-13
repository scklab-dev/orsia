"""NASA Earthdata: CMR discovery and explicitly authenticated dissemination."""

from orsia_data.services.earthdata.auth import EarthdataAuth
from orsia_data.services.earthdata.client import EarthdataClient
from orsia_data.services.earthdata.cmr import (
    EarthdataProduct,
    download_earthdata,
    earthdata_links,
    search_earthdata,
)

__all__ = [
    "EarthdataAuth",
    "EarthdataClient",
    "EarthdataProduct",
    "download_earthdata",
    "earthdata_links",
    "search_earthdata",
]
