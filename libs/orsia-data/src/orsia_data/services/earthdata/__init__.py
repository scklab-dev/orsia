"""NASA Earthdata: CMR discovery and explicitly authenticated dissemination."""

from .auth import EarthdataAuth
from .client import EarthdataClient
from .cmr import EarthdataProduct, download_earthdata, earthdata_links, search_earthdata

__all__ = [
    "EarthdataAuth",
    "EarthdataClient",
    "EarthdataProduct",
    "download_earthdata",
    "earthdata_links",
    "search_earthdata",
]
