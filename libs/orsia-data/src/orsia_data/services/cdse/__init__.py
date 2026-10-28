"""Copernicus Data Space: explicit OData/STAC discovery and token lifecycle."""

from .auth import CdseAuth
from .client import CdseClient
from .odata import download_cdse, search_sentinel2_l2a

__all__ = ["CdseAuth", "CdseClient", "download_cdse", "search_sentinel2_l2a"]
