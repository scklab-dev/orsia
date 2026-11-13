"""Copernicus Data Space: explicit OData/STAC discovery and token lifecycle."""

from orsia_data.services.cdse.auth import CdseAuth
from orsia_data.services.cdse.client import CdseClient
from orsia_data.services.cdse.odata import download_cdse, search_sentinel2_l2a

__all__ = ["CdseAuth", "CdseClient", "download_cdse", "search_sentinel2_l2a"]
