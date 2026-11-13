"""Service-independent STAC documents, assets and explicit API traversal."""

from orsia_data.stac.assets import asset_href, select_assets
from orsia_data.stac.catalog import read_document
from orsia_data.stac.client import StacClient

__all__ = ["StacClient", "asset_href", "read_document", "select_assets"]
