"""Service-independent STAC documents, assets and explicit API traversal."""

from .assets import asset_href, select_assets
from .catalog import read_document
from .client import StacClient

__all__ = ["StacClient", "asset_href", "read_document", "select_assets"]
