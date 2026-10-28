"""Select STAC asset metadata and resolve asset URLs for explicit access."""

from copy import deepcopy
from typing import Any
from urllib.parse import urljoin


def select_assets(
    item: dict[str, Any], *, role: str | None = None, media_type: str | None = None
) -> dict[str, Any]:
    """Return owned copies of all matching assets, retaining keys and extensions.

    No ranking, band inference or automatic choice when several assets match.
    Missing roles/types do not match an explicit filter.

    Args:
        item: Native STAC Item mapping containing an optional ``assets`` mapping.
        role: Optional asset role required among each asset's roles.
        media_type: Optional exact asset ``type`` value to select.

    Returns:
        Mapping of matching original asset keys to independent metadata copies.
    """
    return {
        key: deepcopy(asset)
        for key, asset in item.get("assets", {}).items()
        if (role is None or role in asset.get("roles", []))
        and (media_type is None or asset.get("type") == media_type)
    }


def asset_href(item: dict[str, Any], key: str, *, document_url: str) -> str:
    """Resolve one selected asset against its actual document URI, without IO.

    Local file, HTTPS and non-HTTP asset schemes remain possible. This is URI
    resolution, not authorization to read a path or send credentials to a host.

    Args:
        item: Native Item mapping with the selected asset's ``href``.
        key: Asset key under the Item's ``assets`` mapping.
        document_url: Actual URI of the Item document used as the relative base.

    Returns:
        URL or path reference resolved against the document URI.

    Raises:
        KeyError: The selected asset or its href is missing from the Item.
    """
    return urljoin(document_url, item["assets"][key]["href"])
