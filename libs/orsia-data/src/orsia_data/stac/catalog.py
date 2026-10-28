"""Read and inspect local STAC documents while preserving native metadata."""

import json
from pathlib import Path
from typing import Any


def read_document(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    """Read a bounded Catalog, Collection, Item or ItemCollection; retain extensions.

    Structural envelope checks only, not full STAC/extension schema validation.
    Relative hrefs remain unchanged; pass the source URI to asset_href explicitly.

    Args:
        path: Existing local JSON document path.
        max_bytes: Positive maximum document size, in bytes.

    Returns:
        Parsed STAC document with its original extension fields and relative links.

    Raises:
        ValueError: Byte bound or STAC envelope is invalid or the file exceeds
            the bound.
        OSError: The local document cannot be read.
        json.JSONDecodeError: File content is invalid JSON.
    """
    if max_bytes <= 0:
        raise ValueError("positive byte bound required")
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("STAC document exceeds byte bound")
    document = json.loads(content)
    validate_document(document)
    return document


def validate_document(document: dict[str, Any]) -> None:
    """Check the supported STAC document envelope for local inspection.

    Args:
        document: Parsed Catalog, Collection, Item or ItemCollection mapping.

    Raises:
        ValueError: Type, links, features or STAC version violate the envelope.
    """
    if not isinstance(document, dict) or document.get("type") not in (
        "Catalog",
        "Collection",
        "Feature",
        "FeatureCollection",
    ):
        raise ValueError("unsupported STAC document envelope")
    if not isinstance(document.get("links", []), list):
        raise ValueError("STAC links must be a list")
    if document["type"] == "FeatureCollection":
        if not isinstance(document.get("features"), list):
            raise ValueError("STAC ItemCollection requires features")
    elif not document.get("stac_version"):
        raise ValueError("STAC document requires stac_version")
