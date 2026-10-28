"""Acquire selected native CAMS files using a caller-configured ADS client."""

from pathlib import Path
from typing import Literal

from ..acquisition import retrieve_dataset
from .cds import Retrieve

CamsProduct = Literal[
    "cams-global-greenhouse-gas-forecasts", "cams-global-atmospheric-composition-forecasts"
]


def download_cams(
    *,
    product: CamsProduct,
    request: dict[str, object],
    retrieve: Retrieve,
    destination: Path,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> Path:
    """Retrieve an explicit CAMS GHG/composition request using a borrowed client.

    Product, cycle, lead, fields, levels, area and format remain caller choices.
    No SDK construction, credential lookup, fallback or scientific conversion.
    Client exceptions propagate; size/digest are checked after SDK completion.
    Nested request values are copied. Publication is atomic/no-clobber; failures
    clean only the owned temporary directory. Parent directories may remain.

    Args:
        product: Supported CAMS greenhouse-gas or atmospheric-composition product.
        request: Explicit native SDK request fields, copied before transfer.
        retrieve: Borrowed SDK retrieval method receiving product, request and path.
        destination: New local destination path for the native file.
        max_bytes: Positive post-transfer size limit in bytes.
        expected_sha256: Optional expected SHA-256 digest for the downloaded file.

    Returns:
        Published destination path after size and optional digest verification.

    Raises:
        ValueError: Product, size bound or expected digest is invalid, or a
            completed transfer violates the requested size or digest constraint.
        FileExistsError: The destination already exists.
    """
    if product not in (
        "cams-global-greenhouse-gas-forecasts",
        "cams-global-atmospheric-composition-forecasts",
    ):
        raise ValueError("unsupported CAMS product")
    return retrieve_dataset(product, request, retrieve, destination, max_bytes, expected_sha256)
