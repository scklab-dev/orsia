"""Acquire selected native ERA5 files using a caller-configured CDS client."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ..acquisition import retrieve_dataset

Era5Product = Literal[
    "reanalysis-era5-pressure-levels", "reanalysis-era5-single-levels", "reanalysis-era5-complete"
]
Retrieve = Callable[[str, dict[str, object], str], object]


def download_era5(
    *,
    product: Era5Product,
    request: dict[str, object],
    retrieve: Retrieve,
    destination: Path,
    max_bytes: int = 128 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> Path:
    """Retrieve explicit native ERA5 selections, e.g. retrieve=client.retrieve.

    No defaults, mutation, edition inference or SDK construction. The borrowed
    client owns credentials, deadlines/retries and concurrency policy. Size/digest
    checks are post-transfer; publication is atomic and does not clobber. Only
    the owned temporary directory is removed on failure; parents may remain.

    Args:
        product: Supported ERA5 native product identifier.
        request: Explicit native SDK request fields, copied before retrieval.
        retrieve: Borrowed client retrieval callable receiving product, request and path.
        destination: New local file path for the native retrieval.
        max_bytes: Positive upper bound on completed file size, in bytes.
        expected_sha256: Optional expected SHA-256 digest for verification.

    Returns:
        Published local destination after transfer and integrity checks.

    Raises:
        ValueError: Product, size bound or expected digest is invalid, or a
            completed transfer fails size or digest verification.
        FileExistsError: Destination already exists.
    """
    if product not in (
        "reanalysis-era5-pressure-levels",
        "reanalysis-era5-single-levels",
        "reanalysis-era5-complete",
    ):
        raise ValueError("unsupported ERA5 product")
    return retrieve_dataset(product, request, retrieve, destination, max_bytes, expected_sha256)
