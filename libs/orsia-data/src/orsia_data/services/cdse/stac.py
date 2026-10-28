"""Construct a STAC catalogue client for the CDSE public endpoint."""

from ...http import JsonSession
from ...stac import StacClient

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"


def stac_client(*, session: JsonSession | None = None) -> StacClient:
    """Create a CDSE catalogue client for explicit STAC searches.

    Args:
        session: Optional borrowed JSON session; the client owns a new session
            when this is omitted.

    Returns:
        STAC client configured for the public CDSE catalogue endpoint.
    """
    return StacClient(STAC_URL, session=session)
