"""Shared explicit catalogue query conventions, not a provider registry."""

from datetime import datetime, timedelta


def geographic_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate a WGS84 catalogue rectangle before encoding a spatial query.

    Args:
        bbox: Ordered (west,south,east,north) degrees in one non-wrapping interval.

    Returns:
        The original geographic bounds tuple unchanged.

    Raises:
        ValueError: Coordinates are outside WGS84 limits or bounds do not increase.
    """
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be a non-wrapping WGS84 (west,south,east,north) rectangle")
    return bbox


def utc_interval(start: datetime, end: datetime) -> tuple[str, str]:
    """Encode a UTC catalogue time interval as ISO-8601 query bounds.

    Args:
        start: Timezone-aware UTC lower datetime bound.
        end: Timezone-aware UTC upper datetime bound, later than start.

    Returns:
        Start/end ISO-8601 strings with a trailing UTC Z marker.

    Raises:
        ValueError: Datetimes lack UTC offset zero or bounds fail to increase.
    """
    if start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0) or start >= end:
        raise ValueError("time interval requires increasing explicit UTC instants")
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")
