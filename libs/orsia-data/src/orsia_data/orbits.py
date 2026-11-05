"""Read, parse and select orbital elements from local GP OMM JSON or TLE bytes."""

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from orsia_geometry.orbits import SGP4MeanElements

if TYPE_CHECKING:
    import satkit


def read_orbital_elements(
    path: Path, *, format: Literal["omm", "tle"], max_bytes: int = 16 * 1024 * 1024
) -> list[SGP4MeanElements]:
    """Read a bounded local UTF-8 GP OMM JSON or numeric-catalogue TLE file.

    Args:
        path: Existing local element file, unchanged while read.
        format: Explicit 'omm' for GP JSON objects/arrays or 'tle' for 2/3-line sets.
        max_bytes: Positive bound on file bytes.

    Returns:
        A fresh list of immutable Geometry-owned scalar records in file order.
        Values use radians, SI seconds and timezone-aware UTC epochs. Source
        extensions and acquisition provenance remain in the original payload.
        Reading never authenticates, downloads or loads propagation auxiliaries.

    Raises:
        ValueError: Size, encoding, records or supported conventions are invalid.
        OSError: The local file cannot be read.
        ImportError: The required SatKit installation cannot load.
    """
    if max_bytes <= 0:
        raise ValueError("positive byte bound required")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("SGP4 element file exceeds byte bound")
    return parse_orbital_elements(payload, format=format)


def parse_orbital_elements(
    payload: bytes, *, format: Literal["omm", "tle"]
) -> list[SGP4MeanElements]:
    """Parse GP OMM JSON or TLE bytes into backend-independent SGP4 mean elements.

    Args:
        payload: Borrowed UTF-8 bytes; caller enforces acquisition size bounds.
        format: Explicit GP OMM JSON or numeric-catalogue TLE format.

    Returns:
        A fresh list of immutable scalar records in payload order. Private SatKit
        parsers interpret native values. GP defaults are Earth/TEME/UTC/SGP4;
        supplied conflicting conventions fail. TLE checksums are mandatory.
        XML/KVN OMM, Alpha-5 TLE and SGP4-XP are unsupported. Input is unchanged.

    Raises:
        ValueError: Payload is empty, malformed or outside supported conventions.
        ImportError: The required SatKit installation cannot load.
    """
    if format not in ("omm", "tle"):
        raise ValueError("format must be omm or tle")
    import satkit

    try:
        text = payload.decode("utf-8")
        if format == "omm":
            value = json.loads(text)
            records = value if isinstance(value, list) else [value]
            if not records or any(not isinstance(r, dict) for r in records):
                raise ValueError("nonempty GP OMM objects required")
            elements = []
            for record in records:
                _validate_omm(record)
                native = satkit.TLE.from_omm(cast("satkit.OMMDict", record))
                elements.append(_scalar_elements(native, name=record.get("OBJECT_NAME") or None))
            return elements
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        names = _validate_tle_lines(lines)
        return [
            _scalar_elements(native, name=name)
            for native, name in zip(
                satkit.TLE.from_lines(lines, check_checksum=True), names, strict=True
            )
        ]
    except ValueError, KeyError, TypeError, OverflowError, RuntimeError:
        raise ValueError("invalid or unsupported SGP4 element payload") from None


def select_orbital_elements(
    elements: Sequence[SGP4MeanElements],
    *,
    object_ids: tuple[int, ...],
    element_epoch_interval_utc: tuple[datetime, datetime],
) -> list[SGP4MeanElements]:
    """Select each requested object's latest mean-element set within the epoch interval.

    Args:
        elements: Borrowed decoded scalar records in source order.
        object_ids: Nonempty distinct positive NORAD identifiers, in desired output order.
        element_epoch_interval_utc: Inclusive ordered timezone-aware UTC epoch bounds.
            The caller chooses these bounds from its prediction-age policy.

    Returns:
        A fresh list in requested object order, sharing immutable records with input.
        The latest epoch inside the interval wins; equal epochs retain input order.
        Selection does no I/O, age arithmetic or propagation-accuracy assessment.

    Raises:
        ValueError: Identifiers/bounds are invalid or a requested object has no accepted set.
    """
    start, end = element_epoch_interval_utc
    if any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in (start, end)) or start > end:
        raise ValueError("ordered timezone-aware UTC interval required")
    if not object_ids or len(set(object_ids)) != len(object_ids) or any(i <= 0 for i in object_ids):
        raise ValueError("distinct positive object identifiers required")
    selected = []
    for object_id in object_ids:
        candidates = [r for r in elements if r.norad_id == object_id and start <= r.epoch <= end]
        if not candidates:
            raise ValueError("requested object has no element set inside accepted epoch bounds")
        selected.append(max(candidates, key=lambda r: r.epoch))
    return selected


def _scalar_elements(native: satkit.TLE, *, name: str | None = None) -> SGP4MeanElements:
    """Snapshot parsed values, undoing native derivative factors and angular units.

    Returns:
        Immutable scalar mean elements at microsecond UTC precision.

    Raises:
        ValueError: The parsed set uses SGP4-XP or has unsupported numerical support.
    """
    if native.ephem_type == 4:
        raise ValueError("SGP4-XP mean elements are unsupported")
    return SGP4MeanElements(
        norad_id=native.satnum,
        epoch=native.epoch.to_datetime().astimezone(UTC),
        eccentricity=native.eccen,
        inclination_rad=math.radians(native.inclination),
        raan_rad=math.radians(native.raan),
        argument_of_perigee_rad=math.radians(native.arg_of_perigee),
        mean_anomaly_rad=math.radians(native.mean_anomaly),
        mean_motion_rad_s=native.mean_motion * (math.tau / 86400),
        bstar_per_earth_radius=native.bstar,
        mean_motion_dot_rad_s2=native.mean_motion_dot * (2 * math.tau / 86400**2),
        mean_motion_ddot_rad_s3=native.mean_motion_dot_dot * (6 * math.tau / 86400**3),
        object_name=name,
    )


def _validate_omm(record: dict[str, Any]) -> None:
    """Enforce the narrower GP conventions before the permissive native parser.

    Raises:
        ValueError: Identity, epoch, conventions or required numbers are unsupported.
    """
    raw_id = record["NORAD_CAT_ID"]
    identifier = int(raw_id)
    exact = (
        raw_id.strip().lstrip("+").isascii() and raw_id.strip().lstrip("+").isdigit()
        if isinstance(raw_id, str)
        else not isinstance(raw_id, bool) and raw_id == identifier
    )
    if not exact or not 0 < identifier <= 2147483647:
        raise ValueError("object identity must be an exact positive integer")
    value = record["EPOCH"]
    if not isinstance(value, str):
        raise ValueError("element epoch must be a UTC string")
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if epoch.tzinfo is not None and epoch.utcoffset() != timedelta(0):
        raise ValueError("element epoch must be UTC")
    for field, convention in (
        ("TIME_SYSTEM", "UTC"),
        ("REF_FRAME", "TEME"),
        ("CENTER_NAME", "EARTH"),
        ("MEAN_ELEMENT_THEORY", "SGP4"),
    ):
        if record.get(field, convention) != convention:
            raise ValueError("unsupported OMM convention")
    for field in (
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "BSTAR",
        "MEAN_MOTION_DOT",
        "MEAN_MOTION_DDOT",
    ):
        if not math.isfinite(float(record[field])):
            raise ValueError("nonfinite mean element")


def _validate_tle_lines(lines: list[str]) -> list[str | None]:
    """Enforce numeric-catalogue fixed columns; delegate numerical parsing and checksums.

    Returns:
        Optional source names, preserving absence independently of native placeholders.

    Raises:
        ValueError: A set is empty, incomplete or outside the numeric-catalogue layout.
    """
    if not lines:
        raise ValueError("empty TLE selection")
    names = []
    index = 0
    while index < len(lines):
        name = None
        if not lines[index].startswith("1 "):
            name = lines[index].removeprefix("0 ")
            index += 1
        names.append(name)
        if index + 1 >= len(lines):
            raise ValueError("incomplete TLE")
        first, second = lines[index : index + 2]
        for prefix, line in (("1 ", first), ("2 ", second)):
            if (
                not line.isascii()
                or len(line) != 69
                or not line.startswith(prefix)
                or not line[2:7].strip().isdigit()
                or not line[-1].isdigit()
            ):
                raise ValueError("unsupported TLE line")
        if first[2:7] != second[2:7]:
            raise ValueError("TLE object mismatch")
        index += 2
    return names
