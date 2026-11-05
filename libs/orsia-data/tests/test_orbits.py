"""Offline orbital-element reading, parsing and selection with synthetic GP fixtures."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from math import tau
from typing import Any, cast

import pytest
from orsia_data import (
    orbits,
    parse_orbital_elements,
    read_orbital_elements,
    select_orbital_elements,
)
from orsia_geometry import SGP4MeanElements

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 2, tzinfo=UTC)
OMM = {
    "NORAD_CAT_ID": "25544",
    "OBJECT_ID": "1998-067A",
    "OBJECT_NAME": "synthetic",
    "EPOCH": "2026-01-01T00:00:00",
    "MEAN_MOTION": 15.5,
    "ECCENTRICITY": 0.001,
    "INCLINATION": 51.6,
    "RA_OF_ASC_NODE": 10.0,
    "ARG_OF_PERICENTER": 20.0,
    "MEAN_ANOMALY": 30.0,
    "BSTAR": 0.0001,
    "MEAN_MOTION_DOT": 0.00001,
    "MEAN_MOTION_DDOT": 0.0,
    "NATIVE_EXTENSION": {"edition": "synthetic"},
}


def tle_payload():
    # Synthetic fields based on the publicly specified fixed-column TLE layout.
    first = "1 25544U 98067A   26001.00000000  .00000000  00000-0  00000-0 0  999"
    second = "2 25544  51.6000  10.0000 0001000  20.0000  30.0000 15.5000000012345"

    def checksum(line):
        assert len(line) == 68
        return str(sum(int(c) if c.isdigit() else int(c == "-") for c in line) % 10)

    return (
        "0 synthetic\n" + first + checksum(first) + "\n" + second + checksum(second) + "\n"
    ).encode()


def test_root_and_module_export_the_same_orbital_functions():
    assert read_orbital_elements is orbits.read_orbital_elements
    assert parse_orbital_elements is orbits.parse_orbital_elements
    assert select_orbital_elements is orbits.select_orbital_elements


def test_read_omm_returns_immutable_scalars_and_fresh_lists(tmp_path):
    path = tmp_path / "native.json"
    path.write_text(json.dumps([OMM]))
    records = read_orbital_elements(path, format="omm")
    record = records[0]
    assert isinstance(record, SGP4MeanElements)
    assert record.norad_id == 25544 and record.epoch == EPOCH
    assert record.object_name == "synthetic"
    assert record.mean_motion_rad_s == pytest.approx(15.5 * tau / 86400)
    assert record.mean_motion_dot_rad_s2 == pytest.approx(0.00001 * 2 * tau / 86400**2)
    assert record.inclination_rad == pytest.approx(51.6 * tau / 360)
    assert record.bstar_per_earth_radius == 0.0001
    with pytest.raises(FrozenInstanceError):
        cast(Any, record).object_name = "changed"
    records.clear()
    assert read_orbital_elements(path, format="omm") == [record]
    assert parse_orbital_elements(json.dumps(OMM).encode(), format="omm") == [record]
    strings = {k: str(v) if isinstance(v, float) else v for k, v in OMM.items()}
    assert parse_orbital_elements(json.dumps(strings).encode(), format="omm") == [record]


def test_tle_checksums_name_and_epoch(tmp_path):
    payload = tle_payload()
    path = tmp_path / "native.tle"
    path.write_bytes(payload)
    (record,) = read_orbital_elements(path, format="tle")
    assert record.object_name == "synthetic" and record.norad_id == 25544
    assert record.epoch == EPOCH
    unnamed = b"\n".join(payload.splitlines()[1:])
    assert parse_orbital_elements(unnamed, format="tle")[0].object_name is None
    for bad in (b"", b"name\n", payload.replace(b"26001.", b"26000."), payload[:-2] + b"9\n"):
        with pytest.raises(ValueError, match="payload"):
            parse_orbital_elements(bad, format="tle")


def test_selection_latest_inclusive_bounds_ties_and_requested_order():
    (record,) = parse_orbital_elements(json.dumps(OMM).encode(), format="omm")
    second = replace(record, norad_id=123)
    old = replace(record, epoch=datetime(2025, 12, 31, tzinfo=UTC))
    newer = replace(record, epoch=END)
    tie = replace(record, object_name="equal epoch")
    values = [old, record, tie, newer, second]
    assert select_orbital_elements(
        values, object_ids=(123, 25544), element_epoch_interval_utc=(EPOCH, EPOCH)
    ) == [second, record]
    assert select_orbital_elements(
        values, object_ids=(25544,), element_epoch_interval_utc=(EPOCH, END)
    ) == [newer]
    assert values == [old, record, tie, newer, second]


@pytest.mark.parametrize(
    "ids,accepted",
    [
        ((), (EPOCH, END)),
        ((1, 1), (EPOCH, END)),
        ((0,), (EPOCH, END)),
        ((999,), (EPOCH, END)),
        ((25544,), (END, EPOCH)),
        ((25544,), (EPOCH.replace(tzinfo=None), END)),
        ((25544,), (END, END)),
    ],
)
def test_selection_failures(ids, accepted):
    records = parse_orbital_elements(json.dumps(OMM).encode(), format="omm")
    with pytest.raises(ValueError):
        select_orbital_elements(records, object_ids=ids, element_epoch_interval_utc=accepted)


@pytest.mark.parametrize(
    "change",
    [
        {"REF_FRAME": "GCRS"},
        {"TIME_SYSTEM": "TAI"},
        {"CENTER_NAME": "MARS"},
        {"MEAN_ELEMENT_THEORY": "KEPLER"},
        {"EPHEMERIS_TYPE": 4},
        {"MEAN_MOTION": 0},
        {"ECCENTRICITY": 1},
        {"INCLINATION": -1},
        {"BSTAR": float("nan")},
        {"NORAD_CAT_ID": "-1"},
        {"EPOCH": "not-a-date"},
        {"EPOCH": "2026-01-01T00:00:00+01:00"},
    ],
)
def test_invalid_omm_fields(change):
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(json.dumps(dict(OMM, **change)).encode(), format="omm")


@pytest.mark.parametrize("payload", [b"[]", b"null", b"{", b"[1]", b"{}", b"\xff"])
def test_invalid_payloads(payload):
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(payload, format="omm")


@pytest.mark.parametrize("epoch", [None, 0])
def test_malformed_epoch_is_value_error(epoch):
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(json.dumps(dict(OMM, EPOCH=epoch)).encode(), format="omm")


def test_fractional_object_identity_is_rejected():
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(json.dumps(dict(OMM, NORAD_CAT_ID=25544.5)).encode(), format="omm")


@pytest.mark.parametrize("field", ["999.0000", "bad-data"])
def test_invalid_tle_orbital_number_is_rejected(field):
    lines = tle_payload().decode().splitlines()
    line = lines[2][:8] + field + lines[2][16:68]
    checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line) % 10
    lines[2] = line + str(checksum)
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(("\n".join(lines) + "\n").encode(), format="tle")


def test_format_and_size_failures(tmp_path):
    path = tmp_path / "native.json"
    path.write_text(json.dumps(OMM))
    for limit in (0, 1):
        with pytest.raises(ValueError, match="byte"):
            read_orbital_elements(path, format="omm", max_bytes=limit)
    with pytest.raises(ValueError, match="format"):
        parse_orbital_elements(b"", format=cast(Any, "xml"))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("format", ["omm", "tle"])
def test_native_derivative_conversion_and_propagation_roundtrip(sign, format):
    from unittest.mock import patch

    import numpy as np
    import satkit
    from orsia_geometry.orbits import sgp4_propagator

    source = dict(
        OMM,
        MEAN_MOTION_DOT=sign * 0.123456789,
        MEAN_MOTION_DDOT=sign * 0.0000123456789,
        EPOCH="2026-01-01T00:00:00.123456Z",
    )
    if format == "omm":
        payload = json.dumps(source).encode()
        original = satkit.TLE.from_omm(cast(satkit.OMMDict, source))
    else:
        lines = tle_payload().decode().splitlines()
        # Fixed native TLE fields with signed nonzero n-dot/2 and n-ddot/6.
        sign_field = "-" if sign < 0 else " "
        first = lines[1][:33] + sign_field + ".12345678 " + sign_field + "12345-4" + lines[1][52:68]
        checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in first) % 10
        lines[1] = first + str(checksum)
        payload = "\n".join(lines).encode()
        original = satkit.TLE.from_lines(lines, check_checksum=True)[0]
    (elements,) = parse_orbital_elements(payload, format=format)
    assert elements.mean_motion_dot_rad_s2 == pytest.approx(
        original.mean_motion_dot * 2 * tau / 86400**2, rel=1e-15, abs=0
    )
    assert elements.mean_motion_ddot_rad_s3 == pytest.approx(
        original.mean_motion_dot_dot * 6 * tau / 86400**3, rel=1e-15, abs=0
    )
    # Isolate scalar/native conversion from frame resources, using actual native SGP4.
    captured = []
    native_sgp4 = satkit.sgp4

    def capture(native, instants, **kwargs):
        captured.append(native)
        return native_sgp4(native, instants, **kwargs)

    with (
        patch("satkit.sgp4", capture),
        patch("orsia_geometry.orbits.transform_states", side_effect=lambda t, p, v, **kw: (p, v)),
    ):
        trajectory = sgp4_propagator(elements, epoch=EPOCH, validity_s=(0, 86400))
        times = np.array([0.0, 1234.0, 86400.0])
        positions, velocities = trajectory(times)
    native = captured[0]
    assert native.mean_motion_dot == pytest.approx(original.mean_motion_dot, rel=1e-15)
    assert native.mean_motion_dot_dot == pytest.approx(original.mean_motion_dot_dot, rel=1e-15)
    expected_p, expected_v = native_sgp4(
        original,
        [EPOCH + timedelta(seconds=float(t)) for t in times],
    )
    assert elements.epoch == original.epoch.to_datetime()
    print(
        format,
        sign,
        "roundtrip max metres / m/s",
        float(np.max(np.abs(positions - expected_p))),
        float(np.max(np.abs(velocities - expected_v))),
    )
    np.testing.assert_allclose(positions, expected_p, atol=1e-6, rtol=0)
    np.testing.assert_allclose(velocities, expected_v, atol=1e-9, rtol=0)


def test_numeric_catalogue_domain_and_alpha5_constraint():
    for identifier in (99999, 100000, 2147483647):
        records = parse_orbital_elements(
            json.dumps(dict(OMM, NORAD_CAT_ID=identifier)).encode(), format="omm"
        )
        assert records[0].norad_id == identifier
    for identifier in (True, 2147483648):
        with pytest.raises(ValueError, match="payload"):
            parse_orbital_elements(
                json.dumps(dict(OMM, NORAD_CAT_ID=identifier)).encode(), format="omm"
            )
    # The Orsia reader deliberately retains its numeric-catalogue TLE scope.
    with pytest.raises(ValueError, match="payload"):
        parse_orbital_elements(tle_payload().replace(b"25544", b"A5544"), format="tle")


def test_selection_keeps_satkit_import_deferred(monkeypatch):
    import builtins

    records = parse_orbital_elements(json.dumps(OMM).encode(), format="omm")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "satkit" or name.startswith("satkit."):
            raise AssertionError("record selection must keep native imports deferred")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert (
        select_orbital_elements(
            records, object_ids=(25544,), element_epoch_interval_utc=(EPOCH, EPOCH)
        )
        == records
    )


def test_fresh_native_parser_and_propagation_preserve_free_threading():
    import subprocess
    import sys

    code = f"""
import os
import sys
from datetime import UTC, datetime
import satkit
from orsia_data import parse_orbital_elements
assert os.environ.get('PYTHON_GIL') != '0' and sys._xoptions.get('gil') != '0'
assert not sys._is_gil_enabled()
values = parse_orbital_elements({json.dumps(OMM).encode()!r}, format='omm')
assert values[0].norad_id == 25544
satkit.sgp4(satkit.TLE.from_omm({OMM!r}), [datetime(2026, 1, 1, tzinfo=UTC)])
assert not sys._is_gil_enabled()
"""
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
