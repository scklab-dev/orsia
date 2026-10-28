"""Local decoding of the molecular-profile subset of RFM atmospheric text files."""

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_GASES = frozenset(
    "N2 O2 CO2 O3 H2O CH4 N2O HNO3 CO NO2 N2O5 CLO HOCL CLONO2 NO HNO4 HCN NH3 "
    "F11 F12 F14 F22 CCL4 COF2 H2O2 C2H2 C2H6 OCS SO2 SF6".split()
)
_UNITS = {"HGT": {"km"}, "PRE": {"mb", "mbar", "hpa"}, "TEM": {"k"}}
_REAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z")


def read_rfm_atmosphere(
    path: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
]:
    """Read native point levels as (altitude_m, pressure_pa, temperature_k, gases).

    Each array is a fresh, independently owned, writable float64 vector of shape
    (n_levels,). Decimal text is decoded in float64, then converted km -> m,
    hPa/mb/mbar -> Pa and ppmv -> mol/mol total air, including water. Gas keys
    are uppercase RFM species labels. Native order, concentrations and levels
    are retained. The caller owns subsequent mutations and physical processing.

    Supported UTF-8 text has NLEV >= 1, HGT/PRE/TEM exactly once, zero or more
    of the 30 molecular species in the MIPAS 2001 collection, and a mandatory
    END marker. Any field order is accepted. Labels are case-insensitive, start
    with '*', and end at whitespace; supported labels fit within 20 characters.
    Label suffixes are descriptive, with optional bracketed standard units.
    Absent unit annotations use the RFM standard units above. Explicit annotations
    must agree. Single-point homogeneous profiles are supported.

    Blank records and full-line '!' comments are accepted between blocks and
    after END; comments within a value block raise ValueError. NLEV's trailing
    text is ignored. Values span records with whitespace or single commas as
    separators and decimal E/D exponents. Repetition, null fields, slash
    termination, inline data comments and data records following END raise ValueError.
    Unknown, aerosol, isotopologue and non-LTE labels also raise ValueError.

    All values must be finite, altitude strictly increasing, pressure positive
    and strictly decreasing, temperature positive, and each gas fraction in
    [0, 1]. Species sums are retained even above one: these historical profiles
    are independent tabulations. Composition closure, interpolation, integration
    and scaling belong to subsequent caller-selected operations. Metadata and
    source identity remain caller-owned. Keep the source stable during reading;
    filesystem and text-decoding errors propagate, invalid profiles raise
    ValueError. Each call closes its local file before returning.

    Args:
        path: Local UTF-8 RFM profile text file.

    Returns:
        Altitude in metres, pressure in pascals, temperature in kelvin and uppercase
        gas fractions in mol/mol total air, all at the native point levels.

    Raises:
        ValueError: If the text structure, units or physical support are invalid.
    """
    with path.open(encoding="utf-8") as stream:
        records = iter(enumerate(stream, start=1))
        header = _next_record(records)
        if header is None:
            raise ValueError("missing RFM level count")
        token = header[1].split()[0]
        if re.fullmatch(r"\+?\d+", token) is None or int(token) < 1:
            raise ValueError("RFM level count must be a positive integer")
        count = int(token)
        fields: dict[str, NDArray[np.float64]] = {}
        while (record := _next_record(records)) is not None:
            line, text = record
            name = _label(text, line)
            if name == "END":
                if _next_record(records) is not None:
                    raise ValueError("unexpected data after RFM END marker")
                break
            if name in fields:
                raise ValueError(f"duplicate RFM field {name} at line {line}")
            fields[name] = _values(records, count, name)
        else:
            raise ValueError("missing RFM END marker")

    if not {"HGT", "PRE", "TEM"}.issubset(fields):
        raise ValueError("RFM atmosphere requires HGT, PRE and TEM")
    height, pressure, temperature = (fields.pop(name) for name in ("HGT", "PRE", "TEM"))
    # Validate before and after conversion, including potential SI overflow.
    with np.errstate(over="ignore"):
        height *= 1000.0
        pressure *= 100.0
    for name, values in (("HGT", height), ("PRE", pressure), ("TEM", temperature)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"RFM {name} must be finite in SI units")
    if np.any(height[1:] <= height[:-1]):
        raise ValueError("RFM altitude must be strictly increasing")
    if np.any(pressure <= 0) or np.any(pressure[1:] >= pressure[:-1]):
        raise ValueError("RFM pressure must be positive and strictly decreasing")
    if np.any(temperature <= 0):
        raise ValueError("RFM temperature must be positive")
    for name, values in fields.items():
        if np.any(values < 0) or np.any(values > 1e6):
            raise ValueError(f"RFM gas {name} must be within [0, 1e6] ppmv")
        values *= 1e-6
    return height, pressure, temperature, fields


def _next_record(records: Iterator[tuple[int, str]]) -> tuple[int, str] | None:
    for line, raw in records:
        text = raw.strip()
        if text and not text.startswith("!"):
            return line, text
    return None


def _label(text: str, line: int) -> str:
    token, *suffix = text.split(maxsplit=1)
    name = token[1:].upper()
    if not token.startswith("*") or name not in _GASES | {"HGT", "PRE", "TEM", "END"}:
        raise ValueError(f"unsupported RFM label {token!r} at line {line}")
    if suffix and name != "END":
        annotation = suffix[0]
        units = re.findall(r"\[([^\[\]]*)\]", annotation)
        if annotation.count("[") != len(units) or annotation.count("]") != len(units):
            raise ValueError(f"malformed RFM units for {name} at line {line}")
        if len(units) > 1 or (units and units[0].strip().lower() not in _UNITS.get(name, {"ppmv"})):
            raise ValueError(f"unsupported RFM units for {name} at line {line}")
    return name


def _values(records: Iterator[tuple[int, str]], count: int, name: str) -> NDArray[np.float64]:
    values: list[float] = []
    for line, raw in records:
        text = raw.strip()
        if not text:
            continue
        # Split commas first to reject null fields rather than silently dropping them.
        parts = text.split(",")
        tokens = text.replace(",", " ").split()
        if any(not part.strip() for part in parts) or any(
            _REAL.fullmatch(token) is None for token in tokens
        ):
            raise ValueError(f"invalid RFM numeric data for {name} at line {line}")
        values.extend(float(token.replace("D", "E").replace("d", "e")) for token in tokens)
        if len(values) > count:
            raise ValueError(f"too many values for RFM {name} at line {line}")
        if len(values) == count:
            array = np.array(values, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"RFM {name} must be finite")
            return array
    raise ValueError(f"truncated RFM {name}: expected {count} values, got {len(values)}")
