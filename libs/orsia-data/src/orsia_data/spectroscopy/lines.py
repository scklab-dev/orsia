"""Explicit local HITRAN2004+ and AER/LNFL line readers, without materialisation."""

from math import isfinite
from pathlib import Path
from typing import Literal

from orsia_radiative import Line, LineDatabase


def read_hitran_lines(
    path: Path,
    *,
    molecule_id: int,
    isotopologue_id: int,
    interval_cm1: tuple[float, float],
) -> LineDatabase:
    """Select HITRAN2004+ 160-character records into a declared covered database.

    Interval [cm-1] is caller-declared completeness, not inferred from line centres.
    Intensities [cm/molecule] already include isotope abundance. No downloads.

    Args:
        path: Local HITRAN2004+ fixed-width text file.
        molecule_id: Selected positive HITRAN molecule identifier.
        isotopologue_id: Selected local HITRAN isotopologue identifier.
        interval_cm1: Inclusive line-centre coverage interval in cm^-1.

    Returns:
        Line database sorted by centre, with the declared identity and coverage;
        an empty covered selection remains a valid database.

    Raises:
        ValueError: If the selected identity, coverage or source records are invalid.
    """
    return _read_lines(path, molecule_id, isotopologue_id, interval_cm1, "hitran160", "reject")


def read_aer_lines(
    path: Path,
    *,
    molecule_id: int,
    isotopologue_id: int,
    interval_cm1: tuple[float, float],
    coupling: Literal["reject", "ignore"] = "reject",
) -> LineDatabase:
    """Select AER/LNFL lines, consuming mandatory negative-IFLAG coefficient rows.

    Default reject disallows selected coupled lines; explicit ignore means an
    independent-line approximation, not full line mixing. UTF-8 '>' headers and
    exact '%%%%%%%%' separators are skipped. Empty covered selections are valid.
    Caller declares interval completeness and retains any processing lineage.

    Args:
        path: Local AER/LNFL line text file.
        molecule_id: Selected positive HITRAN molecule identifier.
        isotopologue_id: Selected local isotopologue identifier.
        interval_cm1: Inclusive caller-declared line-centre coverage in cm^-1.
        coupling: Reject selected coupled lines or explicitly ignore their coupling.

    Returns:
        Line database sorted by centre, including an empty covered selection.

    Raises:
        ValueError: If the coupling policy, identity, coverage or source records are
            invalid, or selected coupled lines use the reject policy.
    """
    return _read_lines(path, molecule_id, isotopologue_id, interval_cm1, "aer", coupling)


def _read_lines(
    path: Path,
    molecule_id: int,
    isotopologue_id: int,
    interval_cm1: tuple[float, float],
    format: Literal["hitran160", "aer"],
    coupling: Literal["reject", "ignore"],
) -> LineDatabase:
    if coupling not in ("reject", "ignore"):
        raise ValueError("unsupported line coupling policy")
    LineDatabase((), molecule_id, isotopologue_id, interval_cm1)
    lines: list[Line] = []
    with path.open(encoding="utf-8" if format == "aer" else "ascii") as stream:
        records = enumerate(stream, 1)
        for number, raw in records:
            record = raw.rstrip("\r\n")
            if not record.strip():
                continue
            if format == "aer" and (record.startswith(">") or record.strip() == "%%%%%%%%"):
                continue
            if len(record) < (160 if format == "hitran160" else 67):
                raise ValueError(f"truncated line record at row {number}")
            try:
                molecule = int(record[:2])
                isotope = _isotope_id(record[2])
                centre = _number(record[3:15])
                if molecule <= 0 or centre <= 0:
                    raise ValueError("line molecule identity and centre must be positive")
                flag = (
                    int(record[98:100].strip() or "0")
                    if format == "aer" and len(record) >= 100
                    else 0
                )
                if flag < 0:
                    _, coupled_row = next(records, (0, ""))
                    if not coupled_row.strip() or coupled_row.startswith(">"):
                        raise ValueError("AER coupled line is missing its coefficient row")
                    _validate_coupling_record(coupled_row.rstrip("\r\n"))
                if molecule != molecule_id or isotope != isotopologue_id:
                    continue
                if not interval_cm1[0] <= centre <= interval_cm1[1]:
                    continue
                if flag and coupling == "reject":
                    raise ValueError(
                        "AER line coupling requires explicit ignore or another evaluator"
                    )
                lines.append(
                    Line(
                        centre,
                        _number(record[15:25]),
                        _number(record[45:55]),
                        _number(record[35:40]),
                        _number(record[40:45]),
                        _number(record[55:59]),
                        _number(record[59:67]),
                    )
                )
            except ValueError as error:
                raise ValueError(f"invalid line record at row {number}: {error}") from error
    return LineDatabase(
        tuple(sorted(lines, key=lambda line: line.centre_cm1)),
        molecule_id,
        isotopologue_id,
        interval_cm1,
    )


def _number(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not isfinite(value):
        raise ValueError("line record numbers must be finite")
    return value


def _validate_coupling_record(record: str) -> None:
    if len(record) < 100:
        raise ValueError("truncated AER line-coupling coefficient row")
    int(record[:2])
    int(record[98:100])
    for lo, hi in ((2, 15), (15, 26), (26, 39), (39, 50), (50, 63), (63, 74), (74, 87), (87, 98)):
        _number(record[lo:hi])


def _isotope_id(token: str) -> int:
    if token == "0":
        return 10
    if token in "123456789":
        return int(token)
    if "A" <= token <= "Z":
        return ord(token) - ord("A") + 11
    raise ValueError("unsupported local isotope token")
