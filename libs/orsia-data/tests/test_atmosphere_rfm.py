"""Synthetic native RFM contracts; authentic files remain caller-owned."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from orsia_data import read_rfm_atmosphere
from orsia_data.atmosphere import read_rfm_atmosphere as domain_read
from orsia_data.atmosphere.rfm import read_rfm_atmosphere as module_read

PROFILE = """! Synthetic profile
3 ! native levels
*HGT [km]
0 1 2
*PRE [mb]
1000 900 800
*TEM [K]
290 280 270
*H2O [ppmv]
10000 2000 100
*CO2 [ppmv]
363 363 362
*END
"""


def test_si_values_ownership_and_exports(tmp_path: Path, monkeypatch) -> None:
    from orsia_data import acquisition

    def forbidden(*args, **kwargs):
        raise AssertionError("local reader attempted network access")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    assert domain_read is module_read is read_rfm_atmosphere
    path = tmp_path / "local.atm"
    path.write_text(PROFILE)
    height, pressure, temperature, gases = read_rfm_atmosphere(path)
    np.testing.assert_array_equal(height, [0, 1000, 2000])
    np.testing.assert_array_equal(pressure, [100000, 90000, 80000])
    np.testing.assert_array_equal(temperature, [290, 280, 270])
    np.testing.assert_array_equal(gases["CO2"], np.array([363, 363, 362]) * 1e-6)
    np.testing.assert_array_equal(gases["H2O"], np.array([10000, 2000, 100]) * 1e-6)
    arrays = [height, pressure, temperature, *gases.values()]
    for values in arrays:
        assert values.dtype == np.float64
        assert values.shape == (3,) and values.flags.owndata and values.flags.writeable
    for left, right in combinations(arrays, 2):
        assert not np.shares_memory(left, right)
    for values in arrays:
        values[:] = -123
    gases.clear()
    again = read_rfm_atmosphere(path)
    assert again[0][0] == 0 and again[1][0] == 100000 and again[2][0] == 290
    assert again[3]["CO2"][0] == 363e-6
    assert path.read_text() == PROFILE


def test_records_case_order_commas_and_fortran_exponents(tmp_path: Path) -> None:
    text = """\n  ! header
+3 arbitrary count description
*co2 description [PPMV]
 3.63d2,3.63D+2

 +362.
! between blocks
*tem
290, 280, 270
*pre [hPa]
1e3 9e2 8e2
*hgt
0. .1E1 +2
*f14 (CF4) [ppmv]
1 2 3
*end finished
! trailing comment

"""
    path = tmp_path / "local.atm"
    path.write_text(text)
    height, pressure, temperature, gases = read_rfm_atmosphere(path)
    np.testing.assert_array_equal(height, [0, 1000, 2000])
    np.testing.assert_array_equal(pressure, [100000, 90000, 80000])
    np.testing.assert_array_equal(temperature, [290, 280, 270])
    assert list(gases) == ["CO2", "F14"]
    np.testing.assert_array_equal(gases["F14"], np.array([1, 2, 3]) * 1e-6)


def test_single_level_and_preserved_nonclosing_composition(tmp_path: Path) -> None:
    path = tmp_path / "one.atm"
    path.write_text(
        "1\n*HGT [km]\n-0.1\n*PRE [mbar]\n1000\n*TEM\n290\n"
        "*N2\n800000\n*O2\n210000\n*H2O\n10000\n*END\n"
    )
    height, _, _, gases = read_rfm_atmosphere(path)
    assert height[0] == -100
    assert float(sum(gas[0] for gas in gases.values())) == pytest.approx(1.02)
    assert gases["N2"][0] == pytest.approx(0.8)


def test_geometry_only_and_fraction_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "local.atm"
    path.write_text(PROFILE[: PROFILE.index("*H2O")] + "*END\n")
    assert read_rfm_atmosphere(path)[3] == {}
    path.write_text(PROFILE.replace("10000 2000 100", "0 1000000 100"))
    np.testing.assert_array_equal(read_rfm_atmosphere(path)[3]["H2O"][:2], [0, 1])


@pytest.mark.parametrize(
    "old,new,match",
    [
        ("3 ! native levels", "0", "level count"),
        ("3 ! native levels", "-3", "level count"),
        ("3 ! native levels", "3.0", "level count"),
        ("3 ! native levels", "many", "level count"),
        ("0 1 2", "0 1", "numeric data"),
        ("0 1 2", "0 1 2 3", "too many"),
        ("0 1 2", "0 1 2\n3", "label"),
        ("0 1 2", "0 1 ! inline\n2", "numeric data"),
        ("0 1 2", "! inside block\n0 1 2", "numeric data"),
        ("0 1 2", "0\n! inside block\n1 2", "numeric data"),
        ("0 1 2", "3*0", "numeric data"),
        ("0 1 2", "0,,1,2", "numeric data"),
        ("0 1 2", "0,1,2,", "numeric data"),
        ("0 1 2", "0 1 2 /", "numeric data"),
        ("0 1 2", "0 1 nan", "numeric data"),
        ("0 1 2", "0 1 inf", "numeric data"),
        ("0 1 2", "0 1 1e999", "finite"),
        ("0 1 2", "0 1 1e308", "finite in SI"),
        ("0 1 2", "0 1 1", "altitude"),
        ("0 1 2", "2 1 0", "altitude"),
        ("0 1 2", "0 2 1", "altitude"),
        ("1000 900 800", "1000 1000 800", "pressure"),
        ("1000 900 800", "800 900 1000", "pressure"),
        ("1000 900 800", "1000 0 800", "pressure"),
        ("1000 900 800", "1000 900 -1", "pressure"),
        ("1000 900 800", "1e308 900 800", "finite in SI"),
        ("290 280 270", "0 280 270", "temperature"),
        ("290 280 270", "290 -1 270", "temperature"),
        ("290 280 270", "290 280 1e999", "finite"),
        ("10000 2000 100", "-1 2000 100", "ppmv"),
        ("10000 2000 100", "1000001 2000 100", "ppmv"),
        ("10000 2000 100", "1e999 2000 100", "finite"),
        ("*HGT [km]", "*HGT [m]", "units"),
        ("*PRE [mb]", "*PRE [Pa]", "units"),
        ("*TEM [K]", "*TEM [C]", "units"),
        ("*H2O [ppmv]", "*H2O [mol/mol]", "units"),
        ("*H2O [ppmv]", "*H2O [ppmv", "units"),
        ("*H2O [ppmv]", "*H2O ppmv]", "units"),
        ("*H2O [ppmv]", "*H2O [ppmv] [ppmv]", "units"),
        ("*H2O [ppmv]", "*H2O[ppmv]", "label"),
        ("*H2O [ppmv]", "*H2O(1) [ppmv]", "label"),
        ("*H2O [ppmv]", "*CO2(1)(2) [K]", "label"),
        ("*H2O [ppmv]", "*AEROSOL [km-1]", "label"),
        ("*H2O [ppmv]", "*UNKNOWN", "label"),
        ("*H2O [ppmv]", "*ABCDEFGHIJKLMNOPQRST_MORE", "label"),
        ("*H2O [ppmv]", "*hgt", "duplicate"),
        ("*CO2 [ppmv]", "*h2o", "duplicate"),
        ("*HGT [km]\n0 1 2\n", "", "requires"),
        ("*PRE [mb]\n1000 900 800\n", "", "requires"),
        ("*TEM [K]\n290 280 270\n", "", "requires"),
        ("*END", "", "END marker"),
        ("*END", "*END\n42", "after RFM END"),
        ("*END", "*END\n*END", "after RFM END"),
    ],
)
def test_invalid_profiles(tmp_path: Path, old: str, new: str, match: str) -> None:
    path = tmp_path / "bad.atm"
    path.write_text(PROFILE.replace(old, new))
    with pytest.raises(ValueError, match=match):
        read_rfm_atmosphere(path)


@pytest.mark.parametrize(
    "text,match", [("", "count"), ("! only\n", "count"), ("3\n*HGT\n0 1", "truncated")]
)
def test_empty_and_truncated(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "bad.atm"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        read_rfm_atmosphere(path)


def test_all_supported_species(tmp_path: Path) -> None:
    species = (
        "N2 O2 CO2 O3 H2O CH4 N2O HNO3 CO NO2 N2O5 ClO HOCl ClONO2 NO HNO4 HCN NH3 "
        "F11 F12 F14 F22 CCl4 COF2 H2O2 C2H2 C2H6 OCS SO2 SF6"
    ).split()
    assert len(species) == 30
    text = PROFILE[: PROFILE.index("*H2O")]
    text += "".join(f"*{name} [ppmv]\n0 0 0\n" for name in species) + "*END\n"
    path = tmp_path / "all.atm"
    path.write_text(text)
    assert set(read_rfm_atmosphere(path)[3]) == {name.upper() for name in species}


def test_io_errors(tmp_path: Path) -> None:
    path = tmp_path / "missing.atm"
    with pytest.raises(FileNotFoundError):
        read_rfm_atmosphere(path)
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_rfm_atmosphere(path)
