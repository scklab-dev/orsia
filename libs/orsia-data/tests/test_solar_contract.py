"""Synthetic CSV and physical solar-table contracts; no provider access."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_data import read_solar_spectrum_csv
from orsia_radiative import SolarIrradianceSpectrum

HEADER = "wavelength (nm),irradiance (W/m^2/nm)\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# nothing\n\n",
        HEADER,
        "<html>error</html>",
        HEADER + "500\n",
        HEADER + "500,1,2\n",
        HEADER + ",1\n",
        HEADER + "500,\n",
        HEADER + "wrong,1\n",
        HEADER + "nan,1\n",
        HEADER + "500,inf\n",
        HEADER + "0,1\n",
        HEADER + "500,-1\n",
        HEADER + "500,1\n500,2\n",
        HEADER + "1000,1\n500,2\n",
        HEADER + '"500,1\n',
        HEADER.rstrip() + ",uncertainty (W/m^2/nm)\n500,1,-1\n",
        HEADER.rstrip() + ",uncertainty (W/m^2/nm),bandwidth (nm)\n500,1,0,0\n",
    ],
)
def test_reject_bad_csv(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_solar_spectrum_csv(path)


def test_io_errors_and_invalid_utf8(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_solar_spectrum_csv(tmp_path / "absent.csv")
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="could not read"):
        read_solar_spectrum_csv(path)


@pytest.mark.parametrize("optional", [False, True])
def test_irregular_grid_ownership_and_missing_fields(tmp_path: Path, optional: bool) -> None:
    path = tmp_path / "native.csv"
    path.write_text(
        HEADER.rstrip()
        + (",uncertainty (W/m^2/nm)" if optional else "")
        + "\n"
        + ("500,1,0\n750,2,0.1\n1000,3,0.2\n" if optional else "500,1\n750,2\n1000,3\n")
    )
    native = read_solar_spectrum_csv(path)
    assert (native.uncertainty_per_nm is not None) == optional
    assert native.bandwidth_nm is None
    for field in (native.wavelengths_nm, native.irradiance_per_nm, native.uncertainty_per_nm):
        if field is not None:
            assert field.flags.writeable
    x = np.array([500.0, 750.0, 1000.0])
    y = np.ones(3)
    native = SolarIrradianceSpectrum(x, y)
    x.fill(4)
    y.fill(5)
    np.testing.assert_array_equal(native.wavelengths_nm, [500, 750, 1000])
    np.testing.assert_array_equal(native.irradiance_per_nm, np.ones(3))


@pytest.mark.parametrize(
    "field,value",
    [
        ("wavelengths_nm", np.array([], dtype=np.float64)),
        ("wavelengths_nm", np.array([1000.0, 500.0])),
        ("irradiance_per_nm", np.ones(1)),
        ("irradiance_per_nm", np.ones((2, 1))),
        ("irradiance_per_nm", np.array([1.0, np.nan])),
        ("uncertainty_per_nm", np.ones(1)),
        ("bandwidth_nm", np.ones(1)),
    ],
)
def test_invalid_arrays(field: str, value: NDArray[np.float64]) -> None:
    values = dict(wavelengths_nm=np.array([500.0, 1000.0]), irradiance_per_nm=np.ones(2))
    values[field] = value
    with pytest.raises(ValueError):
        SolarIrradianceSpectrum(**values)
