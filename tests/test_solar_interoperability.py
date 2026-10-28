"""Synthetic CSV-to-irradiance interoperability, not authentic HSRS qualification."""

import csv
import sys
from pathlib import Path

import numpy as np
from orsia_data.solar import read_solar_spectrum_csv
from orsia_radiative import SpectralGrid, solar_irradiance


def test_solar_csv_rows_jacobian_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    path.write_text("# synthetic\n\nwavelength (nm),irradiance (W/m^2/nm)\n500,1\n750,2\n1000,3\n")
    native = read_solar_spectrum_csv(path)
    with path.open(newline="") as stream:
        rows = list(
            csv.reader(line for line in stream if line.strip() and not line.startswith("#"))
        )
    raw = np.array([[float(cell) for cell in row] for row in rows[1:]])
    np.testing.assert_array_equal(native.wavelengths_nm, raw[:, 0])
    np.testing.assert_array_equal(native.irradiance_per_nm, raw[:, 1])
    target = SpectralGrid("wavenumber_cm1", np.arange(10000.0, 20001.0, 1000.0))
    actual = solar_irradiance(native, target)
    # Independent density Jacobian and linear interpolation in inverse centimetres.
    expected = np.interp(
        target.points, (1e7 / raw[:, 0])[::-1], (raw[:, 1] * raw[:, 0] ** 2 / 1e7)[::-1]
    )
    np.testing.assert_allclose(actual.values, expected, rtol=1e-14)
    assert actual.values.size == 11 and np.all(np.isfinite(actual.values))
    assert np.all(actual.values >= 0)
    replay = solar_irradiance(read_solar_spectrum_csv(path), target)
    np.testing.assert_array_equal(replay.values, actual.values)
    assert not sys._is_gil_enabled()
