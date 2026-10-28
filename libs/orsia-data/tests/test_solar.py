from pathlib import Path

import numpy as np
from orsia_data import read_solar_spectrum_csv
from orsia_radiative import SolarIrradianceSpectrum


def test_read_solar_csv_preserves_optional_fields_and_owns_arrays(tmp_path: Path) -> None:
    path = tmp_path / "solar.csv"
    path.write_text(
        "# local fixture\n"
        "wavelength (nm),irradiance (W/m^2/nm),uncertainty (W/m^2/nm),bandwidth (nm)\n"
        "202.0,1.0,0.1,0.025\n203.0,2.0,0.2,0.025\n"
    )
    spectrum = read_solar_spectrum_csv(path)
    assert isinstance(spectrum, SolarIrradianceSpectrum)
    assert spectrum.wavelengths_nm.dtype == np.float64
    np.testing.assert_array_equal(spectrum.uncertainty_per_nm, [0.1, 0.2])
    np.testing.assert_array_equal(spectrum.bandwidth_nm, [0.025, 0.025])
    assert spectrum.irradiance_per_nm.flags.writeable
