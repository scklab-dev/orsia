"""Public file-to-source-to-transfer path using an explicitly synthetic CSV."""

from pathlib import Path

import numpy as np
from orsia_data.solar import read_solar_spectrum_csv
from orsia_radiative import (
    Atmosphere,
    ObservationGeometry,
    RadiativeScene,
    SpectralGrid,
    SurfaceState,
    compose_atmospheric_optics,
    scale_spectral_irradiance,
    solar_irradiance,
    solar_source,
    transfer,
)


def test_offline_solar_transfer_and_linear_illumination(tmp_path: Path) -> None:
    (tmp_path / "native.csv").write_text(
        "wavelength (nm),irradiance (W/m^2/nm),uncertainty (W/m^2/nm),bandwidth (nm)\n"
        "500,1,0.1,0.025\n750,2,0.2,0.025\n1000,3,0.3,0.025\n"
    )
    native = read_solar_spectrum_csv(tmp_path / "native.csv")
    # Regular point sampling in the radiative coordinate, not wavelength bins.
    target = SpectralGrid("wavenumber_cm1", np.linspace(10000.0, 20000.0, 11))
    irradiance = solar_irradiance(native, target)
    source = solar_source(irradiance, direction=(30.0, 0.0))
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]),
        np.array([100000.0, 90000.0]),
        np.array([95000.0]),
        np.array([296.0]),
        np.array([2e29]),
        np.array([1.9e29]),
        np.array([1e28]),
    )
    scene = RadiativeScene(
        atmosphere, SurfaceState(target, reflectance=np.full(11, 0.2)), ObservationGeometry(10.0)
    )
    optics = compose_atmospheric_optics(atmosphere, source)
    nominal = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=source, optics=optics
    )
    np.testing.assert_allclose(
        nominal.values, 0.2 * np.cos(np.deg2rad(30.0)) / np.pi * irradiance.values
    )
    scaled = scale_spectral_irradiance(irradiance, distance_au=2.0, reference_distance_au=1.0)
    distant_source = solar_source(scaled, direction=(30.0, 0.0))
    distant = transfer.solar.reflected_radiance(
        scene, max_order=0, delta_m_moments=None, source=distant_source, optics=optics
    )
    np.testing.assert_array_equal(distant.values, nominal.values / 4)
    assert nominal.grid is target and distant.grid is target
    assert native.uncertainty_per_nm is not None and native.bandwidth_nm is not None
