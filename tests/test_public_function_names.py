"""Keep public function names aligned with their defining modules and root exports."""

import importlib

import orsia_instruments
import orsia_radiative
import orsia_uncertainty
import pytest

FUNCTIONS = (
    (orsia_radiative, "solar.source", "materialize_solar_irradiance", "solar_irradiance", True),
    (orsia_radiative, "spectroscopy.line_tables", "materialize_line_set", "line_set", True),
    (
        orsia_instruments,
        "integration",
        "prepare_optical_response",
        "sampled_optical_response",
        True,
    ),
    (
        orsia_radiative,
        "spectroscopy.cross_sections",
        "prepare_cross_sections",
        "cross_sections",
        True,
    ),
    (orsia_radiative, "transfer.formal", "prepare_propagation", "propagation_coefficients", False),
    (orsia_uncertainty, "covariance", "prepare_whitening", "whitener", True),
    (orsia_uncertainty, "covariance", "prepare_solve", "covariance_solver", True),
    (orsia_uncertainty, "operations", "prepare_gaussian", "gaussian_sampler", True),
    (orsia_uncertainty, "covariance", "apply_covariance", "covariance_product", True),
    (orsia_instruments, "optics.response", "assemble_imaging_optics", "imaging_optics", False),
    (
        orsia_instruments,
        "acquisition",
        "along_track_raster_exposure_bounds",
        "raster_exposure_bounds",
        True,
    ),
    (
        orsia_instruments,
        "imaging.response",
        "separable_gaussian_cross_track_response",
        "gaussian_cross_track_response",
        True,
    ),
    (
        orsia_radiative,
        "optics.atmosphere",
        "compose_atmospheric_optics_on_grid",
        "atmospheric_optics",
        True,
    ),
    (
        orsia_radiative,
        "atmosphere.hydrostatic",
        "specific_humidity_from_mixing_ratio",
        "specific_humidity",
        True,
    ),
    (
        orsia_radiative,
        "atmosphere.layers",
        "geopotential_to_geometric_height",
        "geometric_height",
        True,
    ),
    (
        orsia_radiative,
        "spectral.conversions",
        "wavelength_nm_to_wavenumber_cm1",
        "wavelength_to_wavenumber",
        True,
    ),
    (
        orsia_radiative,
        "spectral.conversions",
        "wavenumber_cm1_to_wavelength_nm",
        "wavenumber_to_wavelength",
        True,
    ),
)


@pytest.mark.parametrize("root,module,old,new,exported", FUNCTIONS)
def test_function_is_owned_by_its_module_and_root_exports_match(root, module, old, new, exported):
    defining_module = importlib.import_module(f"{root.__name__}.{module}")
    function = getattr(defining_module, new)
    assert function.__name__ == new
    assert function.__module__ == defining_module.__name__
    assert not hasattr(defining_module, old)
    assert not hasattr(root, old)
    assert old not in root.__all__
    if exported:
        assert getattr(root, new) is function
        assert new in root.__all__
    else:
        assert not hasattr(root, new)
        assert new not in root.__all__
