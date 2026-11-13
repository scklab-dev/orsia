"""Public function ownership and root export contracts."""

import importlib

import orsia_dispersion
import orsia_instruments
import orsia_radiative
import orsia_uncertainty
import pytest

FUNCTIONS = (
    (orsia_dispersion, "sampling", "sample_fields", True),
    (orsia_dispersion, "evaluation", "concentration", True),
    (orsia_dispersion, "evaluation", "layer_columns", True),
    (orsia_dispersion, "evaluation", "pixel_layer_moments", True),
    (orsia_dispersion, "puffs.population", "discretize_puff_emissions", True),
    (orsia_dispersion, "puffs.population", "split_puffs", True),
    (orsia_dispersion, "puffs.transport", "advance_puffs", True),
    (orsia_dispersion, "puffs.simulation", "simulate_puffs", True),
    (orsia_dispersion, "puffs.evaluation", "puff_concentration", True),
    (orsia_dispersion, "puffs.evaluation", "puff_layer_columns", True),
    (orsia_dispersion, "puffs.evaluation", "puff_path_columns", True),
    (orsia_dispersion, "particles", "simulate_particles", True),
    (orsia_dispersion, "particles", "discretize_particle_emissions", True),
    (orsia_radiative, "solar.source", "solar_irradiance", True),
    (orsia_radiative, "spectroscopy.line_tables", "line_set", True),
    (orsia_instruments, "photodetection.coefficients", "sample_optical_acceptance", True),
    (orsia_radiative, "spectroscopy.cross_sections", "resample_layer_cross_sections", True),
    (orsia_radiative, "transfer.formal", "propagation_coefficients", False),
    (orsia_uncertainty, "covariance", "whiten", True),
    (orsia_uncertainty, "covariance", "factor_covariance", True),
    (orsia_uncertainty, "covariance", "sample_gaussian", True),
    (orsia_uncertainty, "covariance", "covariance_product", True),
    (orsia_instruments, "optics.response", "imaging_optics", False),
    (orsia_instruments, "acquisition.schedule", "raster_exposure_bounds", True),
    (orsia_radiative, "optics.atmosphere", "atmospheric_optics", True),
    (orsia_radiative, "atmosphere.hydrostatic", "specific_humidity", True),
    (orsia_radiative, "atmosphere.layers", "geometric_height", True),
    (orsia_radiative, "spectral.conversions", "wavelength_to_wavenumber", True),
    (orsia_radiative, "spectral.conversions", "wavenumber_to_wavelength", True),
)


@pytest.mark.parametrize("root,module,name,exported", FUNCTIONS)
def test_function_is_owned_by_its_module_and_root_exports_match(root, module, name, exported):
    defining_module = importlib.import_module(f"{root.__name__}.{module}")
    function = getattr(defining_module, name)
    assert callable(function)
    assert function.__module__ == defining_module.__name__
    if exported:
        assert getattr(root, name) is function
        assert name in root.__all__
    else:
        assert name not in root.__all__
