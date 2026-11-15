"""Public function ownership and root export contracts."""

import importlib

import orsia_dispersion
import orsia_instruments
import orsia_inverse
import orsia_radiative
import orsia_uncertainty
import pytest

FUNCTIONS = (
    (orsia_instruments, "channels", "refine_channels", False),
    (orsia_instruments, "response.spatial", "form_detector_images", False),
    (orsia_instruments, "response.spatial", "form_detector_images_with_tangents", False),
    (orsia_inverse, "diagnostics", "linear_target_uncertainty", False),
    (orsia_dispersion, "transport", "validate_tensors", False),
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


@pytest.mark.parametrize(
    "module,old,new",
    [
        ("orsia_instruments.tolerances", "monte_carlo_native", "electron_error_statistics"),
        ("orsia_instruments.tolerances", "monte_carlo_dn", "sample_uncertain_readout"),
        ("orsia_instruments.channels", "exchange_channels", "refine_channels"),
        ("orsia_instruments.response.spatial", "sample_fourier_exposures", "form_detector_images"),
        (
            "orsia_instruments.response.spatial",
            "sample_fourier_exposure_tangents",
            "form_detector_images_with_tangents",
        ),
        (
            "orsia_radiative.atmosphere.levels",
            "integrate_log_linear_columns",
            "integrate_atmospheric_columns",
        ),
        ("orsia_dispersion.puffs.population", "puff_eigenvectors", "puff_principal_axes"),
        ("orsia_instruments.optics.response", "constant_or_default", "validation_scalar"),
        (
            "notebooks.support.notebook_ground_pixel",
            "physical_kernel",
            "scene_to_electron_response",
        ),
    ],
)
def test_role_based_names_replace_old_declarations_without_aliases(module, old, new):
    owner = importlib.import_module(module)
    assert not hasattr(owner, old)
    assert getattr(owner, new).__name__ == new
    assert getattr(owner, new).__module__ == module
    root = importlib.import_module(module.split(".")[0])
    assert not hasattr(root, old)


def test_scene_notebook_exposes_model_without_callback_factory():
    support = importlib.import_module("notebooks.support.spatial_scene")
    assert not hasattr(support, "make_callbacks")
    assert callable(support.SyntheticRadianceModel)


@pytest.mark.parametrize(
    "module",
    ["orsia_instruments.response.fourier", "orsia_inverse.design", "orsia_dispersion.forcing"],
)
def test_superseded_modules_are_absent(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize(
    "module,name",
    [
        ("orsia_instruments.channels", "ChannelSearchResult"),
        ("orsia_inverse.diagnostics", "LinearTargetUncertainty"),
    ],
)
def test_moved_return_types_have_the_same_owner_as_their_operations(module, name):
    assert getattr(importlib.import_module(module), name).__module__ == module
