"""Canonical ownership and boundaries of the domain-oriented module layout."""

import ast
import importlib
import inspect
from dataclasses import MISSING, fields
from pathlib import Path

import numpy as np
import orsia_instruments as instruments
import pytest

# Explicit inventory of the root contract, including additive capabilities.
ROOT_EXPORTS = {
    "optics.camera": (
        "ImagerSizing FilterStrip FilterCamera sized_focal_length filter_strip filter_camera "
        "FocalPlaneImager focal_plane_imager"
    ),
    "imaging.geolocation": (
        "SurfaceProjection SurfaceAcquisitionGrid "
        "frame_exposure_bounds camera_acquisition_grid ground_integration_bounds "
        "wgs84_surface_projection_from_camera spectrometer_acquisition_grid"
    ),
    "imaging.response.band": "NativeBandResponse group_native_bands",
    "imaging.response.joint": (
        "SpatialSpectralResponse spatial_spectral_response stack_frame_electrons"
    ),
    "imaging.radiometry": (
        "SurfaceRadiance SurfaceRadianceCoverage FrameMeasurements frame_response simulate_frames"
    ),
    "imaging.calibration": (
        "CalibratedFrames calibrate_frames CalibratedElectrons calibrate_frame_electrons"
    ),
    "imaging.reconstruction": "reconstruct_radiance_cube",
    "acquisition": (
        "Acquisition ReadSchedule LinearTrajectory RasterReadSchedule "
        "raster_read_schedule select_read_schedule select_raster_reads "
        "ground_fixed_pointing ground_scroll_pointing "
        "read_schedule rectangular_acquisition along_track_acquisition "
        "along_track_exposure_bounds along_track_read_schedule "
        "raster_exposure_bounds along_track_raster_reads "
        "along_track_strip_templates"
    ),
    "charge": "ExpectedElectrons combine_source_paths",
    "los": (
        "LineOfSightSamples sample_lines_of_sight "
        "RadianceCoverage LineOfSightRadiance sample_line_of_sight_radiance"
    ),
    "detector": (
        "DetectorElectronics DetectorProperties DetectorOperatingPoints DetectorReadings "
        "detector_operating_points "
        "correct_digital_readings digitize electron_covariance electron_moments "
        "pixel_quantum_efficiency sample_readout regular_pixel_bounds regular_detector"
    ),
    "imaging.acquisition": ("ExposureSequence exposure_sequence image_acquisition"),
    "imaging.photodetection": (
        "ImageIndices ImagePhotoelectrons ImageQuantumEfficiency "
        "combine_image_source_paths image_photoelectrons image_charge"
    ),
    "imaging.measurements": "ImageMeasurements",
    "imaging.response.instrument": (
        "GroundGrid SceneRadiance DetectorResponse InstrumentResponse "
        "detector_response instrument_response "
        "gaussian_cross_track_response gaussian_along_track_response "
        "response_electrons"
    ),
    "imaging.readout": "image_readout",
    "imaging.simulation": "simulate_images",
    "photoelectrons": (
        "OpticalAcceptance QuantumEfficiency expected_electrons photoelectron_response "
        "sample_optical_acceptance "
        "line_of_sight_expected_electrons"
    ),
    "optics.response": "ImageResponse ImagingOptics OpticalResponse",
    "optics.psfs": (
        "PointSpreadFunction GaussianPSF TabulatedPSF SpatialDensity gaussian_psf airy_psf"
    ),
    "optics.pixels": ("gaussian_pixel_fraction pixel_acceptance"),
    "optics.filters": (
        "gaussian_filter lorentzian_filter super_gaussian_filter rectangular_filter "
        "longpass_filter shortpass_filter notch_filter lvf_centre tabulated_transmission "
        "filter_imager spatial_filter_imager lvf_imager"
    ),
    "optics.dispersion": "polynomial_image_map dispersive_response dispersive_imager",
    "optics.spectrometer": "DispersiveSpectrometer dispersive_spectrometer",
    "optics.two_beam": "two_beam_ports interferometric_imager",
    "optics.cavities": "cavity_transmission cavity_imager",
    "scene": "RadianceField sample_radiance uniform_radiance_field",
    "tolerances": (
        "NativeSignalBudget analogue_dn_budget central_difference monte_carlo_dn "
        "monte_carlo_native native_difference native_sensitivities native_tolerance_budget"
    ),
}


def test_root_export_inventory_matches_canonical_owners():
    names = [name for group in ROOT_EXPORTS.values() for name in group.split()]
    assert len(names) == len(set(names)) == len(instruments.__all__)
    assert set(instruments.__all__) == set(names)


@pytest.mark.parametrize("module,names", ROOT_EXPORTS.items())
def test_root_exports_are_the_canonical_objects(module, names):
    canonical = importlib.import_module(f"orsia_instruments.{module}")
    for name in names.split():
        value = getattr(canonical, name)
        assert getattr(instruments, name) is value
        assert value.__module__ == canonical.__name__


def test_detector_properties_name_and_removed_export():
    canonical = importlib.import_module("orsia_instruments.detector")
    assert instruments.DetectorProperties is canonical.DetectorProperties
    assert instruments.DetectorProperties.__name__ == "DetectorProperties"
    assert not hasattr(instruments, "DetectorMaps")
    assert not hasattr(canonical, "DetectorMaps")
    assert "DetectorMaps" not in instruments.__all__


def test_detector_properties_has_ten_required_fields():
    names = (
        "pixel_bounds_m",
        "pitch_m",
        "quantum_efficiency",
        "dark_electrons_per_s",
        "read_noise_electrons",
        "full_well_electrons",
        "electrons_per_dn",
        "bias_dn",
        "nonlinearity",
        "adc_bits",
    )
    parameters = inspect.signature(instruments.DetectorProperties).parameters
    assert tuple(parameters) == names
    assert tuple(field.name for field in fields(instruments.DetectorProperties)) == names
    for parameter in parameters.values():
        assert parameter.default is inspect.Parameter.empty
    for field in fields(instruments.DetectorProperties):
        assert field.default is MISSING
        assert field.default_factory is MISSING


def test_detector_properties_requires_values_for_empty_construction():
    with pytest.raises(TypeError, match="missing 10 required positional arguments"):
        instruments.DetectorProperties()  # pyrefly: ignore[missing-argument]


@pytest.mark.parametrize(
    "omitted",
    [
        "pixel_bounds_m",
        "pitch_m",
        "quantum_efficiency",
        "dark_electrons_per_s",
        "read_noise_electrons",
        "full_well_electrons",
        "electrons_per_dn",
        "bias_dn",
        "nonlinearity",
        "adc_bits",
    ],
)
def test_detector_properties_requires_each_field(omitted):
    arguments = dict(
        pixel_bounds_m=np.array([[[-1.0, 1.0, -1.0, 1.0]]]),
        pitch_m=None,
        quantum_efficiency=0.7,
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        full_well_electrons=1e6,
        electrons_per_dn=2,
        bias_dn=100,
        nonlinearity=0,
        adc_bits=16,
    )
    del arguments[omitted]
    with pytest.raises(TypeError, match=f"required positional argument: '{omitted}'"):
        instruments.DetectorProperties(**arguments)


@pytest.mark.parametrize(
    "name",
    [
        "image_photoelectrons",
        "image_readout",
        "simulate_images",
        "calibrate_frames",
    ],
)
def test_detector_is_a_required_keyword_argument(name):
    function = getattr(instruments, name)
    parameters = inspect.signature(function).parameters
    detector = parameters["detector"]
    assert detector.kind is inspect.Parameter.KEYWORD_ONLY
    assert detector.default is inspect.Parameter.empty
    assert detector.annotation is instruments.DetectorProperties
    arguments = {
        parameter.name: object()
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty and parameter.name != "detector"
    }
    if name == "calibrate_frames":
        arguments["use_expectation"] = True
    with pytest.raises(TypeError, match="required keyword-only argument: 'detector'"):
        function(**arguments)


@pytest.mark.parametrize(
    "module", ["line_of_sight", "line_of_sight_radiance", "imaging.band_response"]
)
def test_obsolete_module_paths_are_unavailable(module):
    name = f"orsia_instruments.{module}"
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(name)
    assert error.value.name == name


def test_response_package_keeps_exports_in_concrete_modules():
    package = importlib.import_module("orsia_instruments.imaging.response")
    assert package.__file__ is not None
    assert Path(package.__file__).name == "__init__.py"
    assert not (Path(instruments.__file__).parent / "imaging/response.py").exists()
    for module in ("imaging.response.instrument", "imaging.response.band"):
        for name in ROOT_EXPORTS[module].split():
            assert not hasattr(package, name)


def test_internal_imports_follow_domain_boundaries_without_cycles():
    root = Path(instruments.__file__).parent
    graph = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        name = ".".join(relative.parts)
        if path.name == "__init__.py":
            continue
        package = "orsia_instruments" + ("." + name.rsplit(".", 1)[0] if "." in name else "")
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            target = target.removeprefix("orsia_instruments.")
            assert target not in {"orsia_instruments", "imaging", "imaging.response", "optics"}
            if name.startswith("optics."):
                assert not target.startswith("imaging.")
            if name in {"charge", "detector"}:
                assert target != "photoelectrons" and not target.startswith(("imaging.", "optics."))
            if name == "imaging.readout":
                assert target != "photoelectrons" and not target.startswith("optics.")
            dependencies.add(target)
        graph[name] = dependencies

    visited = set()

    def visit(name, active):
        assert name not in active, f"import cycle at {name}"
        if name not in visited:
            for target in graph.get(name, ()):
                visit(target, active | {name})
            visited.add(name)

    for name in graph:
        visit(name, set())
