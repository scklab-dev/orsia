"""Canonical ownership and boundaries of the domain-oriented module layout."""

import ast
import importlib
from pathlib import Path

import orsia_instruments as instruments
import pytest

# Explicit inventory of the root contract, including additive capabilities.
ROOT_EXPORTS = {
    "acquisition": (
        "Acquisition ReadSchedule LinearTrajectory "
        "ground_fixed_pointing ground_scroll_pointing linear_scan_acquisition "
        "read_schedule rectangular_acquisition along_track_acquisition "
        "along_track_exposure_bounds along_track_read_schedule "
        "raster_exposure_bounds along_track_raster_reads "
        "along_track_strip_templates"
    ),
    "charge": "ExpectedElectrons combine_source_paths",
    "line_of_sight": "LineOfSightSamples PlatformState sample_lines_of_sight sphere_intersections",
    "detector": (
        "DetectorElectronics DetectorMaps DetectorOperatingPoints DetectorReadings "
        "correct_digital_readings digitize electron_covariance electron_moments "
        "pixel_quantum_efficiency sample_readout"
    ),
    "imaging.acquisition": (
        "ExposureSequence exposure_sequence image_acquisition regular_pixel_bounds"
    ),
    "imaging.charge": (
        "ImageIndices ImagePhotoelectrons ImageQuantumEfficiency "
        "combine_image_source_paths image_photoelectrons image_charge"
    ),
    "imaging.measurements": "ImageMeasurements",
    "imaging.response": (
        "GroundGrid SceneRadiance DetectorResponse InstrumentResponse "
        "detector_response instrument_response "
        "gaussian_cross_track_response gaussian_along_track_response "
        "response_electrons"
    ),
    "imaging.readout": "image_readout",
    "imaging.simulation": "simulate_images",
    "integration": (
        "QuantumEfficiency expected_electrons photoelectron_response sampled_optical_response "
        "line_of_sight_expected_electrons"
    ),
    "optics.response": "ImageResponse ImagingOptics OpticalResponse",
    "optics.psfs": "PointSpreadFunction gaussian_psf tabulated_psf",
    "optics.pixels": ("gaussian_pixel_fraction modulated_pixel_response tabulated_pixel_response"),
    "optics.filters": (
        "gaussian_filter lorentzian_filter super_gaussian_filter rectangular_filter "
        "longpass_filter shortpass_filter notch_filter lvf_centre tabulated_transmission "
        "filter_imager spatial_filter_imager lvf_imager"
    ),
    "optics.dispersion": "polynomial_image_map dispersive_response dispersive_imager",
    "optics.two_beam": "two_beam_ports interferometric_imager",
    "optics.cavities": "cavity_transmission cavity_imager",
    "scene": "RadianceField sample_radiance uniform_radiance_field",
    "line_of_sight_radiance": "RadianceCoverage LineOfSightRadiance sample_line_of_sight_radiance",
    "tolerances": (
        "NativeSignalBudget analogue_dn_budget central_difference monte_carlo_dn "
        "monte_carlo_native native_difference native_sensitivities native_tolerance_budget"
    ),
}


def test_root_export_inventory_matches_canonical_owners():
    names = [name for group in ROOT_EXPORTS.values() for name in group.split()]
    assert len(names) == len(set(names)) == len(instruments.__all__)
    assert set(instruments.__all__) == set(names)


@pytest.mark.parametrize(
    "old_name",
    [
        "PhysicalReads",
        "physical_reads",
        "along_track_physical_reads",
        "DatedObservationRays",
        "dated_ray_acquisition",
        "RayScene",
        "RayCoverage",
        "sample_ray_scene",
        "ray_expected_electrons",
    ],
)
def test_old_line_of_sight_exports_are_removed(old_name):
    assert not hasattr(instruments, old_name)
    assert old_name not in instruments.__all__


@pytest.mark.parametrize("old_module", ["dated_rays", "ray_scene"])
def test_old_line_of_sight_modules_are_removed(old_module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"orsia_instruments.{old_module}")


@pytest.mark.parametrize(
    "name",
    [
        "GroundResponse",
        "ground_response",
        "apply_ground_response",
        "FactorizedSceneRadiance",
        "apply_scene_response",
        "along_track_instrument_response",
        "optical_response",
    ],
)
def test_instrument_response_uses_renamed_root_contract(name):
    assert not hasattr(instruments, name)
    assert name not in instruments.__all__


@pytest.mark.parametrize("module,names", ROOT_EXPORTS.items())
def test_root_exports_are_the_canonical_objects(module, names):
    canonical = importlib.import_module(f"orsia_instruments.{module}")
    for name in names.split():
        value = getattr(canonical, name)
        assert getattr(instruments, name) is value
        assert value.__module__ == canonical.__name__


@pytest.mark.parametrize(
    "module,old_name",
    (
        ("imaging.response", "apply_instrument_response"),
        ("imaging.charge", "external_image_photoelectrons"),
    ),
)
def test_renamed_instrument_functions_have_only_canonical_names(module, old_name):
    canonical = importlib.import_module(f"orsia_instruments.{module}")
    assert not hasattr(canonical, old_name)
    assert not hasattr(instruments, old_name)
    assert old_name not in instruments.__all__


def test_quadrature_has_no_root_alias():
    assert not hasattr(instruments, "interval_quadrature")
    assert "interval_quadrature" not in instruments.__all__


@pytest.mark.parametrize(
    "module",
    [
        "arrays",
        "responses",
        "psfs",
        "tabulations",
        "image_charge",
        "image_sequence",
        "imaging_optics",
        "imaging.ground",
    ],
)
def test_removed_facades_are_not_importable(module):
    with pytest.raises(ModuleNotFoundError, match=f"orsia_instruments.{module}"):
        importlib.import_module(f"orsia_instruments.{module}")


def test_imaging_package_has_no_compatibility_exports():
    package = importlib.import_module("orsia_instruments.imaging")
    for names in ROOT_EXPORTS.values():
        for name in names.split():
            assert not hasattr(package, name)


def test_internal_imports_follow_domain_boundaries_without_cycles():
    root = Path(instruments.__file__).parent
    wrappers = {
        "arrays",
        "responses",
        "psfs",
        "tabulations",
        "image_charge",
        "image_sequence",
        "imaging_optics",
    }
    graph = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        name = ".".join(relative.parts)
        if path.name == "__init__.py" or name in wrappers:
            continue
        package = "orsia_instruments" + ("." + name.rsplit(".", 1)[0] if "." in name else "")
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            target = target.removeprefix("orsia_instruments.")
            assert target not in wrappers | {"orsia_instruments", "imaging", "optics"}
            if name.startswith("optics."):
                assert not target.startswith("imaging.")
            if name in {"charge", "detector"}:
                assert target != "integration" and not target.startswith(("imaging.", "optics."))
            if name == "imaging.readout":
                assert target != "integration" and not target.startswith("optics.")
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
