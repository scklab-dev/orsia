"""Scientific ownership and dependency direction contracts."""

import ast
import importlib
from pathlib import Path

import orsia_radiative as radiative
import pytest

OWNERS = {
    "atmosphere.layers": (
        "Absorber LayerAtmosphere geometric_height pressure_level_layer_means remap_layer_columns"
    ),
    "atmosphere.hydrostatic": (
        "gas_columns_from_fraction hydrostatic_atmosphere specific_humidity"
    ),
    "profiles": "LayerProfileBasis evaluate_profile",
    "surface": "SurfaceState",
    "geometry": "ObservationGeometry",
    "columns": "ColumnScene",
    "spectral.grid": "SpectralDomain SpectralGrid convert_spectral_grid wavenumber_grid",
    "spectral.conversions": "density_per_nm_to_per_cm1 reverse_spectral_points",
    "spectral.interpolation": "interpolate_point_density",
    "solar.source": "SolarSource solar_source solar_irradiance",
    "solar.spectrum": "SolarIrradianceSpectrum",
    "spectroscopy.lines": "Line LineSet line_absorption",
    "spectroscopy.line_tables": "SpectralLineTable PartitionFunctionTable line_set",
    "spectroscopy.cross_sections": (
        "LayerCrossSections resample_layer_cross_sections table_absorption"
    ),
    "spectroscopy.continuum": "WaterContinuumReference water_continuum_absorption",
    "spectroscopy.components": "AbsorptionComponent",
    "optics.absorption": "Absorption compose_absorption compose_absorption_on_grid",
    "optics.phases": "henyey_greenstein_phase rayleigh_phase",
    "optics.scattering": "ScatteringProperties AerosolOptics RayleighComponent AerosolComponent",
    "optics.atmosphere": "AtmosphericOptics compose_atmospheric_optics validate_atmospheric_optics",
}


@pytest.mark.parametrize("module,names", OWNERS.items())
def test_root_api_uses_defining_modules(module: str, names: str) -> None:
    canonical = importlib.import_module(f"orsia_radiative.{module}")
    for name in names.split():
        symbol = getattr(radiative, name)
        assert symbol is getattr(canonical, name)
        assert symbol.__module__ == canonical.__name__
        assert name in radiative.__all__


@pytest.mark.parametrize("package", ("spectral", "solar", "spectroscopy", "optics"))
def test_namespace_packages_keep_implementations_in_defining_modules(package: str) -> None:
    module = importlib.import_module(f"orsia_radiative.{package}")
    for owner, names in OWNERS.items():
        if owner.startswith(f"{package}."):
            for name in names.split():
                assert not hasattr(module, name)


def test_atmosphere_package_preserves_record_imports() -> None:
    from orsia_radiative import atmosphere
    from orsia_radiative.atmosphere import layers

    for name in ("Absorber", "AirBroadener", "LayerAtmosphere"):
        assert getattr(atmosphere, name) is getattr(layers, name)
        assert name in atmosphere.__all__


def test_column_remapping_public_surface() -> None:
    from orsia_radiative.atmosphere import hydrostatic, layers

    assert radiative.remap_layer_columns is layers.remap_layer_columns
    assert "remap_layer_columns" in radiative.__all__
    assert hydrostatic._pressure_layer_overlap is layers._pressure_layer_overlap


def test_atmosphere_dependencies_follow_construction_order() -> None:
    root = Path(radiative.__file__).parent / "atmosphere"
    allowed = {
        "layers": {"orsia_radiative.profiles"},
        "hydrostatic": {
            "orsia_radiative.atmosphere.layers",
        },
    }
    for module, dependencies in allowed.items():
        for node in ast.walk(ast.parse((root / f"{module}.py").read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0
                if node.module and node.module.startswith("orsia_"):
                    assert node.module in dependencies
                    for alias in node.names:
                        if alias.name.startswith("_"):
                            assert (module, node.module, alias.name) == (
                                "hydrostatic",
                                "orsia_radiative.atmosphere.layers",
                                "_pressure_layer_overlap",
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("orsia_"):
                        assert alias.name in dependencies
