"""Scientific ownership and deliberately removed submodule contracts."""

import ast
import importlib
from pathlib import Path

import orsia_radiative as radiative
import pytest

OWNERS = {
    "atmosphere.models": "Absorber Atmosphere",
    "atmosphere.layers": ("geometric_height pressure_level_layer_means remap_layer_columns"),
    "atmosphere.hydrostatic": (
        "gas_columns_from_fraction hydrostatic_atmosphere specific_humidity"
    ),
    "profiles": "LayerProfileBasis evaluate_profile",
    "surface": "SurfaceState",
    "geometry": "ObservationGeometry",
    "scene": "RadiativeScene",
    "spectral.grid": "SpectralDomain SpectralGrid convert_spectral_grid wavenumber_grid",
    "spectral.conversions": "density_per_nm_to_per_cm1 reverse_spectral_points",
    "spectral.interpolation": "interpolate_point_density",
    "solar.source": "SolarSource solar_source solar_irradiance",
    "solar.spectrum": "SolarIrradianceSpectrum",
    "spectroscopy.lines": "Line LineSet line_absorption",
    "spectroscopy.line_tables": "LineDatabase PartitionFunctionTable line_set",
    "spectroscopy.cross_sections": "CrossSectionField PreparedCrossSections table_absorption",
    "spectroscopy.continuum": "WaterContinuumReference water_continuum_absorption",
    "spectroscopy.components": "AbsorptionComponent",
    "optics.absorption": "Absorption compose_absorption compose_absorption_on_grid",
    "optics.phases": "henyey_greenstein_phase rayleigh_phase",
    "optics.scattering": "ScatteringProperties AerosolOptics RayleighComponent AerosolComponent",
    "optics.atmosphere": "AtmosphericOptics compose_atmospheric_optics validate_atmospheric_optics",
}
REMOVED = (
    "atmosphere.column",
    "atmospheric_columns",
    "sources",
    "arrays",
    "state",
    "optical",
    "absorption",
    "continuum",
    "line_tables",
    "spectral_grid",
    "solar_spectrum",
)


@pytest.mark.parametrize("module,names", OWNERS.items())
def test_root_api_uses_defining_modules(module: str, names: str) -> None:
    canonical = importlib.import_module(f"orsia_radiative.{module}")
    for name in names.split():
        symbol = getattr(radiative, name)
        assert symbol is getattr(canonical, name)
        assert symbol.__module__ == canonical.__name__
        assert name in radiative.__all__


@pytest.mark.parametrize(
    "module,old_name",
    (
        ("transfer.solar", "series_of_scatterers"),
        ("atmosphere.hydrostatic", "materialize_hydrostatic_atmosphere"),
    ),
)
def test_renamed_radiative_functions_have_only_canonical_names(module: str, old_name: str) -> None:
    canonical = importlib.import_module(f"orsia_radiative.{module}")
    assert not hasattr(canonical, old_name)
    assert not hasattr(radiative, old_name)
    assert old_name not in radiative.__all__


@pytest.mark.parametrize("module", REMOVED)
def test_removed_modules_are_not_importable(module: str) -> None:
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(f"orsia_radiative.{module}")
    assert error.value.name == f"orsia_radiative.{module}"


@pytest.mark.parametrize("package", ("spectral", "solar", "spectroscopy", "optics"))
def test_packages_do_not_reexport_old_symbols(package: str) -> None:
    module = importlib.import_module(f"orsia_radiative.{package}")
    for owner, names in OWNERS.items():
        if owner.startswith(f"{package}."):
            for name in names.split():
                assert not hasattr(module, name)


def test_modules_do_not_depend_on_removed_state_or_arrays() -> None:
    root = Path(radiative.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module != "orsia_radiative.state"
                assert node.module != "orsia_radiative.arrays"


def test_atmosphere_package_preserves_record_imports() -> None:
    from orsia_radiative import atmosphere
    from orsia_radiative.atmosphere import models

    for name in ("Absorber", "AirBroadener", "Atmosphere"):
        assert getattr(atmosphere, name) is getattr(models, name)
        assert name in atmosphere.__all__


def test_column_remapping_public_surface() -> None:
    from orsia_radiative import atmosphere
    from orsia_radiative.atmosphere import hydrostatic, layers

    assert radiative.remap_layer_columns is layers.remap_layer_columns
    assert "remap_layer_columns" in radiative.__all__
    for module in (radiative, atmosphere, layers, hydrostatic):
        assert not hasattr(module, "pressure_layer_overlap")
    for module in (radiative, atmosphere):
        assert not hasattr(module, "_pressure_layer_overlap")
    assert hydrostatic._pressure_layer_overlap is layers._pressure_layer_overlap


def test_atmosphere_dependencies_follow_construction_order() -> None:
    root = Path(radiative.__file__).parent / "atmosphere"
    allowed = {
        "models": {"orsia_radiative.profiles"},
        "layers": set(),
        "hydrostatic": {
            "orsia_radiative.atmosphere.models",
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
