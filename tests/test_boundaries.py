"""Small regression cases for the static package import guard."""

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_boundary = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_boundaries.py"))
ALLOWED = cast(set[str], _boundary["ALLOWED"])
imported_roots = cast(Callable[[Path], set[str]], _boundary["imported_roots"])
data_protocol_violations = cast(Callable[[Path], list[str]], _boundary["data_protocol_violations"])


@pytest.mark.parametrize(
    "statement",
    [
        "from ..services import cdse",
        "from orsia_data import providers",
        "import orsia_data.surface",
        "from .. import atmosphere",
    ],
)
def test_protocol_layers_reject_named_services_and_readers(tmp_path: Path, statement: str) -> None:
    module = tmp_path / "client.py"
    module.write_text(f"def hidden():\n    {statement}\n", encoding="utf-8")
    assert data_protocol_violations(module)
    module.write_text("from ..http import JsonSession\n", encoding="utf-8")
    assert not data_protocol_violations(module)


def test_nested_forbidden_import_is_detected(tmp_path: Path) -> None:
    module = tmp_path / "example.py"
    module.write_text("def call():\n    from orsia_data import model\n", encoding="utf-8")
    assert imported_roots(module) - ALLOWED == {"orsia_data"}


def test_hapi_is_allowed_only_in_the_explicit_acquisition_adapter(tmp_path: Path) -> None:
    module = tmp_path / "example.py"
    module.write_text("def fetch():\n    import hapi\n", encoding="utf-8")
    assert imported_roots(module) - ALLOWED == {"hapi"}
    assert _boundary["OPTIONAL_IMPORTS"] == {
        "libs/orsia-data/src/orsia_data/services/hitran.py": {"hapi"},
        "libs/orsia-data/src/orsia_data/atmosphere/grib.py": {"eccodes", "cfgrib"},
        "libs/orsia-data/src/orsia_data/atmosphere/netcdf.py": {"h5netcdf"},
        "libs/orsia-data/src/orsia_data/geospatial.py": {"rasterio", "pyproj"},
        "libs/orsia-data/src/orsia_data/surface/thermal.py": {"h5py", "pyproj"},
    }
    assert all("hapi" not in layer[1] for layer in _boundary["LAYERS"].values())


@pytest.mark.parametrize("name", ["eccodes", "cfgrib", "h5netcdf", "rasterio", "pyproj", "h5py"])
def test_atmosphere_backends_are_not_allowed_in_numerical_modules(
    tmp_path: Path, name: str
) -> None:
    module = tmp_path / "example.py"
    module.write_text(f"def read():\n    import {name}\n", encoding="utf-8")
    assert imported_roots(module) - ALLOWED == {name}
    assert all(name not in layer[1] for layer in _boundary["LAYERS"].values())


def test_xarray_belongs_only_to_data() -> None:
    assert "xarray" in _boundary["LAYERS"]["orsia-data"][1]
    for name in ("orsia-radiative", "orsia-uncertainty"):
        assert "xarray" not in _boundary["LAYERS"][name][1]


def test_parent_relative_import_is_detected(tmp_path: Path) -> None:
    module = tmp_path / "example.py"
    module.write_text("from ..orsia_instruments import detector\n", encoding="utf-8")
    assert imported_roots(module) - ALLOWED == {"__outside_package__"}


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_relative_imports_respect_the_regular_package_depth(tmp_path: Path, depth: int) -> None:
    package = tmp_path
    for index in range(depth):
        package /= f"package_{index}"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "example.py"
    prefix = "." * depth
    module.write_text(
        f"from {prefix}optical import constituent\nfrom {prefix} import optical\n",
        encoding="utf-8",
    )
    assert imported_roots(module) == set()
    module.write_text(f"from {prefix}.orsia_instruments import detector\n", encoding="utf-8")
    assert imported_roots(module) - ALLOWED == {"__outside_package__"}
