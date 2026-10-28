"""Check package dependency declarations and import directions without loading scientific code."""

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {
    "orsia-inverse": (
        "orsia_inverse",
        {"scipy"},
        {"numpy>=2.3,<3", "scipy>=1.16,<2"},
    ),
    "orsia-numerics": (
        "orsia_numerics",
        {"scipy", "threadpoolctl"},
        {"numpy>=2.3,<3", "scipy>=1.16,<2", "threadpoolctl>=3.6,<4"},
    ),
    "orsia-instruments": (
        "orsia_instruments",
        {"scipy", "orsia_numerics", "orsia_radiative", "orsia_uncertainty"},
        {
            "numpy>=2.3,<3",
            "scipy>=1.16,<2",
            "orsia-numerics==0.1.0",
            "orsia-radiative==0.1.0",
            "orsia-uncertainty==0.1.0",
        },
    ),
    "orsia-uncertainty": (
        "orsia_uncertainty",
        {"scipy"},
        {"numpy>=2.3,<3", "scipy>=1.16,<2"},
    ),
    "orsia-radiative": (
        "orsia_radiative",
        {"scipy", "orsia_numerics"},
        {"numpy>=2.3,<3", "scipy>=1.16,<2", "orsia-numerics==0.1.0"},
    ),
    "orsia-data": (
        "orsia_data",
        {"scipy", "xarray", "orsia_radiative"},
        {"numpy>=2.3,<3", "scipy>=1.16,<2", "xarray>=2026.1,<2027", "orsia-radiative==0.1.0"},
    ),
}
ALLOWED = sys.stdlib_module_names | {
    "numpy",
    "scipy",
    "orsia_uncertainty",
    "orsia_radiative",
}
# The optional provider belongs only to its explicit acquisition adapter, not
# numerical packages or other data modules. Runtime checks verify lazy import.
OPTIONAL_IMPORTS = {
    "libs/orsia-data/src/orsia_data/services/hitran.py": {"hapi"},
    "libs/orsia-data/src/orsia_data/atmosphere/grib.py": {"eccodes", "cfgrib"},
    "libs/orsia-data/src/orsia_data/atmosphere/netcdf.py": {"h5netcdf"},
    "libs/orsia-data/src/orsia_data/geospatial.py": {"rasterio", "pyproj"},
    "libs/orsia-data/src/orsia_data/surface/thermal.py": {"h5py", "pyproj"},
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_depth = 0
    for directory in path.parents:
        if not (directory / "__init__.py").is_file():
            break
        package_depth += 1
    # Keep standalone-file checks conservative while allowing imports within
    # the regular nested packages used by the workspace.
    package_depth = max(1, package_depth)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > package_depth:
                roots.add("__outside_package__")
            elif node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def data_protocol_violations(path: Path) -> list[str]:
    """Keep STAC and shared HTTP/file transport independent of named services/readers."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {
        "services",
        "providers",
        "atmosphere",
        "surface",
        "terrain",
        "solar",
        "spectroscopy",
    }
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        else:
            continue
        if any(forbidden.intersection(name.split(".")) for name in names):
            violations.append(
                f"{path.name}:{node.lineno} imports a service/reader from a protocol layer"
            )
    return violations


def main() -> None:
    violations: list[str] = []
    workspace = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if set(workspace["tool"]["uv"]["workspace"]["members"]) != {f"libs/{name}" for name in LAYERS}:
        violations.append("workspace members do not match implemented packages")
    module_count = 0
    for name, (own_module, allowed_layers, expected_dependencies) in LAYERS.items():
        package = ROOT / "libs" / name
        metadata = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
        if (
            metadata["project"]["name"] != name
            or set(metadata["project"]["dependencies"]) != expected_dependencies
        ):
            violations.append(f"{name} runtime dependencies do not match declared boundary")
        if name == "orsia-data" and metadata["project"].get("optional-dependencies", {}).get(
            "hitran"
        ) != ["hitran-api==1.3.0.0"]:
            violations.append("orsia-data HITRAN extra does not match the optional provider")
        if name == "orsia-data" and metadata["project"].get("optional-dependencies", {}).get(
            "atmosphere"
        ) != ["cfgrib>=0.9.15,<0.10", "eccodes>=2.47,<3", "h5netcdf>=1.6,<2", "h5py>=3.15,<4"]:
            violations.append("orsia-data atmosphere extra does not match the optional readers")
        if name == "orsia-data" and metadata["project"].get("optional-dependencies", {}).get(
            "geospatial"
        ) != ["rasterio>=1.5,<2", "pyproj>=3.7,<4", "h5py>=3.15,<4"]:
            violations.append("orsia-data geospatial extra does not match the optional readers")
        modules = sorted((package / "src" / own_module).rglob("*.py"))
        if not modules:
            violations.append(f"{name} has no modules")
        module_count += len(modules)
        allowed = set(sys.stdlib_module_names) | {"numpy", own_module} | allowed_layers
        for path in modules:
            module_allowed = allowed | OPTIONAL_IMPORTS.get(
                path.relative_to(ROOT).as_posix(), set()
            )
            for root in imported_roots(path):
                if root not in module_allowed:
                    violations.append(f"{path.relative_to(ROOT)} imports forbidden {root}")
    if violations:
        raise SystemExit("\n".join(violations))
    data = ROOT / "libs/orsia-data/src/orsia_data"
    for path in [data / "http.py", data / "acquisition.py", *sorted((data / "stac").glob("*.py"))]:
        violations.extend(data_protocol_violations(path))
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"checked {module_count} package modules across {len(LAYERS)} packages")


if __name__ == "__main__":
    main()
