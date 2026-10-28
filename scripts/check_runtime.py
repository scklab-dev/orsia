"""Verify the development runtime after representative native imports."""

import importlib
import json
import os
import sys
import sysconfig


def main() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
        raise RuntimeError("CPython 3.14 free-threaded is required")
    if not sysconfig.get_config_var("Py_GIL_DISABLED") or sys._is_gil_enabled():
        raise RuntimeError("A free-threaded build with the GIL disabled is required")
    if os.environ.get("PYTHON_GIL") == "0" or sys._xoptions.get("gil") == "0":
        raise RuntimeError(
            "Do not force the GIL off: native import compatibility must be observable"
        )

    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "numpy.linalg",
        "scipy",
        "scipy.constants",
        "scipy.linalg",
        "scipy.optimize",
        "scipy.special",
        "scipy.integrate",
        "scipy.interpolate",
        "scipy.io",
        "orsia_numerics.execution",
        "orsia_radiative",
        "orsia_radiative.transfer.native.propagation_native",
        "orsia_instruments",
        "orsia_instruments.quadrature",
        "orsia_inverse",
        "orsia_data",
    ):
        module = importlib.import_module(name)
        if name in ("numpy", "scipy"):
            versions[name] = module.__version__
        if sys._is_gil_enabled():
            raise RuntimeError(f"Importing {name} enabled the GIL")

    if {
        "cfgrib",
        "eccodes",
        "h5netcdf",
        "h5py",
        "netCDF4",
        "cdsapi",
        "rasterio",
        "pyproj",
    } & sys.modules.keys():
        raise RuntimeError("Orsia imports must not eagerly load optional data backends")

    print(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                "gil_enabled": sys._is_gil_enabled(),
                "imports": versions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
