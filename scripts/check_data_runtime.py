"""Observe optional raster/HDF imports on the required unforced runtime."""

import importlib
import json
import os
import sys
import sysconfig


def main() -> None:
    if sys.version_info[:2] != (3, 14) or not sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("CPython 3.14 free-threaded is required")
    if os.environ.get("PYTHON_GIL") == "0" or sys._xoptions.get("gil") == "0":
        raise RuntimeError("do not force the GIL off")
    observations = {}
    for name in ("numpy", "orsia_data", "rasterio", "rasterio.warp", "pyproj", "h5py"):
        module = importlib.import_module(name)
        observations[name] = {
            "version": getattr(module, "__version__", ""),
            "gil_enabled": sys._is_gil_enabled(),
        }
    print(json.dumps(observations, indent=2))
    if sys._is_gil_enabled():
        raise RuntimeError("optional data imports enabled the GIL; backend is not qualified")


if __name__ == "__main__":
    main()
