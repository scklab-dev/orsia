"""Observe optional atmospheric native imports without forcing the GIL off."""

import importlib
import json
import os
import sys
import sysconfig


def main() -> None:
    if sys.version_info[:2] != (3, 14) or not sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("CPython 3.14 free-threaded is required")
    if os.environ.get("PYTHON_GIL") == "0" or sys._xoptions.get("gil") == "0":
        raise RuntimeError("Do not hide native compatibility by forcing the GIL off")
    if len(sys.argv) != 2 or sys.argv[1] not in ("cfgrib", "h5netcdf"):
        raise ValueError("select one backend: cfgrib or h5netcdf")
    if sys._is_gil_enabled():
        raise RuntimeError("GIL must initially be disabled")
    observations = {}
    for name in ("numpy", "scipy.io", "xarray", sys.argv[1]):
        importlib.import_module(name)
        observations[name] = {"gil_enabled": sys._is_gil_enabled()}
    print(json.dumps(observations, indent=2))
    if sys._is_gil_enabled():
        raise RuntimeError("optional atmospheric backend enabled the GIL")


if __name__ == "__main__":
    main()
