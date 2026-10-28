"""Build serial preparation and propagation; compilation failures are not hidden."""

import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "orsia_radiative.transfer.native.propagation_native",
            ["src/orsia_radiative/transfer/native/propagation_native.cpp"],
            depends=[
                "src/orsia_radiative/transfer/native/orders_native.h",
                "src/orsia_radiative/transfer/native/thermal_native.h",
            ],
            cxx_std=17,
            extra_compile_args=(
                ["/O2", "/fp:strict"]
                if sys.platform == "win32"
                else ["-O3", "-fno-fast-math", "-ffp-contract=off"]
            ),
        )
    ]
)
