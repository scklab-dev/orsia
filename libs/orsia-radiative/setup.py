"""Build serial preparation and propagation; compilation failures are not hidden."""

import sys

from orsia_numerics import get_include
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    include_dirs=[get_include()],
    ext_modules=[
        Pybind11Extension(
            "orsia_radiative.transfer.native.transfer_native",
            ["src/orsia_radiative/transfer/native/transfer_native.cpp"],
            depends=[
                "src/orsia_radiative/transfer/native/orders_native.h",
                "src/orsia_radiative/transfer/native/thermal_native.h",
                "src/orsia_radiative/transfer/native/solar_tangent_native.h",
                "src/orsia_radiative/transfer/native/solar_tangent_kernel.h",
                "src/orsia_radiative/transfer/native/scattering_tangent_native.h",
                "src/orsia_radiative/transfer/native/thermal_kernel.h",
                "src/orsia_radiative/transfer/native/thermal_derivatives.h",
                "src/orsia_radiative/transfer/native/tangent_bindings.h",
            ],
            cxx_std=17,
            extra_compile_args=(
                ["/O2", "/fp:strict"]
                if sys.platform == "win32"
                else ["-O3", "-fno-fast-math", "-ffp-contract=off"]
            ),
        ),
        Pybind11Extension(
            "orsia_radiative.optics.absorption_native",
            ["src/orsia_radiative/optics/absorption_native.cpp"],
            depends=[
                "src/orsia_radiative/optics/absorption_kernel.h",
                "src/orsia_radiative/transfer/native/tangent_bindings.h",
            ],
            cxx_std=17,
            extra_compile_args=(
                ["/O2", "/fp:strict"]
                if sys.platform == "win32"
                else ["-O3", "-fno-fast-math", "-ffp-contract=off"]
            ),
        ),
    ],
)
