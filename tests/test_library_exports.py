"""Public library roots expose their own functions, not dependency facades."""

import importlib
import inspect

import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "orsia_data",
        "orsia_instruments",
        "orsia_inverse",
        "orsia_numerics",
        "orsia_radiative",
        "orsia_uncertainty",
    ],
)
def test_root_functions_belong_to_the_library(package_name):
    package = importlib.import_module(package_name)
    for name, value in vars(package).items():
        if not name.startswith("_") and inspect.isroutine(value):
            assert value.__module__.split(".")[0] == package_name, name
