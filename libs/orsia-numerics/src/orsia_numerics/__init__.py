"""Explicit numerical execution and tangents, with distributable native headers."""

from pathlib import Path


def get_include() -> str:
    """Locate the installed Numerics headers for downstream native builds.

    Returns:
        Absolute directory containing ``orsia_numerics/tangent.h``. This path
        belongs to the installed package and requires no source checkout.
    """
    return str(Path(__file__).resolve().parent / "include")
