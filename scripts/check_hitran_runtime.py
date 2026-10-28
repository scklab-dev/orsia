"""Check lazy Orsia imports and the optional HAPI runtime, without network access."""

import sys

import orsia_data  # noqa: F401 - exercise the public import boundary

if "hapi" in sys.modules:
    raise SystemExit("orsia_data must not eagerly import HAPI")
if sys._is_gil_enabled():
    raise SystemExit("GIL is enabled before importing HAPI")

import hapi  # type: ignore[import-not-found]  # noqa: E402

if sys._is_gil_enabled():
    raise SystemExit("GIL is enabled after importing HAPI")
if hapi.HAPI_VERSION != "1.3.0.0":
    raise SystemExit("unexpected HAPI version")
print("HAPI 1.3.0.0 imports on CPython free-threaded with the GIL disabled")
