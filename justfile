set shell := ["bash", "-uc"]

default:
    @just --list

# Optional data-family readers are tested separately from the base import surface.
fmt-data:
    uv run --locked --all-packages ruff format libs/orsia-data/src libs/orsia-data/tests tests/test_core_imports.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src libs/orsia-data/tests tests/test_core_imports.py

test-data:
    uv run --locked --all-packages --extra geospatial pytest --no-cov libs/orsia-data/tests

coverage-data:
    uv run --locked --all-packages --extra geospatial pytest --cov-reset --cov=orsia_data --cov-report=term-missing libs/orsia-data/tests

typecheck-data:
    uv run --locked --all-packages --extra geospatial --extra hitran pyrefly check

setup:
    rustup toolchain install nightly-2025-10-24 --profile minimal
    uv python install "$(cat .python-version)"
    uv sync --locked --all-packages

# Local authentic inputs, dedicated kernel and executed notebook (network acquisition is explicit).
prepare-five-band:
    uv run --locked --all-packages --extra hitran --extra atmosphere python -m notebooks.support.prepare_five_band

# Import retained authentic spectroscopy offline into one new collection.
prepare-spectroscopy:
    uv run --no-sync --locked --all-packages --extra atmosphere python -m notebooks.support.prepare_spectroscopy

# Public trace-gas enrichment; offline receipt verification is the default.
prepare-five-band-enriched *args:
    uv run --locked --all-packages --extra hitran --extra atmosphere python -m notebooks.support.prepare_five_band_enriched {{args}}

# Reference workers use the same locked development environment as the notebook.
test-sasktran2-reference *args:
    MPLBACKEND=Agg uv run --locked --all-packages --extra atmosphere pytest --no-cov tests/test_notebook_sasktran2_cache.py tests/test_five_band_notebook.py tests/test_notebook_companions.py {{args}}

test-five-band-enrichment:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere pytest --no-cov tests/test_five_band_enrichment.py tests/test_five_band_notebook.py

execute-five-band:
    uv run --no-sync --locked --all-packages --extra atmosphere python -m ipykernel install --prefix outputs/jupyter --name orsia-314t --display-name "Orsia CPython 3.14 free-threaded"
    JUPYTER_PATH="$PWD/outputs/jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}" MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere jupyter execute --inplace --timeout=3600 --kernel_name=orsia-314t notebooks/era5_hitran_sasktran2_five_bands.ipynb

# Local enriched radiance through public optical and detector APIs.
execute-instrument-responses:
    uv run --no-sync --locked --all-packages --extra atmosphere python -m ipykernel install --prefix outputs/jupyter --name orsia-314t --display-name "Orsia CPython 3.14 free-threaded"
    JUPYTER_PATH="$PWD/outputs/jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}" MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere jupyter execute --inplace --timeout=3600 --kernel_name=orsia-314t notebooks/era5_hitran_instrument_responses.ipynb
    uv run --no-sync --locked --all-packages python -m notebooks.support.notebook_instrument_responses

test-instrument-responses:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m pytest --no-cov tests/test_instrument_response_notebook.py tests/test_instrument_notebook_chains.py tests/test_notebook_companions.py
    uv run --no-sync --locked --all-packages ruff check notebooks/support tests/test_instrument_response_notebook.py tests/test_instrument_notebook_chains.py tests/test_notebook_companions.py
    uv run --no-sync --locked --all-packages ruff format --check notebooks/support tests/test_instrument_response_notebook.py tests/test_instrument_notebook_chains.py tests/test_notebook_companions.py

# Replay current selected-read chains against the latest full notebook execution.
replay-instrument-response:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m notebooks.support.replay_instrument_response

fmt-instrument-response:
    uv run --no-sync --locked --all-packages ruff format libs/orsia-instruments/src/orsia_instruments/acquisition.py libs/orsia-instruments/src/orsia_instruments/photoelectrons.py libs/orsia-instruments/src/orsia_instruments/imaging/response/instrument.py libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_instrument_response.py notebooks/support/notebook_ground_pixel.py notebooks/support/replay_instrument_response.py tests/test_instrument_response_notebook.py
    uv run --no-sync --locked --all-packages ruff check --fix libs/orsia-instruments/src/orsia_instruments/acquisition.py libs/orsia-instruments/src/orsia_instruments/photoelectrons.py libs/orsia-instruments/src/orsia_instruments/imaging/response/instrument.py libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_instrument_response.py notebooks/support/notebook_ground_pixel.py notebooks/support/replay_instrument_response.py tests/test_instrument_response_notebook.py

test-instrument-response:
    uv run --no-sync --locked --all-packages pytest --no-cov libs/orsia-instruments/tests/test_instrument_response.py

typecheck-instrument-response:
    uv run --no-sync --locked --all-packages pyrefly check libs/orsia-instruments/src/orsia_instruments/acquisition.py libs/orsia-instruments/src/orsia_instruments/photoelectrons.py libs/orsia-instruments/src/orsia_instruments/imaging/response/instrument.py libs/orsia-instruments/tests/test_instrument_response.py notebooks/support/notebook_ground_pixel.py notebooks/support/replay_instrument_response.py

coverage-instruments:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m pytest --cov-reset --cov=orsia_instruments --cov-report=term-missing libs/orsia-instruments/tests tests/test_instrument_response_notebook.py

# Regenerate Python resolution after an explicit dependency declaration change.
lock:
    uv lock

lock-check:
    uv lock --check

# Keep extraction formatting scoped, preserving unrelated user work.
fmt-numerics:
    uv run --locked --all-packages ruff format libs/orsia-numerics libs/orsia-instruments/src/orsia_instruments/optics/psfs.py libs/orsia-instruments/src/orsia_instruments/imaging/photodetection.py libs/orsia-instruments/src/orsia_instruments/imaging/measurements.py libs/orsia-uncertainty/src/orsia_uncertainty/covariance.py tests/test_core_imports.py
    uv run --locked --all-packages ruff check --fix libs/orsia-numerics libs/orsia-instruments/src/orsia_instruments/optics/psfs.py libs/orsia-instruments/src/orsia_instruments/imaging/photodetection.py libs/orsia-instruments/src/orsia_instruments/imaging/measurements.py libs/orsia-uncertainty/src/orsia_uncertainty/covariance.py tests/test_core_imports.py

# Explicit orbital-data acquisition is mocked in this focused suite.
fmt-orbital-data:
    uv run --locked --all-packages ruff format libs/orsia-data/src/orsia_data/orbits.py libs/orsia-data/src/orsia_data/services/space_track.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_orbits.py libs/orsia-data/tests/test_space_track.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src/orsia_data/orbits.py libs/orsia-data/src/orsia_data/services/space_track.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_orbits.py libs/orsia-data/tests/test_space_track.py

test-orbital-data:
    SATKIT_OFFLINE=1 uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests/test_orbits.py libs/orsia-data/tests/test_space_track.py

# Fixed local resources; this numerical qualification recipe never authenticates.
test-orbital *args:
    SATKIT_DATA="$PWD/outputs/geometry-auxiliaries" SATKIT_JPLEPHEM_FILE="$PWD/outputs/geometry-auxiliaries/lnxp1900p2053.421" SATKIT_OFFLINE=1 ORSIA_TEST_ORBITAL=1 uv run --locked --all-packages pytest --no-cov libs/orsia-geometry/tests/test_orbital_reference.py libs/orsia-instruments/tests/test_line_of_sight.py {{args}}

# Inspect the current checkout without changing tracked files.
worktree-status:
    git status --short --untracked-files=all
    git diff --stat
    git diff --check

worktree-diff:
    git diff --no-ext-diff --no-color

# Native-resolution halves for visual inspection of the 1600 px export.
geometry-diagram-detail:
    uv run --locked --all-packages python -c 'from pathlib import Path; from PIL import Image; image = Image.open("libs/orsia-geometry/docs/diagrams/architecture.png"); output = Path("outputs/geometry-diagram-review"); output.mkdir(parents=True, exist_ok=True); middle = image.width // 2; image.crop((0, 0, middle, image.height)).save(output / "left.png"); image.crop((middle, 0, image.width, image.height)).save(output / "right.png"); print("Review image:", image.size, "native-resolution halves in", output)'

# Supplied-trajectory geometry and preserved acquisition consumers.
fmt-geometry:
    uv run --locked --all-packages ruff format libs/orsia-geometry libs/orsia-instruments/src/orsia_instruments/los.py tests/test_core_imports.py
    uv run --locked --all-packages ruff check --fix libs/orsia-geometry libs/orsia-instruments/src/orsia_instruments/los.py tests/test_core_imports.py

test-geometry:
    uv run --locked --all-packages pytest --no-cov libs/orsia-geometry/tests libs/orsia-instruments/tests/test_line_of_sight.py tests/test_core_imports.py

coverage-geometry:
    uv run --locked --all-packages pytest --cov-reset --cov=orsia_geometry --cov-report=term-missing libs/orsia-geometry/tests

# Focused numerical kernels and explicit CPU execution contracts.
test-numerics:
    uv run --locked --all-packages pytest --no-cov libs/orsia-numerics/tests

# Focused regression for the SciPy-backed numerical methods and contracts.
test-scipy:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_scipy_methods.py libs/orsia-radiative/tests/test_thermal.py libs/orsia-radiative/tests/test_formal.py libs/orsia-radiative/tests/test_optical.py libs/orsia-uncertainty/tests

fmt:
    uv run --locked --all-packages ruff format notebooks/support tests libs/*/src libs/*/tests
    uv run --locked --all-packages ruff check --fix notebooks/support tests libs/*/src libs/*/tests

fmt-check:
    uv run --locked --all-packages ruff format --check notebooks/support tests libs/*/src libs/*/tests

lint:
    uv run --locked --all-packages ruff check notebooks/support tests libs/*/src libs/*/tests

typecheck:
    uv run --locked --all-packages --extra geospatial --extra hitran pyrefly check

test:
    # Keep atmospheric reader imports out of the numerical test process.
    # Match files because pytest still traverses directories listed in testpaths.
    uv run --locked --all-packages --extra geospatial pytest --ignore-glob='libs/orsia-data/tests/*' --ignore=tests/test_five_band_notebook.py --cov-fail-under=0
    uv run --locked --all-packages --extra geospatial pytest libs/orsia-data/tests tests/test_five_band_notebook.py --cov-append
    uv run --locked --all-packages coverage report --include='libs/orsia-geometry/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-numerics/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-uncertainty/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-radiative/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-data/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-instruments/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-inverse/src/*'

check: fmt-check lint typecheck test

build:
    for package in orsia-geometry orsia-numerics orsia-uncertainty orsia-radiative orsia-data orsia-instruments orsia-inverse; do uv build --package "$package"; done

test-inverse:
    uv run --locked --all-packages pytest --no-cov libs/orsia-inverse/tests

test-uncertainty:
    uv run --locked --all-packages pytest --no-cov libs/orsia-uncertainty/tests

fmt-inverse:
    uv run --locked --all-packages ruff format libs/orsia-inverse
    uv run --locked --all-packages ruff check --fix libs/orsia-inverse

test-instruments:
    uv run --locked --all-packages pytest --no-cov libs/orsia-instruments/tests tests/test_instrument_examples.py tests/test_lvf_scan.py tests/test_instrument_composition.py

# Preserve bounded synthetic refinement, statistics and 30 s / 512 MiB cost checks.
# These are opt-in local regressions, not scientific or mission qualification.
test-instrument-characterization:
    ORSIA_TEST_INSTRUMENT_CHARACTERIZATION=1 uv run --locked --all-packages pytest --no-cov tests/test_instrument_characterization.py

fmt-instrument-tests:
    uv run --locked --all-packages ruff format tests/instrument_cases.py tests/test_instrument_examples.py tests/test_instrument_characterization.py tests/test_instrument_composition.py tests/test_lvf_scan.py tests/test_sasktran2_chain.py tests/test_solar_interoperability.py
    uv run --locked --all-packages ruff check --fix tests/instrument_cases.py tests/test_instrument_examples.py tests/test_instrument_characterization.py tests/test_instrument_composition.py tests/test_lvf_scan.py tests/test_sasktran2_chain.py tests/test_solar_interoperability.py

# Bounded integrated image convergence, source composition and tolerance evidence.
test-image-closure:
    uv run --locked --all-packages pytest --no-cov libs/orsia-instruments/tests/test_image_closure.py

fmt-image-closure:
    uv run --locked --all-packages ruff format libs/orsia-instruments/tests/test_image_closure.py
    uv run --locked --all-packages ruff check --fix libs/orsia-instruments/tests/test_image_closure.py

# Keep formatting scoped to the image API rather than unrelated user changes.
fmt-image-api:
    uv run --locked --all-packages ruff format libs/orsia-instruments/src/orsia_instruments/imaging libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_imaging.py
    uv run --locked --all-packages ruff check --fix libs/orsia-instruments/src/orsia_instruments/imaging libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_imaging.py

# Scoped formatting for interchangeable optical laws and their tests.
fmt-psf:
    uv run --locked --all-packages ruff format libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_psfs.py
    uv run --locked --all-packages ruff check --fix libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_psfs.py

# Scoped formatting for interchangeable spectral laws and their tests.
fmt-optical-laws:
    uv run --locked --all-packages ruff format libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_spectral_filters.py libs/orsia-instruments/tests/test_imager_laws.py
    uv run --locked --all-packages ruff check --fix libs/orsia-instruments/src/orsia_instruments/optics libs/orsia-instruments/src/orsia_instruments/__init__.py libs/orsia-instruments/tests/test_spectral_filters.py libs/orsia-instruments/tests/test_imager_laws.py

test-radiance-chain:
    uv run --locked --all-packages pytest --no-cov tests/test_sasktran2_chain.py

fmt-radiance-chain:
    uv run --locked --all-packages ruff format tests/test_sasktran2_chain.py
    uv run --locked --all-packages ruff check tests/test_sasktran2_chain.py

# Packaged serial recurrence and the existing formal/SOS/thermal contracts.
test-propagation:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_propagation*.py libs/orsia-radiative/tests/test_formal.py libs/orsia-radiative/tests/test_sos*.py libs/orsia-radiative/tests/test_thermal.py

# Refresh the editable native extension after changing C++ source.
rebuild-propagation:
    uv sync --locked --all-packages --reinstall-package orsia-radiative

# Native thermal physics, interval limits, binding and concurrent-call regression.
test-thermal-native *args:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_thermal*.py {{args}}

typecheck-thermal-native:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py

fmt-thermal-native:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py libs/orsia-radiative/setup.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py libs/orsia-radiative/setup.py

# Scoped optical-law checks preserve unrelated edits and the installed environment.
fmt-scattering-laws:
    uv run --no-sync --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/optics/scattering.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_scattering_laws.py tests/test_notebook_scattering_laws.py
    uv run --no-sync --locked --all-packages ruff check --fix libs/orsia-radiative/src/orsia_radiative/optics/scattering.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_scattering_laws.py tests/test_notebook_scattering_laws.py

check-scattering-laws:
    uv run --no-sync --locked --all-packages ruff format --check libs/orsia-radiative/src/orsia_radiative/optics/scattering.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_scattering_laws.py tests/test_notebook_scattering_laws.py
    uv run --no-sync --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/optics/scattering.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_scattering_laws.py tests/test_notebook_scattering_laws.py
    uv run --no-sync --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/optics/scattering.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_scattering_laws.py tests/test_notebook_scattering_laws.py
    uv run --no-sync --locked --all-packages pytest --cov-reset --cov=orsia_radiative --cov-report=term-missing libs/orsia-radiative/tests tests/test_notebook_scattering_laws.py tests/test_core_imports.py
    # Atmospheric reader imports can enable the GIL; keep notebook checks separate.
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages pytest --no-cov tests/test_five_band_notebook.py tests/test_notebook_aerosol.py
    uv run --no-sync --locked --all-packages python -c 'import sys; import numpy; import scipy; import orsia_radiative; import orsia_radiative.transfer.native.transfer_native; print(sys.version); print("GIL enabled after native imports:", sys._is_gil_enabled()); assert sys.version_info[:2] == (3, 14); assert not sys._is_gil_enabled()'

# Complete solar calls, including established recurrence checks.
test-solar-calls:
    uv run --locked --all-packages python -m pytest --no-cov libs/orsia-radiative/tests

fmt-solar-calls:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/src/orsia_radiative/transfer/solar.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_solar_calls.py libs/orsia-radiative/tests/test_spectral_native.py libs/orsia-radiative/tests/test_rayleigh_native.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_sos.py libs/orsia-radiative/tests/test_sos_delta_m.py

typecheck-solar-calls:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/src/orsia_radiative/transfer/solar.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_solar_calls.py libs/orsia-radiative/tests/test_spectral_native.py libs/orsia-radiative/tests/test_rayleigh_native.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_sos.py libs/orsia-radiative/tests/test_sos_delta_m.py

typecheck-propagation:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_propagation*.py

fmt-propagation:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_propagation*.py libs/orsia-radiative/setup.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/transfer_native.pyi libs/orsia-radiative/tests/test_propagation*.py libs/orsia-radiative/setup.py

test-solar-boundary:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_solar_boundary.py

typecheck-solar-boundary:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_boundary.py

fmt-solar-boundary:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_boundary.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_boundary.py

# Layer-batched Rayleigh orders and scalar-beam/reference-sweep regressions.
test-solar-accumulation:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_solar_accumulation.py

typecheck-solar-accumulation:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_accumulation.py

fmt-solar-accumulation:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_accumulation.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_solar_accumulation.py

# Focused numerical regression without the full-suite coverage denominator.
test-redistribution:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_redistribution.py libs/orsia-radiative/tests/test_sos.py libs/orsia-radiative/tests/test_sos_numerics.py

fmt-redistribution:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_redistribution.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/tests/test_redistribution.py

test-sos-refinement:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_sos_refinement.py

# Fixed-N recurrence, independent rays, singular source/optical boundary and refinement.
test-sos:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_sos*.py libs/orsia-radiative/tests/test_angular_geometry.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_redistribution.py

# LTE Schwarzschild transport, radiometry and explicit-grid contracts.
test-thermal:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_thermal.py libs/orsia-radiative/tests/test_radiometry.py

# Offline readers, local-water convention and bounded numerical contracts.
test-spectroscopy:
    uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests libs/orsia-radiative/tests/test_water_continuum.py libs/orsia-radiative/tests/test_spectroscopy_contract.py libs/orsia-radiative/tests/test_scipy_methods.py

# Genuine optional HAPI fetch/parser with synthetic HTTP bytes, never live fetch.
test-hitran-api:
    uv run --locked --all-packages --extra hitran pytest --no-cov libs/orsia-data/tests/test_hitran_api.py

# Offline atmospheric clients/files and analytic column-conservation cases.
test-atmosphere:
    uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/tests/test_atmospheric_columns.py

test-atmosphere-backends:
    ORSIA_TEST_ATMOSPHERE_BACKENDS=1 ECCODES_DIR="$(dirname "$(dirname "$(command -v grib_ls)")")" uv run --locked --all-packages --extra atmosphere pytest --no-cov libs/orsia-data/tests/test_atmosphere_native.py

fmt-atmosphere:
    uv run --locked --all-packages ruff format libs/orsia-data/src/orsia_data/atmosphere libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/src/orsia_radiative/atmosphere libs/orsia-radiative/tests/test_atmospheric_columns.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-radiative/src/orsia_radiative/__init__.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src/orsia_data/atmosphere libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/src/orsia_radiative/atmosphere libs/orsia-radiative/tests/test_atmospheric_columns.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-radiative/src/orsia_radiative/__init__.py

# Avoid formatting unrelated in-progress comparison work.
fmt-spectroscopy:
    uv run --locked --all-packages ruff format libs/orsia-data/src libs/orsia-data/tests libs/orsia-radiative/src/orsia_radiative/spectroscopy libs/orsia-radiative/src/orsia_radiative/optics/absorption.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_water_continuum.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src libs/orsia-data/tests libs/orsia-radiative/src/orsia_radiative/spectroscopy libs/orsia-radiative/src/orsia_radiative/optics/absorption.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_water_continuum.py

# RFM decoding and bounded MIPAS transfer; synthetic, offline contracts.
test-mipas:
    uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py

typecheck-mipas:
    uv run --locked --all-packages pyrefly check libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py

fmt-mipas:
    uv run --locked --all-packages ruff format libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/src/orsia_data/atmosphere/__init__.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/src/orsia_data/atmosphere/__init__.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py

# Focused offline contracts for publication portability.
test-notebook-portability:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m pytest --no-cov tests/test_notebook_portability.py
