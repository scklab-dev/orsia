set shell := ["bash", "-uc"]

default:
    @just --list

# Optional data-family readers are tested separately from the base import surface.
fmt-data:
    uv run --locked --all-packages ruff format libs/orsia-data/src libs/orsia-data/tests scripts/check_data_runtime.py scripts/check_boundaries.py scripts/check_runtime.py tests/test_boundaries.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src libs/orsia-data/tests scripts/check_data_runtime.py scripts/check_boundaries.py scripts/check_runtime.py tests/test_boundaries.py

test-data:
    uv run --locked --all-packages --extra geospatial python scripts/check_data_runtime.py
    uv run --locked --all-packages --extra geospatial pytest --no-cov libs/orsia-data/tests

coverage-data:
    uv run --locked --all-packages --extra geospatial pytest --cov-reset --cov=orsia_data --cov-report=term-missing libs/orsia-data/tests

typecheck-data:
    uv run --locked --all-packages --extra geospatial pyrefly check

clean-install-data:
    bash scripts/check_data_install.sh

setup:
    uv python install "$(cat .python-version)"
    uv sync --locked --all-packages

# Explicit authentic acquisitions populate ignored outputs/ for both notebooks.
prepare-five-band:
    uv run --locked --all-packages --extra hitran --extra atmosphere python scripts/prepare_five_band.py

prepare-five-band-enriched *args:
    uv run --locked --all-packages --extra hitran --extra atmosphere python scripts/prepare_five_band_enriched.py {{args}}

setup-sasktran2-reference:
    UV_PROJECT_ENVIRONMENT=outputs/sasktran2-env uv sync --locked --only-group sasktran2-reference --python 3.14.3+gil --no-build

execute-five-band:
    uv run --no-sync --locked --all-packages --extra atmosphere python -m ipykernel install --prefix outputs/jupyter --name orsia-314t --display-name "Orsia CPython 3.14 free-threaded"
    JUPYTER_PATH="$PWD/outputs/jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}" MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere jupyter execute --inplace --timeout=3600 --kernel_name=orsia-314t notebooks/era5_hitran_sasktran2_five_bands.ipynb

execute-instrument-responses:
    uv run --no-sync --locked --all-packages --extra atmosphere python -m ipykernel install --prefix outputs/jupyter --name orsia-314t --display-name "Orsia CPython 3.14 free-threaded"
    JUPYTER_PATH="$PWD/outputs/jupyter/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}" MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere jupyter execute --inplace --timeout=3600 --kernel_name=orsia-314t notebooks/era5_hitran_instrument_responses.ipynb
    uv run --no-sync --locked --all-packages python -m notebooks.support.notebook_instrument_responses

# Replay against the user's own completed instrument execution, without new provider calls.
replay-instrument-response:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m scripts.replay_instrument_response

test-notebooks:
    MPLBACKEND=Agg uv run --no-sync --locked --all-packages --extra atmosphere python -m pytest --no-cov notebooks/tests

# Regenerate Python resolution after an explicit dependency declaration change.
lock:
    uv lock

# Keep extraction formatting scoped, preserving unrelated user work.
fmt-numerics:
    uv run --locked --all-packages ruff format libs/orsia-numerics libs/orsia-instruments/src/orsia_instruments/optics/psfs.py libs/orsia-instruments/src/orsia_instruments/imaging/charge.py libs/orsia-instruments/src/orsia_instruments/imaging/measurements.py libs/orsia-uncertainty/src/orsia_uncertainty/covariance.py scripts/check_boundaries.py scripts/check_runtime.py
    uv run --locked --all-packages ruff check --fix libs/orsia-numerics libs/orsia-instruments/src/orsia_instruments/optics/psfs.py libs/orsia-instruments/src/orsia_instruments/imaging/charge.py libs/orsia-instruments/src/orsia_instruments/imaging/measurements.py libs/orsia-uncertainty/src/orsia_uncertainty/covariance.py scripts/check_boundaries.py scripts/check_runtime.py

# Focused numerical kernels and explicit CPU execution contracts.
test-numerics:
    uv run --locked --all-packages pytest --no-cov libs/orsia-numerics/tests

# Focused regression for the SciPy-backed numerical methods and contracts.
test-scipy:
    uv run --locked --all-packages pytest --no-cov libs/orsia-radiative/tests/test_scipy_methods.py libs/orsia-radiative/tests/test_thermal.py libs/orsia-radiative/tests/test_formal.py libs/orsia-radiative/tests/test_optical.py libs/orsia-uncertainty/tests

fmt:
    uv run --locked --all-packages ruff format scripts tests libs/*/src libs/*/tests
    uv run --locked --all-packages ruff check --fix scripts tests libs/*/src libs/*/tests

fmt-check:
    uv run --locked --all-packages ruff format --check scripts tests libs/*/src libs/*/tests

lint:
    uv run --locked --all-packages ruff check scripts tests libs/*/src libs/*/tests

typecheck:
    uv run --locked --all-packages --extra geospatial pyrefly check

boundaries:
    uv run --locked --all-packages python scripts/check_boundaries.py

runtime:
    uv run --locked --all-packages python scripts/check_runtime.py

test:
    uv run --locked --all-packages --extra geospatial python scripts/check_data_runtime.py
    uv run --locked --all-packages --extra geospatial pytest
    uv run --locked --all-packages coverage report --include='libs/orsia-numerics/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-uncertainty/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-radiative/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-data/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-instruments/src/*'
    uv run --locked --all-packages coverage report --include='libs/orsia-inverse/src/*'

check: fmt-check lint typecheck boundaries runtime test

build:
    for package in orsia-numerics orsia-uncertainty orsia-radiative orsia-data orsia-instruments orsia-inverse; do uv build --package "$package"; done

clean-install:
    bash scripts/check_clean_install.sh

test-inverse:
    uv run --locked --all-packages pytest --no-cov libs/orsia-inverse/tests

test-uncertainty:
    uv run --locked --all-packages pytest --no-cov libs/orsia-uncertainty/tests

fmt-inverse:
    uv run --locked --all-packages ruff format libs/orsia-inverse scripts/check_boundaries.py
    uv run --locked --all-packages ruff check --fix libs/orsia-inverse scripts/check_boundaries.py

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

# Replay generated HDF5/GRIB files against installed wheels, outside workspace resolution.
clean-install-atmosphere:
    ORSIA_TEST_ATMOSPHERE_BACKENDS=1 ECCODES_DIR="$(dirname "$(dirname "$(command -v grib_ls)")")" bash scripts/check_clean_install.sh

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
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py

fmt-thermal-native:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py libs/orsia-radiative/setup.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/thermal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_thermal_native.py libs/orsia-radiative/tests/thermal_reference.py libs/orsia-radiative/setup.py

# Complete solar calls, including established recurrence checks.
test-solar-calls:
    uv run --locked --all-packages python -m pytest --no-cov libs/orsia-radiative/tests

fmt-solar-calls:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/src/orsia_radiative/transfer/solar.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_solar_calls.py libs/orsia-radiative/tests/test_spectral_native.py libs/orsia-radiative/tests/test_rayleigh_native.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_sos.py libs/orsia-radiative/tests/test_sos_delta_m.py

typecheck-solar-calls:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/orders.py libs/orsia-radiative/src/orsia_radiative/transfer/solar.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_solar_calls.py libs/orsia-radiative/tests/test_spectral_native.py libs/orsia-radiative/tests/test_rayleigh_native.py libs/orsia-radiative/tests/test_composable_transfer.py libs/orsia-radiative/tests/test_sos.py libs/orsia-radiative/tests/test_sos_delta_m.py

typecheck-propagation:
    uv run --locked --all-packages pyrefly check libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_propagation*.py

fmt-propagation:
    uv run --locked --all-packages ruff format libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_propagation*.py libs/orsia-radiative/setup.py
    uv run --locked --all-packages ruff check libs/orsia-radiative/src/orsia_radiative/transfer/formal.py libs/orsia-radiative/src/orsia_radiative/transfer/native/propagation_native.pyi libs/orsia-radiative/tests/test_propagation*.py libs/orsia-radiative/setup.py

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

# Optional HAPI import/runtime and genuine API with synthetic HTTP bytes, never live fetch.
test-hitran-api:
    uv run --locked --all-packages --extra hitran python scripts/check_hitran_runtime.py
    uv run --locked --all-packages --extra hitran pytest --no-cov libs/orsia-data/tests/test_hitran_api.py

# Offline atmospheric clients/files and analytic column-conservation cases.
test-atmosphere:
    uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/tests/test_atmospheric_columns.py

runtime-atmosphere:
    ECCODES_DIR="$(dirname "$(dirname "$(command -v grib_ls)")")" uv run --locked --all-packages --extra atmosphere python scripts/check_atmosphere_runtime.py cfgrib
    uv run --locked --all-packages --extra atmosphere python scripts/check_atmosphere_runtime.py h5netcdf

test-atmosphere-backends:
    ORSIA_TEST_ATMOSPHERE_BACKENDS=1 ECCODES_DIR="$(dirname "$(dirname "$(command -v grib_ls)")")" uv run --locked --all-packages --extra atmosphere pytest --no-cov libs/orsia-data/tests/test_atmosphere_native.py

fmt-atmosphere:
    uv run --locked --all-packages ruff format libs/orsia-data/src/orsia_data/atmosphere libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/src/orsia_radiative/atmosphere libs/orsia-radiative/tests/test_atmospheric_columns.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-radiative/src/orsia_radiative/__init__.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src/orsia_data/atmosphere libs/orsia-data/tests/test_atmosphere*.py libs/orsia-radiative/src/orsia_radiative/atmosphere libs/orsia-radiative/tests/test_atmospheric_columns.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-radiative/src/orsia_radiative/__init__.py

# Avoid formatting unrelated in-progress comparison work.
fmt-spectroscopy:
    uv run --locked --all-packages ruff format libs/orsia-data/src libs/orsia-data/tests libs/orsia-radiative/src/orsia_radiative/spectroscopy libs/orsia-radiative/src/orsia_radiative/optics/absorption.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_water_continuum.py scripts/check_boundaries.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src libs/orsia-data/tests libs/orsia-radiative/src/orsia_radiative/spectroscopy libs/orsia-radiative/src/orsia_radiative/optics/absorption.py libs/orsia-radiative/src/orsia_radiative/__init__.py libs/orsia-radiative/tests/test_water_continuum.py scripts/check_boundaries.py

# RFM decoding and bounded MIPAS transfer; synthetic, offline contracts.
test-mipas:
    uv run --locked --all-packages pytest --no-cov libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py

typecheck-mipas:
    uv run --locked --all-packages pyrefly check libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py

fmt-mipas:
    uv run --locked --all-packages ruff format libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/src/orsia_data/atmosphere/__init__.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py
    uv run --locked --all-packages ruff check --fix libs/orsia-data/src/orsia_data/atmosphere/rfm.py libs/orsia-data/src/orsia_data/services/mipas.py libs/orsia-data/src/orsia_data/atmosphere/__init__.py libs/orsia-data/src/orsia_data/__init__.py libs/orsia-data/tests/test_atmosphere_rfm.py libs/orsia-data/tests/test_mipas_acquisition.py
