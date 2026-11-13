set shell := ["bash", "-uc"]
set positional-arguments

# List the root lifecycle commands.
default:
    @just --list

setup:
    rustup toolchain install nightly-2025-10-24 --profile minimal
    uv python install "$(cat .python-version)"
    uv sync --locked --all-packages

# With no paths, use the maintained Python source and test trees.
fmt *paths:
    if [ "$#" -eq 0 ]; then set -- notebooks/support tests libs/*/src libs/*/tests; fi; uv run --locked --all-packages ruff format "$@"
    if [ "$#" -eq 0 ]; then set -- notebooks/support tests libs/*/src libs/*/tests; fi; uv run --locked --all-packages ruff check --fix "$@"

fmt-check *paths:
    if [ "$#" -eq 0 ]; then set -- notebooks/support tests libs/*/src libs/*/tests; fi; uv run --locked --all-packages ruff format --check "$@"

lint *paths:
    if [ "$#" -eq 0 ]; then set -- notebooks/support tests libs/*/src libs/*/tests; fi; uv run --locked --all-packages ruff check "$@"

# With no arguments, check the configured project with its optional import extras.
typecheck *args:
    uv run --locked --all-packages --extra geospatial --extra hitran pyrefly check "$@"

# Delegate the standard suite and coverage gates to tests/justfile.
test:
    just --justfile tests/justfile test

check: fmt-check lint typecheck test

build:
    for package in orsia-geometry orsia-numerics orsia-uncertainty orsia-radiative orsia-data orsia-instruments orsia-inverse orsia-dispersion orsia-scene; do uv build --package "$package"; done

# Run bounded examples and validation scripts in the locked workspace.
python +args:
    uv run --locked --all-packages python "$@"

# Resolve declared dependency changes without hand-editing the lockfile.
lock:
    uv lock

# Refresh the editable native extension after changing C++ source.
rebuild-native:
    uv sync --locked --all-packages --reinstall-package orsia-radiative
