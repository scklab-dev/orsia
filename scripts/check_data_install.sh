#!/usr/bin/env bash
set -euo pipefail

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for package in orsia-numerics orsia-radiative orsia-data; do
    uv build --package "$package" --out-dir "$tmp/dist"
done
uv venv --python "$(cat .python-version)" "$tmp/environment"
uv pip install --python "$tmp/environment/bin/python" --find-links "$tmp/dist" 'orsia-data[geospatial]==0.1.0' pytest
mkdir "$tmp/tests"
cp libs/orsia-data/tests/test_{aerosols,geospatial,scene_providers,surface_products,bearer_transfer,terrain,stac,service_clients,atmosphere_rfm,mipas_acquisition}.py "$tmp/tests/"
cp scripts/check_data_runtime.py "$tmp/check_runtime.py"
(
    cd "$tmp"
    "$tmp/environment/bin/python" -I check_runtime.py
    "$tmp/environment/bin/python" -I -m pytest --noconftest --rootdir "$tmp" "$tmp/tests"
)
