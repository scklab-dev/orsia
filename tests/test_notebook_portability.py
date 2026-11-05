"""Offline contracts for clone-local inputs and saved notebook metadata."""

import json
import re
from pathlib import Path

import pytest

from notebooks.support import prepare_five_band, replay_instrument_response

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("credentials", [{}, {"CDSAPI_URL": "example"}, {"CDSAPI_KEY": "example"}])
def test_cds_credentials_are_read_from_repository_root(monkeypatch, tmp_path, credentials):
    observed = []

    def read_credentials(path):
        observed.append(path)
        return credentials

    def forbid_client(**kwargs):
        raise AssertionError("missing credentials must fail before creating a provider client")

    monkeypatch.setattr(prepare_five_band, "BASE", tmp_path)
    monkeypatch.setattr(prepare_five_band, "dotenv_values", read_credentials)
    monkeypatch.setattr(prepare_five_band.cdsapi, "Client", forbid_client)
    with pytest.raises(ValueError, match="Set CDSAPI_URL and CDSAPI_KEY"):
        prepare_five_band.main()
    assert observed == [ROOT / ".env"]


def test_replay_requires_a_locally_executed_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_instrument_response, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="evidence.json"):
        replay_instrument_response.main()
    assert not (tmp_path / "outputs/instrument-response-replay").exists()


@pytest.mark.parametrize(
    "name",
    ["era5_hitran_instrument_responses.ipynb", "era5_hitran_sasktran2_five_bands.ipynb"],
)
def test_embedded_metadata_and_outputs_have_no_local_paths_or_credentials(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    saved = json.dumps(
        {
            "metadata": notebook["metadata"],
            "cells": [
                {"metadata": cell["metadata"], "outputs": cell.get("outputs", [])}
                for cell in notebook["cells"]
            ],
        }
    )
    assert not re.search(r"/home/|/Users/|/tmp/|[A-Za-z]:[/\\]", saved)
    assert not re.search(r"(?:api[_-]?key|password|secret|token|authorization)\s*[=:]", saved, re.I)
