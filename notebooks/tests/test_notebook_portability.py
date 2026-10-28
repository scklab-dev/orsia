"""Offline contract checks for clone-local notebook inputs and archived outputs."""

import json
import re
from pathlib import Path

import pytest

from scripts import prepare_five_band, replay_instrument_response

ROOT = Path(__file__).resolve().parents[2]


def test_cds_credentials_are_read_from_repository_root(monkeypatch, tmp_path):
    observed = []

    def read_credentials(path):
        observed.append(path)
        return {}

    monkeypatch.setattr(prepare_five_band, "BASE", tmp_path)
    monkeypatch.setattr(prepare_five_band, "dotenv_values", read_credentials)
    with pytest.raises(ValueError, match="Set CDSAPI_URL and CDSAPI_KEY"):
        prepare_five_band.main()
    assert observed == [ROOT / ".env"]


def test_replay_requires_a_locally_executed_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_instrument_response, "REFERENCE", tmp_path)
    monkeypatch.setattr(replay_instrument_response, "OUT", tmp_path / "replay")
    with pytest.raises(FileNotFoundError, match="Run execute-instrument-responses first"):
        replay_instrument_response.main()
    assert not (tmp_path / "replay").exists()


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
