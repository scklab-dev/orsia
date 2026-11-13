"""Offline contracts for clone-local inputs and saved notebook metadata."""

import json
import re
from pathlib import Path

import pytest

from notebooks.support import replay_instrument_response

ROOT = Path(__file__).resolve().parents[1]


def test_replay_requires_a_locally_executed_reference(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_instrument_response, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="evidence.json"):
        replay_instrument_response.main()
    assert not (tmp_path / "outputs/instrument-response-replay").exists()


@pytest.mark.parametrize(
    "name",
    [
        "era5_hitran_instrument_responses.ipynb",
        "era5_hitran_sasktran2_five_bands.ipynb",
        "gaussian_puff_dispersion.ipynb",
        "spatial_scene.ipynb",
    ],
)
def test_embedded_metadata_and_outputs_have_no_local_paths_or_credentials(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    saved = {
        "metadata": notebook["metadata"],
        "cells": [
            {"metadata": cell["metadata"], "outputs": cell.get("outputs", [])}
            for cell in notebook["cells"]
        ],
    }
    for text in _saved_strings(saved):
        _assert_portable_text(text)


def test_portability_scan_accepts_labelled_line_breaks():
    _assert_portable_text("Synthetic inputs:\nNo provider acquisition.\n")
    assert list(_saved_strings({"outputs": [{"text": ["first\n", "second\n"]}]})) == [
        "first\n",
        "second\n",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "/home/user/input",
        "/Users/user/input",
        "/tmp/input",
        r"C:\Users\user",
        "C:/Users/user",
        "token=private",
    ],
)
def test_portability_scan_rejects_private_paths_and_credential_assignments(text):
    with pytest.raises(AssertionError):
        _assert_portable_text(text)


def _assert_portable_text(text):
    assert not re.search(r"/home/|/Users/|/tmp/|[A-Za-z]:[/\\]", text)
    assert not re.search(r"(?:api[_-]?key|password|secret|token|authorization)\s*[=:]", text, re.I)


def _saved_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _saved_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _saved_strings(item)
