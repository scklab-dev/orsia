"""Keep teaching examples executable without contacting a provider or service."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("numerics", "uncertainty", "radiative", "data", "instruments", "inverse")


@pytest.mark.parametrize("library", LIBRARIES)
def test_guide_examples(library: str) -> None:
    path = ROOT / f"libs/orsia-{library}/src/orsia_{library}/README.md"
    blocks = re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)
    assert len(blocks) == (5 if library == "data" else 3)
    namespace: dict[str, object] = {"__name__": "__main__"}
    with patch("socket.socket", side_effect=AssertionError("network forbidden in examples")):
        for number, code in enumerate(blocks, 1):
            # A virtual filename must not masquerade as a missing library source
            # in coverage reports; calls into real library modules stay measured.
            compiled = compile(code, f"<{path}:example-{number}>", "exec")
            if library == "data" and number > 2:
                continue  # Explicitly opt-in network examples: syntax only.
            if library in ("data", "numerics"):
                namespace = {"__name__": "__main__"}  # Documented standalone blocks.
            exec(compiled, namespace)


@pytest.mark.parametrize("library", LIBRARIES)
def test_guide_local_links(library: str) -> None:
    path = ROOT / f"libs/orsia-{library}/src/orsia_{library}/README.md"
    text = path.read_text()
    assert text.count("```") % 2 == 0
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        assert "://" not in link, "Guide references should stay within the local documentation"
        target, _, anchor = link.partition("#")
        destination = (path.parent / target) if target else path
        assert destination.is_file(), (path, link)
        if anchor:
            headings = re.findall(r"^#+ (.+)$", destination.read_text(), re.M)
            slugs = {
                re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-") for heading in headings
            }
            assert anchor in slugs, (path, link)
