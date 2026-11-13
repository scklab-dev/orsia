"""Keep teaching examples executable without contacting a provider or service."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = (
    "geometry",
    "numerics",
    "uncertainty",
    "radiative",
    "data",
    "instruments",
    "inverse",
    "dispersion",
    "scene",
)
EXAMPLES = (
    # Geometry's orbital example requires native auxiliary fixtures in test-orbital.
    ("libs/orsia-geometry/README.md", "Intersect a ray with interpolated terrain"),
    *((f"libs/orsia-{library}/README.md", None) for library in LIBRARIES if library != "geometry"),
    ("libs/orsia-numerics/docs/execution.md", "Keep task inputs independent"),
    ("libs/orsia-uncertainty/docs/covariance.md", "Explicit preparation for repeated use"),
    ("libs/orsia-uncertainty/docs/source-projection.md", None),
    ("libs/orsia-instruments/docs/observation-cubes.md", "Run the two-band example"),
    ("libs/orsia-instruments/docs/observation-cubes.md", "Geometry-independent radiometry"),
    (
        "libs/orsia-radiative/docs/state-and-spectra.md",
        "Convert coordinates and densities together",
    ),
    ("libs/orsia-data/docs/services.md", "Inspect local metadata without downloading"),
    ("libs/orsia-instruments/docs/acquisition-and-detector.md", "Sample already integrated charge"),
    ("libs/orsia-instruments/docs/imaging.md", "Two sources, one detector readout"),
    ("libs/orsia-instruments/docs/line-of-sight.md", None),
    ("libs/orsia-instruments/docs/slit-spectrometer.md", None),
    ("libs/orsia-instruments/docs/spectral-imagers.md", None),
    ("libs/orsia-instruments/docs/wavelength-response.md", None),
    ("libs/orsia-inverse/docs/estimation.md", "Check a derivative in fitted coordinates"),
    ("libs/orsia-dispersion/docs/transport-and-evaluation.md", None),
)
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "DEVELOPMENT.md",
    ROOT / "docs/spatial-scenes.md",
    *(ROOT / f"libs/orsia-{library}/README.md" for library in LIBRARIES),
    *(
        path
        for library in LIBRARIES
        for path in sorted((ROOT / f"libs/orsia-{library}/docs").glob("*.md"))
    ),
)


@pytest.mark.parametrize(("relative_path", "section"), EXAMPLES)
def test_guide_examples(
    relative_path: str, section: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    path = ROOT / relative_path
    text = path.read_text()
    if section is not None:
        heading = re.search(rf"^(#+) {re.escape(section)}$", text, re.M)
        assert heading is not None, (path, section)
        text = text[heading.end() :]
        text = re.split(rf"^#{{1,{len(heading[1])}}} ", text, maxsplit=1, flags=re.M)[0]
    blocks = re.findall(r"^```python\n(.*?)^```", text, re.M | re.S)
    assert blocks, f"No runnable examples found in {path}"
    namespace: dict[str, object] = {"__name__": "__main__"}
    with patch("socket.socket", side_effect=AssertionError("network forbidden in examples")):
        for number, code in enumerate(blocks, 1):
            # A virtual filename must not masquerade as a missing library source
            # in coverage reports; calls into real library modules stay measured.
            compiled = compile(code, f"<{path}:example-{number}>", "exec")
            exec(compiled, namespace)
    if relative_path == "libs/orsia-geometry/README.md":
        expected_output = re.search(r"^```text\n(.*?)^```", text, re.M | re.S)
        assert expected_output is not None, "Terrain example must show its expected output"
        assert capsys.readouterr().out.strip() == expected_output[1].strip()


@pytest.mark.parametrize("library", LIBRARIES)
def test_single_library_readme(library: str) -> None:
    package = ROOT / f"libs/orsia-{library}"
    assert sorted(package.rglob("README.md")) == [package / "README.md"]


@pytest.mark.parametrize("path", DOCUMENTS)
def test_guide_local_links(path: Path) -> None:
    text = path.read_text()
    assert text.count("```") % 2 == 0
    for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        if "://" in link:
            continue
        target, _, anchor = link.partition("#")
        destination = (path.parent / target) if target else path
        assert destination.is_file(), (path, link)
        if anchor:
            headings = re.findall(r"^#+ (.+)$", destination.read_text(), re.M)
            slugs = {
                re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-") for heading in headings
            }
            assert anchor in slugs, (path, link)
