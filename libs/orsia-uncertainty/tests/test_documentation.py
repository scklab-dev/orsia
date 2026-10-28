"""Executable public examples and local documentation/diagram consistency."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (ROOT / "README.md", ROOT / "src/orsia_uncertainty/README.md")


@pytest.mark.parametrize("document", DOCUMENTS)
def test_documented_examples_and_local_links(document: Path) -> None:
    text = document.read_text()
    namespace: dict[str, object] = {}
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    for index, code in enumerate(blocks):
        exec(compile(code, f"<{document}:example-{index}>", "exec"), namespace)
    assert sum(line.startswith("```") for line in text.splitlines()) % 2 == 0
    assert all(line == line.rstrip() for line in text.splitlines())
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        path = target.split("#", 1)[0]
        if path and "://" not in path:
            assert (document.parent / path).exists(), target


def test_architecture_cell_references_geometry_and_fixed_svg_colours() -> None:
    diagram = ROOT / "docs/diagrams/architecture.drawio"
    model = ET.parse(diagram)
    for page in model.findall("diagram"):
        cells = page.findall("mxGraphModel/root/mxCell")
        ids = [cell.attrib["id"] for cell in cells]
        assert len(ids) == len(set(ids))
        by_id = {cell.attrib["id"]: cell for cell in cells}
        for cell in cells:
            for reference in ("parent", "source", "target"):
                if reference in cell.attrib:
                    assert cell.attrib[reference] in by_id
            assert not (cell.get("vertex") == cell.get("edge") == "1")
            if cell.get("vertex") == "1" or cell.get("edge") == "1":
                geometry = cell.find("mxGeometry")
                assert geometry is not None
                if cell.get("vertex") == "1":
                    assert float(geometry.attrib["width"]) > 0
                    assert float(geometry.attrib["height"]) > 0
                else:
                    assert geometry.get("relative") == "1"
                    source = by_id[cell.attrib["source"]]
                    target = by_id[cell.attrib["target"]]
                    # These edges join siblings or cross the single package boundary.
                    expected_parent = (
                        source.get("parent")
                        if source.get("parent") == target.get("parent")
                        else "1"
                    )
                    assert cell.get("parent") == expected_parent
    assert "layout.py" not in diagram.read_text()
    svg = diagram.with_suffix(".svg").read_text()
    assert "color-scheme: light" in svg and "light-dark(" not in svg
