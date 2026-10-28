"""Local notebook composition and figure extraction for the instrument demonstration."""

import ast
import base64
import json
from pathlib import Path


def preparation_sources(path: Path) -> dict[str, str]:
    """Select current five-band input preparation, preserving its numerical statements.

    The comparison notebook remains the authority for gases and solver controls.
    Selection follows its stable cell IDs and excludes reference-only setup and
    display calls, keeping this consumer focused on local atmospheric inputs.
    """
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
    selected = {}
    reference_names = {
        "cached_reference",
        "REFERENCE_PYTHON",
        "REFERENCE_CACHE",
        "REFERENCE_CACHE_MODE",
        "TIMING_RUNS",
        "show_atmosphere",
    }
    for name in ("imports", "config", "atmosphere", "materialize"):
        tree = ast.parse(cells[name])
        nodes = []
        for node in tree.body:
            if name == "atmosphere" and isinstance(node, ast.Assign):
                if any(isinstance(item, ast.Name) and item.id == "fig" for item in ast.walk(node)):
                    break
            if name == "materialize" and isinstance(node, ast.FunctionDef):
                if node.name != "prepare_case":
                    continue
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            if names & reference_names:
                continue
            nodes.append(node)
        selected[name] = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    return selected


def load_radiative_inputs(root: Path) -> dict:
    """Read verified local files through the canonical notebook's preparation cells."""
    path = root / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"
    namespace = {}
    for name, source in preparation_sources(path).items():
        exec(compile(source, f"{path}:{name}", "exec"), namespace)
    return namespace


def extract_figures(root: Path) -> list[Path]:
    """Extract the executed canonical notebook's embedded PNGs for local review."""
    path = root / "notebooks/era5_hitran_instrument_responses.ipynb"
    target = root / "outputs/instrument-responses/figures"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for cell in json.loads(path.read_text())["cells"]:
        for number, output in enumerate(cell.get("outputs", [])):
            png = output.get("data", {}).get("image/png")
            if png:
                destination = target / f"{cell['id']}-{number}.png"
                destination.write_bytes(base64.b64decode(png))
                paths.append(destination)
    return paths


if __name__ == "__main__":
    for path in extract_figures(Path(__file__).resolve().parents[2]):
        print(path)
