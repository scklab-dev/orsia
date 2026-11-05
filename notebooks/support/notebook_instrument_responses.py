"""Extract figures from the executed instrument demonstration."""

import base64
import json
from pathlib import Path


def extract_figures(root: Path) -> list[Path]:
    """Extract the executed canonical notebook's embedded PNGs for local review."""
    path = root / "notebooks/era5_hitran_instrument_responses.ipynb"
    target = root / "outputs/instrument-responses/figures/chains"
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
