# Publication candidate manifest

## Source and selection

This isolated working tree was assembled from tracked blobs at Orsia commit `bc1cd72da7c1be77ed76b084687a8c712f4ad9f7`. Its Git repository starts with one independent root commit. The original Orsia Git history remains in its original checkout.

- Include six complete `libs/orsia-*/` trees: distribution metadata, library source, synthetic fixtures, library tests, guides and locally produced Draw.io source and SVG/PNG exports. The native C++ source is part of the radiative distribution.
- Include the root `.gitignore`, `.python-version`, `devbox.json`, `devbox.lock`, `pyproject.toml` and `uv.lock`; include the development `scripts/check_*` files and 16 root test files selected from tracked HEAD.
- Include seven library composition and data-service Markdown guides from `docs/`: `atmosphere-to-radiance.md`, `atmosphere-to-thermal-radiance.md`, `atmospheric-data.md`, `data-products.md`, `data-services.md`, `hsrs.md` and `mipas-reference-atmospheres.md`.
- Adapt `justfile` to the selected source and tests. Replace the root README with a candidate-specific entry point. Update local documentation and one test for the selected file set. Add the official Apache License, Version 2.0 text from `https://www.apache.org/licenses/LICENSE-2.0.txt` at the root and in each distribution; declare `Apache-2.0` in the root and six package metadata files. Each built wheel and source distribution carries its package license file.
- Exclude the original `.git` and `.agents/` trees, `AGENTS.md`, the original root README, historical `reports/` (Markdown and JSON), the complete `notebooks/` tree, notebook preparation and replay scripts, five notebook-dependent root tests, and the five-band/SASKTRAN2 comparison documents. `notebooks/data/gesat-toulouse-2025.json` has unresolved Space-Track provenance. Local inputs, caches, compiled binaries, environments and other ignored outputs remain outside this candidate.

## Owner review before distribution

- Confirm authority to license all library sources, tests, synthetic fixtures, diagrams and documentation under Apache-2.0, including generated SVG/PNG exports and any contributor-owned work. This stage records a proposed license, rather than rights clearance.
- Confirm the selected file set and public-facing scientific claims. Authentic provider files and historical measurement reports are outside the candidate; the included docs describe bounded library behavior and provider contracts.
- Review the development dependency lock and the `sasktran2-reference` dependency group: the group remains in the original locked configuration while notebook execution is outside this candidate.

This manifest records selection and review needs; build and test evidence belongs to the handoff from this staging operation.
