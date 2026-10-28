# Source manifest

## Source and selection

This source tree was assembled from tracked blobs at Orsia commit `bc1cd72da7c1be77ed76b084687a8c712f4ad9f7`. Its Git history starts with one independent root commit. The original Orsia Git history remains in its original checkout.

- The tree includes six complete `libs/orsia-*/` trees: distribution metadata, library source, synthetic fixtures, library tests, guides and locally produced Draw.io source and SVG/PNG exports. The native C++ source is part of the radiative distribution.
- The root includes `.gitignore`, `.python-version`, `devbox.json`, `devbox.lock`, `pyproject.toml`, `uv.lock`, the development `scripts/check_*` files and 16 root test files selected from tracked HEAD.
- The `docs/` directory includes seven library composition and data-service Markdown guides: `atmosphere-to-radiance.md`, `atmosphere-to-thermal-radiance.md`, `atmospheric-data.md`, `data-products.md`, `data-services.md`, `hsrs.md` and `mipas-reference-atmospheres.md`.
- The selected `justfile` targets the included source and tests. The root README introduces the selected software. Local documentation and one test reflect the selected file set. The root and each distribution include the Apache License, Version 2.0 text; the root and six package metadata files declare `Apache-2.0`. Each built wheel and source distribution carries its package license file.
- The original checkout retains its `.git` and `.agents/` trees, `AGENTS.md`, original root README, historical `reports/` (Markdown and JSON), complete `notebooks/` tree, notebook preparation and replay scripts, five notebook-dependent root tests, and five-band/SASKTRAN2 comparison documents. This selection focuses on reusable library software; notebook dependencies, executed outputs and scenario citations have a separate scope. The excluded `notebooks/data/gesat-toulouse-2025.json` names Space-Track GP_HISTORY and Skyfield as its data and processing sources. Local inputs, caches, compiled binaries, environments and other ignored outputs remain in their original locations.

## License and scope

The owner reviewed the selected file set and confirmed Apache-2.0 for all included components: library sources, tests, synthetic fixtures, diagrams and their SVG/PNG exports, and documentation. The included documentation describes bounded library behavior and provider contracts. Scientific validation of instrument-space agreement, retrieval performance and engineering feasibility is a separate activity. The development lock retains the `sasktran2-reference` dependency group for notebook reference work in the original checkout.
