# Source manifest

## Source and selection

This delivery selects tracked files from Orsia development commit `48bcb3fc06c51b23bcc6cc47fc70333fc3fa88a0` plus the accompanying contributor-documentation and ignore-rule corrections. It updates public commit `c63f38428bb130d31107d0be5210a0205818c601` on the existing public branch. Development Git history remains outside the delivery.

- The nine complete library trees are Geometry, Numerics, Uncertainty, Radiative, Data, Instruments, Inverse, Dispersion and Scene. Each tree contains its metadata, source, tests, fixtures and maintained guides. Native C++ sources and shared headers are included.
- Workspace configuration and dependency locks, root tests, examples, validation scripts, architecture assets, four executed notebooks, notebook support modules, recipes and scenario JSON files are included. Locks retain their source bytes; delivery preparation does not resolve dependencies.
- The root and all nine distributions include the unchanged Apache License, Version 2.0 text. All ten project metadata files declare `Apache-2.0`. The notebook provenance and attribution text from the preceding public snapshot is retained.
- Contributor documentation is self-contained. The migration guide describes public responsibilities and maintained checks without requiring the private design or development commit history. Its historical verification record remains distinct from current delivery checks.
- Current CDS credential handling reads the ignored repository-root `.env`, disables interpolation and gives environment values precedence. Missing settings allow the CDS client's own configuration defaults. Current credential tests supersede the older public-only fail-fast tests. The public-only replay/saved-output portability tests remain and cover all four notebooks.
- The instrument notebook has one saved stdout path shortened to a repository-relative path. Code cells, scientific arrays, embedded images, execution counts and metadata retain the selected source values. Delivery preparation does not rerun the provider-dependent notebooks.
- `.agents/`, `AGENTS.md` and `.DS_Store` are excluded and ignored at any directory depth. Credentials, private provider inputs, caches, generated execution archives and development Git history remain local. Ignore rules supplement the explicit tracked-source selection; they do not remove previously committed files.

## License and scope

The owner reviewed the initial selected file set and confirmed Apache-2.0 for its library sources, tests, synthetic fixtures, diagrams and documentation. The publication owner confirms rights and applicable attributions for the historical saved figures in the included complete notebooks. The repository's Apache-2.0 declaration covers Orsia-authored notebook sources and figures; provider inputs used to produce embedded graphics retain their own terms and attribution requirements. The [notebook provenance guide](notebooks/README.md#provenance-and-attribution) identifies ERA5, HITRAN and Space-Track inputs and records the `current` edition label for HITRAN. Redistribution of separately acquired provider data follows the applicable provider terms. Scientific validation of instrument-space agreement, retrieval performance and engineering feasibility is a separate activity.
