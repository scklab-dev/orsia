# Developing Orsia

Use this reference for workspace setup, focused checks and the full software gate. The [root README](README.md) introduces the libraries and example choices. Run the commands below from the Orsia repository root.

## Environment ownership

| Configuration | Owns |
| --- | --- |
| [devbox.json](devbox.json) and [devbox.lock](devbox.lock) | Development tools and shell initialization. |
| [.python-version](.python-version) | The CPython 3.14.3 free-threaded interpreter pin. |
| [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) | Workspace members, development dependencies, Python resolution and check configuration. |
| `libs/*/pyproject.toml` | Each library's runtime and build requirements. |

The root is a non-package UV workspace. Its `.venv` belongs to this repository. Devbox pins Git, Just, ripgrep, UV, rustup, ecCodes and the gfortran/NetCDF Fortran tools for the optional MT_CKD comparison. `UV_PYTHON_PREFERENCE=only-managed` keeps host Python out of interpreter selection. Run recipes through Devbox to use those tools; preserve lockfiles unless a dependency change is intended.

Prerequisite: Devbox with Nix configured. Setup installs the pinned Rust toolchain and interpreter, then synchronizes the locked workspace:

```bash
MAKEFLAGS=-j2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 UV_CONCURRENT_BUILDS=2 devbox run -- just setup
```

Setup can download tools and packages. `UV_NO_SYNC=1` preserves an already provisioned Python environment during `uv run` checks; it requires the selected dependencies to be installed. `UV_OFFLINE=1` additionally prevents UV network access. Neither variable disables network access inside application code or scientific backends. Data's product acquisition remains explicit; Geometry's native adapters can also acquire public auxiliaries.

Ruff, Pyrefly, pytest, pytest-cov, Coverage.py, NumPy/SciPy and SASKTRAN2 are workspace development dependencies. Scientific distributions declare their own runtime requirements independently. SASKTRAN2 is a development/reference dependency and runs in separate worker processes using the same workspace interpreter; its imports stay outside the calling Orsia process.

## Commands and arguments

The [root justfile](justfile) owns lifecycle commands:

| Recipe | Purpose |
| --- | --- |
| `setup` | Install the pinned toolchain/interpreter and synchronize the workspace. |
| `fmt [paths...]` | Format Python files and apply lint fixes. |
| `fmt-check [paths...]`, `lint [paths...]` | Check formatting or lint without changing source files. |
| `typecheck [args...]` | Run Pyrefly with geospatial/HAPI import extras; default to the configured project. |
| `test` | Run the split-process suite and aggregate/per-library coverage gates. |
| `check` | Run formatting, lint, typing and the full test gate. |
| `build` | Build all nine library distributions. |
| `python <args...>` | Run an example or validation script in the locked workspace. |
| `lock` | Resolve an intended dependency change with UV. |
| `rebuild-native` | Reinstall the editable radiative extension after C++ changes. |

With no paths, formatting and lint recipes select `notebooks/support`, `tests`, `libs/*/src` and `libs/*/tests`. Arguments preserve shell quoting. Run parameterized recipes separately because trailing arguments belong to that recipe. For an existing environment:

```bash
UV_NO_SYNC=1 devbox run -- just fmt libs/orsia-inverse
UV_NO_SYNC=1 devbox run -- just typecheck libs/orsia-inverse/src/orsia_inverse/nonlinear/map.py
UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_core_imports.py -k 'native or pinned'
```

[tests/justfile](tests/justfile) owns focused and opt-in tests. Its `pytest` recipe requires at least one argument, selects geospatial extras by default and disables coverage. Override the dependency set with `extra=atmosphere` before `pytest` when needed. Keep atmospheric/data-reader tests separate from numerical concurrency tests, as described in the [coverage policy](#test-coverage-policy).

[notebooks/justfile](notebooks/justfile) owns preparation, execution and replay. The [notebook guide](notebooks/README.md) gives required inputs, acquisition effects and execution commands. Execution and replay use `--no-sync`; execution updates the canonical notebooks in place.

List each command interface with:

```bash
devbox run -- just --list
devbox run -- just --justfile tests/justfile --list
devbox run -- just --justfile notebooks/justfile --list
```

Devbox starts `run` commands at the repository root. Retain the explicit `--justfile` path when launching a child interface from another directory. Inside a Devbox shell, `just` in `tests/` or `notebooks/` selects that directory's interface. Both child interfaces execute from the root and list recipes when no recipe is supplied. Paths passed to recipes remain root-relative.

## Resource and platform controls

Devbox defaults OpenMP, OpenBLAS, MKL and macOS Accelerate to one numerical-library thread. Leave capacity for other work when selecting worker counts. Bound native builds with `MAKEFLAGS=-j2`, `CARGO_BUILD_JOBS=2`, `CMAKE_BUILD_PARALLEL_LEVEL=2` and `UV_CONCURRENT_BUILDS=2`. Use `MPLBACKEND=Agg` for headless tests, including `check`, to avoid GUI backends in synthetic notebook checks.

The [Numerics executor](libs/orsia-numerics/docs/execution.md#explicit-cpu-execution) requires supported native thread controls: pthreads OpenBLAS or Apple Accelerate BLAS/LAPACK with the public macOS 15+ APIs, plus one optional OpenMP runtime. It uses Linux affinity when both affinity APIs are available; otherwise construction emits `RuntimeWarning` and workers run without pinning, including on macOS. Common executor tests simulate affinity and native controls where needed. Numerical integration tests also exercise unpinned execution with real NumPy/SciPy. On macOS with Accelerate loaded, a dedicated test checks worker-local BLAS/LAPACK modes and unchanged caller settings. Real pinning/OpenMP integration tests require Linux and are skipped on macOS. Notebook runtime records use `affinity: null` when the operating system lacks an affinity API.

GIL checks run in fresh processes after relevant native imports. Leave `PYTHON_GIL` unset: forced-off overrides invalidate those checks. Import success and a disabled GIL leave concurrent native-library safety unqualified. Applications own pool lifetimes, task decomposition, stable shared inputs and disjoint writable outputs.

### Native builds

The pinned SASKTRAN2 development dependency requires Rust `nightly-2025-10-24` for its source build. Setup installs the toolchain into repository-local `.devbox/rustup` and `.devbox/cargo`. UV builds SASKTRAN2 in the shared free-threaded environment. Reference workers record GIL state after imports; process isolation contains import effects but leaves backend thread safety unqualified.

The radiative extension uses C++17 and pybind11. Its package declares isolated build requirements; the workspace lock also pins pybind11 for development. The Devbox initialization hook appends `-mtls-dialect=gnu` to `CXXFLAGS` on Linux x86-64, preserving existing flags. This Linux GCC TLS workaround leaves macOS and Linux arm64 flags unchanged and stays outside the package's numerical compiler policy. Use `rebuild-native` after C++ changes; it changes the environment by reinstalling `orsia-radiative`.

## Focused and optional tests

Run only the scopes needed for the changed behavior during development. The full `test`/`check` commands retain the repository-wide gates.

| Recipe in `tests/justfile` | Scope and prerequisites |
| --- | --- |
| `pytest <args...>` | Explicit test selection; coverage disabled. Keep reader and numerical concurrency scopes separate. |
| `test-hitran-api` | Optional HAPI with intercepted synthetic HTTP bytes. |
| `test-atmosphere-backends` | Generated-file reader checks with atmosphere extras and Devbox ecCodes. |
| `test-orbital [args...]` | Native Geometry/Instruments checks using existing DE421/EOP files in `outputs/geometry-auxiliaries`, with acquisition disabled. Missing fixtures fail. |
| `test-instrument-characterization` | Opt-in synthetic refinement/statistics and Linux `/proc`-based 30 s / 512 MiB resource checks. |

For mocked orbital acquisition, use `SATKIT_OFFLINE=1 UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest libs/orsia-data/tests/test_orbits.py libs/orsia-data/tests/test_space_track.py`. Geometry's ordinary SatKit auxiliary acquisition remains available outside the fixed offline recipe. Native resource and scientific evidence limits are in the [Geometry contract](libs/orsia-geometry/docs/geometry.md#resources-provider-boundary-and-execution).

### Documentation checks

Run the existing guide suite after changes to introductory examples or links:

```bash
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_library_guides.py
```

The suite checks local links and anchors in the root README, this development reference, library READMEs and package-local thematic guides. It executes selected Python examples with sockets blocked. Geometry's README requires native auxiliaries and is checked separately with `devbox run -- just --justfile tests/justfile test-orbital -k readme -s`. Numerics' example executes without pinning and emits a warning when an affinity API is unavailable. These checks cover documentation/software contracts; live providers, scientific accuracy and native thread safety have separate qualification requirements.

Library-source Ruff rules check module/class/function documentation, selected layout, existing Google-style `Args` completeness and explicitly selected preview return/yield/exception rules. [pyproject.toml](pyproject.toml) lists the exact rule codes. Human review covers missing argument sections, field/parameter meanings, units, ownership, semantic accuracy and exceptions raised by callees. Tests and notebook companions retain code lint without library docstring requirements.

## Test coverage policy

Run the full software gate in the existing environment with:

```bash
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just check
```

`test` runs two pytest processes in the same `.venv` to isolate optional native-import effects. The first excludes `libs/orsia-data/tests` and `tests/test_five_band_notebook.py`; the second runs those scopes and appends coverage. Some synthetic fixtures import xarray/cftime and enable the GIL during collection, so this separation preserves meaningful numerical and concurrency checks. Atmospheric readers defer their direct SciPy/h5netcdf/ecCodes imports. xarray remains a workspace dependency for preparation, fixtures and SASKTRAN2.

The first pytest process resets prior coverage. The second enforces the aggregate threshold, then separate reports enforce each library's threshold. [pyproject.toml](pyproject.toml) requires at least 90% combined statement-and-branch coverage overall and independently in each of the nine libraries. The percentage combines executed statements and branch destinations in a single threshold. Every module under the library `src/` trees contributes, including unimported modules. Tests and development scripts are outside that denominator; no project-specific coverage omissions or exclusion pragmas are configured.

Reuse existing checks, add cases for uncovered risks or explicit requirements, and report the exact scope run. Software coverage measures execution; instrument agreement, retrieval performance and engineering feasibility need their own evidence.

## Local artifacts and contribution scope

Maintained notebooks keep preparation, display, reference-cache and replay helpers in [notebooks/support](notebooks/support/README.md). Tests and fixtures live in `tests/` and `libs/*/tests/`. Package-local `docs/` directories own detailed scientific contracts and editable diagrams. The workspace design SVG lives in [assets/architecture.svg](assets/architecture.svg).

Ignored `outputs/` holds local inputs, original provider bytes, notebook/reference caches, execution receipts and replay baselines. Preserve inputs and receipt lineage needed for replay. Regenerable plots and caches can be removed when their documented workflow can recreate them. Treat other retained local work as owned by its author; credentials and copyrighted raw products stay outside version control.

Keep contributions scoped to the intended change and preserve unrelated work. Use the library guides, public contracts and maintained tests to assess behavior. Keep credentials, acquired provider data, caches, generated outputs and macOS `.DS_Store` files outside version control.
