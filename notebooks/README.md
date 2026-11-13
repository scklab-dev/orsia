# Orsia notebooks

These notebooks demonstrate public Orsia APIs with visible scientific calls, figures and diagnostics. Choose the self-contained puff-to-plume tutorial for dispersion or the spatial-scene tutorial for the connection from a frozen plume to native camera reads. The five-band comparison covers radiative transfer; the instrument notebook covers acquisition, detector noise and reconstruction.

## Choose a notebook

| Notebook | Question and products | Main limits |
| --- | --- | --- |
| [Gaussian puff dispersion](gaussian_puff_dispersion.ipynb) | Move and spread one puff, build a plume from continuous emissions, and compare concentration with height-integrated mass. | Synthetic uniform wind and diffusivity; finite-duration passive source in free space; no downloads. |
| [Spatial scene and native camera reads](spatial_scene.ipynb) | Inspect a frozen puff, compare direct-solar paths, follow callback frame/sign/shape conversion, and read native expected electrons and two sensitivities. | Synthetic spectroscopy and motion; direct solar only; reduced teaching quadrature; no retrieval or reconstruction. |
| [Five-band transport comparison](era5_hitran_sasktran2_five_bands.ipynb) | Compare Orsia and SASKTRAN2 radiance in UV, VNIR, SWIR, MWIR and LWIR windows; inspect spectra, residuals and complete-call timing tables. | Shared-input scene comparison with prescribed solar aerosol and supported-band water continuum; UV remains a lines-only water control. |
| [Six instrument-response chains](era5_hitran_instrument_responses.ipynb) | Follow uniform filter, mosaic, LVF, slit, two-beam and Fabry-Perot designs from radiance to electrons, noisy digital numbers (DN) and reconstructed products. | Illustrative optical layouts and supplied calibration; bounded 2-by-2 target patch. The two-beam inverse is rank-deficient and has unresolved quadrature sensitivity. |

Read the notebooks without executing them to inspect their sources and saved examples. All four notebooks contain saved outputs and execution counts. These records describe the retained runs and their inputs; fresh local results require explicit execution. Existing local exports and reference records retain their recorded inputs and scope. The detailed guides retain the scientific assumptions and provenance:

- [Five-band inputs, physics and reference evidence](five-band.md).
- [Instrument model, products and numerical limits](instrument-responses.md).

## Prerequisites

Run all commands below from the Orsia repository root. The [development environment](../README.md#development-environment) supplies Devbox, Just and the pinned CPython 3.14 free-threaded workspace. The dispersion and spatial-scene notebooks need NumPy, SciPy, Matplotlib and Jupyter. The five-band and instrument notebooks also need atmospheric reader dependencies and the local inputs described below. SASKTRAN2 reference workers use the same workspace interpreter in separate processes.

Keep numerical libraries single-threaded and builds bounded. Set these variables in the shell used for preparation, execution and tests:

```sh
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAKEFLAGS=-j2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 UV_CONCURRENT_BUILDS=2
```

For a new environment, run `devbox run -- just setup`. For the five-band and instrument notebooks, prepare atmospheric inputs as described below. An already provisioned notebook environment can use `export UV_NO_SYNC=1` to preserve its installed optional dependencies during checks and preparation. All execution recipes use `uv run --no-sync`.

### Use retained atmospheric inputs

The five-band and instrument notebooks require the enriched acquisition manifest, offline spectroscopy collection and separately receipted MT_CKD 4.3 water coefficients. Their default locations are `outputs/five-band-enriched-inputs/manifest.json`, `outputs/spectroscopy/five-band/` and `outputs/spectroscopy/mt-ckd-4.3/manifest.json`. The manifest also references baseline files under `outputs/five-band-inputs/`. These ignored files are local prerequisites and are absent from a clean checkout.

With those acquisition files provisioned, verify the enriched receipt and import the spectroscopy collection and retained native water coefficients offline on first use:

```sh
UV_NO_SYNC=1 devbox run -- just --justfile notebooks/justfile prepare-five-band-enriched
UV_NO_SYNC=1 devbox run -- just --justfile notebooks/justfile prepare-spectroscopy --water-continuum /path/to/retained-MT_CKD-4.3.nc
```

Later calls omit `--water-continuum` to verify the existing references offline. The water import copies native bytes and records their SHA-256 and import time; acquisition date and source authentication remain caller-owned. Notebook cells verify both the separate receipt and native-file hash, then call `read_mt_ckd_water` with the actual layer-temperature envelope. Missing files and checksum mismatches stop these operations. Keep each receipt with its referenced files; execution verifies hashes again.

### Acquire missing atmospheric inputs explicitly

Acquisition contacts external providers and writes local data under `outputs/`. Run it only when you intend to download data and have the required provider access. Copy [`.env.example`](../.env.example) to the ignored repository-root `.env` and supply your own `CDSAPI_URL` and `CDSAPI_KEY`. The [baseline helper](support/prepare_five_band.py) reads these two settings from that file with interpolation disabled; existing environment values take precedence. If settings are absent, the CDS client applies its own configuration defaults.

```sh
devbox run -- just --justfile notebooks/justfile prepare-five-band
devbox run -- just --justfile notebooks/justfile prepare-five-band-enriched --download
devbox run -- just --justfile notebooks/justfile prepare-spectroscopy --water-continuum /path/to/retained-MT_CKD-4.3.nc
```

Obtain missing water coefficients separately through the existing explicit [`download_mt_ckd_water(edition="4.3", destination=...)` interface](../libs/orsia-data/docs/readers-and-acquisition.md#scientific-acquisition), then pass the retained native file to offline preparation. Notebook execution never downloads coefficients.

The baseline recipe requests the `hitran` and `atmosphere` extras. If `UV_NO_SYNC=1` is set, those dependencies must already be installed. Preparation acquires ERA5 meteorology, HITRAN lines, solar irradiance and the added trace-gas references. Provider editions labelled `current` can change; local receipts identify the retained bytes. See [input provenance](five-band.md#input-preparation-and-provenance) before changing the data or interpretation.

## Run a notebook

Choose one execution recipe:

```sh
devbox run -- just --justfile notebooks/justfile execute-dispersion
devbox run -- just --justfile notebooks/justfile execute-spatial-scene
devbox run -- just --justfile notebooks/justfile execute-five-band
devbox run -- just --justfile notebooks/justfile execute-instrument-responses
```

Each recipe registers a project-local kernel, starts a fresh kernel, and executes its canonical notebook in place. Dispersion uses `orsia-dispersion-314t` under `outputs/dispersion-notebook/jupyter/` with a 120-second cell timeout. The atmospheric notebooks use `orsia-314t` under `outputs/jupyter/` with a 3600-second cell timeout. The recipes use a headless backend; the notebooks produce inline figures. Successful execution replaces the notebook's saved outputs, counts and timestamps. Preserve a copy first if you need the previous execution record. A cell failure returns an error and leaves the previously saved notebook intact; inspect the terminal traceback before retrying. Instrument figure export runs after notebook execution, so an export failure can leave an updated notebook with incomplete exported figures.

### Gaussian puff dispersion

[From one puff to a plume](gaussian_puff_dispersion.ipynb) defines every input in its cells and runs without atmospheric files or provider access. It saves three figures and checks constant-forcing transport, Gaussian peak concentration, release accounting and a finite-domain mass integral. The recipe caps native numerical threads at one and writes its kernel under `outputs/dispersion-notebook/`; it preserves the existing environment with `--no-sync`.

Run its focused regression checks independently of the atmospheric notebooks:

```sh
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_dispersion_notebook.py
```

### Spatial scene

[From a frozen plume to native camera reads](spatial_scene.ipynb) explains [spatial_scene.py](support/spatial_scene.py) through seven figures and small executable examples. It uses synthetic inputs without downloads. The [spatial-scene guide](../docs/spatial-scenes.md) retains the physical contracts and continuous coverage argument. Saved figures distinguish chart samples from native detector arrays; the finite-difference check covers two fixed-geometry parameter directions.

[`SyntheticRadianceModel`](support/spatial_scene.py) owns shared radiative composition and background preparation for its bound `radiance` and `radiance_tangent` methods. The separate curved-terrain illustration uses [`SplineTerrain`](support/terrain.py), which owns the terrain grid, cubic interpolant and closed domain. Its `height` and `gradient` methods query that interpolant. The camera example retains affine terrain and its continuous coverage proof.

The recipe registers `orsia-spatial-scene-314t` under `outputs/spatial-scene-notebook/jupyter/`, uses a 120-second cell timeout, caps native numerical threads at one and preserves the installed environment with `--no-sync`. It executes the canonical notebook in place with saved inline figures. Run its isolated headless regression check with:

```sh
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_spatial_scene_notebook.py
```

The test executes the actual cells, checks their assertions and PNG displays, and verifies retained execution counts. Numerical agreement checks the synthetic discrete calculation; physical qualification and quadrature convergence require separate evidence.

### Five-band transport comparison

Edit [five-band.json](data/five-band.json) for spectral sampling, physics, solver controls and cache policy. The scenario loader validates fields and input hashes. UV/VNIR/SWIR include gas absorption, Rayleigh scattering and a prescribed aerosol column; MWIR/LWIR are dark absorption-only controls. VNIR/SWIR/MWIR/LWIR pair pedestal-subtracted local H2O lines with one matching-grid MT_CKD self/foreign contribution. The retained water table ends at 20000 cm-1; UV is outside its complete-stencil support and retains plain truncated H2O lines without continuum. The current aerosol/continuum configuration still needs dedicated convergence and physical-model validation.

Successful execution saves the atmospheric profile plot, five spectrum/residual figures, difference summaries and timing tables in the notebook. Reference records live under `outputs/sasktran2-reference-cache/`. Cache reuse retains original reference timings and computation dates; choose `refresh` or `off` when fresh timing comparisons are required. [Reference caching and timing scope](five-band.md#persistent-sasktran2-references) explain cache identity and invalidation.

### Instrument responses

The instrument notebook uses an aerosol-free gas/Rayleigh atmosphere with pedestal-subtracted H2O local lines and one matching-grid MT_CKD self/foreign water continuum, the [local passage scenario](data/gesat-toulouse-2025.json) and six explicit instrument chains. The displayed 5 km by 5 km scene supplies context; the default execution reconstructs a bounded target region and illuminated neighboring cells. The [instrument guide](instrument-responses.md) explains read selection, achieved responses, supplied calibration and inverse limitations.

The execution recipe writes these products under `outputs/instrument-responses/`:

| Product | Contents |
| --- | --- |
| `figures/chains/` | PNG figures extracted from the executed notebook. |
| `spectral-scene.npz` | Atmospheric and instrument grids and retained spectral radiance. |
| `chains.npz` | Selected electrons/DN, native identities and timing, estimated coordinates, responses, products, coverage, references and inverse arrays. |
| `evidence.json` | Source/input hashes, refinement results, budgets, timings and observed GIL/thread settings. |
| `notebook-source.txt` | Cell IDs, types and sources without image payloads. |

After a successful full execution, replay the instrument cells against the retained spectral scene:

```sh
devbox run -- just --justfile notebooks/justfile replay-instrument-response
```

Replay verifies instrument source hashes and compares seeded arrays against `chains.npz`. It writes separate arrays, evidence and `comparison.json` under `outputs/instrument-response-replay/`, leaves canonical displays unchanged, and excludes atmospheric recomputation. Missing inputs, changed source hashes, failed numerical checks or mismatched arrays stop replay.

## Checks and interpretation

Run focused tests in the provisioned environment, keeping reader-dependent tests in their separate process:

```sh
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile extra=atmosphere pytest tests/test_notebook_sasktran2_cache.py tests/test_five_band_notebook.py tests/test_notebook_companions.py
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile extra=atmosphere pytest tests/test_instrument_response_notebook.py tests/test_instrument_notebook_chains.py
```

These tests cover input integrity, array propagation, cache behavior and bounded synthetic numerical composition. The instrument chain tests use 33 spectral samples and reduced integration orders, and mock figure creation. PNG extraction tests inspect stored bytes; these checks do not render or visually validate notebook figures. Full notebook execution and inspection provide separate evidence for the default grids and displays.

Software composition, numerical convergence, instrument-space agreement and geophysical retrieval accuracy have different validation scopes. These notebooks do not establish mission performance, hardware feasibility or concurrent native-library safety. Use the detailed guides for unresolved sensitivities and historical evidence; use each execution receipt for its actual inputs, settings and timings.

### Trace-gas enrichment methodology

The [enrichment methodology](five-band.md#trace-gas-enrichment-methodology) retains species selection, concentration conversion, laboratory temperature/pressure assumptions and source provenance. This link also serves the notebook's existing methodology reference.

## Read next

- [Dispersion library](../libs/orsia-dispersion/README.md): passive transport and mass-field evaluation.
- [Scene library](../libs/orsia-scene/README.md): spatial radiance, sensitivities and observation callbacks.
- [Radiative library](../libs/orsia-radiative/README.md): physical conventions, spectroscopy and transfer APIs.
- [Instruments library](../libs/orsia-instruments/README.md): detector, optical response, acquisition and reconstruction contracts.
- [Inverse library](../libs/orsia-inverse/README.md): rank-aware estimation and diagnostic interpretation.
- [Data library](../libs/orsia-data/README.md): readers, acquisition and spectroscopy collections.
- [Notebook support](support/README.md): preparation, display, replay and reference-worker responsibilities.
- [Notebook recipes](justfile) and [test recipes](../tests/justfile): maintained command definitions.

## Saved execution scope

All four notebooks retain outputs and execution counts from their saved runs. Fresh local results require explicit execution with the documented inputs. The delivery shortens one saved instrument archive path to a repository-relative path. Code cells, arrays, image payloads, counts and execution metadata retain the selected source values. Run `MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_notebook_portability.py` for offline saved-output and missing-baseline checks. Credential configuration is covered by `tests/test_five_band_notebook.py` in the separate reader-dependent phase.

## Provenance and attribution

The saved notebook graphics are author-created outputs derived in part from ERA5 and HITRAN inputs. The instrument scene also uses the [Space-Track-derived scenario](data/gesat-toulouse-2025.json). The publication owner confirms rights and applicable attributions for these historical saved figures in the complete notebooks. The repository's Apache-2.0 declaration covers Orsia notebook code and author-created figures; provider terms and attribution apply to the underlying inputs.

- **ERA5:** The notebooks use a pressure-level profile for 2024-01-15 12:00 UTC at 0° latitude and longitude. The [ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) lists DOI `10.24381/cds.bd0915c6` and a [CC-BY licence](https://cds.climate.copernicus.eu/licences/cc-by). Attribution for the derived figures: “Contains modified Copernicus Climate Change Service information (ERA5 profile dated 2024-01-15).” The notebook authors alone bear responsibility for use of Copernicus information or data in these notebooks. The [ECMWF citation guide](https://confluence.ecmwf.int/spaces/CKB/pages/197704114/How+to+acknowledge+and+cite+a+Climate+Data+Store+CDS+catalogue+entry+and+the+data+published+as+part+of+it) calls for dataset citation and Copernicus attribution. The [Copernicus product licence](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) excludes datasets flagged with separate terms (clause 1.5); section 5 specifies attribution for products covered by that licence. The local manifest records the ERA5 request and labels the edition `current`.
- **HITRAN:** The spectral inputs come from HITRAN. The [HITRAN citation policy](https://hitran.org/citepolicy/) requests the article for the database edition actually used. The local manifest labels the acquisition `current`; verify its actual edition with the provider before selecting an article to cite.
- **Scenario geometry:** The [scenario snapshot](data/gesat-toulouse-2025.json) records Space-Track GP_HISTORY elements propagated with Skyfield. [Space-Track documentation](https://www.space-track.org/documentation) requires appropriate citation for published analyses based on USSPACECOM data. Cite Space-Track alongside the scenario geometry.

Local acquisitions also include LISIRD, MIPAS, Zenodo and Bremen material. Redistribution of separately acquired provider data follows the applicable provider terms. The [candidate manifest](../CANDIDATE-MANIFEST.md#license-and-scope) records the owner's confirmation for the included historical saved figures.
