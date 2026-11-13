# Orsia

Orsia is a work in progress. It provides reusable scientific Python libraries for remote-sensing observation modelling, instrument simulation, uncertainty propagation and retrieval. Applications can compose atmospheric radiance, observation geometry and detector response, then solve an inverse problem with their own physical model and derivatives.

The repository contains nine separately packaged libraries in a non-package UV workspace, targeting CPython 3.14 free-threaded. Applications and studies own scene and dataset selection, instrument assumptions, model composition and acceptance criteria. Library guides define the implemented contracts and their scientific limits.

## High-level design

![Caller-owned composition selects Data, Dispersion, Scene, Geometry, Radiative, Instruments and callback-based Inverse; Uncertainty and Numerics provide shared capabilities.](assets/architecture.svg)

The figure includes all nine libraries. Solid arrows show selected application calls; the dashed arrow shows Inverse evaluating caller-supplied prediction and Jacobian functions. [Dispersion](libs/orsia-dispersion/README.md) supplies transported excess mass fields; [Scene](libs/orsia-scene/README.md) composes supplied fields, terrain and optical inputs into direct radiance and explicit sensitivities on flat local surface queries. The scientific components are independently selectable within their declared dependencies. Their grouping describes responsibilities; arrows between library internals are omitted. [Full-size editable SVG](assets/architecture.svg).

- **Application composition:** the caller connects scientific values and operations, defines the forward model, supplies coordinate conventions and controls execution. Inverse accepts prediction/Jacobian callbacks or an explicit linear matrix; it has no Instruments dependency.
- **Resources and calculation:** Data owns explicit scientific-product acquisition, local readers and geospatial preparation. Pure numerical kernels consume materialised values. Geometry also has lazy SatKit orbital/frame adapters that can acquire public auxiliary files and use process-global caches; their resource contract applies when selected.
- **Shared capabilities:** Uncertainty supplies dense and source-projected covariance algebra to callers and consuming libraries. Numerics provides explicit tangent arithmetic, dense Jacobian assembly, general forward/transpose operator adapters and shared native headers, plus optional application-owned CPU scheduling with controlled native threads. CPU affinity is used when available; its absence produces a warning and execution without pinning. Supported controls include Apple Accelerate BLAS/LAPACK on macOS 15 and later; see the [executor contract](libs/orsia-numerics/docs/execution.md). Scientific calls remain sequential unless the application schedules them.

## Library guides

Each guide introduces its library with an example and links to detailed numerical contracts, ownership rules and evidence limits. The [consolidated refactor migration](docs/refactor-migration.md) records the implemented ownership, breaking API changes and validation scope.

| Library | Responsibility |
| --- | --- |
| [Dispersion](libs/orsia-dispersion/README.md) | Stationary plumes, Gaussian/compact puffs and bounded particles, with shared sampling and supported point/cell/path mass integrals. |
| [Scene](libs/orsia-scene/README.md) | Physical gas/terrain/optical composition, frozen cell scenes, path-resolved radiance and explicit sensitivities. |
| [Data](libs/orsia-data/README.md) | Explicit acquisition, local atmosphere/solar/spectroscopy/orbit/product readers and geospatial preparation. |
| [Geometry](libs/orsia-geometry/README.md) | Explicit frames and point/cell support, trajectories, attitudes, ray composition and analytic/callable terrain with bounded intersection searches. |
| [Radiative](libs/orsia-radiative/README.md) | Atmospheric state and spectroscopy, optical composition, reflected-solar and non-scattering LTE radiance with explicit tangents, physical spectral means. |
| [Instruments](libs/orsia-instruments/README.md) | Acquisition, optics, detector readout, uncertain calibration, native observation cubes and separate ground-grid reconstruction. |
| [Inverse](libs/orsia-inverse/README.md) | Fixed-covariance nonlinear Gaussian estimation, rank-aware linear inversion and local retrieval diagnostics. |
| [Uncertainty](libs/orsia-uncertainty/README.md) | Dense/source-projected covariance and transport, low-rank solves, signed bias, cross-covariances and explicit-RNG ensembles. |
| [Numerics](libs/orsia-numerics/README.md) | Explicit tangents and dense Jacobians, general linear operators, shared native headers and optional CPU execution with native thread controls and available affinity. |

The timed [instrument line-of-sight contract](libs/orsia-instruments/docs/line-of-sight.md#time-frames-support-and-ownership) uses elapsed SI seconds from a timezone-aware UTC epoch and positions, velocities and rays in the geocentric celestial reference system (GCRS). Terrestrial inputs require explicit dated frame conversion. Detailed units, shapes, support and mutation rules belong to the linked library references.

## Getting started

Choose a guide for the calculation you need, then run its Python blocks in order with the relevant distribution and dependencies installed. The [development environment](#development-environment) installs the workspace libraries together.

| Start with | What it demonstrates | Inputs and resources |
| --- | --- | --- |
| [Radiative solar example](libs/orsia-radiative/README.md#minimal-solar-example) | A transparent atmosphere and Lambertian reflection. | Synthetic in-memory values. |
| [Instrument image example](libs/orsia-instruments/README.md#minimal-image-example) | A uniform radiance field, optical filter, detector and seeded noisy reads. | Synthetic in-memory values. |
| [Native observation cube](libs/orsia-instruments/docs/observation-cubes.md) | Actual DN calibration, shifted band supports, fixed accumulation, masks and common gain uncertainty, with dense/projected agreement. | Synthetic in-memory inputs; optional saved support/covariance figure. |
| [Numerics tangent example](libs/orsia-numerics/README.md#minimal-tangent-example) | A dense Jacobian and a directional derivative from one explicit callback. | Synthetic in-memory values; no executor required. |
| [Inverse fit](libs/orsia-inverse/README.md#minimal-example) | Caller-defined prediction/Jacobian functions and local posterior diagnostics. | Synthetic observations and covariance. |
| [Geometry ground trace](libs/orsia-geometry/README.md#construct-a-leo-trajectory-and-ground-trace) | A designed orbit, independent attitudes and WGS84 intersections. | SatKit with dated orbital auxiliaries; the guide gives an offline-fixture recipe. |
| [Spatial atmosphere and plume acquisition](docs/spatial-scenes.md) | Terrain-adjusted columns, independent solar/view paths, native camera images and signed electron Jacobians. | Configurable synthetic in-memory fields; offline execution. |
| [Maintained notebooks](notebooks/README.md) | Dispersion and spatial-scene tutorials, five-band radiative comparison and six instrument-response chains. | Self-contained synthetic tutorials or prepared atmospheric inputs; acquisition and execution are separate operations. |

The Data, Uncertainty and Numerics guides provide additional focused examples. The [guide tests](tests/test_library_guides.py) execute dependency-light examples with sockets blocked; the orbital example has a separate native-fixture check. See [documentation checks](DEVELOPMENT.md#documentation-checks) for the commands.

## Development environment

Prerequisite: Devbox with Nix configured. From this repository root, provision the pinned toolchain, interpreter and locked workspace:

```bash
MAKEFLAGS=-j2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 UV_CONCURRENT_BUILDS=2 devbox run -- just setup
```

Setup can download tools and dependencies. For an already provisioned environment, use `UV_NO_SYNC=1` to prevent package synchronization during checks. Common commands are:

```bash
UV_NO_SYNC=1 devbox run -- just lint
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just --justfile tests/justfile pytest tests/test_library_guides.py
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just check
```

`check` runs formatting, lint, typing and the full split-process test/coverage gate. Use focused checks while working, and the full gate for repository-wide validation. [DEVELOPMENT.md](DEVELOPMENT.md) covers environment ownership, optional backends, native builds, platform limits, recipe arguments and coverage policy.

## Scientific maturity and limits

Orsia has executable examples, synthetic regressions and bounded reference comparisons. Their conclusions apply to the tested inputs, observables and numerical settings. Scientific qualification remains case-specific:

- [Radiative transfer](libs/orsia-radiative/docs/transfer.md#evidence-and-limits) includes experimental fixed-order solar scattering and absorption-only LTE thermal transfer. Order, angle, layer and spectral convergence need separate assessment; broad atmospheric accuracy remains unqualified.
- [Data interoperability](libs/orsia-data/README.md#resource-and-evidence-boundaries) depends on the reader and provider. Local synthetic checks leave authentic dataset coverage and live-provider behavior subject to the documented evidence limits.
- [Instrument simulation](libs/orsia-instruments/README.md#physical-conventions-and-limits) and [inverse estimation](libs/orsia-inverse/README.md#fit-and-diagnostic-rules) retain explicit support, calibration, rank and model-validity conditions. Output sampling, physical resolution, identifiability and geophysical retrieval accuracy require distinct evidence.
- [Native execution](libs/orsia-numerics/docs/execution.md) requires explicit resource control and stable inputs. A disabled GIL or successful software gate leaves concurrent native-library safety, scientific accuracy and hardware feasibility unqualified.

The [notebook guide](notebooks/README.md) records the assumptions and unresolved numerical sensitivities of the larger demonstrations. Software coverage policy and contributor guidance are in the [development reference](DEVELOPMENT.md#test-coverage-policy).

## License

Orsia-authored source, tests, synthetic fixtures, documentation, notebooks, diagrams, saved notebook figures and diagram SVG/PNG exports are distributed under Apache-2.0. External inputs used to generate notebook figures retain their provider terms and attribution requirements. See the root [LICENSE](LICENSE), each library's `LICENSE` and the [notebook provenance guide](notebooks/README.md#provenance-and-attribution).
