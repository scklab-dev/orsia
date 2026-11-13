# Consolidated Orsia refactor: implementation and migration

**Status: implementation and software verification complete.** The scientific and optional-backend limits below remain applicable.

This guide records the implemented ownership boundaries and breaking API changes in the nine-library workspace. The maintained library guides, public contracts and tests provide the current reference. Common root exports remain available under their accepted names; obsolete defining modules and replaced names have no compatibility shims.

## Implemented boundaries and contracts

| Area | Implemented responsibility and contract | Evidence and scope |
|---|---|---|
| Nine library boundaries | Geometry owns supports; Dispersion owns mass fields; Radiative owns material, coefficients and transfer; Scene owns physical spatial composition; Instruments owns measurement; Inverse consumes numerical models. | Root ownership, dependency-cycle, base-import and installed-package checks. No new study application or provider acquisition is embedded in numerical code. |
| Point and cell support | `PointLattice`, `RectangularCells`, `LayeredCells` and explicit `LocalCartesianFrame` preserve coordinates and datum declarations. Mass-to-molecular conversion retains the complete cell support. | Equal-shape translated, reordered, resized and differently framed support cases; finite mass/moment and unit checks. |
| Spectral values and conversions | `Radiance` and `SpectralIrradiance` accept nm or cm-1 with matching density units. Existing point/density conversion functions accept an explicit spectral `axis`. | Reciprocal round trips, disconnected coverage, signed multidimensional tangents, mutable-input isolation and discrete response-pairing checks. The cm-1 accessor and restricted consumers reject nm rather than relabelling values. |
| Atmospheric representations | `LayerAtmosphere` and `LevelAtmosphere` remain distinct. `AtmosphericColumnIntegrals` separates material reconstruction from spectroscopy and uses optional zero-based interval indices. | Partial-layer, endpoint, species-order and water-density-product tests retain their distinct numerical laws. |
| Coefficient ownership | `LayerCrossSections` and `WaterContinuumCoefficients` retain exact spectral/layer support and declared evaluation conditions. Explicit resampling returns the same cross-section concept. | Snapshot, singleton, gap, exact-condition and water-line/continuum pairing checks. Historical coefficients without recovered T/p provenance remain labelled numerical references. |
| Physical scenes | `DirectSolarScene` owns frozen cell-scene preparation, physical query validation, explicit continuation and gas/surface directions. Separate path-resolved direct/tangent functions live in `orsia_scene.radiance`; molecular composition lives in `gas`. | Independent direct-transfer, centroid-sign, exact-subset, frame/time/mask, null-amplitude and finite-difference cases. Arbitrary path-column directions retain their existing function contract. |
| Dispersion semantics | `StationaryGaussianPlume`, puff and particle fields retain their existing model scopes. `path_columns` carries a puff field's reflection and compact cutoff into Scene. | Independent path/concentration comparisons, support rejection and preserved sampling/RNG tests. Raw particles still require explicit finite-cell integration. |
| Spectral transfer and averaging | `transfer.extinction` owns spectral exposure means and explicit log-tangent approximation. Application parameter binding and band integration are separate. | Reference value/slope, off-reference differences, direct formulas and signed derivative tests. The old mixed band renderer is retained only as a labelled test reference. |
| Native photon integration | `photodetection.frames` shares read/spectral-block coefficients between expectation, characterization, tangents and joint operators. Full-domain quadrature is prepared before slicing. | Block-size, disconnected-domain, units, native-order, normalization, product-rule and transpose checks. Geometry/time directions need total radiance query-motion terms. |
| Response representations | `GridElectronResponse` replaces duplicate constructor-labelled records. `SpectralElectronResponse`, `NativeBandResponse` and `SpatialSpectralResponse` retain their distinct bases and information. | Nonsquare interpolation, exact support/frame, constructor parity and operator tests. No inheritance hierarchy or false equivalence between marginals and joint response is introduced. |
| Filter ownership and coaddition | `FilterPassband`, `FilterElectronResponse`, actual `ReadSchedule` durations and explicit QE/transmission/etendue inputs replace the mixed parameter bundle. Coaddition is an explicit retained linear map. | Previous per-nm formulas, fixed-area profile directions, one-time preparation and read-noise covariance accounting. No read chronology or on-chip accumulation model is inferred. |
| Spatial response | Gaussian/pixel transfer is shared by replicated-boundary filtering and explicit zero-extended Fourier sampling. Fourier uses point lattices, explicit reference pitch/active widths and preserve/clip/reject policies. | Dense inverse-FFT, displacement, mask, signed-ringing and policy tests. Optional FINUFFT plumbing is tested with a deterministic fake; native FINUFFT is unavailable locally. |
| Surface-field resampling | Geometry's `resample_point_field` supplies explicit bilinear point-field resampling with reject/nearest continuation and conservative endpoint validity. Scene supplies reflectance; Instruments supplies physical contraction. | Signed and nonsquare field tests replace the mixed `surface_anchor_charge` helper. No band-name convention, inferred centre alignment or hidden coadd division remains in production. |
| Observation products | `BandRadianceCube` is defined in Instruments' observation family. `SpectralBand` and `spectral_band` remain Radiative physical spectral means. Native logical observation cubes retain their separate geometry and uncertainty semantics. | Product axis, NaN/validity, ownership, calibration and reconstruction checks. Registered means and native-band supports remain distinct. |
| Inverse contracts | Common `GaussianEstimate`, `FitTermination`, explicit quadratic priors, separate/fused model evaluation and shared local information coexist with distinct MAP and damped algorithms. | Analytic linear cases, rank/bounds/failure handling, fixed-geometry derivatives and actual-source uncertainty transport. Numerical damping is excluded from posterior information. |
| Scientific names | Scientific names are applied consistently to definitions, exports, callers, tests, guides and affected diagrams. | Canonical-owner tests, stale-name searches, executable examples and unchanged numerical reference data. Names alone establish no scientific validity. |

## Responsibility homes

Related implementation files are grouped within a responsibility home. Instruments uses domain packages for `acquisition`, `photodetection`, `calibration` and `observation`. This keeps low-level read/charge records independent of detector and simulation orchestration and preserves an acyclic import graph.

- `acquisition.schedule` owns immutable read/raster identities, local scan timing and its constant-trajectory descriptor. `acquisition.quadrature` owns sampled acquisitions and coordinate-measure rules. `acquisition.images`, `surface` and `los` retain the separate image, projected-surface and GCRS boundaries.
- `photodetection.charge`, `coefficients`, `images` and `frames` separate physical charge records, coefficient arithmetic and the two numerical illumination interfaces. `frame_photoelectrons` supplies the shared surface expectation/characterization boundary used by `simulation.simulate_frames`.
- `calibration.radiometry` and `calibration.frames` retain generic and native-frame supplied-calibration contracts. `observation.native` owns native logical products; `observation.measurements` owns measurement rasters and `BandRadianceCube`.
- `response` owns spectral, point-grid, spatial, Fourier, joint and achieved-band representations. `sampling` owns angular and surface radiance samples. `registration` owns estimation, signed resampling, reconstruction and covariance projection.
- `simulation` composes photon integration and image readout. `detector` retains physical detector properties, noise and electronics. The former broad `imaging` source family is removed.
- Radiative's `columns`, `spectral.bands` and `optics.aerosol_mixture` own their named concepts. Scene's `gas`, `optical_depth`, `projection`, `surface` and `radiance` own physical composition. Numerics uses separate `tangents` and `operators` modules.

Coaddition uses explicit matrices and the existing linear/covariance algebra to retain the map and transport covariance and bias.

## Important API changes

| Previous surface | Current surface |
|---|---|
| `Atmosphere`, `AtmosphericColumn`, `AtmosphericPath` | `LayerAtmosphere`, `LevelAtmosphere`, `AtmosphericColumnIntegrals` |
| `CrossSectionField`, `PreparedCrossSections`, `cross_sections` | `LayerCrossSections`, `resample_layer_cross_sections` |
| `AtmosphericAbsorption` | Separate state, cross sections and water-continuum coefficients |
| Scene `SceneRadiance`, `RenderedScene` | Physical `DirectSolarScene` and instrument-owned charge/products |
| `DetectorResponse`, `InstrumentResponse`, `GroundGrid` | `GridElectronResponse` and explicit Geometry `PointLattice` |
| Instruments `SceneRadiance`, `RadianceField` | `SurfaceRadianceSamples`, `AngularRadianceSamples` |
| `SpectralResponseParameters`, `WavelengthElectronResponse` | Separate physical owners and `FilterElectronResponse` |
| Raw-vector surface callback | `radiance(xy_m, view_local, times_s, spectral: SpectralGrid)` |
| Two-item coefficient iterator | `(read_slice, spectral_slice, coefficients)` |
| `SurfaceGrid`, implicit Fourier clipping | `PointLattice`, explicit physical widths and negative policy |
| `RadiativeScene`, Radiative `RadianceCube` | `ColumnScene`, Instruments `BandRadianceCube` |
| `StationaryGaussian`, `discretize_emissions` | `StationaryGaussianPlume`, `discretize_puff_emissions` |
| `LineDatabase`, `lognormal_mixture_spectrum` | `SpectralLineTable`, `lognormal_aerosol_optics` |
| `pointing_rates`, `log_normal_interval` | `attitude_interval_speeds`, `normal_interval_log_probability` |
| `SharedCalibration`, `DesignInformation`, `ExchangeSearch` | `SharedContrastOffset`, `LinearTargetUncertainty`, `ChannelSearchResult` |
| `estimate_shared_calibration`, `design_information` | `estimate_shared_contrast_offset`, `linear_target_uncertainty` |
| `posterior_estimate_covariance`, `curvature_operator` | `covariance_from_marginals`, `masked_second_difference_operator` |
| `estimate_joint_rows`, `prepare_registration_image` | `estimate_row_displacements`, `highpass_log_radiance` |
| `MetricSensitivity`, `summarize_metric_sensitivity`, `interpolate_p50` | `FiniteIntervalSensitivity`, `finite_interval_sensitivity`, `half_probability_crossing` |

Geometry-bearing field arrays use `(y,x,...)`; point coordinates use `(x,y)`. The translated point-grid response deliberately retains the established scan-native `(template,target_x,target_y)` output order. The ground-pixel reference experiment explicitly transposes its retained `(along,cross)` arrays at the library boundary. Gas conversion uses molecules/m2 with an explicit molar-mass/Avogadro conversion. These are numerical/schema changes, not spelling-only migrations.

Native frame/observation products, angular illumination samples, `SolarSource` and column-transfer inputs retain their existing cm-1 density convention. Physical `Radiance` and `SpectralIrradiance` records, lower-level filter/photon calculations and physical cell queries accept their declared nm or cm-1 coordinate. Use `convert_spectral_grid` with the corresponding density-conversion function to convert a spectrum explicitly. `reverse_spectral_points` and both density converters accept `axis`, defaulting to the last array axis. Their nonspectral axes and signed tangent values are preserved. The `wavenumbers_cm1` accessor rejects nm records. The [spectral reference](../libs/orsia-radiative/docs/state-and-spectra.md#convert-physical-spectrum-records) contains executable examples.

## Maintained software evidence

The following tests and scripts cover the implemented responsibilities and public contracts:

| Scope | Representative maintained evidence |
|---|---|
| Boundaries and public homes | `tests/test_core_imports.py`, `tests/test_library_exports.py`, `tests/test_public_function_names.py`, Instruments `tests/test_module_layout.py`, Radiative `tests/test_radiative_module_layout.py`, and both installed-distribution scripts. |
| Support, atmosphere and spectra | Geometry `tests/test_grids.py` and `tests/test_point_field_resampling.py`; Radiative `tests/test_layer_cross_sections.py`, `tests/test_level_columns.py`, `tests/test_spectral.py`, `tests/test_spectral_grid.py` and `tests/test_extinction.py`. |
| Physical scene and bounded measurement | Dispersion `tests/test_field_paths.py`; Scene `tests/test_direct_solar_scene.py`, `tests/test_spatial_optical_depth.py` and `tests/test_centroid_projection.py`; Instruments `tests/test_spectral_blocks.py`; `tests/test_scene_instrument_blocks.py`. |
| Response, coaddition and products | Instruments `tests/test_filter_electron_response.py`, `tests/test_grid_response_contract.py`, `tests/test_fourier_sampling.py`, `tests/test_joint_operator.py`, `tests/test_observations.py` and `tests/test_radiometric_operations.py`. |
| Estimation and ownership | Inverse `tests/test_nonlinear_contract.py`, `tests/test_estimation.py` and `tests/test_damped_spatial_design.py`, plus the support/spectrum/scene mutation and concurrent-reuse tests. |
| Consumer migration and software evidence | `tests/test_library_guides.py`, executable library-guide tests, synthetic notebook tests, archived-package probes and unchanged historical numerical fixtures. |
| Names | Canonical export/owner tests and a source inventory checking the proposed replacements and absence of superseded declarations. Dictionary-only renames preserve unaffected keys. |

Paths in a named library row are relative to that library directory. The limits below qualify the retained software evidence. Native FINUFFT parity, realistic-scale benchmarks, approximation-error qualification and external serialized consumers remain outside the verified software evidence.

## Retained evidence and limits

Historical scientific reference fixtures are unchanged. The log-tangent renderer survives only in `libs/orsia-scene/tests/reference_log_tangent.py`, with its original approximation identified. Its mixed mapping is not a public API. The refactor checks preserved the then-retained notebook outputs and metadata while migrating source cells. Later notebook executions regenerated the saved displays now included in this delivery. Saved execution records and current local verification have separate scopes; see the [notebook guide](../notebooks/README.md#saved-execution-scope).

Public Python class/module names changed. External applications and qualified Python pickles require their own explicit migration.

The software gate checks formatting, lint, typing, tests and per-library coverage. Installed-distribution probes build source archives and wheels, install outside workspace resolution and exercise public contracts. Core native-import GIL behavior is checked without forcing the GIL off. Optional geospatial and platform-specific paths retain their existing separate test phases and limits; disabled GIL alone does not establish general native thread safety.

FINUFFT is not installed, so accelerated native parity and its GIL behavior are unqualified. Final diagram verification includes XML/graph checks, exports and visual inspection; an official compatible XSD is unavailable locally. No realistic-scale performance benchmark or independent atmospheric/retrieval qualification is inferred from these checks.

## Historical refactor verification record

The refactor's final `just check` completed successfully after the spectral-coordinate correction and obsolete-module assertion update. Its numerical phase passed 5,548 tests with 35 skips; the separate Data/notebook phase passed 686 tests with 18 skips. Combined statement-and-branch coverage was 97.24% overall and at least 94.92% in each of the nine libraries. Formatting, lint and configured Pyrefly checks passed. Existing typing suppressions and warnings remain configured; these checks establish no claim of warning-free code.

Both archived-distribution scripts completed successfully on the refactor source. They build source archives and wheels, install outside workspace resolution, exercise representative scene/instrument and tangent/estimation compositions, and check the free-threaded runtime after native imports. The scene probe also verifies the nm/cm-1 spectrum contract and spectral-axis conversion. The separate offline orbital recipe passed 47 tests using existing local auxiliary files.

The four affected architecture diagrams, for Dispersion, Instruments, Inverse and Numerics, have editable Draw.io sources and regenerated SVG/PNG exports. XML/graph and label checks passed; individual exports and affected guide previews were inspected. The Numerics figure now names `tangents.py` and `operators.py`. Documentation checks found no missing relative file links or prohibited prose characters in the affected documents, changed notebook source text and diagram labels. Source inventory checks found the target declarations and no superseded declarations in production. Scene reference fixtures and the then-retained notebook outputs/metadata matched the refactor checkpoint.

The historical refactor verification used the existing synchronized environment with offline reuse from the Orsia root. A new checkout requires the documented locked setup before using these environment overrides:

```bash
export DEVBOX_USE_VERSION=0.18.5 DEVBOX_OFFLINE=1
export UV_NO_SYNC=1 UV_OFFLINE=1 UV_PYTHON="$PWD/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
export RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
devbox run -- just check
RAYON_NUM_THREADS=1 UV_CONCURRENT_BUILDS=2 devbox run -- just python scripts/validate_scene_distribution.py
RAYON_NUM_THREADS=1 UV_CONCURRENT_BUILDS=2 devbox run -- just python scripts/validate_tangent_distribution.py
devbox run -- just --justfile tests/justfile test-orbital
```

Standard-suite skips cover optional backend fixtures and platform-specific execution/resource checks. The offline orbital recipe separately covers its opt-in cases. Native FINUFFT, Linux affinity/resource protocols and authenticated provider operations were not qualified by this macOS execution. Archive outputs are retained under `outputs/scene-validation/distribution` and `outputs/tangent-distribution`; diagram review copies are under `outputs/refactor-layout/diagram-qa`. Exact numerical cases, tolerances and seeds remain in the cited tests and scripts.
