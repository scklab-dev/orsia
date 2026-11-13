# Native observation cubes

`calibrate_observations` and `form_observation_cube` preserve native observation supports while calibrating and accumulating radiances. `ObservationCube.values` has logical `(line,sample,band)` axes. Equal line/sample indices across bands can have different ground footprints. The caller supplies the organization and fixed contribution weights; these functions perform no common-ground-grid registration.

## Run the two-band example

The [executable example](../../../examples/native_observation_cube.py) uses two spatially offset detector/filter rows, two exposures, supplied DN and independent read variance. A dimensionless common gain parameter has standard deviation 0.01. Three masked reads produce a renormalized one-read average and one empty cube cell. All calibration coefficients and local-plane geometry are synthetic.

Run from the repository root in the existing environment:

```bash
MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just python examples/native_observation_cube.py --figure outputs/native-observation-cube.png
```

The saved figure shows native support rectangles and valid-observation covariance. The script checks dense/source-projected agreement and prints radiances `[[[10.0, 15.0], [20.0, NaN]]]`, valid flat indices `[0, 1, 2]`, and marginal variances `[0.51, 0.1475, 1.04]`. Radiances use W m^-2 sr^-1 per (cm^-1); covariance uses squared radiance units. Native read labels in the plot are zero-based columns of the retained contribution matrix.

The following block executes the same example without writing a figure, from the repository root:

```python
import runpy

example = runpy.run_path("examples/native_observation_cube.py")
cube = example["run_example"]()
assert cube.values.shape == (1, 2, 2)
assert cube.native_weights.shape == (3, 8)
```

## Geometry-independent radiometry

`orsia_instruments.calibration.radiometry` supplies `calibrate_readings`, `accumulate_observations`, `RadiometricObservations` and `RadiometricParameterMaps`. These operations use numerical read values and error maps independently of detector geometry. Native image calibration and cube formation reuse these operations after checking native alignment, electronic linearity, flags and response compatibility. A direct numerical caller owns those physical checks.

`calibrate_readings(readings, *, sensitivity, read_covariance, gain=1.0, offset=0.0, dark_electrons=0.0, ...)` accepts a float64 read vector. `sensitivity` is a finite nonnegative `(reads,)` electron/radiance vector. Gain, offset and fixed dark charge accept finite scalars or `(reads,)` maps; gain is positive and dark charge is nonnegative. The formula and derivatives are given below. Supported input values must be finite. Negative output radiances remain valid.

Fixed calibration omits `parameters`, `parameter_covariance` and `read_parameter_cross_covariance`. The output has empty parameter names/units and `(valid observations,0)` derivatives. Explicit empty maps and float64 `(0,)` or `(0,0)` parameter covariance are also supported. Fixed calibration contributes no artificial parameter-error source.

`RadiometricObservations.values` and `valid` retain full one-dimensional output order. `indices` lists increasing valid coordinates. Covariance, signed bias and derivative rows follow `indices`. `accumulate_observations(observations, weights)` accepts fixed float64 CSR `(outputs,input observations)` normalized nonnegative weights. It applies the same value/error/sensitivity transport as cube formation and leaves empty rows invalid. Rebuild weights explicitly when the input validity selection changes. Signed interpolation weights belong to a separate resampling contract. Direct accumulation leaves physical band and spatial-support compatibility to the caller.

To retain calibration correlations across separately acquired blocks, concatenate read values and derivative rows in caller order. Use the same parameter columns and one parameter covariance for all blocks. Read covariance may itself be `ProjectedCovariance` with common source projections spanning the blocks. Calibrating each block independently and combining their covariance as independent sources discards the shared correlations.

For independent read errors, a bounded caller can calibrate and accumulate each read block separately, retain each output's parameter derivatives, and assemble those derivatives with one shared parameter covariance. The application owns block support, integer-DN summation and physical response checks. This block recipe does not represent detector-read correlations across blocks or nonzero read/parameter cross covariance; use a joint source contract for those inputs. Keep parameter names, units and column order identical across blocks. The common parameter source contributes cross-block output correlations after assembly.

This synthetic example averages two blocks that share one fractional gain error:

```python
import numpy as np
from scipy.sparse import csr_array
from orsia_instruments.calibration.radiometry import RadiometricParameterMaps, accumulate_observations, calibrate_readings
from orsia_uncertainty import materialize_covariance

readings = np.concatenate((np.array([10.0, 12.0]), np.array([20.0, 24.0])))
maps = RadiometricParameterMaps(
    ("common-gain",), ("1",), np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1)),
)
calibrated = calibrate_readings(
    readings,
    sensitivity=np.ones(4),
    read_covariance=np.ones(4),
    parameters=maps,
    parameter_covariance=np.array([0.01]),
    read_parameter_cross_covariance="zero",
    parameter_bias=np.array([-0.02]),
)
averaged = accumulate_observations(
    calibrated, csr_array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]),
)
covariance = materialize_covariance(averaged.covariance, max_bytes=32)
np.testing.assert_allclose(averaged.values, [11, 22])
np.testing.assert_allclose(covariance, np.eye(2) * 0.5 + np.outer([11, 22], [11, 22]) * 0.01)
np.testing.assert_allclose(averaged.bias, [-0.22, -0.44])
```

## Shared spectral response tables

Legacy `NativeBandResponse` construction retains `(reads,spectral)` `spectral_coefficients` and requires no caller changes. To share characterized responses, supply `(responses,spectral)` coefficients and an int64 `(reads,)` `response_indices` map. Every read selects one table row. The table retains genuinely varying shapes as separate rows; construction performs no approximate merging or implicit read-by-spectral expansion.

Optional `(reads,)` `read_sensitivity` gives absolute electrons per unit radiance. Without this argument, each read uses its selected table row's coefficient sum. With this argument, the selected row supplies normalized shape and `read_sensitivity` supplies absolute normalization. Zero sensitivity marks a dark read. A zero-mass table row cannot support a positive sensitivity. Spatial weights must still sum to one for responsive reads and zero for dark reads.

`response.sensitivity` returns an owned `(reads,)` normalization vector. `response.normalized_responses` returns normalized table rows, with zero-mass rows left zero. `response.response_rows(indices)` maps selected native indices to table rows. `response.coefficients_for_reads(indices)` explicitly allocates physical coefficients in selected-read order; its output can scale with reads times spectral samples. Use the table accessors when this expansion is unnecessary.

Grouping, calibration, cube formation and reconstruction operate on the mapped table rows. Pairwise spectral checks use distinct response rows and preserve their existing tolerances. `group_native_bands` changes descriptive labels while retaining table coefficients and sensitivity. Native `frame_response` and `simulate_frames` continue to construct dense responses, preserving existing array consumers and independently varying instrument responses. Callers can supply compact tables when their characterization establishes shared shapes.

## Calibrate one realization

Import these records and operations from `orsia_instruments`. `RadiometricParameterMaps` is defined in `calibration.radiometry`; native observation records and formation are defined in `observation.native`. `CalibratedFrames`, `calibrate_frames`, `BandRadianceCube` and `reconstruct_radiance_cube` retain their existing contracts, including the latter's separate common-grid registration.

`calibrate_observations(images, response, *, detector, read_covariance, parameters=None, parameter_covariance=None, read_parameter_cross_covariance=None, ...)` checks electronic correction with the supplied nominal detector using `calibrate_frame_electrons`. It delegates radiometric calibration to `calibrate_readings`, using `response.sensitivity` for normalization. `realisation` selects one DN realization; expected charge has index zero. Negative calibrated radiances remain valid. Zero response, an explicit false `read_valid`, or masked saturation/ADC flags remove a read from the error coordinate order and leave its native value as NaN. Empty valid support raises `ValueError`.

For native DN d, gain g in electrons/DN, pedestal o in DN, fixed dark charge k in electrons and normalization s in electrons per unit radiance:

\[
v = \frac{g(d-o)-k}{s},\quad A=\frac{\partial v}{\partial d}=\frac{g}{s},\quad B=\frac{(d-o)\,\partial g-g\,\partial o-v\,\partial s}{s}.
\]

`RadiometricParameterMaps` names the parameters and their units. Its `gain`, `offset` and `normalization` arrays have shape `(native reads,parameters)` and contain derivatives of g, o and s. For a common fractional gain error, supply one dimensionless parameter and set `gain[:,0]` to the nominal gain. A pedestal error uses `offset`; its derivative has a negative sign in the corrected radiance. A scalar response-normalization error uses `normalization`. Dark calibration is fixed in this interface. Expected-electron calibration permits normalization derivatives only; nonzero gain or pedestal maps raise `ValueError`.

`read_covariance` follows all native reads, including masked reads, and uses DN² for DN input or electrons² for expected charge. It accepts float64 variance vectors, validated dense covariance or `ProjectedCovariance`. `parameter_covariance` follows map columns and uses their corresponding parameter-unit products; it accepts float64 variance vectors or validated dense covariance matrices. Read covariance must exclude the parameter contribution supplied separately here.

For uncertain calibration, the required cross-covariance argument is either the explicit string `"zero"` or a finite `(native reads,parameters)` matrix. Fixed calibration can omit the argument. A nonzero matrix represents read errors correlated with the estimated calibration errors. The transported covariance is

\[
C_v=A C_d A^T+B C_\theta B^T+A C_{d\theta}B^T+B C_{d\theta}^T A^T.
\]

The joint source block must satisfy source PSD validation, followed by the [source-projection marginal check](../../orsia-uncertainty/docs/source-projection.md#representation-and-ownership). That check rejects every computed negative or nonfinite marginal without clipping Q, including negatives allowed by source-space tolerance. The optional `read_bias` and `parameter_bias` are finite signed offsets. They produce `bias=A b_d+B b_\theta`, separately from covariance. Nonfinite transported offsets or their sum raise `ValueError`, even when all input offsets are finite. `parameter_sensitivity` retains B in valid-read order and keeps parameter names and units.

`representation="projected"` is the default. Explicit zero cross-covariance preserves read sources and a separate calibration block, with a sparse diagonal/selection read map and dense calibration sensitivities. `representation="dense"` materializes the same model. With uncertain parameters, a matrix cross declaration uses one joint dense source block, even if its entries are zero; use `"zero"` for the scalable independent-block path. Projected read input requires `"zero"` for uncertain calibration; correlated read/parameter errors need a supplied joint source through the [source-projection API](../../orsia-uncertainty/docs/source-projection.md). `max_dense_bytes` checks each joint/output dense allocation before construction. It excludes existing source storage, covariance-validation workspaces and total process memory.

## Form the logical cube

`form_observation_cube(calibrated, geometry, *, shape, band_names, native_weights, ...)` accepts a float64 SciPy `csr_array` with shape `(prod(shape),native reads)`. Rows address every logical cell in flattened C order; columns retain the full native read order. A selection or permutation uses one unit weight per supported row. An accumulation uses fixed nonnegative weights summing to one per supported row. Empty rows become invalid cells. At least one supported output cell is required.

Every contributor must be a valid calibrated read with the output band's physical label. All contributing normalized spectral coefficient vectors in a band must agree pairwise within `response_tolerance` in L1 distance, default `1e-8`. Exactly equal normalized rows are deduplicated before checking all remaining pairs. Thus repeated responses avoid quadratic pair comparisons; K distinct rows still require O(K² S) comparisons for S spectral samples. Deduplication uses exact equality, with no tolerance-based grouping or representative-only acceptance. Weights do not mix bands. Invalid-read contributions raise an error; the caller must rebuild normalized weights after a mask change. Weights estimated from noisy observations require additional uncertainty propagation outside this fixed-weight contract.

The valid-row accumulation matrix W transforms values, errors and sensitivities together:

\[
u=Wv,\quad C_u=W C_v W^T,\quad b_u=Wb_v,\quad B_u=WB.
\]

Cube formation validates the input bias and transported bias for finite float64 support; numerical overflow raises `ValueError`.

`ObservationCube.observation_indices` is `flatnonzero(valid.ravel(order="C"))`. Covariance rows/columns, bias entries and sensitivity rows all follow these indices. `native_weights` retains only valid rows but all native columns. `geometry` retains native coordinates, bounds, rays, timing and raster correspondence; `response` retains spectral coefficients and the spatial marginal. Thus a common gain error remains common after averaging. Read count, weight sum and effective independent-sample count cannot replace this covariance.

## Ownership and evidence limits

Calibration and assembly factories allocate their result arrays and sparse contribution maps. Native geometry and response are shared; direct dataclass construction borrows inputs. `RadiometricParameterMaps` also borrows its maps. Keep all writable buffers stable during calls and rebuild derived products after changes. Geometry must retain native identities and sensor quadrature. Covariance is conditional on the supplied geometry, response support, validity selection and weights. Position uncertainty requires an appropriate physical sensitivity before it can contribute to radiometric covariance.

The spectral and spatial margins of `NativeBandResponse` do not encode a general joint spectral-spatial kernel. Scalar normalization preserves the achieved photon-weighted band interpretation under the existing response assumptions. No registration, deconvolution, geometry-error propagation or mission-specific grid selection is inferred.

The [native tests](../tests/test_observations.py) compare dense and projected paths, joint cross terms, signed bias, finite-difference derivatives, masks and common gain errors. A fixed-seed 80,000-draw joint small-error ensemble checks first-order covariance and nonlinear remainder at fixed support. The [numerical and compact-response tests](../tests/test_radiometric_operations.py) cover zero-parameter calibration, shared read/calibration sources across two blocks, dense/compact agreement and bounded traced scratch storage. These are synthetic linearization/software checks. They leave flight calibration, clipping-selection statistics, retrieval performance and engineering feasibility unqualified.


## Ground-registered band products

`BandRadianceCube(x_m, y_m, bands, values, coverage, valid)` stores band means with axes `(y,x,band)`. The x and y axes are finite increasing cell centres in metres. A single centre is supported. The caller declares the surface chart, terrestrial frame and datum externally; numerical coordinates alone do not identify a map projection.

`values` has units W m^-2 sr^-1 per (cm^-1). Every valid value is finite and has positive coverage. Invalid elements contain NaN. `coverage` is finite and nonnegative, with meaning established by the producer. The fixed-filter reconstruction uses dimensionless accumulated normalized spatial response weight. Coverage is distinct from physical covered area.

The constructor validates axes, shapes, coverage and NaN/validity consistency. It snapshots coordinate, value, coverage and validity arrays into writable storage. The band tuple retains its supplied objects; keep their buffers stable during use. Cube construction performs no resampling or scene construction.

The [fixed-filter camera guide](filter-camera.md) describes an implemented producer. Its reconstruction retains compatible achieved spectral responses and uses separately supplied estimated geometry. Missing bands remain invalid with NaN values. The record is defined in `orsia_instruments.observation.measurements` and exported from `orsia_instruments`; calibration and reconstruction remain instrument-owned operations.
