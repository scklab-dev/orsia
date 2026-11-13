# Bounded spectral radiometry

The surface-acquisition functions share one photon-coefficient generator. `frame_response_coefficients` now yields `(read_slice, spectral_slice, coefficients)`. Its coefficients have `(read,node,spectral)` axes within the two slices. `simulate_frames`, `frame_response`, joint matrix/operator construction and `frame_electron_tangents` consume this same calculation.

## Physical callback boundary

`SurfaceRadiance` receives `(xy_m, view_local, times_s, spectral)`: `(Q,2)` local metre coordinates, `(Q,3)` outward surface-to-sensor unit directions, `(Q,)` elapsed seconds and a `SpectralGrid`. Output is finite nonnegative `(Q,K)` radiance in W/m2/sr per declared spectral coordinate. Each block retains the full grid's declared physical domain while selecting its requested points. Reads are flattened before nodes in C order. Callback results are borrowed without mutation.

This replaces the former raw cm-1 vector argument. Update the callback to read `spectral.points` and enforce its physical coordinate contract. The separate GCRS line-of-sight and resident angular-field interfaces retain their existing signatures. A cm-1-prepared `DirectSolarScene.radiance` can be passed directly to `simulate_frames`; no callback wrapper or Scene import inside Instruments is required.

The coverage callable receives the complete acquisition and full spectral grid before any radiance call. It returns a Boolean `(reads,)` certificate for complete footprints, time intervals, directions and spectral coverage. It must establish frame and UTC-epoch agreement with the supplied scene. Valid sampled nodes alone do not establish full coverage.

Native `FrameMeasurements` and `NativeBandResponse` retain their existing cm-1 observation contract. Lower-level `photon_conversion_weights`, `photoelectron_response`, the shared coefficient iterator and `frame_electron_tangents` support either nm or cm-1 radiance density. The selected function's contract determines the admitted representation.

## Quadrature before blocking

1. **Objective:** construct the complete spectral photon-conversion rule. **Inputs:** increasing positive samples `s[k]`, their coordinate and explicitly declared coverage intervals. **Outputs:** `q[k]`, in coordinate-measure/J. **Processing:** require at least two samples in every interval, including its two endpoints. Within each interval, add half of each adjacent sample separation to both endpoint weights, starting from zero. No weight bridges a gap. Divide each weight by photon energy: `h*c*100*s[k]` for cm-1, or `h*c/(s[k]*1e-9)` for nm. Reject unknown coverage, unsampled interval endpoints or nonfinite conversion. `photon_conversion_weights` performs this calculation once before the coefficient iterator partitions the spectrum.
2. **Objective:** construct a response block. **Inputs:** a read slice, spectral slice, immutable acquisition nodes, sampled optical fractions `A[r,n,k]`, collecting area `S`, throughput `T[r,k]`, QE `Q[r,k]` and the selected full-domain weights `q[k]`. **Outputs:** `K[r,n,k]`, in electrons per spectral-radiance unit. **Processing:** use `K=A*weights_sr_s*S*T*Q*q`. Each physical or quadrature factor enters once. Fractions are finite in `[0,1]`; area and photon weights are finite and nonnegative. `weighted_photoelectron_response` implements this multiplication. A one-point block is valid because its quadrature weight came from the full domain. Arithmetic overflow fails explicitly.
3. **Objective:** integrate values and retain native characterization. **Inputs:** response blocks and physical radiance `L[r,n,k]`. **Outputs:** expected source electrons `e[r]`, spectral coefficients `B[r,k]` and normalized spatial marginals `W[r,n]`. **Processing:** accumulate `e[r]+=sum_(n,k in block)(K*L)`, assign `B[r,k]=sum_n(K)` and accumulate `M[r,n]+=sum_(k in block)(K)`. After all blocks, set `W[r,n]=M[r,n]/sum_n(M[r,n])`; zero-response rows receive zero. This normalization occurs after complete spectral accumulation. Detector realization and calibration follow in their separate owners.

`photoelectron_response(grid, acceptance, ...)` remains the full-grid convenience operation. It selects exact acceptance wavelengths, prepares the complete photon weights and delegates to the weighted kernel. Supplied acceptance snapshots can retain another wavelength order; selection restores the grid's order without interpolation.

Both `read_batch_size` and `spectral_block_size` are memory controls. Positive spectral sizes are optional; `None` preserves a full-spectrum callback per read batch. Read batches are traversed first, then increasing spectral slices within each batch. Different partitions may change floating-point accumulation, so numerical comparisons use tolerances.

Optical and radiance callbacks must evaluate the same physical law independently of slice width, including singleton slices. A callback that normalizes its output using the currently requested sample range violates this requirement. The full-spectrum default preserves the previous evaluation scope for callers that have not adopted spectral blocking.

## Signed physical and response derivatives

`frame_electron_tangents(camera, grid, spectral, radiance_tangent, coverage, ...)` returns an `ExpectedElectrons` and a fresh `(read,direction)` derivative array. The radiance callback returns the nominal `(Q,K)` values and signed `(Q,K,P)` total derivatives. Direction count and order must remain constant across blocks; `P=0` is supported.

An optional `response_tangent(read_slice, spectral_slice)` supplies signed `(read,node,spectral,P)` coefficient derivatives. It includes derivatives of the same physical and quadrature factors represented by the nominal coefficients. It receives no mutable response buffer from the integrator.

**Objective:** propagate the complete fixed-query product rule. **Inputs:** matching nominal coefficients/radiance and their signed direction blocks. **Outputs:** `de[r,p]`. **Processing:** accumulate `de[r,p]+=sum_(n,k)(K[r,n,k]*dL[r,n,k,p]+dK[r,n,k,p]*L[r,n,k])`. Omitted response derivatives contribute zero. Validate physical nominal values, finite signed derivatives and matching direction counts before contraction; reject nonfinite accumulated derivatives. Dark current, random realization and calibration are excluded.

Geometry or timing perturbations can move the physical queries. Their complete derivative needs those motion terms in `dL`, in addition to `dK`. A gas/surface-only scene tangent does not supply geometry derivatives. Select an appropriate total-derivative callback or an explicit external finite-difference composition for such directions.

The spatial-scene notebook companion now uses this shared operation for its electron means and physical Jacobian. The radiance-only `simulate_frames` path still computes no tangents. Noise sampling occurs once after expected-charge assembly in native order; floating-point changes in the means can affect exact random draws even when the RNG traversal is unchanged.

## Joint operators and memory scope

`spatial_spectral_response` and `spatial_spectral_operator` accept the same optional spectral block size. They retain their declared piecewise-constant spatial-cell and cm-1 spectral-bin basis. Forward products accumulate all blocks into the same native rows. Transpose products accumulate with the same signed or unsigned supplied cotangents, preserving all-port and selected-read order. The physical transpose remains unnormalized.

With `R` reads per batch, `N` nodes and `K_b` spectral samples, dense coefficient and radiance scratch scale as `R*N*K_b`. Tangent scratch also depends on `P`. Retained acquisition arrays, full-domain weights, requested native spectral/spatial marginals, output arrays, callback-owned state and native-library workspaces remain additional storage. Matrix/operator budget checks retain their documented scope; these bounds are storage estimates, not performance measurements.

Tests compare complete and blocked coefficients, electron means, spatial normalization and joint transpose products. They also exercise disconnected intervals, reciprocal-coordinate quadrature convergence, signed response terms, physical `DirectSolarScene` integration, invalid callbacks and zero directions. These checks establish bounded numerical contracts; realistic-scale runtime, scientific accuracy and retrieval performance require separate evidence.
