# Point-lattice electron responses and radiance samples

`GridElectronResponse` represents one positive map from sampled spectral radiance to native expected photoelectrons. `detector_response`, `instrument_response` and `along_track_response` now construct this same record. Their geometry inputs remain different. A construction route no longer selects a different response type.

The defining modules are [`response.grid`](../src/orsia_instruments/response/grid.py), [`sampling.surface`](../src/orsia_instruments/sampling/surface.py) and [`sampling.angular`](../src/orsia_instruments/sampling/angular.py). Root imports expose their public records and existing constructor/evaluation functions.

## Represented information

| Record | Retained information | Numerical ownership |
|---|---|---|
| `GridElectronResponse` | Native `ReadSchedule`, `SpectralGrid`, `PointLattice`, complete per-read bounds and `(read,y,x,spectral)` photon coefficients. | Read schedule and lattice are shared immutable owners. Bounds and coefficients are independent writable float64 copies; keep them stable during use. |
| `SurfaceRadianceSamples` | `PointLattice`, spectral grid and `(y,x,spectral)` radiance in W/m2/sr per grid coordinate. | Lattice/grid are shared. Values are independent writable float64 copies and must remain stable during response application. |
| `AngularRadianceSamples` | Time/u/v knots, a cm-1 grid and `(time,u,v,spectral)` radiance. | Existing copy/dtype rules are retained; numerical arrays remain writable and stable during interpolation. |

A point lattice contains increasing metre x/y coordinates and an explicit local frame. Its value axes are `(y,x,...)`. It supplies no finite cell area or terrain. Bilinear grid-response construction requires at least two knots and finite positive interval widths on each axis. A surface-sample record can hold singleton axes independently; a response that interpolates those axes still needs adequate support.

`support_bounds_m[read]` is `(xmin,xmax,ymin,ymax)` in the lattice frame. It must enclose the complete physical integration support, including unsampled exposure endpoints, and fit inside the lattice. The constructors also check all supplied quadrature nodes. The caller remains responsible for the complete-support declaration and the actual coordinate/frame agreement of projected nodes.

`GridElectronResponse` coefficients multiply radiance densities, with the declared physical and quadrature factors already included. `response_electrons` applies no additional QE, area, exposure or spectral width. The spectral grid can use nm or cm-1 and must contain the endpoint-sampled declared integration intervals required by photon conversion. It is separate from the spatial point lattice.

## Construction routes

- `detector_response(..., ground_xy_m=..., lattice=..., support_bounds_m=...)` consumes an existing optical acquisition and its explicitly corresponding ground nodes. `ground_xy_m` uses `(read,node,coordinate=2)` with `(x,y)` components.
- `instrument_response(..., lattice=...)` constructs the existing constant-height, constant-speed local scan and derives nodes and complete swept support. Its x coordinate is along track and its y coordinate is cross track. The caller declares that coordinate system through the supplied lattice frame. The operation performs no frame conversion.
- `along_track_response(..., lattice=...)` stacks per-read optical scan templates. Stencil coordinates are offsets about the target centre in the declared local axes. Native identities and exposure intervals remain attached to their physical reads.

**Objective:** construct positive bilinear response coefficients. **Inputs:** declared lattice, projected quadrature nodes and per-node photon coefficients `P[r,n,k]`. **Outputs:** `K[r,j,i,k]`, initially zero. **Processing:** bracket each node coordinate between adjacent x knots `i,i+1` and y knots `j,j+1`. Define fractions `a=(x-x_i)/(x_(i+1)-x_i)` and `b=(y-y_j)/(y_(j+1)-y_j)`. Add `P` times `(1-a)*(1-b)`, `a*(1-b)`, `(1-a)*b` and `a*b` to `(j,i)`, `(j,i+1)`, `(j+1,i)` and `(j+1,i+1)`. Exact upper-edge nodes use the last bracket with fraction one. Require complete declared coverage before this projection. Construction visits x offsets 0 then 1 and, within each, y offsets 0 then 1, retaining the supplied node order in accumulation. Negative fractions, extrapolation and padding are excluded by the support checks.

The three constructors retain the existing physical photon arithmetic. The consolidated record changes spatial array order to `(y,x)` while preserving the physical meaning of x and y.

## Applying sampled radiance

`response_electrons(response, samples)` accepts `SurfaceRadianceSamples` directly. It requires exactly matching spectral points and coordinate type. Direct application requires equal ordered lattice coordinates on an admitted common local frame. A synthetic frame matches through the explicitly shared frame owner; equal frame names alone are insufficient. Anchored frame comparison follows Geometry's exact origin/basis/datum contract. No implicit reprojection or resampling occurs.

Raw `(y,x,spectral)` arrays remain available when the caller asserts the response's sample support. A nonnegative factorization uses `(factor,y,x)` maps and `(factor,spectral)` spectra. Supplied `scene_lattice` makes raw-array support explicit and is required for translated application. Negative physical values, mismatched axes or incompatible support fail with `ValueError`.

**Objective:** form physical expected electrons. **Inputs:** `K[r,j,i,k]` and supported radiance `L[j,i,k]`, or nonnegative maps `M[c,j,i]` and spectra `S[c,k]`. **Outputs:** `e[r]`. **Processing:** calculate `e[r]=sum_(j,i,k)(K*L)` or `e[r]=sum_(c,j,i,k)(K*M*S)`. Return `ExpectedElectrons` with the original physical read schedule and an independent source-labelled count array. These operations retain static, direction-independent radiance as an explicit caller assumption.

## Translation reuse and native order

Translated application requires a stationary scene, translation-invariant acquisition/optics and a centred odd-sized stencil. `target_indices=(y_indices,x_indices)` selects a Cartesian product of target points. The scene lattice must share the response frame and have matching uniform spacing. Stencil offsets and scene spacing use relative tolerance `1e-12` and absolute tolerance `1e-12 m`; the stencil must have odd length and be centred at zero. Every target must have the complete stencil halo; invalid edges are rejected rather than padded.

`method="direct"` contracts each complete patch independently. `method="correlation"` applies each stencil to the corresponding factor or spectral plane and samples the same target points. Both produce the same numerical map within floating-point summation tolerance.

**Native scan order remains `(template,target_x,target_y)` flattened in C order.** The calculation uses `(y,x)` field arrays internally and explicitly restores that native sequence at output. This preserves the existing along-track scan convention and physical sample identities. Caller-supplied native reads are mandatory; their count and setting/port sequence must agree with the templates. Duration comparisons use relative tolerance `1e-11` and absolute tolerance `1e-14 s`. Physical clock values and the translation-invariance assertion remain caller-owned. Spatial field axes and native detector-read order are separate contracts.

## Angular sampling

`AngularRadianceSamples` replaces the former angular record name without changing its scientific law. `sample_radiance` retains trilinear time/u/v interpolation and rejects extrapolation. `uniform_radiance_field` constructs uniform samples over explicitly supplied bounds. The entire angular support rectangle lies inside the forward unit disk. Direction cosines and local paraxial slopes remain distinct acquisition conventions, and times must share the acquisition epoch. This path retains its single endpoint-sampled cm-1 interval; the rename adds no spectral interpolation or atmospheric model.

## Breaking migration and evidence

- Replace `DetectorResponse` and `InstrumentResponse` with `GridElectronResponse`. Replace `GroundGrid` and bare response axes with `PointLattice(x_m, y_m, frame)` from Geometry. Supply an actual frame declaration; the constructors invent none.
- Replace Instruments' former `SceneRadiance` with `SurfaceRadianceSamples`. Transpose former `(along,cross,spectral)` values to `(y,x,spectral)` when x denotes along track. Transpose the two spatial coefficient axes in persisted response arrays correspondingly.
- Replace angular `RadianceField` imports with `AngularRadianceSamples`. Its fields, units, writable ownership, interpolation law and existing public function names are unchanged.
- Use `response.grid`, `sampling.surface` and `sampling.angular` as defining modules. The old `imaging.response.instrument` and `scene` modules are removed without aliases. Qualified Python pickle names require an explicit external migration; no generic compatibility promise is made.

The retained notebook ground-pixel experiment keeps its established `(along,cross)` array and native-scan conventions. Its owning adapter transposes values and coefficients explicitly at the library boundary, preserving retained numerical data and report interpretation.

Tests cover nonsquare lattices, an independent bilinear photon oracle, equal-shape coordinate/frame/spectral mismatches, both response-construction routes, writable snapshot ownership, complete translated halos and direct/correlation agreement in native order. These establish bounded software and numerical contracts. They do not establish physical frame provenance, atmospheric accuracy, instrument calibration or image-scale performance.
