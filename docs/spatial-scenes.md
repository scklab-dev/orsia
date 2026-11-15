# Spatial atmosphere and plume acquisition

The executable [spatial-scene example](../notebooks/support/spatial_scene.py) connects a prescribed gas plume, background atmosphere, affine terrain and spectral surface reflectance to native filter-camera images. It uses [Scene](../libs/orsia-scene/README.md) for two explicit modes: `independent_column` and `heterogeneous_direct_solar`. Both modes return physical radiance and plume-amplitude/reflectance tangents through the same detector response.

Run from the Orsia workspace with its provisioned environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just python -m notebooks.support.spatial_scene
```

The program prints a bound on excluded plume mass, native expected-electron arrays with shape `(2,2,2)`, and Jacobians with shape `(2,2,2,2)`. The first three axes are frame, detector row and detector column. The last Jacobian axis is dimensionless plume amplitude followed by additive dimensionless Lambertian reflectance. Noise sampling is absent; `simulate_frames` retains its optional caller-owned random-generator interface.

## Ownership and reusable operations

Reusable physical composition lives in `orsia-scene`. The example retains synthetic cross sections, spectral selection, reflectance law, frame rotation, parameter choices and the camera-specific coverage certificate. Geometry, Dispersion and Radiative retain their numerical responsibilities; Instruments retains acquisition and detector response. Scene consumes in-memory arrays and narrow callables without providers, hidden time advancement or random draws. Inputs remain unchanged and outputs have fresh writable storage. Keep borrowed arrays and bound application inputs stable during use, including concurrent calls.

| Owner and public import | Operation and representation |
| --- | --- |
| `orsia_scene.puff_molecular_columns` | Combine supplied layered background amounts and frozen plumes with explicit species mapping, kg/mol conversion, scales and arbitrary signed parameter directions. |
| `orsia_scene.surface_radiance` | Compose affine or callable terrain, molecular-column-only callbacks and layer-resolved cross sections into radiance through horizontal-affine independent columns or separate displaced paths. |
| `orsia_scene.surface_radiance_tangent` | Return the same radiance with explicit gas-column and reflectance sensitivities through shared geometry and physical validation. |
| `orsia_geometry.affine_surface_points` | Map finite `(...,2)` metre chart coordinates to `(...,3)` points and one upward normal for `z=height_m+slopes[0]*x+slopes[1]*y`. |
| `orsia_geometry.intersect_affine_terrain` | Intersect matching exterior `(...,3)` origins/unit directions with that same affine terrain. Return Boolean hits and metre points; misses contain NaNs. A caller-selected minimum incidence rejects grazing support. |
| `orsia_geometry.segment_layer_paths` | Split `(paths,3)` upward rays at horizontal height interfaces. Return start/end arrays `(segments,3)`, integer offsets `(paths+1,)` and source-layer indices `(segments,)`. Call separately for solar and viewing paths. |
| `orsia_radiative.clip_atmosphere_height` | Reconstruct the actual retained `LayerAtmosphere` above one terrain height under constant within-layer molecular density and temperature, with pressure linear between height interfaces. |
| `orsia_radiative.retained_layer_fractions` | Return source-layer fractions for other extensive constituents on the same constant-density reconstruction, including fixed scattering depths. Remove zero-fraction layers before transfer. |
| `orsia_radiative.integrate_path_optical_depth` | Convert `(segments,species)` molecular slant amounts and `(segments,species,spectral)` cross sections into separate `(paths,spectral)` extinction depths. Optional segment scattering losses enter once. |
| `orsia_radiative.integrate_path_optical_depth_tangent` | Propagate signed `(segments,species,k)` amount directions at fixed cross sections, paths and scattering losses. |
| `orsia_radiative.transfer.paths.direct_surface_radiance` | Compute Lambertian direct reflection from separate solar/viewing slant depths. |
| `orsia_radiative.transfer.paths.direct_surface_radiance_tangent` | Propagate signed depth and reflectance seeds with a final direction axis, at fixed illumination, geometry and visibility. |
| `orsia_instruments.photodetection.frames.frame_response_coefficients` | Shared generator of `(read_slice,spectral_slice,coefficients)` blocks. `simulate_frames` and `frame_electron_tangents` use the same full-domain photon quadrature; the latter contracts physical and optional response directions. |

Geometry also shares its ray validation through `orsia_geometry.rays.validate_rays`; sphere, ellipsoid and affine-terrain intersections use the same finite/unit-vector contract. `frame_response` supplies spectral coefficients for uniform-radiance contraction; the example uses the full joint coefficients for spatially varying derivatives.

### Shared flat radiance queries

`SyntheticRadianceModel` owns the transfer mode, parameter values, a shallow immutable input-mapping snapshot and shared background preparation. Its `radiance` method calls `surface_radiance`; `radiance_tangent` calls `surface_radiance_tangent` with explicit sensitivities. Borrowed arrays and physical objects must remain stable. Callers construct the model and pass its bound `radiance` or `radiance_tangent` method directly. The example binds its population to `GaussianPuffField(state, "free")`, preserving its declared unbounded Gaussian free-space law. Direct plume evaluation supplies no tangent seed to `puff_molecular_columns`, so direct simulation computes no sensitivity arrays. Both callbacks accept `(xy_m, view_local, times_s, spectral)` with `(N,2)`, `(N,3)`, `(N,)` arrays and a `SpectralGrid`. The synthetic model requires its declared cm-1 coordinate. Radiance returns `(N,K)`; the tangent callback additionally returns `(N,K,2)`. Views are outward unit vectors in the scene's fixed Cartesian chart. The projection callback converts dated GCRS rays and returns `(xy_m, view_local, hits)`. Instruments flattens reads then nodes in C order and restores its integration axes; `frame_electron_tangents` contracts the same physical direction block with the shared photon coefficients. The [migration contract](../libs/orsia-scene/README.md#flat-callback-migration) records removed APIs and the independently pinned external consumer.

### Terrain and atmosphere contracts

The connected acquisition example retains affine terrain and its analytical coverage proof. The separate application adapter [`SplineTerrain`](../notebooks/support/terrain.py) owns one cubic interpolant, immutable grid snapshots and a closed rectangular domain. Its `height` and `gradient` methods can be passed to Geometry and Scene callback APIs; `horizontal_bounds_m` supplies matching intersection bounds. This SciPy support module adds no Geometry dependency and supplies no full-ray visibility certificate. The Scene library also accepts plain height and gradient callables in heterogeneous mode, with per-point normals and an explicit full-ray visibility policy. See the [callable-terrain guide](../libs/orsia-geometry/docs/terrain.md) for interpolation and bounded intersection, and the [Scene visibility contract](../libs/orsia-scene/README.md#callable-terrain-and-visibility) for shadow and occlusion rejection. Generic sampled intersections cannot replace the example's continuous coverage certificate.

Height edges increase bottom to top in one Cartesian vertical datum. A surface must lie in `[bottom,top)`. Exact interior interfaces remove complete lower layers. A partial first layer retains its actual thickness; no zero-thickness padding is introduced. Below-bottom and at/above-top surfaces raise `ValueError`.

`clip_atmosphere_height` scales dry-air, water, total-air and absorber molecular columns by the retained height fraction. Total air remains dry air plus water. Temperature stays constant in each source layer. The new surface pressure follows linear interpolation between source interfaces; a partially clipped layer receives the new midpoint pressure. Uncut layer pressures are preserved. This reconstruction has no general hydrostatic guarantee. Existing pressure-overlap remapping remains a separate constant-amount-per-Pa reconstruction.

The returned atmosphere contains the retained background only. Scene instead composes molecular arrays directly: it segments paths at actual retained height interfaces, scales background density by segment length, and integrates and converts plume excess once. State-dependent spectroscopy must be reevaluated after pressure, temperature or composition changes. Perturbation bases require explicit remapping; the clipping operation rejects existing bases.

A clipped atmosphere can feed existing column solar scattering and non-scattering thermal APIs under their existing contracts. Scene's two implemented modes use the direct Lambertian kernel. The example previously used `max_order=0, delta_m_moments=None` column transfer; saved regression values check that the extraction preserves its absorption-only output and sensitivities. A nonzero slope is accepted by the heterogeneous mode. Independent columns retain a horizontal boundary and apply solar/view secants to vertical optical depths. Scene adds no scattering or thermal mode.

### Paths and direct reflection

Use `surface_radiance` when only physical radiance is needed. Its column callback receives segment endpoints in metres and source-layer indices, then returns `(segments,species)` molecular amounts in molecules/m2. It accepts no sensitivity inputs and returns one `(queries,spectral)` array. `surface_radiance_tangent` retains the tuple-returning column callback and explicit reflectance seeds for radiance plus sensitivities. Both functions support empty queries and the same terrain, visibility and sampled-support constraints. The [Scene example](../libs/orsia-scene/README.md#minimal-example) demonstrates both contracts.

Rays are straight in one Cartesian frame. Outward solar directions point from the surface toward the Sun; incoming solar photons traverse these segments in reverse. Outgoing viewing directions point toward a sensor beyond the atmospheric top. The instrument's scene-directed GCRS direction is reversed after dated chart conversion.

Segment offsets are nondecreasing, start at zero and end at the total segment count. Repeated offsets represent empty paths with zero optical depth. Molecular columns use molecules/m2; cross sections use m2/molecule. `path_columns` preserves the supplied puff field boundary/cutoff and already includes segment length in its kg/m2 output. Neither another path length nor a plane-parallel air-mass factor multiplies those amounts.

The direct-path calculation is `L = rho * E * mu / pi * exp(-tau_solar) * exp(-tau_view)`. Here `rho` is spectral Lambertian reflectance, `E` is beam-normal irradiance in W m-2 per (cm-1), and `mu` is the surface-normal incidence cosine. Depths already contain geometry. Output radiance is W m-2 sr-1 per (cm-1). The lower-level Radiative kernel accepts zero incidence and returns zero. Scene requires positive local incidence and rejects shadowed or occluded queries under its visibility policy. The connected example excludes shadow and view-obstruction transitions from its continuous support.

Direct values and tangents use binary scaling to retain representable float64 results when transmittance underflows or unscaled seed sums and products overflow. The tangent operation rejects final derivatives outside finite float64 support. Zero albedo retains its reflectance derivative; zero incidence returns zero value and derivative.

Optional segment scattering depths attenuate the direct beam. Diffuse sources, heterogeneous multiple scattering, terrain-to-terrain exchange, refraction and overhangs remain outside these operations. The example rejects unknown transfer combinations, including heterogeneous scattering. It computes no local diffuse addition to heterogeneous direct attenuation.

## Explicit synthetic fixture

`synthetic_inputs()` returns fresh editable application inputs. Construct the radiance model after choosing those inputs, then keep borrowed arrays and physical objects stable. Reconstruct the model to change its configuration; the direct and tangent methods use the same parameter values. These values define a numerical reference case and establish no mission requirement or instrument recommendation.

| Quantity | Fixture choice |
| --- | --- |
| LayerAtmosphere | Heights `[0,500,1500] m`; pressures `[100000,90000,70000] Pa`; temperatures `[290,280] K`; constant dry-air density `2e25 molecules/m3`; water/dry-air molecular ratio `0.01`. |
| Gas and conversion | One CH4-labelled trace absorber at dry-air molecular fraction `2e-6`; plume molar mass `0.01604 kg/mol`; kg/m2 multiplied by `Avogadro/molar_mass_kg_mol`. |
| Spectroscopy | Smooth synthetic cross section centred at `4050 cm-1`, fixed under parameter variation. It is independent of pressure, temperature and abundance and has no HITRAN or measured-spectrum claim. |
| Prescribed plume | One free-space Gaussian at `(150,0,800) m`, mass `20 kg`, diagonal standard deviations `(180,200,120) m`, frozen at elapsed time zero. |
| Terrain and support | Default horizontal terrain at `250 m`, cutting the first layer; supported affine slopes have norm at most `0.1`. The acquisition certificate requires all configured atmospheric height edges within `[0,1500] m` and terrain inside that atmosphere. Application chart support is `[-3000,3000] m` in x and y. |
| Illumination and surface | Outward Sun vector `(0.6,0,0.8)`; beam-normal irradiance `0.02 W m-2 per (cm-1)`; reflectance baseline `0.25` plus smooth spatial/spectral contrast and a fitted offset. |
| Time and frame | UTC epoch `2023-05-16T00:00:00Z`, support `[0,0.3] s`; analytic chart-to-GCRS z rotation at `0.02 rad/s`, with GCRS chart origin `(6400000,0,0) m`. |
| Sensor | Height `5000 m`, above model top; chart x speed `800 m/s`; fixed local oblique pointing of `0.3 rad`; two exposures of `0.04 s`, starting at `0` and `0.1 s`. |
| Camera | Two rows and two columns; pitch and active size `0.002 m`; focal length `0.3 m`; f-number `4`; Gaussian PSF width `0.0003 m`; omitted PSF mass `1e-6`; QE `0.7`; two spectral strip laws. |
| Demonstration quadrature | 33 spectral points over `[4000,4100] cm-1`, time order 2, angular order 21 in each sensor direction. Coarse test grids are labelled software fixtures. |

The frame law defines a synthetic moving Cartesian slab embedded in GCRS. It is not an Earth-fixed ITRF transformation or an orbital ephemeris. Projection computes chart coordinates and outward views with the same dated rotation, origin and affine terrain. Physical radiance receives these local queries directly. Earth applications must supply their own dated transformation and valid auxiliary support through Geometry's existing frame interfaces.

The plume is a prescribed frozen field evaluated above terrain. The model supplies no terrain-aware transport or reflecting boundary at the affine surface. It clips atmospheric support without renormalizing plume mass. `plume_excluded_mass_bound` computes a conservative kg bound from Gaussian half-space probabilities for terrain, model top and horizontal domain. For the default flat case, the excluded fraction is below `3e-6`; changed terrain or puff inputs require a new mass-accounting assessment. The trace-gas calculation holds pressure, temperature and air density fixed.

## Continuous coverage proof and restrictions

`make_acquisition` returns a coverage callback bound to the exact stable grid constructed by its known analytic laws. It conservatively rejects another grid, even one with identical sampled nodes, because those nodes do not establish the underlying continuous law. The caller must rebuild the acquisition and certificate after any geometry change. Array mutation while callbacks are in use is outside the example contract.

For the retained sensor rectangle, `abs(u),abs(v)<=0.02`. Fixed local pointing then gives upward view cosine greater than `0.8`. Surface slope norm is at most `0.1`; therefore the affine intersection denominator stays separated from zero. Surface-to-observer directions satisfy `normal_z*0.8 - norm(normal_xy)*0.6 > 0`. A positive Sun-normal dot product establishes an unshadowed affine surface along the entire solar ray. This continuous argument excludes grazing or visibility transitions.

The chart sensor moves linearly while its local attitude stays fixed; the synthetic GCRS rotation cancels exactly on chart inversion. For slopes at most `0.1` and supported heights between 0 and 1500 m, surface x/y speed is below `1600 m/s`. Intersection distance is below `5000/0.74 m`. Differentiating the affine intersection with the sensor direction map bounds the chart infinity-norm angular derivative below `20000 m`. Those conservative values are passed to acquisition. Its padded `bounds_m` therefore encloses the continuous surface footprint under these restrictions.

The coverage callback restricts the configured atmospheric height edges to `[0,1500] m`, inclusive, in the synthetic chart datum. This restriction supplies the height premise of the fixed motion and angular bounds. Other configured height domains return false, including domains with negative terrain. Affine terrain extrema over each padded rectangle occur at corners. The coverage callback checks those extrema against atmospheric support. It pads the footprint by `(top-minimum_height)*max(0.75,max(abs(sun_xy))/sun_z)` in both horizontal coordinates. The factor `0.75` bounds all view slopes when view cosine is at least `0.8`. This enclosure includes both complete displaced atmospheric paths and the convex rectangle between their endpoints.

Coverage also checks complete exposure intervals, epoch identity and every declared spectral-domain interval, including edges. Any unsupported or unproven read returns false, causing `simulate_frames` or `frame_electron_tangents` to fail before physical radiance evaluation. Valid sampled nodes alone cannot certify a read. Direct standalone queries separately check shapes, time, spectrum, chart positions, outward directions and path endpoints.

## Sensitivities and numerical evidence

The analytical derivative boundary fixes transport, plume shape, geometry, terrain, visibility, spectral cross sections, background, irradiance and the complete physical instrument response. Plume amplitude multiplies only converted excess molecular amounts. Reflectance offset adds one to its spectral derivative. Signed derivatives share the actual forward spatial, temporal, spectral, photon, pupil, throughput and QE coefficients. Noise draws stay outside the Jacobian.

The [Scene tests](../libs/orsia-scene/tests/test_scene_composition.py), [Geometry tests](../libs/orsia-geometry/tests/test_terrain.py), [Radiative tests](../libs/orsia-radiative/tests/test_terrain_paths.py) and [connected tests](../tests/test_spatial_scene.py) cover these references:

| Reference | Observable and acceptance |
| --- | --- |
| Known affine plane and chart inversion | Analytic surface points and collinearity; metre residuals near float64 roundoff. Misses cover upward, below-surface and grazing rays. |
| Full/partial atmospheric layers | Exact retained fractions, dry-plus-water identity, absorber conservation, pressure reconstruction, exact interfaces and rejected uncovered heights. |
| Constant fields and ragged paths | Geometric segment lengths, independently summed species/scattering depths, empty-path and zero-direction-count contracts. |
| Gaussian layer/path conversion | Independent adaptive quadrature of concentration on the same support; relative agreement within `2e-10` for path molecular columns. |
| Zero plume and homogeneous atmosphere | Independent-column composition, separate-path transfer and an independent homogeneous radiance formula produce expected electrons within `1e-14` relative tolerance. |
| Extraction regression | Saved pre-extraction expected electrons and signed sensitivities agree within `2e-14` relative tolerance for horizontal independent columns and horizontal/sloped heterogeneous paths. These saved values establish behavior preservation. |
| General library inputs | Two species, multiple plumes including a shared species, zero/one/three signed parameter directions, dated geometry frame law, flat query axes and fresh output storage. |
| Radiance-only API | Array-returning column callbacks, direct/tangent value equality in both modes, partial layers on sloped heterogeneous terrain, transparent/opaque limits, query permutations and empty queries. |
| Direction and frame handling | Separate oblique rays sample distinct plume amounts; changed viewing direction changes radiance; dated frame inversion and query permutations preserve the frozen-field result. |
| Continuous support | Reject full exposure, declared spectral edges and displaced atmospheric paths when sampled source coordinates remain valid. Reject pre-binding atmospheric height domains outside `[0,1500] m`. |
| Extreme direct attenuation | Independent Decimal formula checks signed derivatives, opaque-primal underflow, large finite illumination, overflowing unscaled seeds, zero albedo, shadow and cancellation; relative tolerance `3e-14` with zero absolute allowance. |
| Composed Jacobians | Centred finite differences of expected native electrons at steps `1e-2`, `1e-3`, `1e-4`; relative tolerance `2e-7`, with `0.02 electron` absolute allowance. Batch sizes 1 and 8 agree. |
| Joint signed response | A separate physical field-dependent spectral-transmission test makes the response nonseparable; its electron Jacobian agrees with full forward finite differences. |
| Refinement | Angular orders 13, 21, 29 are compared separately; 21-to-29 relative electron change is below `2e-6`. Spectral 33-to-65 and temporal 2-to-4 comparisons each use `2e-5` relative tolerance. Analytic path amounts are invariant under layer subdivision within `1e-13`. |

These checks establish bounded numerical and software evidence. They leave authentic spectroscopy, heterogeneous diffuse transfer, terrain transport, retrieval performance, uncertainty reconstruction and engineering feasibility unqualified. Geometry/transport/response derivatives and visibility transitions require separate contracts.

Run the focused tests with `devbox run -- just --justfile tests/justfile pytest libs/orsia-scene/tests libs/orsia-geometry/tests/test_terrain.py libs/orsia-radiative/tests/test_terrain_paths.py tests/test_spatial_scene.py`. The installed-distribution check is `devbox run -- just python scripts/validate_scene_distribution.py`; it builds archived packages offline, installs them into a temporary environment under `outputs/scene-validation`, and verifies the direct and tangent README calls, native imports and both connected modes with workspace resolution disabled. The script checks that native imports leave the GIL disabled without forcing it off. The ordinary bounded headless `just check` remains the repository completion gate.
