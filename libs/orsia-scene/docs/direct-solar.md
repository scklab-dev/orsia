# Frozen direct-solar cell scenes

[`DirectSolarScene`](../src/orsia_scene/radiance.py) evaluates physical radiance on explicit local surface queries. Its shared preparation contains a `SpatialOpticalDepth`, two surface-reflectance anchors, beam-normal irradiance and a solar direction. Construction also fixes the spatial method, continuation rule, clear-path assertion, UTC epoch and closed frozen interval. Instrument filters, detector properties, acquisition grouping and product-analysis masks remain outside this owner.

## Construct and bind

This executable example uses synthetic thermodynamics, cross sections and geometry. It illustrates the public contract without asserting atmospheric accuracy.

```python
from datetime import UTC, datetime
import numpy as np
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells
from orsia_radiative import LayerCrossSections, LevelAtmosphere, SpectralGrid
from orsia_scene import DirectSolarScene, GasLayerColumns, SpatialOpticalDepth

frame = LocalCartesianFrame("synthetic cells", "synthetic altitude")
points = PointLattice.regular((2, 2), (10.0, 10.0), (0.0, 0.0), frame)
support = LayeredCells(
    RectangularCells.around_points(points, (10.0, 10.0)),
    np.array([0.0, 20.0]), np.zeros((2, 2)),
)
spectral = SpectralGrid("wavelength_nm", np.array([500.0, 600.0, 700.0]))
state = LevelAtmosphere(
    support.height_edges_m, np.array([1e5, 9e4]), np.full(2, 280.0),
    np.zeros(2), ("CH4",), np.full((2, 1), 2e-6),
)
table = LayerCrossSections(
    "CH4", "all", "cross_section", spectral, np.array([280.0]),
    np.array([9.5e4]), np.array([[1e-23, 2e-23, 3e-23]]),
    height_edges_m=support.height_edges_m,
)
optical_depth = SpatialOpticalDepth(state, (table,), spectral=spectral, support=support)
scene = DirectSolarScene(
    optical_depth, anchor_wavelengths_nm=(500.0, 700.0),
    surface_reflectance=np.full((2, 2, 2), 0.3),
    beam_normal_irradiance=np.ones(3), solar_direction=np.array([0.0, 0.0, 1.0]),
    epoch=datetime(2026, 1, 1, tzinfo=UTC), validity_s=(0.0, 1.0),
    method="layer_centroid", continuation="finite", visibility="assume_clear",
)
gas = GasLayerColumns(support, "CH4", np.full((2, 2, 1), 1e21), np.full((2, 2, 1), 10.0))
bound = scene.with_gas((gas,))
xy = points.xy_m.reshape(-1, 2)
views = np.broadcast_to([0.0, 0.0, 1.0], (4, 3))
times = np.full(4, 0.5)
values = bound.radiance(xy, views, times, spectral)
assert values.shape == (4, 3) and np.all(values >= 0)
seeded = bound.with_tangents(np.ones((1, 1)), np.zeros((2, 2, 2, 1)))
primal, derivative = seeded.radiance_tangent(xy, views, times, spectral)
np.testing.assert_array_equal(primal, values)
assert derivative.shape == (4, 3, 1) and np.all(derivative < 0)
```

`with_gas` retains the exact complete support, including cell bounds, local frame, terrain and layer interfaces. Equal shapes alone are insufficient. Species must exist in the atmospheric declaration; CO2 is optional. Multiple contributions can share a species. Nonzero reference H2O columns with prepared water continuum requires a physical density-product update and is rejected by this fixed-coefficient binding.

Each binding shares unchanged physical preparation, including precomputed species coefficients. Array inputs retained at construction and tangent binding are independent immutable snapshots. `with_gas(())` removes excess gas. Optional `scales` supplies nonnegative amplitudes for the reference column fields, defaulting to one. A zero amplitude retains reference centroids and permits a nonzero absolute-amplitude tangent at the null scene. Every gas rebind clears the previous tangent block because contribution order can change. Calls allocate independent writable outputs and scratch; they perform no I/O, stochastic transport, random draws or implicit cache mutation.

## Query support and reconstruction

The callback is `radiance(xy_m, view_local, times_s, spectral)`. Coordinates have `(Q,2)` axes in local `(x,y)` metres. Views have `(Q,3)` axes and point outward from the surface toward a sensor above the atmosphere. Times have `(Q,)` axes in elapsed seconds from `scene.epoch`; endpoints of `scene.validity_s` are admitted. `SpectralGrid` identifies the density coordinate and exact prepared samples. Changing coordinate or interpolating coefficients requires explicit upstream preparation. Empty queries and repeated queries are valid.

Output has `(Q,K)` axes in W/m2/sr per declared spectral coordinate, either nm or cm-1. The scene uses the irradiance supplied in that same coordinate. Point coefficients and reflectance receive no density Jacobian. Spectral-density conversion belongs to the source preparation when changing coordinates.

The physical support must be regular tiled cells with `(y,x)` array order. Queries use half-open cells `[lower,upper)`, including lower faces and excluding the outer upper boundary. Reconstructed radiance is constant within each destination cell for a fixed direction. A query in a masked physical cell raises `ValueError`. Processing masks remain independent instrument or application data.

The cell model treats each surface facet as horizontal and the background atmosphere as a local vertical column. `visibility="assume_clear"` asserts unobstructed solar and viewing paths. Cell membership and sampled direction checks do not certify terrain visibility or continuous exposure coverage. The application must compare frame, UTC epoch, full time interval, spectral support and complete footprints before connecting a scene to an acquisition. The separate [path-resolved functions](../README.md#public-contracts) retain their terrain and visibility interfaces.

## Numerical construction

1. **Objective:** select supported physical values. **Inputs:** surface coordinates, outward views, elapsed times, requested spectral samples and the shared cell preparation. **Outputs:** cell indices `(y_q,x_q)`, spectral indices `k`, background depths `tau_bg[q,k]`, reflectance `rho[q,k]` and irradiance `E[k]`. **Processing:** validate the closed time interval, unit upward views and exact spectral subset. Select cells by right-sided edge search and reject outside or masked queries. `SpatialOpticalDepth.at_pixels` contracts only selected cell coefficients and spectral bases. For wavelengths between anchors `a<b`, let `f=(lambda-a)/(b-a)` and set `rho=(1-f)*rho_a+f*rho_b`. Wavenumber queries use `lambda=1e7/nu` for this dimensionless surface law. Extrapolation outside the anchor interval fails.
2. **Objective:** construct separate incident and outgoing extinction. **Inputs:** background depths, each gas column field `N_g[y,x,l]`, its centroid height `H_g` above local terrain, fixed cross sections `sigma_g[l,k]`, solar direction `s` and view `v_q`. **Outputs:** `tau_s[q,k]` and `tau_v[q,k]`. **Processing:** initialize `tau_s=tau_bg/s_z` and `tau_v=tau_bg/v_z`. In `independent_column` mode, set `C_g=N_g[y_q,x_q,:] @ sigma_g`, then add `C_g/s_z` and `C_g/v_z`. In `layer_centroid` mode, project `N_g` with the [declared centroid law](../README.md#explicit-centroid-continuation), using slopes `(-s_y/s_z,-s_x/s_z)` and `(-v_y/v_z,-v_x/v_z)` in array `(y,x)` order. Select destination cells, contract layers with cross sections, then apply the corresponding secant. Multiply each nominal gas contribution by its supplied amplitude; tangent contributions use the reference field times the supplied amplitude seed, including at zero amplitude. The negative sign follows the outward-ray convention: a ray from a surface point reaches height `H` at `xy_surface+H*direction_xy/direction_z`. Reuse each distinct view projection within the current batch.
3. **Objective:** evaluate direct reflected radiance. **Inputs:** the preceding slant depths, reflectance, irradiance and horizontal-facet incidence `s_z`. **Outputs:** `L[q,k]`. **Processing:** use Radiative's shared direct transfer kernel for `L=rho*E*s_z/pi*exp(-tau_s-tau_v)`. Its scaled arithmetic retains representable products. The cell scene introduces no diffuse radiation, spectral integration or optical blur.
4. **Objective:** propagate a bound physical direction block. **Inputs:** signed gas-scale seeds `a[g,p]` and surface-anchor seeds `(y,x,2,P)` with fixed geometry, times, centroids, illumination and spectroscopy. **Outputs:** `dL[q,k,p]` and the same primal radiance. **Processing:** multiply each gas contribution to incident/outgoing depth by its scale seed and sum contributions. Interpolate surface seeds with the same affine basis to obtain `drho`. Pass both depth seeds and `drho` to Radiative's direct tangent kernel, which implements `dL=(E*s_z/pi)*exp(-tau_s-tau_v)*(drho-rho*(dtau_s+dtau_v))` with scaled signed arithmetic. Unsupported geometry or timing directions are absent from this interface; callers must select a separately defined method for them.

The centroid calculation is an explicit within-cell approximation. It loses outgoing material under finite continuation and requires uniform amounts and active heights under infinite-uniform continuation. It does not resolve the full spatial gas distribution or terrain-dependent ray intersections. Averaging radiance after evaluation differs from evaluating radiance at an averaged column or direction.

## Measurement boundary and evidence

Pass the bound physical methods to Instruments' `simulate_frames` or `frame_electron_tangents`, with a caller-owned complete acquisition coverage certificate. Native frame/observation products retain their cm-1 contract; the lower-level photon and tangent integrators support either declared coordinate. The existing `photoelectron_response` coefficients already include time, angular and full-domain spectral quadrature, photon energy, area, throughput and QE. Slice these prepared coefficients when testing spectral blocks; recomputing trapezoidal endpoints independently in each block changes the integral. Apply spatial filtering to the radiance product once, under its separately selected boundary rule.

Tests cover independent transfer formulas, outward-view projection sign, finite-difference gas/surface derivatives, exact spectral subsets, nm/cm-1 density conversion, support and mask failures, owned inputs, repeated concurrent evaluations and a bounded instrument-coefficient composition. Block comparisons allow floating-point summation differences. The historical log-tangent band fixture remains a test-only approximation reference. These checks establish the stated numerical cases; atmospheric accuracy, realistic acquisition performance and retrieval qualification remain separate.
