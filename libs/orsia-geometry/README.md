# Orsia Geometry

Construct observation rays from a trajectory and independent platform/instrument attitudes, then intersect the rays with analytic spheres or the fixed WGS84 reference ellipsoid. Local height-field operations also evaluate terrain points and normals and search for intersections with caller-supplied surfaces.

![Classical elements construct a GCRS trajectory; compute_lines_of_sight combines independent attitudes and instrument geometry and returns ITRF rays for separate WGS84 intersections.](docs/diagrams/architecture.svg)

The left-to-right path follows the example below. Arrows pass numerical inputs or evaluated arrays; the trajectory stage supplies positions and velocities at the requested times. Independent attitude laws enter `compute_lines_of_sight`, which composes the rays and transforms both arrays into the selected output frame. Surface intersection remains a separate operation; the surface-model node denotes Geometry's fixed WGS84 constants. The diagram shows this usage path, with orbital adapters at construction and inside the ITRF ray computation.

## Construct a LEO trajectory and ground trace

This example defines an illustrative designed low Earth orbit (LEO) from classical osculating elements at `2023-05-16T20:00:00Z`. The semimajor axis is 700 km above the WGS84 equatorial radius, with eccentricity 0.001 and inclination 98 degrees. These chosen elements describe a hypothetical orbit; a real spacecraft calculation needs dated orbit estimates and an appropriate propagation model.

`NumericalPropagator` accepts the following element order in GCRS (geocentric celestial reference system):

| Element | Meaning | Unit |
| --- | --- | --- |
| `a` | Semimajor axis | m |
| `e` | Eccentricity | dimensionless |
| `i` | Inclination | rad |
| `RAAN` | Right ascension of ascending node | rad |
| `arg_periapsis` | Argument of periapsis | rad |
| `true_anomaly` | True anomaly at epoch | rad |

The constructor numerically integrates once and returns a callable `NumericalPropagator` object. Calling `trajectory(times_s)` evaluates retained dense output and returns two independent writable float64 arrays: GCRS positions in metres and velocities in metres per second, each shaped `(times, 3)`. Times are physical elapsed SI seconds from the UTC epoch. The closed `validity_s` interval includes both endpoints; times outside the interval fail.

`NumericalPropagator` exposes fixed scalar configuration and an independent copy through `trajectory.elements`. Construct a new object to change force settings or time support. Use the direct classes listed in the [constructor migration](docs/geometry.md#callable-object-migration).

The example uses the default EGM96 degree 2/order 0 force model and RKV98 integration, with absolute tolerance `1e-4` in state units and relative tolerance `1e-11`. Sun/Moon gravity, tides, relativity, space weather, drag and radiation pressure are disabled. Use `gravity_degree`, `gravity_order`, `sun_gravity` and `moon_gravity` to select the supported force terms. Use `absolute_tolerance` and `relative_tolerance` to refine integration while preserving those terms. Public calls use scalar controls, UTC datetimes and NumPy arrays. See the [orbital adapter contract](docs/geometry.md#orbital-adapters) for supported ranges, fixed native choices and migration from the removed `settings` keyword.

Install `orsia-geometry[orbital]` for this example; the extra supplies SatKit 0.24.1. The base package requires NumPy only. The [workspace setup](../../README.md#development-environment) supplies the development environment. SatKit can acquire its normal public auxiliary files in process. Earth-orientation parameter (EOP) coverage must include the requested dates and the propagator's padded nodes. Missing or held EOP values fail explicitly.

Run the three Python blocks in this section in order in the same session.

### 1. Define the trajectory and attitudes

```python
from datetime import UTC, datetime

import numpy as np
from orsia_geometry import NumericalPropagator, compute_lines_of_sight, ellipsoid_intersections, radial_nadir
from orsia_geometry.constants import WGS84_SEMIMAJOR_AXIS_M

# Osculating [a, e, i, RAAN, arg_periapsis, true_anomaly]; metres and radians.
epoch = datetime(2023, 5, 16, 20, tzinfo=UTC)
validity_s = (0.0, 60.0)
elements = np.array([WGS84_SEMIMAJOR_AXIS_M + 700000.0, 0.001, *np.deg2rad([98.0, 0.0, 0.0, 0.0])])
trajectory = NumericalPropagator(elements, epoch=epoch, validity_s=validity_s)


def platform_attitude(times_s):
    return radial_nadir(*trajectory(times_s))


def instrument_attitude(times_s):
    return np.broadcast_to(np.eye(3), (len(times_s), 3, 3))
```

`radial_nadir` points +platform-z toward Earth's centre and sets +x along transverse inertial motion. This is radial nadir; ellipsoid-normal nadir requires another attitude law. The identity instrument attitude fixes the instrument to the platform. Both attitude callbacks return `(times, 3, 3)` column-vector rotations, respectively platform-to-GCRS and instrument-to-platform. Replace either law independently to model tracking or scanning.

### 2. Compute the lines of sight in ITRF

Sample 31 times over the 60-second interval and use the same three instrument directions at each time. `times_s` has shape `(31,)`; `directions_i` has shape `(3, 3)`, with one unit direction per row. `compute_lines_of_sight(..., output_frame="ITRF")` combines the GCRS trajectory and independent attitudes, then rotates both ray origins and directions into ITRF (international terrestrial reference frame). The returned origins are in metres and the directions are unit vectors. Both arrays have shape `(times, rays, coordinates)`, here `(31, 3, 3)`, with final-axis coordinate order x, y, z.

```python
times_s = np.linspace(0.0, 60.0, 31)
# Boresight +instrument-z and two rays about 1 mrad off-axis.
directions_i = np.array(
    [[0.0, 0.0, 1.0], [0.001, 0.0, np.sqrt(1 - 1e-6)], [-0.001, 0.0, np.sqrt(1 - 1e-6)]]
)
ray_origins_itrf_m, ray_directions_itrf = compute_lines_of_sight(
    times_s,
    epoch=epoch,
    validity_s=validity_s,
    trajectory=trajectory,
    platform_attitude=platform_attitude,
    instrument_attitude=instrument_attitude,
    directions_i=directions_i,
    mount_offset_p_m=np.zeros(3),
    output_frame="ITRF",
)
```

The default `output_frame="GCRS"` output computation uses NumPy only and adds no backend imports or auxiliary acquisition. Supplied callbacks retain their own dependencies and effects. `"ITRF"` and `"TEME"` select the existing lazy SatKit adapter. The operation rotates each origin and direction at its requested time; ray coordinates need no velocity-derivative term or shifted dates. Returned arrays own independent writable float64 storage, including empty time/ray axes. Inputs and callback outputs remain unchanged.

Invalid frame names, epochs or time support raise `ValueError` before callbacks or backend use. ITRF/TEME output raises `RuntimeError` when EOP coverage or backend evaluation fails. An `ImportError` indicates that the required SatKit installation cannot load. Callback exceptions propagate unchanged. SatKit can acquire public auxiliary files and populate process-global caches; serialize native evaluation, configuration changes and auxiliary refresh. See the [resource contract](docs/geometry.md#resources-provider-boundary-and-execution).

### 3. Intersect the rays with the fixed WGS84 surface

`ellipsoid_intersections(origins_m, directions)` fixes the WGS84 reference ellipsoid internally at ellipsoidal height zero. Supply Earth-centred ITRF Cartesian x, y, z origins in metres and matching ITRF unit directions. The reference surface excludes terrain and atmospheric ray bending. `orsia_geometry.constants` defines the WGS84 scalars once and derives the immutable `WGS84_SEMIAXES_M` tuple in ITRF x, y, z order. See the [surface contract](docs/geometry.md#analytic-surfaces) for definitions and source references.

`ellipsoid_intersections` selects the nearest forward intersection for each ray. The `hit` mask has shape `(times, rays)`, here `(31, 3)`. The corresponding `ground_points_itrf_m` array has shape `(31, 3, 3)` and contains Earth-centred ITRF Cartesian x, y, z coordinates in metres. Latitude and longitude require a separate geodetic conversion. Misses have a false mask entry and NaN coordinates.

```python
hit, ground_points_itrf_m = ellipsoid_intersections(ray_origins_itrf_m, ray_directions_itrf)
print(f"{hit.sum()} hits; ITRF points shape: {ground_points_itrf_m.shape}")
```

The example produces 93 hits, one for each of three rays at each of 31 times. The README execution test checks array shapes, all-hit status and ellipsoid membership. Those checks establish execution and surface consistency; orbit knowledge, attitude knowledge, terrain and atmospheric ray bending remain application inputs or limitations.

## Intersect a ray with interpolated terrain

![A caller supplies a consistent height and gradient model to surface_points and intersect_terrain; the operations return local points and normals or a hit mask and bounded-search intersections.](docs/diagrams/terrain.svg)

Arrows pass supplied callables or numerical values. The two operations share a terrain model and can be called independently. The application owns interpolation, support and distance samples; finite sampling can miss contacts between nodes. [Full-size SVG](docs/diagrams/terrain.svg) · [Editable Draw.io source](docs/diagrams/terrain.drawio).

This standalone example constructs a curved synthetic terrain grid, evaluates a surface point and its normal, then intersects a downward ray with the same surface. It requires NumPy, SciPy and Orsia Geometry from the development environment and uses no orbital data.

All coordinates use one local Cartesian frame: x and y are horizontal, z is upward, and distances and heights are in metres. The grid has shape `(9, 7)`, with axes `(x, y)`. The height and gradient callbacks use the same cubic interpolator, so the slopes describe the interpolated surface.

```python
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from orsia_geometry import intersect_terrain, surface_points

x_m = np.linspace(-100.0, 100.0, 9)
y_m = np.linspace(-80.0, 80.0, 7)
xx_m, yy_m = np.meshgrid(x_m, y_m, indexing="ij")
heights_m = 20.0 + 0.002 * xx_m**2 + 0.001 * yy_m**2
interpolator = RegularGridInterpolator(
    (x_m, y_m), heights_m, method="cubic", bounds_error=True,
    solver_args={"rtol": 1e-13, "atol": 1e-13},
)


def height(xy_m):
    return interpolator(xy_m).reshape(xy_m.shape[:-1])


def gradient(xy_m):
    return np.stack(
        [interpolator(xy_m, nu=order).reshape(xy_m.shape[:-1])
         for order in ((1, 0), (0, 1))],
        axis=-1,
    )


terrain_points_m, terrain_normals = surface_points(
    np.array([[30.0, -20.0]]), height=height, gradient=gradient,
)
terrain_hit, terrain_intersections_m = intersect_terrain(
    np.array([[30.0, -20.0, 100.0]]),  # One origin, shape (1, 3), in metres.
    np.array([[0.0, 0.0, -1.0]]),     # One downward unit direction.
    height=height, gradient=gradient,
    horizontal_bounds_m=(x_m[0], x_m[-1], y_m[0], y_m[-1]),
    distance_nodes_m=np.linspace(0.0, 100.0, 11),
    distance_tolerance_m=1e-8, height_tolerance_m=1e-9,
    minimum_incidence_cosine=0.1,
)
print(f"Terrain height: {terrain_points_m[0, 2]:.2f} m")
print("Upward unit normal:", terrain_normals[0].round(4).tolist())
print(f"Ray hit: {bool(terrain_hit[0])}")
print("Intersection [x, y, z] (m):", terrain_intersections_m[0].round(2).tolist())
```

Expected output:

```text
Terrain height: 22.20 m
Upward unit normal: [-0.1191, 0.0397, 0.9921]
Ray hit: True
Intersection [x, y, z] (m): [30.0, -20.0, 22.2]
```

`height` maps horizontal coordinates shaped `(..., 2)` to heights shaped `(...)`. `gradient` returns dimensionless slopes `(dz/dx, dz/dy)` shaped `(..., 2)`; `nu` selects each derivative of the interpolator. The reshapes preserve these contracts for both single coordinates and batches. Here, the surface points, normals and intersections each have shape `(1, 3)`, and the hit mask has shape `(1,)`. The upward unit normal follows `(-dz/dx, -dz/dy, 1)` after normalization.

The search samples forward distances from 0 to 100 m at 10 m spacing, then refines the first detected contact candidate. The ray reaches the terrain after 77.8 m. Every searched point must stay inside the declared horizontal bounds; this vertical ray keeps x and y fixed and has one crossing. For general rays, finite samples can miss crossings or tangencies between nodes. A rejected grazing candidate ends the search, and a miss does not certify clear visibility. See the [callable-terrain guide](docs/terrain.md) for support, tolerances, first-intersection limits and another interpolation adapter.

## Use GP OMM or TLE mean elements

For an existing spacecraft's GP OMM or TLE product, use the distinct `SGP4Propagator` route. These products contain SGP4 mean elements and must not be supplied as classical osculating elements to `NumericalPropagator`.

Install `orsia-data[orbital]` and read a local file with Data's `read_orbital_elements(path, format="omm")` or `read_orbital_elements(path, format="tle")`. Import the reader and `select_orbital_elements` from `orsia_data` or `orsia_data.orbits`. Both file formats return a list of Geometry-owned immutable `SGP4MeanElements` records. Use `select_orbital_elements` to select each requested object's latest set within the accepted element-epoch interval, in requested object order. Pass a selected record directly to `SGP4Propagator`. Supply the evaluator's UTC `epoch` and accepted `validity_s` explicitly. The returned callback has the same GCRS position/velocity contract as the example, after SGP4 propagation in TEME and a dated state transformation. The [Data orbital guide](../orsia-data/docs/orbital-elements.md#scalar-records-and-selection) defines the reader and selection contracts. Local files require no provider account; element availability alone establishes no prediction accuracy.

## Point and finite-cell supports

`LocalCartesianFrame` declares Cartesian axes and a vertical datum. Synthetic charts match only through the same object. Anchored charts specify an ITRF origin and a right-handed basis together; `same_local_frame` compares origin, basis and datum exactly. Names are descriptive. Geometry performs no implicit reprojection or datum conversion.

`PointLattice` owns increasing `x_m` and `y_m` axes; values use `(y,x,...)`. `RectangularCells` owns lower/upper `(x,y)` corners with `(*pixels,2)` axes and derives full areas. `LayeredCells` adds vertical interfaces and terrain and derives clipped volumes. These records own immutable numerical snapshots. Their `matches` methods compare the complete support exactly, without hidden tolerances. Constructing cells around points requires explicit cell widths and converts geometry only; point values and cell averages retain distinct numerical meanings.

```python
import numpy as np
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells

frame = LocalCartesianFrame("synthetic scene", "synthetic altitude")
points = PointLattice.regular((2, 3), (50.0, 50.0), (0.0, 0.0), frame)
cells = RectangularCells.around_points(points, (50.0, 50.0))
support = LayeredCells(cells, np.array([0.0, 100.0, 300.0]), np.zeros(points.shape))
assert support.shape == (2, 3, 2)
np.testing.assert_array_equal(cells.area_m2, np.full((2, 3), 2500.0))
```

## Responsibilities and dependencies

Geometry requires NumPy; orbital/frame evaluation additionally requires the `orbital` extra. Array-only operations use NumPy without loading SatKit. `los.compute_lines_of_sight` checks closed support and composes a supplied GCRS trajectory with independent attitude callbacks. Its default GCRS output stays in the NumPy core; ITRF/TEME output uses the lazy frame adapter internally. `rays.compose_rays` also supports arrays in a caller-named common frame. `attitude` supplies radial nadir, target tracking, scan-axis rotations and sampled angular-speed estimates. `surfaces` owns sphere intersections, fixed-WGS84 intersections and WGS84 target visibility. `surface_visibility(observers_m, targets_m)` requires exterior ITRF observers and matching WGS84 height-zero targets, all in metres.

The `orbits` and `frames` adapters import the optional orbital SatKit dependency lazily and use its ordinary in-process auxiliary management. Serialize native operations, backend configuration changes and auxiliary refresh; concurrent native scientific safety remains unqualified. Instruments consumes Geometry for UTC-epoch/GCRS acquisition rays and owns acquisition quadrature. Geometry has no dependency on Instruments, Data or authenticated providers. The [numerical contract](docs/geometry.md) defines axes, time, ownership, singular cases, resource boundaries and qualification limits.

For existing dated rays, `transform_rays` rotates matching `(...,3)` origins and directions with exactly `(...)` elapsed times. `project_local_east_north` projects ITRF points onto an explicit height-zero WGS84 origin's fixed east/north metre chart. `WGS84SurfaceProjection(epoch=..., validity_s=..., chart_origin_itrf_m=...)` composes both operations with WGS84 intersections into a callable object returning `(xy_m, view_local, hits)`. Outward unit views use fixed chart-origin east/north/up axes after dated GCRS-to-ITRF rotation. `WGS84SurfaceProjection.from_ray` returns the same class and selects the chart origin from one explicit `(3,)` GCRS ray and scalar elapsed time; construction uses SatKit and requires EOP coverage. Reuse the same callback for truth and estimated geometry. See the [chart contract](docs/geometry.md#fixed-local-eastnorth-chart) and the executable [fixed-filter camera walkthrough](../orsia-instruments/docs/filter-camera.md#walkthrough-two-strips-over-earth). Continuous visibility and support bounds remain caller responsibilities.

The [spatial-scene guide](../../docs/spatial-scenes.md) documents `affine_surface_points`, `intersect_affine_terrain` and `segment_layer_paths`. These array-only operations supply a reversible affine-terrain chart, surface normals and ragged straight paths split at horizontal atmospheric interfaces. The connected example reuses the same terrain for acquisition and radiance; its explicit continuous-support bounds and synthetic dated frame law remain application-owned.

For curved single-valued terrain, use `surface_points` with height and gradient callables. Workspace applications can use the [`SplineTerrain` support model](../../notebooks/support/terrain.py) to share one interpolant and its closed domain through bound methods. `intersect_terrain` searches explicit bounded distance samples and refines the first detected contact candidate. Finite samples cannot certify hidden crossings or tangencies. The [callable-terrain guide](docs/terrain.md) defines the domain, tolerance, grazing and first-intersection limits and provides runnable SciPy interpolation adapters.

## Run examples and checks

Run the standalone terrain example and its expected-output check with `devbox run -- just --justfile tests/justfile pytest tests/test_library_guides.py -k geometry` from the workspace root.

With the existing DE421 and IERS EOP fixtures under `outputs/geometry-auxiliaries`, execute all README Python blocks against actual SatKit from the workspace root. The offline recipe fails if required files are absent:

```bash
devbox run -- just --justfile tests/justfile test-orbital -k readme -s
```

SatKit's normal public auxiliary management remains available outside the fixed local test recipe; offline operation is a test-fixture choice.

`devbox run -- just --justfile tests/justfile pytest libs/orsia-geometry/tests libs/orsia-instruments/tests/test_line_of_sight.py tests/test_core_imports.py` covers array geometry, supplied trajectories and orbital contracts without requiring native orbital resources. The opt-in `test-orbital` scope executes this README and separately exercises native SatKit against bounded ERFA, GMAT and Vallado references. The workspace guide tests retain dependency-light examples and local-link checks. `test` and `check` include Geometry in the aggregate and per-library 90% combined statement-and-branch coverage gates. Software checks establish no mission, orbit-knowledge or engineering qualification. Authenticated Space-Track access belongs to Data and has only mocked/local qualification here.
