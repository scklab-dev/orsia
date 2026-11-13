# Callable terrain geometry

`surface_points` evaluates a single-valued height field and its upward unit normals. `intersect_terrain` searches bounded straight rays for sampled contact candidates. Both functions use a caller-selected Cartesian metre frame and vertical datum. The model supports curved height fields and excludes overhangs, refraction and coordinate conversion.

## Callable contract

Supply two ordinary callables: `height(xy_m)` maps `(...,2)` horizontal coordinates to `(...)` heights in metres, and `gradient(xy_m)` returns matching `(...,2)` dimensionless derivatives in `(dz/dx,dz/dy)` order. Callers ensure that both callables represent the same surface and interpolation convention. `surface_points` returns fresh writable float64 points and upward unit normals, both shaped `(...,3)`. The upward normal is proportional to `(-dz/dx,-dz/dy,1)`.

Inputs and callback outputs remain unchanged. Callbacks borrow their arguments and must keep their state stable during each operation. Concurrent callers must supply thread-safe callbacks. Nonfinite values or incompatible output shapes raise `ValueError`; callback exceptions propagate. Supply valid coordinates to `surface_points`, or enforce support in the callbacks. The function has no separate domain parameter.

The analytical `affine_surface_points` and `intersect_affine_terrain` APIs retain their existing signatures and behavior. The affine point evaluator returns one `(3,)` normal; the generic evaluator returns one normal per point. `segment_layer_paths` remains the shared horizontal-layer path operation.

## Cohesive application adapter

Workspace applications can use [`SplineTerrain`](../../../notebooks/support/terrain.py) to own a sampled `(x,y)` height grid, one cubic interpolant and its closed `horizontal_bounds_m`. Supply `terrain.height` and `terrain.gradient` to the functions above. Both methods derive values from the same interpolant, preserve scalar and batch query shapes, and reject extrapolation. Grid arrays are independent immutable snapshots; outputs are writable. The [spatial notebook](../../../notebooks/spatial_scene.ipynb) includes a separate curved-surface illustration while retaining affine camera geometry and its coverage proof. Full-ray visibility remains an explicit application responsibility.

The self-contained adapters below demonstrate interoperability with independently selected SciPy interpolators. They remain runnable with installed Geometry and SciPy outside the workspace.

## Runnable interpolation example

Run the following Python blocks in order. The example requires NumPy, SciPy and Orsia Geometry, as supplied by the development environment. SciPy remains an application dependency for interpolation; the Geometry terrain functions use NumPy only.

The example samples a quadratic terrain on a metre grid with array axes `(x,y)`. It uses cubic interpolation and derivatives of the same interpolant. SciPy's [`RegularGridInterpolator.__call__`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.__call__.html) supports `nu` derivatives for `slinear`, `cubic` and `quintic`. The adapters restore the scalar query shape because SciPy adds a leading axis for a single coordinate vector.

```python
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from orsia_geometry import surface_points, intersect_terrain

x = np.linspace(-100.0, 100.0, 9)
y = np.linspace(-80.0, 80.0, 7)
xx, yy = np.meshgrid(x, y, indexing="ij")
heights_m = 20.0 + 0.002 * xx**2 + 0.001 * yy**2
interpolator = RegularGridInterpolator(
    (x, y), heights_m, method="cubic", bounds_error=True,
    solver_args={"rtol": 1e-13, "atol": 1e-13},
)


def height(xy_m):
    return interpolator(xy_m).reshape(xy_m.shape[:-1])


def gradient(xy_m):
    return np.stack([
        interpolator(xy_m, nu=order).reshape(xy_m.shape[:-1])
        for order in ((1, 0), (0, 1))
    ], axis=-1)


xy = np.array([[0.0, 0.0], [30.0, -20.0]])
points, normals = surface_points(xy, height=height, gradient=gradient)
np.testing.assert_allclose(points[:, 2], [20.0, 22.2], atol=1e-10)
np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0)
assert normals.shape == points.shape == (2, 3)

# Vertical rays have a single root for this height field.
origins = points + np.array([0.0, 0.0, 50.3])
directions = np.tile([0.0, 0.0, -1.0], (len(points), 1))
hit, intersections = intersect_terrain(
    origins, directions, height=height, gradient=gradient,
    horizontal_bounds_m=(x[0], x[-1], y[0], y[-1]),
    distance_nodes_m=np.linspace(0.0, 70.0, 15),
    distance_tolerance_m=1e-8, height_tolerance_m=1e-9,
    minimum_incidence_cosine=0.1, max_iterations=64,
)
assert hit.all()
np.testing.assert_allclose(intersections, points, atol=1e-9)
```

A [`RectBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.ev.html) adapter uses the same point and derivative contracts. Evaluate the spline only within its declared support; these adapters add an explicit domain check.

```python
spline = RectBivariateSpline(x, y, heights_m, kx=3, ky=3, s=0)


def require_domain(xy_m):
    if (np.any(xy_m[..., 0] < x[0]) or np.any(xy_m[..., 0] > x[-1])
            or np.any(xy_m[..., 1] < y[0]) or np.any(xy_m[..., 1] > y[-1])):
        raise ValueError("outside spline terrain support")


def spline_height(xy_m):
    require_domain(xy_m)
    return spline.ev(xy_m[..., 0], xy_m[..., 1])


def spline_gradient(xy_m):
    require_domain(xy_m)
    return np.stack([
        spline.ev(xy_m[..., 0], xy_m[..., 1], dx=dx, dy=dy)
        for dx, dy in ((1, 0), (0, 1))
    ], axis=-1)


spline_points, spline_normals = surface_points(
    xy, height=spline_height, gradient=spline_gradient,
)
np.testing.assert_allclose(spline_points, points, atol=1e-10)
np.testing.assert_allclose(spline_normals, normals, atol=1e-10)
```

## Bounded intersection policy

`intersect_terrain` takes matching `(...,3)` origins and unit directions. Supply at least two finite, strictly increasing `distance_nodes_m`, starting at zero. All rays share these forward metre distances. The closed rectangle `horizontal_bounds_m=(xmin,xmax,ymin,ymax)` must contain every complete requested search interval, including the origin and final node. Domain violations raise before callback evaluation, even when an earlier part of the ray could intersect terrain. Choose the domain and search length together; the function neither clips intervals nor extrapolates heights.

The residual is ray height minus terrain height. An origin below terrain by more than `height_tolerance_m` returns a miss. Otherwise, the function selects the first sampled residual at or below that tolerance. If the absolute residual is within tolerance, that sample becomes the candidate. Otherwise, the preceding positive residual and selected negative residual define the candidate bracket. Bisection requires continuous height and stops at an exact zero, or when the bracket width and vertical residual meet both supplied tolerances. Exhausting `max_iterations` raises `ValueError` for the call.

The candidate must satisfy `-normal dot direction >= minimum_incidence_cosine`, with a positive threshold at most one. A rejected grazing or exiting candidate ends that ray's search. The function never skips that candidate to accept a later crossing. Supported surface origins hit at distance zero. Outputs are fresh Boolean `(...)` hits and `(...,3)` points on the rays; misses have NaN coordinates. Sampled contacts use a vertical tolerance, so they can lie slightly above or below the exact height field.

**Finite sampling cannot certify the first physical intersection of an arbitrary callable.** Crossings and tangencies can occur between two positive samples. A bracket can contain multiple roots, and bisection can select any of them. A first-intersection claim requires independent evidence that preceding intervals contain no contact and the selected bracket contains one root. The vertical-ray example satisfies this condition because horizontal coordinates remain fixed. Refining the distance grid provides a convergence experiment, without proving the absence of narrower features.

A miss means that this sampling policy found no supported candidate inside the requested interval. It does not establish clear terrain visibility. Scene therefore requires an explicit visibility policy for callable terrain; see the [Scene contract and example](../../orsia-scene/README.md#callable-terrain-and-visibility).

## Validation scope

`test_height_fields.py` checks affine agreement, curved polynomial terrain, SciPy derivative adapters, shapes, ownership, bounded search, multiple crossings, grazing contacts, hidden contacts, domain errors and refinement failures. Its guide test executes both Python blocks. These synthetic checks establish the documented numerical behavior. Terrain-data accuracy, gradient consistency for other interpolants, and visibility certification remain application responsibilities.
