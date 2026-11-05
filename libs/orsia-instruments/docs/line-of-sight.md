# Observe a surface from a moving platform

Use GCRS lines of sight when radiance depends on where a moving pixel looks. `sample_lines_of_sight` transforms dated sensor directions into GCRS rays. A scene provider supplies spectral radiance on those rays; `line_of_sight_expected_electrons` integrates it into charge, and `sample_readout` optionally samples noisy DN.

The [Geometry API](../../orsia-geometry/docs/geometry.md) owns `compute_lines_of_sight` and separate analytic intersections. Instruments supplies read scheduling and temporal/angular quadrature and uses the operation's default GCRS output for scene integration. For direct terrestrial rays, Geometry's [LEO ground-trace example](../../orsia-geometry/README.md#construct-a-leo-trajectory-and-ground-trace) requests `output_frame="ITRF"` inside `compute_lines_of_sight`, then intersects the returned arrays separately.

## Example: one pixel crossing a brighter surface

The example uses an origin-centred sphere of radius 1 m and a platform at `(2, 0.2*t, 0)` metres with fixed attitude. One pixel looks approximately along GCRS -x. Surface radiance increases with the intersection's y coordinate, so a later exposure sees a brighter patch and produces more charge.

These dimensions and motion define a bounded synthetic scene. The metre-scale sphere illustrates geometry; it represents no mission orbit or Earth model. Time is elapsed SI seconds since `2000-01-01T00:00:00+00:00`, a timezone-aware UTC epoch. The synthetic trajectory and sphere are defined directly in GCRS, with the sphere at the frame origin. Valid time is `[0,1.5]` s; the FOV is `[-0.01,0.01]` in each sensor direction cosine; spectral support is `[4900,5100]` cm^-1.

Run the following blocks in order with the distribution and dependencies installed, or in the [development environment](../../../README.md#development-environment).

### 1. Sample one pixel at two exposure times

The mount maps instrument +z to GCRS -x. Each call makes a separate 0.5 s exposure of the same physical pixel. The platform moves continuously during each exposure; the sensor-frame pixel geometry and integration weights remain the same between exposures.

```python
from datetime import UTC, datetime

import numpy as np
from orsia_geometry import sphere_intersections
from orsia_instruments import (
    DetectorElectronics,
    ReadSchedule,
    gaussian_pixel_fraction,
    line_of_sight_expected_electrons,
    sample_lines_of_sight,
    sample_readout,
)
from orsia_radiative import SpectralDomain, SpectralGrid

EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def trajectory(times_s):
    position = np.column_stack(
        (
            np.full_like(times_s, 2.0),
            0.2 * times_s,
            np.zeros_like(times_s),
        )
    )
    velocity = np.broadcast_to([0.0, 0.2, 0.0], position.shape).copy()
    return position, velocity


def platform_attitude(times_s):
    return np.broadcast_to(np.eye(3), (len(times_s), 3, 3)).copy()


def instrument_attitude(times_s):
    mount = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    return np.broadcast_to(mount, (len(times_s), 3, 3)).copy()


def pixel_rays(start_s):
    reads = ReadSchedule(
        ("read-0",),
        ("pixel-0",),
        ("band",),
        ("direct",),
        np.array([[start_s, start_s + 0.5]]),
    )
    return sample_lines_of_sight(
        reads,
        pixel_bounds_m=np.array([[-0.02, 0.02, -0.02, 0.02]]),
        fov_bounds_uv=np.array([[-0.01, 0.01, -0.01, 0.01]]),
        epoch=EPOCH,
        validity_s=(0.0, 1.5),
        trajectory=trajectory,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        mount_offset_p_m=np.zeros(3),
        focal_length_m=1.0,
        time_order=3,
        angular_order=3,
    )


early_rays = pixel_rays(0.0)
late_rays = pixel_rays(1.0)
grid = SpectralGrid(
    "wavenumber_cm1",
    np.array([4900.0, 5000.0, 5100.0]),
    SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),)),
)
```

### 2. Use surface intersections to set radiance

`sphere_intersections` supplies hit status and coordinates. The toy surface emits `1e-8 * (1 + y / radius)` W m^-2 sr^-1 (cm^-1)^-1, independent of wavelength and viewing angle. Missed rays receive a faint constant space radiance of `1e-10` in the same units. Intersections therefore determine both the selected scene component and the surface brightness.

This small adapter tabulates radiance for one acquisition. Its callback verifies the requested rays and spectral grid before returning the table. Rebuild the adapter whenever geometry changes. A general provider can instead evaluate intersections or other radiance physics directly from its callback arguments.

The coverage attestation is restricted to rays constructed by `pixel_rays` with the fixed state and mount above. Over the full valid time and FOV, each ray reaches x=0 with `abs(y) < 0.321 m` and `abs(z) < 0.021 m`. The transverse distance is therefore below 0.33 m, so every ray intersects the sphere. This bound includes exposure endpoints and FOV boundaries. The table values use sampled intersections; the coverage statement uses the continuous analytic motion.

```python
def surface_charge(rays):
    hit, points_gcrs_m = sphere_intersections(
        rays.origins_gcrs_m, rays.directions_gcrs, radius_m=1.0
    )
    surface = 1e-8 * (1.0 + np.nan_to_num(points_gcrs_m[..., 1]) / 1.0)
    values = np.broadcast_to(
        np.where(hit, surface, 1e-10)[..., None],
        (*hit.shape, len(grid.points)),
    ).copy()

    epoch = rays.epoch  # The scene closure binds absolute-time interpretation.

    def scene(origins_gcrs_m, directions_gcrs, times_s, wavenumbers_cm1):
        assert epoch == EPOCH
        np.testing.assert_array_equal(origins_gcrs_m, rays.origins_gcrs_m)
        np.testing.assert_array_equal(directions_gcrs, rays.directions_gcrs)
        np.testing.assert_array_equal(times_s, rays.acquisition.times_s)
        np.testing.assert_array_equal(wavenumbers_cm1, grid.points)
        return values.copy()

    def coverage(current, spectral_grid):
        assert current is rays  # This provider is bound to this acquisition.
        bounds = current.acquisition.exposure_bounds_s
        fov = current.acquisition.direction_bounds_uv
        return (
            (bounds[:, 0] >= 0.0)
            & (bounds[:, 1] <= 1.5)
            & np.all(fov[:, [0, 2]] >= -0.01, axis=1)
            & np.all(fov[:, [1, 3]] <= 0.01, axis=1)
            & (spectral_grid.points[0] >= 4900.0)
            & (spectral_grid.points[-1] <= 5100.0)
        )

    def response(read, wavelengths_m, acquisition):
        accepted = gaussian_pixel_fraction(
            acquisition.pixel_bounds_m[read],
            acquisition.image_xy_m[read],
            0.002,
        )
        return np.broadcast_to(accepted[:, None], (len(accepted), len(wavelengths_m)))

    charge = line_of_sight_expected_electrons(
        rays,
        grid,
        scene,
        coverage,
        response,
        collecting_area_m2=1e-6,
        throughput=0.7,
        quantum_efficiency=0.8,
        source_name="surface",
    )
    return charge, hit, points_gcrs_m, values


early, early_hit, early_points, early_radiance = surface_charge(early_rays)
late, late_hit, late_points, late_radiance = surface_charge(late_rays)
assert early_hit.all() and late_hit.all()
assert np.all(late_points[..., 1] > early_points[..., 1])
assert np.all(late_radiance > early_radiance)
assert late.total[0] > early.total[0] > 0.0
np.testing.assert_array_equal(
    early_rays.acquisition.weights_sr_s,
    late_rays.acquisition.weights_sr_s,
)
```

The assertions follow the causal chain: platform motion changes the surface intersections, intersection coordinates change radiance, and integrated charge increases. The response integrates a Gaussian PSF over the finite active pixel. The accepted angular patch is explicitly truncated to the stated FOV; light outside that patch is excluded from this toy calculation.

### 3. Sample detector output once charge is known

```python
readings = sample_readout(
    late,
    DetectorElectronics(dark_electrons_per_s=0.1, read_noise_electrons=0.5),
    np.random.default_rng(71),
)
assert late.total.shape == (1,) and readings.dn.shape == (1, 1)
assert not readings.full_well_saturated.any() and not readings.adc_clipped.any()
print(late.total[0] > early.total[0], readings.dn.shape)
```

Expected output: `True (1, 1)`. The comparison concerns pre-dark mean charge; a noisy realisation need not preserve the ordering of two means. The three-point spectral grid and quadrature orders illustrate the software flow. They establish no convergence or instrument-performance claim.

## Time, frames, support and ownership

`ReadSchedule` retains ordered sample, pixel, setting and port identities with positive exposure intervals in seconds. Intervals may overlap. The same functions accept focal-plane pixels from image, slit or along-track layouts.

Supply `epoch`, a timezone-aware UTC instant, and `validity_s`, the finite closed interval common to trajectory and both attitude laws. Numerical times are elapsed SI seconds from that instant, including across leap seconds. The `trajectory(times_s)` callback returns GCRS `(times,3)` positions in metres and velocities in metres per second. Independent `platform_attitude(times_s)` and `instrument_attitude(times_s)` callbacks return proper `(times,3,3)` platform-to-GCRS and instrument-to-platform rotations. The caller owns interpolation, frame conversion and continuous validity. Every complete exposure must lie within `validity_s`; unsupported endpoints fail before any callback runs. The optical origin is `position_gcrs + platform_attitude @ mount_offset_p_m`; the direction is `platform_attitude @ instrument_attitude @ sensor_direction_i`. A moving instrument attitude changes pointing independently of the fixed platform-frame optical offset.

`LineOfSightSamples.epoch` identifies the numerical times' reference instant. Scene callbacks receive origins, directions, elapsed seconds and spectral points; they bind their absolute-time conversion to this epoch. Coverage callbacks receive the complete ray record, including its epoch. For terrestrial inputs, explicitly transform Earth-fixed positions, velocities, platform attitude and scene geometry at the requested dates. Geometry's optional `transform_states` and `frame_rotations` support this conversion. Supplying Earth-fixed arrays under GCRS field names gives incorrect geometry.

The sensor boresight is +instrument-z toward the scene. Each per-read FOV rectangle `(umin,umax,vmin,vmax)` contains instrument-frame direction cosines in the forward unit disk. A node direction is `(u,v,sqrt(1-u²-v²))`. The GCRS ray points from optical origin toward the scene. A sight-line radiance provider receives **that travel direction** and produces unpolarized spectral radiance in `W m^-2 sr^-1 (cm^-1)^-1` on the caller's increasing, fully sampled connected wavenumber grid. Provider adapters perform any conversion from incoming-photon or solver-specific look conventions. The virtual forward image plane uses `(f*u/w,f*v/w)` metres with `w=sqrt(1-u²-v²)`; callers match this coordinate orientation to the selected optical response law.

Temporal and angular Gauss rules integrate `dt du dv / w` with weights in `s sr`. The weights express a coordinate measure; `response(read,wavelengths_m,acquisition)` supplies a separate physical accepted-intensity fraction with axes `(nodes,wavenumbers)`. `photoelectron_response` multiplies this fraction by projected collecting area, throughput, QE, trapezoidal `cm^-1` spectral measure and photon-energy conversion once. `RadianceField` uses path-dependent local u/v conventions. `LineOfSightRadiance` receives GCRS 3D origins and directions, with its distinct `RadianceCoverage` contract. Its output has axes `(reads,nodes,wavenumbers)`; origins and directions have `(reads,nodes,3)` axes, times have `(reads,nodes)` and the grid is a 1-D increasing vector in cm^-1.

With `entrance_bounds_m=(reads,4)`, `sample_lines_of_sight` samples focal-plane entrance rectangles in metres instead of a direction-cosine rectangle. The corresponding solid-angle Jacobian is `f/(f**2+x**2+y**2)**1.5`. Detector `pixel_bounds_m` remain separate output apertures. The supplied FOV must enclose the complete entrance rectangle, including direction-cosine extrema at axis crossings; an under-enclosing FOV raises `ValueError`. The [slit-spectrometer path](slit-spectrometer.md) uses this mode for hard finite slits, with conservative FOV-corner support and focal-plane Gauss quadrature using the exact solid-angle Jacobian.

The scene provider's `coverage(rays,grid)` returns a Boolean `(reads,)` attestation for **each complete continuous exposure interval, the complete transformed pixel FOV and the spectral interval**, including endpoints and boundaries. The integrator checks every read's attestation before sampling any radiance. The rays also carry `support_times_s` with interval endpoints and quadrature times and GCRS corner rays at those times; providers can inspect them when constructing a conservative support bound. A provider certifies the continuous interval between sampled times using its own orbit interpolation and scene domain policy. The optical acceptance law and FOV rectangle jointly define the intended light support; callers establish that the selected FOV encloses the physically relevant PSF tails and stray light. Spatial resolution, angular/time/spectral quadrature convergence and provider coverage require case-specific evidence.

Input geometry and scene callbacks are borrowed, pure and stable during a call; `LineOfSightSamples` owns writable copies of its numerical ray arrays, and its `Acquisition` owns immutable optical node snapshots and references an immutable `ReadSchedule`. The dataclass shares that acquisition object by identity. Integration retains that geometry-free schedule with owned source vectors. `ExpectedElectrons.reads` and `DetectorReadings.reads` retain the physical identities and timing, while `sample_readout` advances the explicit caller-owned RNG. Combining source paths through `combine_source_paths` aligns physical identities and exposure intervals before Poisson charge sampling.

## Surface and coverage failures

`sphere_intersections` uses an origin-centred sphere in the supplied GCRS. It returns a `(reads,nodes)` hit mask and `(reads,nodes,3)` coordinates in metres, with NaNs for misses. Choose the radius and corresponding datum explicitly. Ray-space radiance remains available for missed rays; the provider decides their radiance. Ground-grid response projection remains a separate derived calculation.

A false coverage value for any read causes `ValueError` before radiance is sampled. Extending an exposure beyond `[0,1.5]` s therefore requires a new support argument and provider. Checking only quadrature-node hits would leave continuous coverage unestablished. The example's analytic bound depends on its fixed linear motion and mount; changing either requires a new bound.

## Evidence and qualification limits

The [executable guide tests](../../../tests/test_library_guides.py) run this example with network access blocked, checking that changed intersections alter radiance and charge. The [line-of-sight tests](../tests/test_line_of_sight.py) check scheduled geometry, sphere hits and misses, rejected support, photon accounting and seeded detector sampling on synthetic inputs.

Real orbit and attitude products require a named source frame, explicit UTC-epoch/elapsed-time conversion, surface datum, temporal interpolation bounds and independently checked scene coverage. An instrument case requires calibrated mount/pixel/PSF/port/throughput/QE laws, convergence budgets, engineering feasibility and instrument-space agreement. The [fixed-filter camera](filter-camera.md) and [slit spectrometer](slit-spectrometer.md) provide instrument-selected surface grids for caller-supplied projection and radiance adapters. Reusable scene-map construction remains outside those paths.
