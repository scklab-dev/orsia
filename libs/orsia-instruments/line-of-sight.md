# Earth-frame line-of-sight acquisition

`sample_lines_of_sight` maps a `ReadSchedule` to sensor-frame optical nodes and Earth-frame sight lines at known times. `line_of_sight_expected_electrons` combines caller-supplied sight-line radiance with `photoelectron_response`. `sample_readout` then converts named mean-electron contributions to seeded physical detector readings. The same functions accept focal-plane pixels from an image, slit, or along-track layout. Read intervals may overlap, and each read retains its sample, pixel, setting, and port identity.

## Time, frames, support and ownership

The example convention uses continuous TAI seconds relative to a **caller-named reference epoch** and Earth-fixed, right-handed Cartesian metres. The caller supplies a `platform_state(times_s)` adapter returning `(times,3)` platform reference positions and `(times,3,3)` proper rotations mapping platform-frame column vectors into the Earth frame. It owns orbital interpolation, time-scale conversion, frame realization, datum and validity interval. The fixed mount offset in platform metres identifies the optical origin, and its proper rotation maps instrument-frame column vectors into platform-frame vectors. The optical origin is `position_e + attitude_e_p @ mount_offset_p_m`; the direction is `attitude_e_p @ mount_rotation_p_i @ sensor_direction_i`.

The sensor boresight is +instrument-z toward the scene. Each per-read FOV rectangle `(umin,umax,vmin,vmax)` contains instrument-frame direction cosines in the forward unit disk. A node direction is `(u,v,sqrt(1-u²-v²))`. The Earth-frame ray points from optical origin toward the scene. A sight-line radiance provider receives **that travel direction** and produces unpolarized spectral radiance in `W m^-2 sr^-1 (cm^-1)^-1` on the caller's increasing, fully sampled connected wavenumber grid. Provider adapters perform any conversion from incoming-photon or solver-specific look conventions. The virtual forward image plane uses `(f*u/w,f*v/w)` metres with `w=sqrt(1-u²-v²)`; callers match this coordinate orientation to the selected optical response law.

Temporal and angular Gauss rules integrate `dt du dv / w` with weights in `s sr`. The weights express a coordinate measure; `response(read,wavelengths_m,acquisition)` supplies a separate physical accepted-intensity fraction with axes `(nodes,wavenumbers)`. `photoelectron_response` multiplies this fraction by projected collecting area, throughput, QE, trapezoidal `cm^-1` spectral measure and photon-energy conversion once. `RadianceField` uses path-dependent local u/v conventions. `LineOfSightRadiance` receives Earth-frame 3D origins and directions, with its distinct `RadianceCoverage` contract.

The scene provider's `coverage(rays,grid)` returns a Boolean `(reads,)` attestation for **each complete continuous exposure interval, the complete transformed pixel FOV and the spectral interval**, including endpoints and boundaries. The integrator checks every read's attestation before sampling any radiance. The rays also carry `support_times_s` with interval endpoints and quadrature times and Earth-frame corner rays at those times; providers can inspect them when constructing a conservative support bound. A provider certifies the continuous interval between sampled times using its own orbit interpolation and scene domain policy. The optical acceptance law and FOV rectangle jointly define the intended light support; callers establish that the selected FOV encloses the physically relevant PSF tails and stray light. Spatial resolution, angular/time/spectral quadrature convergence and provider coverage require case-specific evidence.

Input geometry and scene callbacks are borrowed, pure and stable during a call; `LineOfSightSamples` owns writable copies of its numerical ray arrays, and its `Acquisition` owns writable copies of optical node arrays and read intervals. The dataclass shares that acquisition object by identity. Returned means own source vectors, while `sample_readout` advances the explicit caller-owned RNG. Combining source paths through `combine_source_paths` aligns physical identities and exposure intervals before Poisson charge sampling.

## Runnable synthetic example

```python
import numpy as np
from orsia_instruments import (
    DetectorElectronics, ReadSchedule, sample_lines_of_sight,
    line_of_sight_expected_electrons, sample_readout, sphere_intersections,
)
from orsia_radiative import SpectralDomain, SpectralGrid

reads = ReadSchedule(
    ("r0", "r1"), ("p0", "p1"), ("band", "band"), ("direct", "direct"),
    np.array([[0., 1.], [0.5, 1.5]]),
)
def platform_state(times):
    position = np.column_stack((2. + 0.1 * times, np.zeros_like(times), np.zeros_like(times)))
    return position, np.broadcast_to(np.eye(3), (len(times), 3, 3))

mount = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
rays = sample_lines_of_sight(
    reads, pixel_bounds_m=np.array([[-0.1, 0.1, -0.1, 0.1], [0.9, 1.5, -0.1, 0.1]]),
    fov_bounds_uv=np.array([[-0.03, 0.03, -0.01, 0.01], [0.75, 0.8, -0.01, 0.01]]),
    platform_state=platform_state, mount_offset_p_m=np.zeros(3),
    mount_rotation_p_i=mount, focal_length_m=1., time_order=3, angular_order=3,
)
hit, hit_points_e_m = sphere_intersections(rays, radius_m=1.)
assert hit.shape == (2, 27) and hit_points_e_m.shape == (2, 27, 3)
grid = SpectralGrid(
    "wavenumber_cm1", np.array([4900., 5000., 5100.]),
    SpectralDomain("wavenumber_cm1", ((4900., 5100.),)),
)
def coverage(rays, grid):
    # This synthetic scene is uniform throughout this entire FOV and time support.
    bounds = rays.acquisition.exposure_bounds_s
    return (bounds[:, 0] >= 0.) & (bounds[:, 1] <= 1.5) & (
        grid.points[0] >= 4900.) & (grid.points[-1] <= 5100.)
def scene(origins_e_m, directions_e, times_s, wavenumbers_cm1):
    return np.full((*times_s.shape, len(wavenumbers_cm1)), 1e-8)
def response(read, wavelengths_m, acquisition):
    # Replace unity by a physically justified pixel/port/setting optical law.
    return np.ones((acquisition.times_s.shape[1], len(wavelengths_m)))

means = line_of_sight_expected_electrons(
    rays, grid, scene, coverage, response, collecting_area_m2=1e-3,
    throughput=0.7, quantum_efficiency=0.8, source_name="earth-scene",
)
readings = sample_readout(means, DetectorElectronics(), np.random.default_rng(71))
assert means.total.shape == (2,) and readings.dn.shape == (1, 2)
```

`sphere_intersections` returns a hit mask and NaN hit coordinates for missed rays; ray-space radiance and charge remain available for both hit and miss cases. A surface-aware provider chooses how its hit status changes radiance. Sphere intersections use an origin-centred sphere in the supplied Earth frame; the caller chooses its radius and corresponding datum. Ground-grid response projection remains a separate derived calculation; the direct signal uses sight-line radiance.

## Qualification scope

This implementation establishes synthetic software coverage of per-read scheduling, sampled Earth-frame sight lines, uniform radiance, optical photon accounting, analytic sphere status and seeded detector readout. Real orbit and attitude products require a named Earth-frame realization, datum, TAI conversion, temporal interpolation bounds and independently checked scene coverage. A mission instrument requires calibrated mount/pixel/PSF/port/throughput/QE laws, convergence budgets, engineering feasibility and instrument-space agreement. Ground-grid projection and provider-specific surface radiance adapters remain design extensions.
