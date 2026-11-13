# Optical response and integration

This reference specifies response fractions, spectral laws, PSFs and explicit optical reuse. [Acquisition and readout](acquisition-and-detector.md) define the surrounding geometry and detector contracts.

## Explicit optical reuse

`sample_optical_acceptance(acquisition, response, wavelengths_m)` evaluates each optical callback once and returns an immutable `OpticalAcceptance` usable in `expected_electrons` or `photoelectron_response`. Reuse it for different radiance fields on the same acquisition and exact wavelength grid; changing radiance does not change the retained optics. Preparation preserves joint spatial/spectral/node dependence, arbitrary custom responses and distinct physical reads, without assuming identical scene spectra.

```python
import numpy as np
from orsia_instruments import AngularRadianceSamples, ReadSchedule, expected_electrons, rectangular_acquisition, sample_optical_acceptance
from orsia_radiative import SpectralDomain, SpectralGrid

grid = SpectralGrid(
    "wavenumber_cm1",
    np.array([4000.0, 5000.0]),
    SpectralDomain("wavenumber_cm1", ((4000.0, 5000.0),)),
)
scene = AngularRadianceSamples(
    grid,
    np.array([0.0, 1.0]),
    np.array([-0.01, 0.01]),
    np.array([-0.01, 0.01]),
    np.ones((2, 2, 2, 2)),
)
other_scene = AngularRadianceSamples(grid, scene.times_s, scene.u, scene.v, 2 * scene.values)
acquisition = rectangular_acquisition(
    reads=ReadSchedule(("r",), ("p",), ("s",), ("direct",), np.array([[0.0, 1.0]])),
    pixel_bounds_m=np.array([[-1.0, 1.0, -1.0, 1.0]]),
    direction_bounds_uv=(-0.01, 0.01, -0.01, 0.01),
    pointing=lambda t: np.zeros((len(t), 2)),
    focal_length_m=1.0,
    time_order=1,
    angular_order=2,
)


def response(read, wavelengths, current):
    return np.full((current.times_s.shape[1], len(wavelengths)), 0.8)


prepared = sample_optical_acceptance(acquisition, response, 0.01 / scene.grid.points)
counts = expected_electrons(scene, acquisition, prepared, collecting_area_m2=1.0)
other_counts = expected_electrons(other_scene, acquisition, prepared, collecting_area_m2=1.0)
np.testing.assert_allclose(other_counts.total, 2 * counts.total)
```

`OpticalAcceptance` retains `(reads,nodes,wavelengths)` physical fractions, exact distinct wavelengths and the original immutable acquisition. Its buffers own immutable snapshots, independent of callback scratch storage. `photoelectron_response(grid, acceptance, collecting_area_m2=..., throughput=..., quantum_efficiency=...)` accepts only exact subsets or permutations of the prepared spectral support; it performs no interpolation. The coefficients apply spectral quadrature, photon conversion, projected area, throughput, QE and acquisition `dOmega dt` once. Reprepare after an optical or geometric change. Scene radiance and electronic parameters remain independent.

Retained fractions require `8 * reads * nodes * wavelengths` bytes plus preparation temporaries. Frame simulation samples, consumes and releases acceptance by read batch, then performs readout in retained physical order. Applications retain acceptance explicitly when reuse across scenes or QE/throughput alternatives is useful. These ownership and execution contracts do not establish a performance improvement for an arbitrary workload.

## Optical laws and integration

### Radiance integration execution

Scene interpolation and photoelectron response coefficients cover all supplied read/quadrature nodes and wavelengths; charge contraction uses one dense NumPy einsum. Retained interpolated radiance and physical response each scale as `(N,Q,K)`. Optical-response callbacks retain their per-read contract, with one full-grid evaluation per read.

`expected_electrons(field,acquisition,response,collecting_area_m2=...,throughput=...,quantum_efficiency=...)` calls `response(read_index,wavelengths_m,acquisition)` once for each read over the complete spectral grid. The response returns `(Q,K)` accepted intensity fractions in [0,1], including finite-pixel collection and the family-specific selection/modulation, but not the supplied projected area, common throughput, QE, spectral quadrature, photon conversion or time/solid-angle weights. Common throughput and array QE are scalars or broadcastable to `(N,spectral)`. Alternatively, QE is a pure callback `(local_read_indices, wavelengths_m) -> (reads,wavenumbers)`, evaluated once on the complete acquisition and spectral grid, with exact shape and range checks. Inputs are borrowed arrays that callbacks must not mutate, wavelengths need not increase, and the callback must reject unsupported wavelengths. QE is bounded to [0,1] for this detector model; gain-producing detectors need another model. Throughput/loss factors must each enter once.

The integral uses trapezoidal weights in cm^-1 and photon energy `h*c*100*wavenumber_cm1`. Per-cm^-1 radiance is never relabelled as per-wavelength radiance. Exposure and pixel area are not multiplied again: dOmega dt is already in acquisition weights, and pixel integration already supplies a collected fraction. The spectral domain is finite supplied support, not an assertion of zero radiance or perfect blocking outside it. Every transmitted order and background within the actual case must be included before photon noise.

`pixel_acceptance` selects one declared `GaussianPSF`, prepared `TabulatedPSF` or density callback. A Gaussian with explicitly uniform scalar/array transmission uses the analytic backend; spatial transmission callbacks and other densities share quadrature. `gaussian_pixel_fraction` analytically integrates a normalized diagonal 2-D Gaussian over a rectangle, retaining lost tails. `dispersive_response` applies linear/quadratic wavelength dispersion, linear smile and keystone to supplied image centroids before that integration. Width changes redistribute photons at fixed throughput; physical slit changes require a separate declared acceptance coupling. Tilted/correlated profiles can be supplied with `spatial=my_density` or `spatial=TabulatedPSF(...)` through the common pixel acceptance contract; coherent propagation and general grating designs remain excluded.

The [complete slit-spectrometer path](slit-spectrometer.md) supplies explicit finite entrance acceptance around `dispersive_imager`. Entrance-plane rays receive polynomial dispersion before analytic Gaussian integration over independent detector rectangles. The complete path retains photon-weighted pixel-specific wavelength responses for supplied calibration and bounded cube assembly.

`pixel_acceptance(..., spatial=GaussianPSF(sigma_m), transmission=callback)` integrates the declared Gaussian relay against `transmission(x_m,y_m,wavelengths_m)` by finite-pixel quadrature. The callback can depend jointly on per-node angles/times, local position, setting and port; no spectral-spatial separability is imposed. Refine quadrature independently of the scene/spectral sampling; narrow images and local fringes can be missed. Accepted fractions above one trigger an under-resolution error, but fractions below one do not establish convergence.

`gaussian_filter` is peak-normalized transmission with wavelength standard deviation `width_m` in metres; its integrated spectral area varies with width. `lorentzian_filter` and `super_gaussian_filter` use `fwhm_m` (full width at half maximum); `rectangular_filter` includes both wavelength bounds. `longpass_filter` and `shortpass_filter` use a logistic edge with `transition_m` measured from 10% to 90% of the supplied stop/pass levels; `notch_filter` uses `fwhm_m` for the full width at half rejection depth. These laws return dimensionless intensity transmission; the [filter implementations](../src/orsia_instruments/optics/filters.py) specify their parameter domains. `lvf_centre` uses a local polynomial coordinate/temperature law and an effective-index angular shift for incidence from air/vacuum. Callers supply coating polarization, leakage, pointing-to-passband shifts and the map from detector/ray coordinates to the filter plane when needed.

### Interchangeable factory example

Each factory below binds a distinct physical selection to the same idealized focal-plane geometry. The returned responses still require an acquisition and wavelength grid for evaluation.

```python
from functools import partial

import numpy as np
from orsia_instruments import GaussianPSF, cavity_imager, dispersive_imager, filter_imager, gaussian_filter, interferometric_imager, lvf_imager, spatial_filter_imager

geometry = dict(focal_length_m=0.02, collecting_area_m2=1e-8, spatial=GaussianPSF(3e-5))
optics_by_name = {
    "filter": filter_imager(
        **geometry, filter=partial(gaussian_filter, centre_m=2e-6, width_m=2e-8)
    ),
    "spatial": spatial_filter_imager(**geometry, filter=lambda x, y, w: np.full_like(w, 0.5)),
    "lvf": lvf_imager(**geometry, reference_m=2e-6, gradient=0, sigma_m=2e-8),
    "interferometric": interferometric_imager(**geometry, opd_m=0, port="+"),
    "cavity": cavity_imager(**geometry, spacing_m=1e-5, order_transmission=0.7),
    "dispersive": dispersive_imager(**geometry, reference_m=2e-6, dispersion=1, transmission=0.7),
}
assert len(optics_by_name) == 6
```

### PSF callback contract

A density callback receives detector offsets from mapped centroid `(nodes,wavelengths,2)` in metres, wavelengths `(wavelengths,)`, incident field positions `(nodes,2)` before dispersion and a global exposure index. It returns finite nonnegative `(nodes,wavelengths)` density in m^-2. The caller establishes whole-plane normalization and convergence by refinement. `GaussianPSF(sigma_m)` declares a normalized diagonal Gaussian with immutable positive scalar or `(x,y)` widths. `TabulatedPSF(x_offsets_m, y_offsets_m, density_per_m2)` prepares immutable axes, values, exact bilinear mass and zero-extrapolating interpolation once. Its finite-support mass must be at most one; truncated mass is retained without implicit normalization. Detector truncation also retains lost light.

```python
from functools import partial

import numpy as np
from orsia_instruments import GaussianPSF, TabulatedPSF, filter_imager

geometry = dict(focal_length_m=0.02, collecting_area_m2=1e-8)
spectral_filter = lambda w: np.full_like(w, 0.7)
gaussian_optics = filter_imager(
    **geometry,
    spatial=GaussianPSF((3e-5, 2e-5)),
    filter=spectral_filter,
)
half_width = 4e-5
tabulated_optics = filter_imager(
    **geometry,
    spatial=TabulatedPSF(
        x_offsets_m=[-half_width, half_width],
        y_offsets_m=[-half_width, half_width],
        density_per_m2=np.full((2, 2), 1 / (2 * half_width) ** 2),
    ),
    filter=spectral_filter,
)
```

`two_beam_ports` returns both nonnegative physical port intensities with their sum equal to supplied transmission. OPD is optical path difference in metres; convert mirror travel according to the supplied optical geometry. Scanning and static encoding are supplied maps. Signed port differences occur only after separate physical readout; OPD samples are not reconstructed spectral channels.

`cavity_transmission` uses the local equal-mirror Airy law, reflectivity in [0,1), explicit order-sorting transmission, spacing, refractive index, internal angle and coating phase. The coefficient `4R/(1-R)^2` is not finesse. Reducing peak independently is a phenomenological loss test, not an unequal-mirror solution. Wedges use a local-cavity approximation without lateral coherence or beam walk-off. No general Fizeau realism is claimed.
