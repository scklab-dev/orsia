# Build a radiance cube from a scanning slit spectrometer

A dispersive slit spectrometer admits a finite strip of the surface through an entrance slit. Optics maps each admitted position and wavelength onto the detector. Detector columns usually distinguish wavelength, while detector rows distinguish position along the slit. Platform motion or an independent pointing law scans the strip across the surface.

This guide follows caller-supplied radiance through instrument-selected surface sampling, native detector frames, supplied calibration and a registered cube. Each detector pixel retains its own achieved spectral band. The cube has pixel-specific band planes, including separate bands for spatial detector rows. Common-band spectral rectification is outside this method.

## Supported physical and inverse model

`dispersive_spectrometer` constructs an ideal hard rectangular entrance slit in a forward focal plane. The entrance focal length maps a direction to undispersed coordinates `(x,y)`. A unit-magnification polynomial map sends those coordinates to detector centroids:

```text
d = wavelength_m - reference_m
detector_x = x + dispersion*d + quadratic_per_m*d**2 + smile*y
detector_y = y + keystone*d
```

Positions and wavelengths are metres. `dispersion`, `smile` and `keystone` are dimensionless; `quadratic_per_m` is in m^-1. The Gaussian point-spread function (PSF) acts after the slit. Orsia integrates that PSF analytically over each finite detector pixel. The full Gaussian tails remain included, and light lost outside active pixels remains lost. Pre-slit blur, anamorphic magnification, multiple diffraction orders and stray light are excluded.

Each native pixel receives the full entrance-slit quadrature. Dispersed detector coordinates never select ground rays. This distinction preserves finite slit width and its spatial/spectral coupling. The entrance-plane solid-angle Jacobian is `f/(f**2+x**2+y**2)**1.5`; the pupil-projection cosine multiplies that solid angle once. Collecting area, throughput, transmission, quantum efficiency (QE), exposure duration, spectral quadrature and photon conversion each enter once.

The supported inverse applies supplied linear detector correction, normalizes by supplied sensitivity, and accumulates compatible achieved-band means at separately estimated surface coordinates. It performs registration and smoothing. It provides neither spatial deconvolution nor spectral unmixing. Finite slit width, smile and keystone can make the joint spatial/spectral response nonseparable even with a constant PSF.

## Walkthrough: an analytic moving surface

Run the following Python blocks in order in one session, with Orsia installed or in the [development environment](../../../README.md#development-environment). The example uses analytic geometry without loading SatKit and requires no files, network access or reusable scene builder. All dimensions and radiance values are illustrative.

### 1. Construct the physical slit and detector

The entrance slit is 10 micrometres wide and 40 micrometres long. The detector has two spatial rows and five spectral columns of 20-micrometre pixels. A 4-micrometre Gaussian PSF, nonzero smile and nonzero keystone produce pixel-specific achieved responses.

The example declares a finite spectral model from 4500 to 5500 cm^-1. The supplied order-sorting transmission is zero outside that closed interval. This model assumes the passband is the only admitted order and excludes out-of-band leakage. Radiance and calibration must cover the complete admitted interval. The grid endpoints alone cannot establish physical spectral completeness; the coverage callback certifies that model assumption.

```python
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
from orsia_instruments import (
    GaussianPSF,
    calibrate_frames,
    dispersive_spectrometer,
    frame_exposure_bounds,
    ground_integration_bounds,
    reconstruct_radiance_cube,
    regular_detector,
    simulate_frames,
    spectrometer_acquisition_grid,
    frame_response,
)
from orsia_radiative import SpectralBand, SpectralDomain, wavenumber_grid


def order_sorting(wavelengths_m):
    wavenumbers_cm1 = 0.01 / wavelengths_m
    return np.where(
        (wavenumbers_cm1 >= 4500.0 - 1e-10) & (wavenumbers_cm1 <= 5500.0 + 1e-10),
        0.8,
        0.0,
    )


detector = regular_detector(
    (2, 5),
    pitch_m=20e-6,
    active_size_m=20e-6,
    centre_xy_m=(0, 0),
    quantum_efficiency=0.7,
    dark_electrons_per_s=2.0,
    read_noise_electrons=0.1,
    full_well_electrons=1e6,
    electrons_per_dn=0.01,
    bias_dn=2000.0,
    nonlinearity=0,
    adc_bits=32,
)
spectrometer = dispersive_spectrometer(
    detector,
    slit_bounds_m=np.array([-5e-6, 5e-6, -20e-6, 20e-6]),
    focal_length_m=0.3,
    collecting_area_m2=0.004,
    spatial=GaussianPSF(4e-6),
    reference_m=2e-6,
    dispersion=200.0,
    smile=0.4,
    keystone=25.0,
    transmission=order_sorting,
    throughput=0.7,
)
spectral = wavenumber_grid(
    np.linspace(4500.0, 5500.0, 65),
    domain=SpectralDomain("wavenumber_cm1", ((4500.0, 5500.0),)),
)
```

### 2. Project the slit and its continuous exposure sweep

The optical origin moves at 0.2 m/s along GCRS x, 300 m above an analytic plane at GCRS z=0. The constant platform rotation points instrument +z towards the plane. Instrument attitude is independently supplied as identity. The surface chart uses the plane's GCRS x/y coordinates in metres.

The trajectory is a local analytic fixture, rather than an Earth orbit. A surface callback can instead transform GCRS rays and intersect a terrestrial surface through Geometry. The caller supplies the fixed surface chart and certifies continuous visibility. Reuse that chart for truth and estimated geometry.

```python
from orsia_instruments import raster_read_schedule

epoch = datetime(2023, 5, 16, tzinfo=UTC)


def trajectory(times_s):
    positions = np.column_stack(
        (
            0.2 * times_s,
            np.zeros_like(times_s),
            np.full_like(times_s, 300.0),
        )
    )
    return positions, np.broadcast_to([0.2, 0.0, 0.0], positions.shape)


def platform_attitude(times_s):
    return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times_s), 3, 3))


def instrument_attitude(times_s):
    return np.broadcast_to(np.eye(3), (len(times_s), 3, 3))


def surface(origins_gcrs_m, directions_gcrs, times_s):
    distance_m = -origins_gcrs_m[..., 2] / directions_gcrs[..., 2]
    points = origins_gcrs_m + distance_m[..., None] * directions_gcrs
    return points[..., :2], distance_m > 0


exposures = frame_exposure_bounds(
    start_s=0.0,
    frame_count=3,
    frame_rate_hz=20.0,
    exposure_s=0.01,
)


def acquisition_grid(trajectory_law):
    return spectrometer_acquisition_grid(
        spectrometer,
        raster_read_schedule(
            exposures, (spectrometer).pixel_bounds_m.shape[:2], port=(spectrometer).optics.port
        ),
        epoch=epoch,
        validity_s=(0.0, 1.0),
        trajectory=trajectory_law,
        platform_attitude=platform_attitude,
        instrument_attitude=instrument_attitude,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        time_order=2,
        slit_order=20,
    )


truth_grid = acquisition_grid(trajectory)
assert truth_grid.image_shape == (3, 2, 5)
assert truth_grid.rays.acquisition.sample_ids[-1] == "s2-p9"
assert np.array_equal(
    truth_grid.rays.acquisition.image_xy_m[0],
    truth_grid.rays.acquisition.image_xy_m[4],
)
print("Surface support (metres):", ground_integration_bounds(truth_grid))
```

For this rotation, the exact surface coordinates are `X=0.2*time_s-1000*x` and `Y=1000*y`. The admitted rays remain near nadir throughout all exposures. The continuous projected speed is exactly 0.2 m/s. A directional infinity-norm Lipschitz bound of 301 m encloses the small local direction domain. The same bound also covers the conservative direction-rectangle corner rays used for support checks.

`SurfaceAcquisitionGrid.xy_m` has shape `(reads,nodes,2)`. The grid retains complete exposure intervals, elapsed seconds from a timezone-aware UTC epoch, GCRS rays, native `(frame,row,column)` indices and conservative per-read surface bounds. `slit_order` refines both entrance coordinates; `time_order` refines exposure motion. Changed orders require new radiance evaluation. The full hard slit bounds scene support for every detector pixel, wavelength and post-slit PSF tail. No PSF-tail scene truncation is needed in this model.

### 3. Evaluate radiance on the instrument-selected coordinates

The analytic source is spatially uniform and emits `wavenumber_cm1*1e-8` W m^-2 sr^-1 per (cm^-1). It is static and independent of viewing direction. The coverage callback certifies the entire continuous surface/time/direction support and the admitted spectral interval before any radiance evaluation.

The attached detector supplies QE and electronics for frame simulation. Its regular factory requires explicit geometry and all eight photoelectric/electronic fields, including for expectation-only calculations. The values in step 1 are illustrative assumptions.

```python
def radiance(xy_m, directions_gcrs, times_s, wavenumbers_cm1):
    return np.broadcast_to(
        wavenumbers_cm1 * 1e-8,
        (*times_s.shape, len(wavenumbers_cm1)),
    )


def coverage(grid, spectral_grid):
    # The analytic source exists on every admitted ray and throughout the passband.
    return np.ones(len(grid.read_bands), dtype=bool)


measurements = simulate_frames(
    spectrometer,
    truth_grid,
    spectral,
    radiance,
    coverage,
    rng=np.random.default_rng(42),
    realisations=2,
)
assert measurements.images.photoelectrons.shape == (3, 2, 5)
assert measurements.images.dn.shape == (2, 3, 2, 5)
```

Complete C-order rasters use `(frame,row,column)` image axes; partial or permuted selections use `(reads,)`. Digital numbers (DN) add a leading realization axis. Charge integration preserves the full spatial/spectral coupling. Detector sampling runs once after complete charge assembly in raster order, independently of integration batch size. Omitting `rng` returns expected electrons with `dn=None`.

### 4. Apply supplied radiometric and wavelength calibration

For this ideal calibration, supply a separately calculated `frame_response`. The response includes photon conversion and spectral quadrature; normalizing its coefficients defines an achieved photon-weighted band. A scalar sensitivity recovers that band mean for spatially uniform spectra. It cannot produce an arbitrary requested energy-response mean or a monochromatic radiance.

An independently characterized instrument can supply different detector QE, wavelength-map coefficients or throughput to `frame_response`. `calibrate_frames` receives the independently estimated detector explicitly. The calibration model must preserve the selected entrance slit, detector rectangles and quadrature. A supplied `NativeBandResponse` can also carry independently measured coefficients on compatible spectral nodes. These processing inputs can change while the simulated measurements stay fixed. Calibration estimation is outside this API.

```python
# Independent calibration records use identical estimates for this closure example.
estimated_detector = replace(detector)
calibration_spectrometer = replace(spectrometer, detector=estimated_detector)
supplied_response = frame_response(
    calibration_spectrometer,
    truth_grid,
    spectral,
)
requested_bands = []
for name in dict.fromkeys(supplied_response.read_bands):
    read = supplied_response.read_bands.index(name)
    coefficients = supplied_response.spectral_coefficients[read]
    requested_bands.append(
        SpectralBand(
            name,
            supplied_response.grid,
            coefficients / coefficients.sum(),
        )
    )

calibrated = calibrate_frames(
    measurements.images,
    supplied_response,
    detector=estimated_detector,
    use_expectation=True,
)
assert calibrated.values.shape == (1, 30)
```

`use_expectation=True` explicitly selects deterministic photoelectrons for this closure check. Omit that option to correct the sampled DN using supplied linear gain, bias and dark maps. Saturated or ADC-clipped DN raise `ValueError`; `mask_flagged=True` instead excludes flagged reads across all realizations and retains invalid NaN values. Nonlinear detector correction requires another method and is rejected. Zero sensitivity gives invalid NaN values.

### 5. Assemble the scan on an independent output grid

Estimated geometry is a separate input. The example recomputes that grid from the same known trajectory. A perturbed trajectory or attitude can change estimated surface coordinates while the native read identities and entrance sensor quadrature remain fixed.

```python
estimated_grid = acquisition_grid(trajectory)
cube = reconstruct_radiance_cube(
    calibrated,
    estimated_grid,
    x_edges_m=np.array([-0.04, -0.03, 0.0, 0.01, 0.02, 0.03, 0.04]),
    y_edges_m=np.array([-0.03, 0.0, 0.03]),
    bands=requested_bands,
)
expected = np.array([np.dot(band.weights, spectral.points * 1e-8) for band in requested_bands])
assert cube.values.shape == (2, 6, 10)
assert cube.valid.any(axis=(0, 1)).all()
assert (~cube.valid).any()
assert np.isnan(cube.values[~cube.valid]).all()
expected_cube = np.broadcast_to(expected, cube.values.shape)
np.testing.assert_allclose(
    cube.values[cube.valid],
    expected_cube[cube.valid],
    rtol=2e-14,
)
print("Cube shape:", cube.values.shape)
```

Expected final output: `Cube shape: (2, 6, 10)`. The cube uses `(y,x,band)` axes and radiance units W m^-2 sr^-1 per (cm^-1). Each `r{row}-c{column}` band identifies one detector pixel's achieved spectral response across the scan. Smile and keystone can produce different responses for the same nominal spectral column in different detector rows. The method keeps those planes separate. Requested common bands that disagree with a selected channel's normalized coefficients raise `ValueError`.

## Cube interpretation and limits

- `NativeBandResponse` retains the achieved photon-weighted spectral coefficients and the normalized flat-spectrum spatial marginal. It retains quadrature times and entrance sensor directions for identity checks. The response does not store a joint kernel for spectral inversion.
- The forward integration retains the full joint spatial/spectral response. Reconstruction distributes each calibrated scalar through its flat-spectrum spatial marginal. Exact band interpretation requires a spatially uniform spectrum within each read, or a separable response. Heterogeneous spectra can have wavelength-dependent footprints and biased apparent spatial/spectral structure. The spatially uniform walkthrough avoids this approximation error; the moving-gradient regression exercises the exact flat-spectrum marginal case.
- Reconstruction assumes a static common-view source over the combined exposures. A time-varying or view-dependent source can be simulated, but this reconstruction does not estimate a time- or direction-resolved cube.
- The native-band cube can have sparse spatial coverage. Each spatial detector row has a separate set of bands. Motion across the slit does not guarantee that every ground cell receives every row-specific band. Missing or quality-masked channels remain invalid NaN values. Gaussian tails can produce tiny positive coverage far from a row's main footprint; use `minimum_coverage` for a caller-selected accumulated-weight cutoff.
- The method requires requested spectral nodes and normalized response weights to agree with calibrated information within `response_tolerance`. It rejects unsupported spectral rectification and super-resolution. A wholly missing requested identity yields an invalid band plane with the requested response retained.
- Full frames share one exposure interval per frame. Rolling shutters, detector binning, calibration estimation, uncertain geometry estimation, common-band spectral rectification and general inversions remain outside this path.
- The detector owns independent read-only geometry and property maps. The spectrometer retains that detector and delegates `pixel_bounds_m` to its active bounds. Slit and response arrays remain writable. Acquisition nodes, read timing and native indices are immutable snapshots, as are canonical spectral coordinates. Keep retained arrays stable while in use. Transmission, radiance, trajectory, attitude, surface and coverage callbacks are borrowed, pure and stable; callback exceptions propagate. The caller owns RNG state.

## Checks and sources

The maintained implementation is in `optics/spectrometer.py`, `imaging/geolocation.py`, `los.py` and `imaging/radiometry.py`. Shared supplied-calibration and cube contracts are in `imaging/calibration.py`, `imaging/response/band.py`, `imaging/reconstruction.py` and Radiative's [bands and cubes](../../orsia-radiative/docs/bands-and-cubes.md). The optical centroid and finite-pixel kernels remain in `optics/dispersion.py`.

[The spectrometer regressions](../tests/test_slit_spectrometer.py) check line placement, finite slit width, independently integrated photon counts, spatial/spectral coupling with smile/keystone, an analytic moving surface, estimated-geometry registration, supplied calibration perturbations, seeded noise, flags, missing coverage and quadrature refinement. [Executable guide tests](../../../tests/test_library_guides.py) run this complete walkthrough with sockets blocked. Run focused instrument checks through `devbox run -- just --justfile tests/justfile pytest libs/orsia-instruments/tests tests/test_instrument_examples.py tests/test_lvf_scan.py tests/test_instrument_composition.py` from the Orsia root.

These checks establish bounded software and analytic-fixture behavior. Scientific instrument performance, orbital accuracy, engineering feasibility, retrieval performance and concurrent native-library safety require separate evidence for the intended use.


`spectrometer_acquisition_grid(spectrometer, schedule, ...)` consumes a caller-created `RasterReadSchedule`. The retained indices identify detector pixels while the finite entrance slit determines incident ray support. Selected reads retain original identity/timing and use a read axis through measurement and calibration. A supplied estimated geometry may differ from simulation geometry when its response retains the same schedule.
