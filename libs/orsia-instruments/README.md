# Orsia Instruments

`orsia-instruments` maps explicitly supported radiance through acquisition geometry, optical response and detector physics to native reads. It supports dense images and GCRS timed lines of sight, finite-pixel filter/dispersive/two-beam/cavity responses, explicit source charge and optional noisy DN. The caller supplies radiance, trajectory assumptions, pointing and accepted support.

`imaging.response.joint.spatial_spectral_operator` provides bounded fixed-response products and their unnormalized physical transpose, with all-port ordering and an optional fixed validity mask. The [joint-response guide](docs/spectral-imagers.md#bounded-response-products) defines retained-input stability and memory-estimate scope.

## Architecture

![Detector geometry and QE feed native charge and readout; supplied calibration supports separate ground registration and native observation cubes with uncertainty, while fixed models supply joint response products.](docs/diagrams/architecture.svg)

Arrows show selected value or model flow. The upper chain integrates radiance to charge and optional noisy digital numbers (DN), with required detector geometry and quantum efficiency (QE). The native-frame path calibrates expected charge or DN with independently supplied characterization. Separate calls provide ground registration or logical observation cubes with retained native supports and propagated read/calibration errors. The joint-response branch builds fixed forward and unnormalized transpose products. Radiance inputs include local angular images (`AngularRadianceSamples`), timed GCRS lines of sight and physical surface callbacks, each with its own geometry and coverage contract. [Full-size SVG](docs/diagrams/architecture.svg) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

The [fixed-filter camera path](docs/filter-camera.md) adds configurable detector strips, physical sizing, an instrument-selected surface grid, native frames, supplied linear calibration and ground-registered band cubes. Its test-local analytic fixtures cover expected charge and explicit seeded noise. The architecture figure includes the shared forward measurement chain and separate calibration and registration calls.

The [scanning slit-spectrometer path](docs/slit-spectrometer.md) integrates finite entrance-slit rays through polynomial dispersion and a supplied post-slit spatial model. It shares native detector readout, supplied calibration and estimated-geometry registration. Every detector pixel retains its achieved photon-weighted band, including row-dependent smile/keystone responses; common-band spectral rectification remains outside native-band registration.

The [detector-owned spectral imager guide](docs/spectral-imagers.md) completes LVF, two-port interferometer and Fabry-Perot chains. All families share `frame_response` and `simulate_frames`. A bounded joint spatial-spectral operator retains actual footprints and separate native port schedules for static piecewise-constant scenes. Inverse owns the covariance-weighted rank-aware solver; Instruments supplies positive physical coefficients.

`wgs84_surface_projection_from_camera` selects a fixed east/north metre chart from the camera's geometric detector centre or a native `(row,column)` pixel centre. It uses the trajectory, both attitude laws, reference time and optional platform-frame mount offset. The helper returns Geometry's `WGS84SurfaceProjection`, also used by the explicit-origin and from-ray constructors. Calling the object returns `(xy_m, view_local, hits)` with outward views in fixed chart-origin east/north/up axes after dated GCRS-to-ITRF rotation. Reuse one returned surface callback for truth and estimated acquisition grids. The [camera walkthrough](docs/filter-camera.md#walkthrough-two-strips-over-earth) defines reference conventions, orbital auxiliary requirements and continuous-support duties.

Surface-grid simulation uses the shared flat callback `radiance(xy_m, view_local, times_s, spectral)`: `(N,2)`, `(N,3)`, `(N,)` inputs and a `SpectralGrid` with K points and `(N,K)` output. Projection supplies required `SurfaceAcquisitionGrid.view_local`; Instruments flattens reads then nodes in C order and restores the integration axes. See the [breaking migration notes](../orsia-scene/README.md#flat-callback-migration) for removed observation adapters and shared tangent integration. The separate timed line-of-sight and image-field interfaces retain their existing contracts.

The [bounded radiometry guide](docs/bounded-radiometry.md) defines spectral/read blocks, disconnected-domain photon quadrature, callback migration and the radiance/response derivative product rule.

The [point-lattice response guide](docs/grid-responses.md) defines `GridElectronResponse`, explicit frames, `(y,x)` field axes, native scan order and the separate angular/surface radiance sample records.

## Minimal image example

Run the example with this distribution and its dependencies installed, or in the [Orsia development environment](../../README.md#development-environment). This synthetic uniform field illuminates a 2-by-2 detector through a Gaussian spectral filter. A caller-owned RNG requests two detector realizations.

```python
from functools import partial

import numpy as np
from orsia_instruments import GaussianPSF, exposure_sequence, filter_imager, gaussian_filter, regular_detector, simulate_images, uniform_radiance_field
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid

radiance = Radiance(
    wavenumber_grid(np.linspace(4900.0, 5100.0, 65)),
    np.full(65, 1e-3),
    ObservationGeometry(0.0),
)
field = uniform_radiance_field(
    radiance,
    time_bounds_s=(0.0, 0.1),
    u_bounds=(-0.004, 0.004),
    v_bounds=(-0.004, 0.004),
    spectral_bounds_cm1=(4900.0, 5100.0),
)
optics = filter_imager(
    focal_length_m=0.02,
    collecting_area_m2=1e-8,
    spatial=GaussianPSF(30e-6),
    filter=partial(gaussian_filter, centre_m=2e-6, width_m=20e-9),
    throughput=0.8,
)
detector = regular_detector(
    (2, 2),
    pitch_m=30e-6,
    active_size_m=30e-6,
    centre_xy_m=(0, 0),
    quantum_efficiency=0.7,
    dark_electrons_per_s=0,
    read_noise_electrons=0,
    full_well_electrons=1e6,
    electrons_per_dn=1,
    bias_dn=100.0,
    nonlinearity=0,
    adc_bits=16,
)
images = simulate_images(
    field,
    pixel_bounds_m=detector.pixel_bounds_m,
    optics=optics,
    exposures=exposure_sequence(
        start_s=0.0,
        duration_s=0.01,
        direction_bounds_uv=(-0.004, 0.004, -0.004, 0.004),
    ),
    detector=detector,
    rng=np.random.default_rng(71),
    realisations=2,
    angular_order=3,
)
assert np.all(images.photoelectrons > 0)
assert images.dn is not None
print(images.photoelectrons.shape, images.dn.shape)
```

Expected output: `(1, 2, 2) (2, 1, 2, 2)`. Photoelectrons use `(exposure,row,column)`; DN adds a leading realization axis. Omitting `rng` returns expected photoelectrons with `dn=None`.

## Sampled wavelength response

[`FilterElectronResponse` and `SpectralElectronResponse`](docs/wavelength-response.md) provide geometry-free per-read spectral coefficients with explicit passband, QE, transmission, accepted-etendue and read-schedule owners. One immutable preparation supplies nominal/trial profiles, sampled unblocked areas and fixed-area centre/FWHM sensitivities. Coaddition is an explicit map applied after native reads. Both nm and cm-1 density measures use the shared spectral quadrature contract.

## Native observation cubes and uncertainty

[`calibrate_observations` and `form_observation_cube`](docs/observation-cubes.md) retain logical `(line,sample,band)` indexing, native supports, contribution weights, calibration sensitivities and signed bias. Gain, pedestal and response-normalization uncertainty propagate with explicitly declared read/calibration cross-covariance. Fixed nonnegative within-band weights transport values and covariance together. These additive APIs preserve band-dependent footprints; `reconstruct_radiance_cube` retains its separate ground-registration contract.

[`calibrate_readings` and `accumulate_observations`](docs/observation-cubes.md#geometry-independent-radiometry) provide the shared geometry-independent numerical operations. Fixed calibration has zero uncertain parameters. Common parameter columns and projected read sources preserve correlations across concatenated read blocks. [`NativeBandResponse`](docs/observation-cubes.md#shared-spectral-response-tables) also accepts shared spectral tables, read-to-response indices and optional absolute per-read sensitivity without implicit read-by-spectral expansion; dense construction remains compatible.

Run the [two-band DN example](../../examples/native_observation_cube.py) from the repository root with `MPLBACKEND=Agg UV_NO_SYNC=1 devbox run -- just python examples/native_observation_cube.py --figure outputs/native-observation-cube.png`. It shows a mask, accumulation and a common gain error, then checks dense/source-projected agreement. The [guide](docs/observation-cubes.md) explains the saved plot, parameter units, valid C-order indexing and fixed-geometry limits.

## Explicit detector API

`DetectorProperties` owns active geometry, pitch, pixel QE and electronics. Its ten fields are required: `pixel_bounds_m`, `pitch_m`, `quantum_efficiency`, `dark_electrons_per_s`, `read_noise_electrons`, `full_well_electrons`, `electrons_per_dn`, `bias_dn`, `nonlinearity` and `adc_bits`. Omitting any field raises `TypeError`. The example's numerical values are illustrative assumptions.

Active bounds have shape `(rows,columns,4)` in `(xmin,xmax,ymin,ymax)` metre order. `shape` is derived from these bounds. `pitch_m` is a positive `(x,y)` pair or explicit `None` for irregular geometry. Columns increase x and rows increase y for regular detectors. Declared pitch must agree with centre spacing on available axes, and active dimensions cannot exceed pitch. Single-row and single-column detectors retain both pitch components.

`regular_detector(shape, pitch_m=..., active_size_m=..., centre_xy_m=..., ...)` builds regular active bounds and requires every geometry and property argument. Pitch and active size accept scalar or `(x,y)` values. `regular_pixel_bounds` remains available from the package root and `orsia_instruments.acquisition.schedule.images` for lower-level geometry calculations.

Geometry and scalar/map properties become independent read-only copies at detector construction. Property maps must exactly match the derived `(rows,columns)` shape; implicit row/column broadcasting is rejected. Scalars apply to every pixel. Keep retained arrays stable during use. Geometry defines active-gap losses directly, and QE enters once before photon noise.

`filter_camera(detector, strips, ...)`, `dispersive_spectrometer(detector, ...)` and `focal_plane_imager(detector, optics, ...)` retain the supplied detector. Their `pixel_bounds_m` properties delegate to that detector. `frame_response` and `simulate_frames` use the instrument's detector and accept no separate `detector` argument. Calibration requires an explicit independently estimated detector; a separately characterized instrument can supply its response. `frame_response` replaces the previous `camera_response` and `spectrometer_response` names without aliases.

The functional image kernels `image_photoelectrons`, `image_readout` and `simulate_images` keep explicit `detector=` arguments, including expectation-only calls. Optical integration also receives explicit bounds and optics. `calibrate_frames` likewise requires `detector=`. `None` is outside these contracts.

This breaking migration retains the earlier `DetectorMaps` to `DetectorProperties` rename without an alias and adds detector-owned geometry. Move geometry into the detector constructor, attach that detector when building a physical instrument, and remove detector keywords from instrument response and frame simulation calls. Update independently supplied calibration detectors and low-level callers with explicit geometry. Numerical photon and readout calculations retain their existing conventions.

## Along-track optical responses

Evaluate `optics.response(exposure, read, wavelengths_m, acquisition)` on the full acquisition to obtain optical accepted fractions with shape `(nodes, wavelengths)`. Finite-pixel losses remain in these fractions. Area, throughput, QE and quadrature enter the downstream photon calculation separately. Keep borrowed optics and acquisition arrays unchanged during evaluation.

For a single optical exposure, pass `functools.partial(optics.response, 0)` to `detector_response`, or supply it as `optical_response=functools.partial(optics.response, 0)` to `instrument_response`. The callback evaluates the supplied full acquisition directly. When preparation depends on that acquisition, supply `optical_factory` instead; its positional and keyword forms remain supported and it runs once. Supply exactly one of `optical_response` and `optical_factory`; missing or ambiguous selection raises `ValueError`.

`along_track_response` stacks per-read optical responses with explicit scan geometry, quadrature and QE. Configure the spatial model on each `ImagingOptics` returned by `optics_for_read`. Optical factories accept `GaussianPSF`, `TabulatedPSF` or a pure spatial density callback. The function evaluates those optics on every acquisition node. The previous Gaussian-prefixed name has been removed without an alias; update imports and calls to `along_track_response`.

`GaussianCrossTrackResponse` and `GaussianCrossTrackAcquisitionResponse` have been removed without aliases. Migrate their callers to the full optical response above and remove the separate `sigma_m` keyword from calls to `along_track_response`. The Gaussian PSF and optical-model APIs remain available.

## Physical conventions and limits

- Acquisition-based radiance is unpolarised W m-2 sr-1 per (cm-1); fields use `(time,u,v,spectral)`. Optical callbacks receive wavelengths in metres. Geometry, full exposure and spectral coverage are explicit. The separate sampled wavelength response uses per-nm radiance and its declared pixel-cone approximation.
- Optical fractions, projected area, QE, time/solid-angle weights and photon conversion each enter once. A Gaussian filter's `width_m` is its standard deviation.
- Photoelectrons precede dark current and electronics. Inspect full-well and ADC flags with sampled DN; applying the detector transfer to mean charge does not give expected clipped DN.
- Dense integration retains full scene/response arrays. Check spatial/time/spectral quadrature and support independently; the small example's angular order is illustrative.
- Input records usually copy numerical fields; custom callbacks are borrowed and must remain pure and stable. RNGs belong to the caller. Detailed result-container ownership is specified in the references.

## Read next

- [Physical spatial scenes](../../docs/spatial-scenes.md): terrain/plume radiance callbacks, conservative continuous coverage and signed derivatives using shared bounded photon coefficients and `frame_electron_tangents`.

- [Native observation cubes](docs/observation-cubes.md): DN calibration, retained native contributions and joint read/calibration uncertainty.
- [Fixed-filter camera and cubes](docs/filter-camera.md): physical layout, PSF/motion support, supplied calibration, compatible achieved bands and estimated-geometry accumulation.
- [Slit spectrometer and native-band cubes](docs/slit-spectrometer.md): finite slit/pixel integration, scan support, supplied wavelength calibration and sparse pixel-specific band planes.
- [LVF, two-port and cavity chains](docs/spectral-imagers.md): detector-owned instruments, supplied characterization, native bands and bounded joint inversion.
- [Complete-image simulation](docs/imaging.md): two source paths, one detector readout, pixels, QE and common noise.
- [Optical response](docs/optics.md): factories, PSFs, response integration and explicit reuse.
- [Acquisition, readout and tolerances](docs/acquisition-and-detector.md): source ownership, local geometry, charge-only sampling and advanced kernels.
- [GCRS line of sight](docs/line-of-sight.md): a moving pixel observes a varying surface, followed by frame and continuous-support contracts.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-instruments/tests tests/test_instrument_examples.py tests/test_lvf_scan.py tests/test_instrument_composition.py` from the workspace root. Bounded synthetic regressions leave realistic internal backgrounds, instrument accuracy, image-scale feasibility and retrieval performance unqualified.

## Finite-grid sampling and measured-image registration

[`response.fourier`](src/orsia_instruments/response/fourier.py) evaluates zero-extended finite Fourier fields on explicit `PointLattice` supports and shared frames. Detector reference pitch, active pixel size and the signed/clipped/rejected reconstruction policy are explicit. The [Fourier sampling guide](docs/fourier-sampling.md) defines coordinates, support, numerical laws and migration from the removed mixed anchor helper. The optional FINUFFT backend requires a separately installed distribution; only dense numerical behavior and backend plumbing were checked locally.

[`imaging.registration`](src/orsia_instruments/registration/estimation.py) estimates constant or joint row-knot corrections from positive measured images. [`RegistrationMap`](src/orsia_instruments/registration/resampling.py) retains signed Keys/Lanczos stencils, complete endpoint support and exact zero-shift identity. Its bounded row extraction, value/variance propagation and structural flags preserve source-cell identities. [`imaging.covariance`](src/orsia_instruments/registration/covariance.py) projects detector variance and supplied local gains into sparse scalar-state covariance. This signed interpolation contract is separate from normalized nonnegative observation accumulation.

[`pointing`](src/orsia_instruments/pointing.py) fits a two-axis Matérn-3/2 process from two RPE constraints and supplies state-space matrices and coadd-averaged correlations. The application selects budgets, cadence and historical root conventions. Uncertainty draws explicit linear state processes from supplied roots and caller-owned generators. Conditional estimator covariance and bounded software agreement leave scene-scale cost and physical pointing accuracy unqualified.
