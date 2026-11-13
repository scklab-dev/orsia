# Complete-image simulation

Use the dense image interface to integrate scene radiance through optics into mean photoelectrons, then optionally sample detector output. The [package introduction](../README.md) provides a minimal single-source image.

## Two sources, one detector readout

This synthetic example combines a target field and a faint background field on the same 2-by-2 detector during one exposure. Each source passes through its own spectral filter and spatial PSF. The two mean-charge images join before dark current and detector noise enter once.

Run this standalone example with the distribution and its dependencies installed, or in the [development environment](../../../README.md#development-environment). Both fields are uniform over the declared local angular and time support; radiance levels use W m^-2 sr^-1 (cm^-1)^-1. The background represents an independently supplied optical path.

```python
from functools import partial

import numpy as np
from orsia_instruments import GaussianPSF, combine_image_source_paths, exposure_sequence, filter_imager, gaussian_filter, image_photoelectrons, image_readout, regular_detector, uniform_radiance_field
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid


def source_field(level):
    radiance = Radiance(
        wavenumber_grid(np.linspace(4900.0, 5100.0, 65)),
        np.full(65, level),
        ObservationGeometry(0.0),
    )
    return uniform_radiance_field(
        radiance,
        time_bounds_s=(0.0, 0.02),
        u_bounds=(-0.004, 0.004),
        v_bounds=(-0.004, 0.004),
        spectral_bounds_cm1=(4900.0, 5100.0),
    )


exposures = exposure_sequence(
    start_s=0.0,
    duration_s=0.01,
    direction_bounds_uv=(-0.004, 0.004, -0.004, 0.004),
)
detector = regular_detector(
    (2, 2),
    pitch_m=30e-6,
    active_size_m=30e-6,
    centre_xy_m=(0, 0),
    quantum_efficiency=0.7,
    dark_electrons_per_s=2.0,
    read_noise_electrons=3.0,
    full_well_electrons=1e6,
    electrons_per_dn=2.0,
    bias_dn=100.0,
    nonlinearity=0,
    adc_bits=16,
)
pixels = detector.pixel_bounds_m
target_optics = filter_imager(
    focal_length_m=0.02,
    collecting_area_m2=1e-8,
    spatial=GaussianPSF(30e-6),
    filter=partial(gaussian_filter, centre_m=2e-6, width_m=20e-9),
    throughput=0.8,
)
background_optics = filter_imager(
    focal_length_m=0.02,
    collecting_area_m2=1e-8,
    spatial=GaussianPSF(40e-6),
    filter=partial(gaussian_filter, centre_m=2.02e-6, width_m=15e-9),
    throughput=0.5,
)
target = image_photoelectrons(
    source_field(1e-3),
    pixel_bounds_m=pixels,
    optics=target_optics,
    exposures=exposures,
    detector=detector,
    source_name="target",
    angular_order=5,
)
background = image_photoelectrons(
    source_field(1e-4),
    pixel_bounds_m=pixels,
    optics=background_optics,
    exposures=exposures,
    detector=detector,
    source_name="background",
    angular_order=5,
)
combined = combine_image_source_paths(target, background)
assert np.all(target.signal.total > 0) and np.all(background.signal.total > 0)
np.testing.assert_allclose(
    combined.signal.total,
    target.signal.total + background.signal.total,
    rtol=1e-14,
)
images = image_readout(
    combined,
    detector=detector,
    rng=np.random.default_rng(71),
    realisations=2,
)
np.testing.assert_array_equal(
    images.contributions["target"],
    target.signal.total.reshape((1, 2, 2)),
)
np.testing.assert_array_equal(
    images.contributions["background"],
    background.signal.total.reshape((1, 2, 2)),
)
np.testing.assert_allclose(
    images.photoelectrons,
    images.contributions["target"] + images.contributions["background"],
    rtol=1e-14,
)
assert images.dn is not None and images.dn.shape == (2, 1, 2, 2)
print(images.photoelectrons.shape, images.dn.shape)
```

Expected output: `(1, 2, 2) (2, 1, 2, 2)`. Photoelectrons and named contributions use `(exposure,row,column)` axes; DN adds a leading realisation axis. Passing `detector` to each integration applies only its QE. `image_readout` applies the remaining detector physics to the combined charge and preserves the pre-dark source means.

For a supplied internal emitter, describe its radiance and remaining optical path explicitly. The background above is an assumed uniform field; the library infers neither emission temperature nor emissivity. The example checks source accounting on a bounded grid. Its support and quadrature orders are illustrative and carry no instrument-accuracy claim.

## Charge alignment, timing and ownership

Mean photoelectrons per physical read form the composition boundary. Source paths may use different geometry, quadrature and spectral grids. Combination requires identical ordered sample/pixel/setting/port identity sequences and exactly equal exposure bounds; it performs no read reordering. Use distinct source names; repeated names, mismatched identities, timing, image shape or output port raise `ValueError`.

| Operation | Input and output |
| --- | --- |
| `image_photoelectrons(scene, pixel_bounds_m=..., optics=..., exposures=..., detector=...)` | Integrate one full radiance scene and optical path into `ImagePhotoelectrons`; a scene callback receives the whole acquisition once. |
| `image_charge(contributions, schedule=...)` | Attach physical read identities to named dense `(exposure,row,column)` arrays or selected `(reads,)` vectors of already integrated photoelectrons, including upstream optics and QE where applicable. |
| `combine_image_source_paths(*paths)` | Join aligned complete charge images, retaining source contributions. |
| `image_readout(charge, detector=..., rng=...)` | Apply dark current, fixed electronics and optional conditional counting/noise to the complete image. |
| `simulate_images(scene, ...)` | Compose optical integration and detector readout for one source. |

`raster_read_schedule(exposure_bounds_s, shape, row_offsets_s=..., setting_ids=..., port=...)` owns the `(frame,row,column)` C-order identity convention and rolling intervals. `RasterReadSchedule` composes its immutable `ReadSchedule` with full `native_shape` and `(reads,3)` native indices. Acquisitions reference the read schedule; image charge, measurements and calibration responses retain the raster correspondence. `select_raster_reads` preserves original IDs, indices and full native shape. Dense reshaping requires complete C-order coverage; partial or permuted selections use `(reads,)` values. Calibration compares measured and estimated identity/timing and uses retained indices to select the independently estimated detector maps. Geometry and detector parameters can differ between simulation and supplied calibration.

`ImagePhotoelectrons.signal` stores named owned `(reads,)` charge vectors before dark current. For complete C-order rasters, `ImageMeasurements.photoelectrons` and `contributions` use `(exposure,row,column)` axes; optional DN and flags use `(realisation,exposure,row,column)`. Partial or permuted selections use `(reads,)` values and `(realisation,reads)` DN/flags in retained read order. Measurement construction and charge functions copy numerical values into independent writable arrays. The raster schedule and its timing remain immutable. Detector geometry and property arrays remain read-only. Mappings have read-only keys and writable independent value copies. Descriptive scalar or per-exposure optical settings come from the first combined path. Custom callbacks are borrowed and must remain pure, stable and nonmutating throughout evaluation.

## Spectral QE and noise

`image_photoelectrons` and `simulate_images` accept `quantum_efficiency(indices, wavelengths_m)`. The callback receives exposure/row/column integer index vectors and the full wavenumber grid converted to wavelengths in metres. It returns an exact `(reads,wavenumbers)` array of finite fractions in `[0,1]`, replacing `DetectorProperties.quantum_efficiency`. At the lower level, `expected_electrons` accepts `quantum_efficiency(local_read_indices,wavelengths_m)` with the same complete output shape. The optical accepted-fraction callback retains its per-read `(nodes,wavenumbers)` contract and evaluates the complete spectral grid for each read. Callbacks own their support policy and borrow their inputs.

QE enters during optical charge integration. `image_charge` accepts means that already include any required QE; `image_readout` does not apply QE again. The detector draws Poisson photoelectron-plus-dark charge, clips it at full well, adds independent and group-common Gaussian read noise, then applies electronics and digitization as specified in the [detector reference](acquisition-and-detector.md#detector-and-perturbations).

Row and column noise groups renew for each exposure and realisation; a frame group spans one exposure. The caller-owned RNG advances in complete-raster order. For expectation-only analysis, omit `rng`: pre-dark means and named contributions remain available, while DN and flags are `None`.

## Physical-description image API

Start with `regular_detector`, an optical factory and `exposure_sequence`, then call `simulate_images` with the detector's active bounds and the explicit detector. Lower-level acquisition, response and detector calls remain available, subject to the optical-description replacement restriction below. No pixel names or setting dictionaries are required. With `filter_imager`, supply the spectral filter separately; there is no need to write a full optical response callback. Labels in the advanced acquisition API identify reads; they do not supply physical parameters.

- `regular_detector((rows, columns), pitch_m=..., active_size_m=..., centre_xy_m=..., ...)` requires all geometry and property values and owns read-only active bounds. `regular_pixel_bounds((rows, columns), pitch_m=..., active_size_m=..., centre_xy_m=...)` remains available for lower-level calculations and generates `(rows,columns,4)` bounds in metres. Pitch and active dimensions are scalar or `(x,y)`, while shape is `(rows,columns)`; columns increase x and rows increase y. The centre is that of the whole array. Active dimensions default to pitch and cannot exceed it. For irregular rectangular geometry, construct `DetectorProperties` with explicit `pixel_bounds_m` and `pitch_m=None`. Detector shape is derived from bounds. Declared `(x,y)` pitch is validated against centre spacing on available axes and retained for single-row or single-column arrays.
- `filter_imager`, `spatial_filter_imager`, `lvf_imager`, `interferometric_imager`, `cavity_imager` and `dispersive_imager` assemble the physical response kernels. All take `focal_length_m`, projected `collecting_area_m2`, one `spatial` model: a declared `GaussianPSF`, prepared `TabulatedPSF` or density callback, and optional scalar `throughput`. The spatial PSF is independent of the spectral filter shape; callbacks receive explicit field, wavelength and exposure coordinates. The [PSF contract](optics.md#psf-callback-contract) specifies offsets, field/wavelength/exposure context, density units, table support and normalization. All factories expose `pixel_order` for general PSFs while retaining analytic Gaussian integration where applicable. Optical losses, QE, collecting area, exposure and pixel acceptance each enter once. The [factory example](optics.md#interchangeable-factory-example) exercises the supported families.
- `filter_imager(..., filter=spectral_filter)` accepts a wavelength-only function: `spectral_filter(wavelengths_m)` receives the complete 1-D float64 wavelength grid in metres, in the grid's converted order, and returns the same shape with finite dimensionless intensity transmission in [0,1]. Invalid values or shapes raise `ValueError` when evaluated, before pixel acceptance; no clipping or normalisation is applied. The filter is uniform across the detector and identical for all exposures. The callable and captured data are borrowed, must remain pure and stable, and must not mutate the input; they are not copied into `settings`. Bind existing laws with `functools.partial`, for example `partial(gaussian_filter, centre_m=2e-6, width_m=20e-9)` or `partial(tabulated_transmission, nodes_m=..., fractions=...)`. Tabulated support and its explicit `outside` policy are preserved. `filter_imager` is the entry point for a supplied uniform spectral law. Gaussian `width_m` is a wavelength standard deviation, not FWHM or spectral-grid spacing.
- LVF reference/sigma, two-beam base OPD and cavity base spacing accept scalars or vectors with exactly one value per exposure; a length-one vector is not implicitly repeated. Settings stay constant in time during each exposure. `lvf_imager` accepts either the existing Gaussian `sigma_m` or `profile(wavelengths_m, centre_m)`, keeping the polynomial centre law separate. It integrates spatial transmission across the active pixel, not only at its centre, using a detector-plane x/y coordinate relative to `reference_position_m`; its convenience law assumes normal incidence with no thermal shift. For other spatial transmission laws, use `spatial_filter_imager(..., filter=local_filter)`. `cavity_imager` requires explicit scalar or wavelength-callable `order_transmission` and also accepts spectral index, reflectivity and coating phase, with an optional additive `spacing_offset_m(x,y)`. Rebuild the optical factory when changing physical parameters: the model exposes a fixed description with immutable numerical setting values. Keep the settings arrays' shape/dtype metadata unchanged. `dataclasses.replace` of individual optical attributes raises `TypeError`. Factories and custom responses use `ImagingOptics(focal_length_m, collecting_area_m2, response, settings, throughput=1, port="direct")`; consistency of a supplied arbitrary closure with its description remains caller-owned.
- `interferometric_imager(..., opd_m=..., port="+")` models two-beam interference with the selected spatial PSF, not arbitrary interferometer architectures. It constructs one physical output per call: `port="+"` and `port="-"` represent separate outputs. Optical path difference is fixed or varies between exposures, remaining constant in time within each exposure; an optional `opd_offset_m(x,y)` adds detector-plane spatial encoding, integrated jointly with the PSF. Phase, visibility and transmission accept scalars or wavelength-only functions. Results retain `port` and the base `settings["opd_m"]`. Simulate both before differencing measured DN; the exposure/OPD axis is not a reconstructed spectral axis. Separate calls do not represent common electronic or scene fluctuations between ports.
- `dispersive_imager` accepts the existing polynomial coefficients or a supplied `image_map(image_xy_m, wavelengths_m)` returning `(nodes,wavelengths,2)` centroids in metres. `transmission` can be scalar or spectral. `polynomial_image_map` exposes the existing mapping law independently of pixel integration.
- `DetectorProperties` requires active `pixel_bounds_m` and explicit `(x,y)` `pitch_m` or `None`. It accepts a scalar or an exact `(rows,columns)` map for QE, dark electrons/s, read-noise sigma in electrons, full well in electrons, electrons/DN, bias DN and dimensionless nonlinearity; ADC width is scalar. No implicit row/column broadcasting is introduced. A `quantum_efficiency(indices, wavelengths_m)` callback on the image integration functions supplies spectral QE and replaces the simple map, as specified in the [spectral QE contract](#spectral-qe-and-noise). The maps are fixed across exposures and indexed numerically for all reads. `image_photoelectrons`, `image_readout` and `simulate_images` require an explicit `detector` keyword argument. Supply geometry and all eight photoelectric/electronic fields when constructing `DetectorProperties`; every field is required, including for expectation-only calculations. Geometry and property maps are independent read-only copies, and maps must match the derived detector shape even when spectral QE replaces their values. Omitting a field raises `TypeError`. The [package example](../README.md#minimal-image-example) shows a complete constructor.
- `exposure_sequence(start_s=..., duration_s=..., direction_bounds_uv=...)` accepts scalar or broadcastable 1-D starts/durations and optional `row_delay_s`. Constant `pointing_uv=(u,v)` defaults to `(0,0)`; a time callback is optional. The caller still declares the accepted sky patch and supplies scene coverage over the entire rolling exposure support. The image map remains the bounded paraxial `f*(direction-pointing)` approximation.
- `simulate_images(scene, pixel_bounds_m=..., optics=..., exposures=..., detector=...)` returns `ImageMeasurements.photoelectrons` with axes `(exposure,row,column)`, before dark current. Without `rng`, DN and saturation flags are `None`; only expected photoelectrons are returned. With an explicit generator, `dn`, `full_well_saturated` and `adc_clipped` have axes `(realisation,exposure,row,column)`. Actual timing is retained in `schedule.reads.exposure_bounds_s` with axes `(reads,2)`, including row delay. Measurement value arrays are independent writable copies; the shared raster schedule is immutable. Input geometry/maps/settings are copied, while custom callbacks are borrowed and must remain pure and stable.
- A scene may be a resident `AngularRadianceSamples` or a caller-owned `field(acquisition)` callback. The callback receives the complete image acquisition once and supplies radiance over its full time and direction support. Source QE may instead be supplied as a callback over complete image indices and spectral grid wavelengths.

The high-level functions accept `row_noise_electrons`, `column_noise_electrons` and `exposure_noise_electrons` as Gaussian sigmas. Row/column groups are independent for each exposure and realisation; the frame component is shared by the exposure. The detector kernel groups full-image row, column and exposure identities before sampling, retaining the physical group covariance in the resulting DN. Do not count the same noise component again through low-level group options.

The complete acquisition is evaluated with full-grid NumPy/SciPy contractions. The caller selects `time_order`, `angular_order` and spatial-response `pixel_order`. Scene interpolation, coefficient tensors and sampled readout use dense storage. Spectral sampling, sky-patch truncation and quadrature convergence require separate checks; image-scale performance requires case-specific qualification.

For a custom instrument, `ImagingOptics` accepts `response(exposure_index, local_read_index, wavelengths_m, acquisition)`, returning accepted fractions with axes `(nodes,wavelengths)`. Numeric exposure indices are supplied directly, without decoding labels. Use the advanced kernels for time/ray-dependent LVF physics, arbitrary rolling schedules or cross-port correlations not represented by these convenience factories. Wavelength-dependent QE and multiple source paths are available through the composable image functions in the [two-source example](#two-sources-one-detector-readout).

## Read schedules without optical geometry

`ReadSchedule` describes ordered sample/pixel/setting/port identities and positive exposure intervals `(reads,2)` in seconds. Pass it directly to `ExpectedElectrons` or `digitize` without optical nodes, directions or weights. `read_schedule(acquisition)` strips geometry explicitly; it returns a `ReadSchedule` input unchanged. `ExpectedElectrons.reads` and `DetectorReadings.reads` retain only this schedule. `electron_moments`, `electron_covariance`, `digitize`, `sample_readout` and the native tolerance functions use physical reads without retaining optical geometry. Optical integration and `LineOfSightSamples` retain `Acquisition` at the ray/optical boundary.

## Replicated-boundary grid filtering

[`apply_spatial_response`](../src/orsia_instruments/response/spatial.py) and `spatial_response_support` own the former Scene optical/pixel filters. They operate on supplied `(y,x,...)` values and Boolean `(y,x)` support. Apply the filter to evaluated radiance or its linear spectral contraction exactly once. Filtering reflectance before spatially varying atmospheric transfer selects a different diagnostic approximation.

1. **Objective:** filter sampled values with explicit replicated boundaries. **Inputs:** finite values, Nyquist MTF `m` in `(0,1]`, Boolean pixel integration and optional `upper_bound`. **Outputs:** a fresh filtered array in the original units. **Processing:** set `sigma=sqrt(-2*log(m))/pi`, pad both spatial axes by `max(16,ceil(8*sqrt(sigma^2+pixel_integration/12)))` using edge replication, and form `H(f_y,f_x)=exp(-2*pi^2*sigma^2*(f_y^2+f_x^2))`. Pixel integration multiplies by `sinc(f_y)*sinc(f_x)`. For `upper_bound=True`, replace negative real impulse-response coefficients with zero, then transform back without renormalization. Multiply the padded spatial FFT by `H`, inverse-transform, take its real part and crop to the original extent. Signed input values remain signed; no output clipping occurs.
2. **Objective:** mark complete finite-grid neighborhoods. **Inputs:** Boolean mask, the same MTF/pixel choice and tail tolerance `t` in `(0,1)`. **Outputs:** a fresh Boolean mask. **Processing:** set `radius=ceil(sigma*sqrt(-2*log(t))+0.5*pixel_integration)` and apply a square minimum filter of width `2*radius+1`, with outside-grid values treated as true. This support convention accompanies replicated boundaries. Zero-extended Fourier sampling retains its separate boundary law.

## Evidence and limits

The [dense imaging tests](../tests/test_dense_imaging.py) check full-grid photon conversion, spectral QE shape, source combination, rolling timing, response projection and common-noise grouping. The [executable guide tests](../../../tests/test_library_guides.py) run the two-source example with network access blocked. These software checks cover synthetic numerical contracts. Physical scene support, optical quadrature convergence, detector-scale feasibility and instrument calibration require case-specific evidence.
