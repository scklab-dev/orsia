# Composing instrument charge and image readout

The composition boundary is mean photoelectrons per physical read. Each source follows its optical path before its charge joins the detector's input. Geometry, quadrature and spectral grids may differ across sources that share physical read identities and exposure timing.

## Complete-image operations

| Operation | Input and output |
| --- | --- |
| `image_photoelectrons(scene, pixel_bounds_m=..., optics=..., exposures=...)` | Integrate one full radiance scene and optical path into `ImagePhotoelectrons`; an optional scene callback receives the whole acquisition once. |
| `image_charge(contributions, exposure_bounds_s=..., row_delay_s=...)` | Attach physical read identities to named `(exposure,row,column)` arrays of already integrated photoelectrons. |
| `combine_image_source_paths(*paths)` | Join complete images of matching shape, native identities, port and read intervals with distinct source names. |
| `image_readout(charge, detector=..., rng=...)` | Apply dark current, fixed detector electronics and optional conditional counting/noise to one complete image. |
| `simulate_images(scene, ...)` | Compose the optical and detector functions above for one source. |

The image acquisition uses `(exposure,row,column)` raster order, with numeric destinations derived by `np.indices(image_shape)`. Its physical read IDs follow `s{exposure}-p{row * columns + column}`, pixel IDs follow `r{row}-c{column}` and setting IDs are exposure indices as strings. This convention aligns charge from `image_photoelectrons` and `image_charge`. Every input has complete image support and evaluation returns one dense image. Rows receive their rolling exposure timing via `row_delay_s`.

`ImagePhotoelectrons.signal` stores named owned `(reads,)` charge vectors before dark current. `ImageMeasurements.photoelectrons` and its named `contributions` have `(exposure,row,column)` axes; optional DN and flags have `(realisation,exposure,row,column)` axes. Returned image arrays own writable storage, while mappings provide read-only keys and writable independent value copies. Image optics settings are descriptive scalar or per-exposure snapshots retained from the first path. Source names and physical read identities must agree at combination time; the first path supplies the descriptive settings.

## Spectral QE and noise

`image_photoelectrons` and `simulate_images` accept `quantum_efficiency(indices, wavelengths_m)`, with exposure/row/column integer index vectors and the full wavenumber grid converted to wavelengths in metres. The callback supplies an exact `(reads,wavenumbers)` array of finite physical fractions in `[0,1]`; it replaces `DetectorMaps.quantum_efficiency`. At the lower level, `expected_electrons` accepts `quantum_efficiency(local_read_indices,wavelengths_m)` with the same complete `(reads,wavenumbers)` shape. The optical accepted-fraction callback retains its per-read `(nodes,wavenumbers)` contract and evaluates the complete spectral grid for each read. Callbacks own their support policy, remain pure throughout evaluation and receive borrowed inputs.

The detector draws Poisson photoelectron-plus-dark charge, clips it at full well, adds independent and group-common Gaussian read noise, then applies gain, bias, rounding and ADC clipping according to `sample_readout`. Row and column groups renew for each exposure and realisation; a frame group spans one exposure. The supplied RNG advances explicitly in complete-raster order. For expectation-only analysis, omit `rng` and use the pre-dark means and source contributions; DN and flags are then `None`.

## Scope of evidence

The [dense imaging tests](tests/test_dense_imaging.py) check full-grid photon conversion, QE shape, source combination, rolling timing, response projection, correlation and a synthetic common-noise case. Software tests establish these numerical contracts on synthetic inputs; physical scene support, optical quadrature convergence, detector-scale feasibility and instrument calibration require case-specific evidence.
