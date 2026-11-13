# Explicit finite Fourier sampling

`response.fourier.sample_fourier_exposures` evaluates a zero-extended finite trigonometric reconstruction. It accepts a regular source `PointLattice`, an output `PointLattice` on the same declared frame, finite `(source_y,source_x,channel)` values and a Boolean source mask. Output point axes can be nonuniform or singleton. A point lattice implies no active pixel area.

Declare `detector_pitch_xy_m` separately as the reference pitch for the Gaussian Nyquist MTF. Declare active widths through `pixel_size_xy_m`, or select `None` to omit pixel integration. Both pairs use physical `(x,y)` metres. Per-channel displacement arrays use `(output_y,exposure,2)` with `(x,y)` components. Output lists preserve channel order and contain `(output_y,output_x,exposure)` values and masks. The caller owns timing, temporal averaging and complete-exposure coverage; this numerical operation evaluates the supplied offsets.

## Reconstruction policy

The required `negative_policy` selects one of three contracts:

- `preserve` retains signed polynomial values. At fixed geometry and masks, this numerical map is linear up to floating-point arithmetic.
- `clip` explicitly replaces every negative reconstructed value with zero. It is nonlinear and has no unconditional transpose or linear-tangent contract.
- `reject` requires nonnegative supported input values. It rejects supported outputs below `-64*float64_epsilon*scale`, where `scale` is the maximum absolute supported input value for that channel. Smaller negative values are corrected to zero. This is a stated numerical acceptance rule, not a rigorous global interpolation-error bound.

Invalid output neighborhoods are zero with false masks. Source values must be finite, including masked positions. Grid frames must agree explicitly; equal names or array shapes do not establish alignment. Source intervals must be regular within relative and absolute tolerances `1e-12` and `1e-12 m`. Reorder descending coordinates and their values before constructing the increasing point lattices.

## Numerical construction

1. **Objective:** prepare optical/pixel transfer and a finite validity neighborhood. **Inputs:** source spacing `(dy,dx)`, explicit detector reference pitches and optional active widths, Nyquist MTF `m` and tail tolerance `t`. **Outputs:** source-sample radii, padding and Fourier transfer. **Processing:** set `sigma=sqrt(-2*log(m))/pi` in reference-pitch samples. Divide physical pitches and widths by source spacing in `(y,x)` order. The neighborhood radius is `sqrt(-2*log(t))*sigma*pitch_ratio + pixel_width/2`, omitting the last term when pixel integration is disabled. Set padding to `ceil(radius)+2` and reject a source whose axis length is at most twice the padding. Use the shared `gaussian_pixel_transfer`: Gaussian transfer is `exp(-2*pi^2*sigma^2*((fy*pitch_y)^2+(fx*pitch_x)^2))`; pixel averaging multiplies by `sinc(fy*width_y)*sinc(fx*width_x)`.
2. **Objective:** sample the finite polynomial. **Inputs:** one masked source channel, its FFT after zero padding, transfer factors and displaced physical output positions. **Outputs:** unmodified reconstructed values. **Processing:** multiply the padded FFT by transfer. Convert each physical query to source fractional indices relative to the first source coordinates and source spacing, then add padding offsets. The dense backend evaluates the positive-exponent Fourier sum and divides by the padded sample count. The optional FINUFFT type-2 path uses the same coefficients and query ordering, `isign=1`, FFT mode order, tolerance `1e-12` and one native thread. A plan belongs to the current call. An unavailable selected backend raises `ImportError`; no fallback changes the method.
3. **Objective:** retain admitted values under the selected reconstruction rule. **Inputs:** reconstructed values, finite radii, source mask and negative policy. **Outputs:** value and validity arrays. **Processing:** require the full chosen rectangle to remain within source half-cell limits `[-0.5,n-0.5]`. Use a summed-area table to require every integer point between the clipped floor/ceiling rectangle bounds to be valid. Apply the negative policy to admitted values, then zero unsupported results. The mask describes this finite Gaussian/pixel neighborhood rule; the global finite-polynomial reconstruction and zero extension remain distinct assumptions.

`apply_spatial_response` retains its separate replicated-edge boundary rule. Both methods reuse the same Gaussian/pixel transfer calculation. Apply optical blur and pixel averaging once to the evaluated radiance product or an equivalent linear spectral contraction. Filtering reflectance before spatially varying transfer remains a different approximation.

## Replacing the mixed anchor operation

`SurfaceGrid`, `surface_anchor_charge` and the old `imaging.fourier` module are removed. Geometry's `resample_point_field` evaluates arbitrary finite `(y,x,...)` fields on explicit source/target point lattices, preserving trailing axes and signed values. Select `boundary="reject"` or the explicit nearest-edge continuation. Optional source validity is conservative: all four selected bounded endpoints must be valid, including zero-weight upper endpoints. Invalid output values are zero. This operation infers no common grid centre and performs no coaddition or clipping.

Scene owns the affine surface-reflectance law. Instruments owns the subsequent physical response contraction. Applications retain explicit read accumulation weights. The old anchor reference is reproduced in tests through these separate operations and its stated identical-read assumption; it is no longer a production helper that combines unrelated responsibilities or uses mission band names.

## Evidence and limits

Tests compare dense integer queries with an independent inverse FFT, check physical displacement components, nonuniform output coordinates, masks and frame failures, and demonstrate signed ringing, explicit clipping and rejection. A deterministic fake verifies the optional backend's argument/order plumbing. FINUFFT is unavailable in the inspected environment, so native accelerated parity and its runtime GIL behavior remain unqualified. No optional backend was installed. Numerical support checks do not establish optical accuracy, acquisition feasibility or retrieval performance.
