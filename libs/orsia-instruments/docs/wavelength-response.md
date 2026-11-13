# Sampled filter electron response

`FilterElectronResponse` prepares geometry-free per-read coefficients on a shared `SpectralGrid`. Physical owners are separate: `FilterPassband` describes a passband, `ReadSchedule` supplies actual read identities and durations, and the caller supplies QE, other optical transmission and accepted etendue. No coadd count is embedded in the preparation. Use explicit accumulation weights after native read evaluation.

## Prepare and evaluate actual reads

The following synthetic example declares three distinct detector reads with explicit timing. The source radiance is uniform within each read's accepted cone and exposure. Values are illustrative and establish no instrument calibration.

```python
import numpy as np
from orsia_instruments import FilterElectronResponse, FilterPassband, ReadSchedule, spectral_electrons, square_pixel_etendue
from orsia_radiative import SpectralDomain, SpectralGrid

wave = np.linspace(496.0, 509.0, 131)
grid = SpectralGrid("wavelength_nm", wave,
    SpectralDomain("wavelength_nm", ((496.0, 509.0),)))
reads = ReadSchedule(
    ("r0", "r1", "r2"), ("p0", "p0", "p1"), ("a", "a", "b"),
    ("direct",) * 3, np.array([[0.0, 0.02], [0.1, 0.12], [0.2, 0.22]]),
)
filters = {
    "a": FilterPassband(500.0, 2.0, "lorentzian", peak_transmission=0.9),
    "b": FilterPassband(504.0, 3.0, "supergaussian", peak_transmission=0.9, exponent=4.0),
}
preparation = FilterElectronResponse(
    grid, reads, filters,
    accepted_etendue_m2_sr=square_pixel_etendue(15e-6, 2.5, 3.0),
    quantum_efficiency=0.7,
    transmission=0.8 * ((wave >= 498.0) & (wave <= 506.0)),
)
response = preparation.coefficients()
signal = spectral_electrons(response, np.full(wave.size, 0.01))
assert signal.reads is reads and signal.total.shape == (3,)
assert np.all(signal.total > 0)
nominal, centre_derivative, width_derivative = preparation.fixed_area_filter_sensitivities()
assert centre_derivative.shape == width_derivative.shape == (3, wave.size)
np.testing.assert_array_equal(nominal.coefficients, response.coefficients)

# Explicit sum and mean maps retain the original native-read correspondence.
accumulation = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
mean_map = accumulation / accumulation.sum(axis=1, keepdims=True)
summed = accumulation @ signal.total
means = mean_map @ signal.total
np.testing.assert_allclose(summed, means * [2, 1])
# Illustrative independent shot noise plus 3-electron read noise on each actual read.
read_covariance = np.diag(signal.total + 9.0)
sum_covariance = accumulation @ read_covariance @ accumulation.T
mean_covariance = mean_map @ read_covariance @ mean_map.T
assert sum_covariance.shape == mean_covariance.shape == (2, 2)
```

## Physical owners and sample contracts

`FilterPassband` stores positive centre and FWHM in nm, a profile choice and peak intensity transmission in `[0,1]`. Super-Gaussian passbands require a positive finite full exponent. Lorentzian passbands use their fixed quadratic denominator and omit the exponent. Blocking remains part of the separately supplied optical transmission.

The preparation selects passbands through `reads.setting_ids`. `filter_ids` exposes first-read-occurrence order; every trial vector and profile-area vector follows that order. Missing settings fail. Reads retain their actual native order even when several reads select the same filter. Their duration is the difference of each supplied exposure interval; no chronology is synthesized.

Accepted etendue is a nonnegative scalar or `(read,)` array in m2 sr. `square_pixel_etendue(size_m, optical_f_number, detector_f_number)` supplies the retained approximation `pi*size_m^2/(4*max(f_numbers)^2)`. Its size is the active square width; using pixel pitch asserts full fill. The helper contributes no exposure, throughput or QE. Other acceptance models can supply their own etendue explicitly.

QE and other optical transmission are finite fractions in `[0,1]`, broadcastable from a scalar, `(spectral,)`, `(read,1)` or `(read,spectral)` array. QE can instead be a pure callback receiving original read indices and wavelengths in metres and returning exactly `(read,spectral)`. It is evaluated once during construction and is not retained. A supplied detector characterization can provide the QE values; no parallel detector configuration is created. The caller chooses the interpolation and outside-support policy of tabulated QE, for example with `tabulated_transmission`. QE must be supplied over all requested samples, including blocked samples.

Preparation retains immutable numerical snapshots and immutable passband declarations. Original mutable arrays or a later change to the supplied mapping do not change an existing preparation. Each evaluation owns independent scratch. `SpectralElectronResponse` retains the immutable read schedule, spectral grid and immutable `(read,spectral)` coefficients in electrons per W/m2/sr per declared spectral coordinate. `spectral_electrons` returns native `ExpectedElectrons`; detector noise and calibration remain separate.

## Profiles, areas and photon coefficients

1. **Objective:** prepare the common spectral and read factors. **Inputs:** the full declared `SpectralGrid`, physical QE/transmission, accepted etendue and read intervals. **Outputs:** photon weights `q[k]`, wavelength-area weights `a[k]` and base factors `B[r,k]`. **Processing:** use the shared endpoint-sampled trapezoidal rule independently inside each declared interval. Set `q` to coordinate weights divided by photon energy. For profile areas, convert a cm-1 grid to increasing wavelength, form nm weights on the same physical domain, then restore output sample order. Set `B[r,k]=QE[r,k]*T[r,k]*q[k]*etendue[r]*duration[r]`. Reject missing coverage, incomplete endpoint sampling and nonfinite represented factors.
2. **Objective:** evaluate unit-peak passband profiles. **Inputs:** wavelength samples `lambda[k]`, selected centres `c[j]`, FWHM `w[j]` and profile shape. **Outputs:** `P[j,k]` and unscaled sampled area `A[j]`. **Processing:** Lorentzian profiles use `P=1/(1+(2*(lambda-c)/w)^2)`. Super-Gaussian profiles use `P=exp(-log(2)*abs(2*(lambda-c)/w)^p)`. Powers at least two reuse `super_gaussian_filter` with `order=p/2`; smaller positive powers use the same explicit law. Set `A[j]=sum_k(a[k]*P[j,k])`. These areas exclude peak scaling, blocking, QE and photon conversion and retain only the sampled domain.
3. **Objective:** construct per-read electron coefficients. **Inputs:** base factors, selected profiles, peak transmission and optional positive fixed areas `F[j]`. **Outputs:** `K[r,k]`. **Processing:** if fixed areas are supplied, require positive sampled trial areas and multiply each profile by `F[j]/A[j]`. Let `j` be the filter selected by `reads.setting_ids[r]`; set `K[r,k]=B[r,k]*peak[j]*P[j,k]`. `coefficients(...)` returns this map as `SpectralElectronResponse`. `profile_areas_nm(...)` independently exposes the unscaled `(filter,)` area diagnostic.
4. **Objective:** evaluate fixed-area centre/FWHM sensitivities. **Inputs:** nominal filters and a positive finite nm step smaller than every centre and FWHM that representably changes each scalar in float64. **Outputs:** nominal response and two signed `(read,spectral)` derivative arrays per nm. **Processing:** evaluate nominal profiles and retain their sampled areas. Evaluate centre-plus, centre-minus, width-plus and width-minus profiles while fixing those nominal areas. Divide each paired coefficient difference by twice the step. All five evaluations share the same prepared QE, blocking, photon and duration factors. Each row differentiates its assigned filter; shared setting identities retain the same physical perturbation.

Fixed-area and fixed-peak perturbations define different derivatives. Fixed-area scaling can increase a finite trial's peak above its nominal value; the caller must establish physical feasibility before treating a finite normalized trial as a realizable passive filter. The normalization rule is retained for the stated sensitivity calculation.

## Coordinates and coaddition

Profiles are point functions of wavelength and receive no density Jacobian. Photon coefficients use the actual nm or cm-1 measure selected by the grid. Source radiance must use that same density measure. Reciprocity changes sample ordering and source-density values; each coordinate's trapezoidal discretization has its own convergence error. Declared spectral gaps are never bridged by area or photon quadrature.

A coadd is an explicit map from native reads to accumulated values. A sum and a normalized mean use different weights, as in the example. Propagate covariance and signed bias through the same map. The mapped values are not new detector readouts: combining charge does not reduce several independent read-noise contributions to one read or establish an on-chip accumulation model. Calibrated radiance requires the corresponding response normalization.

## Migration and evidence

`SpectralResponseParameters`, `WavelengthElectronResponse` and the old `spectral_response` module are removed. Import `FilterElectronResponse` and `SpectralElectronResponse` from `response.spectral`, `FilterPassband` from `optics.filters`, and the square-pixel helper from `optics.pixels`, or use their root exports. Replace `evaluate` with `coefficients`, obtain unblocked areas through `profile_areas_nm`, and use `fixed_area_filter_sensitivities` for the preserved derivative convention. Supply actual read intervals and explicit accumulation weights instead of implicit coadd counts. Qualified Python pickle compatibility is not provided.

Tests retain the previous per-nm analytic coefficient formula and profile-area/finite-difference checks, with old sums reproduced through explicit read accumulation. They also check unequal read durations, native order, one-time QE evaluation, mutable-input isolation, concurrent reuse, reciprocal-coordinate convergence and invalid physical inputs. These establish bounded numerical contracts; instrument accuracy, feasible filter designs, realistic-scale performance and retrieval quality remain unqualified.
