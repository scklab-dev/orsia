# Spectral extinction means

The functions in [`transfer.extinction`](../src/orsia_radiative/transfer/extinction.py) evaluate supplied spectral source radiance under explicit vertical optical depths, air masses and averaging weights. The source carries its declared density coordinate and units unchanged. Read grouping, response integration, surface parameterization and retrieval coordinates belong to the caller.

## Inputs and outputs

`source_radiance` and `optical_depth` are aligned nonempty float64 vectors `(spectral,)`. Source values and dimensionless depths are finite and nonnegative. `air_masses` and `averaging_weights` use `(samples,)`; masses are positive, weights are nonnegative and at least one weight is positive. Operations borrow stable inputs and return fresh writable arrays. Invalid physical domains raise `ValueError`.

`exposure_mean_radiance` returns a spectral mean. `exposure_mean_radiance_tangent` also returns signed `(spectral,direction)` derivatives from explicit source and optical-depth seeds. Geometry, illumination choices, masses and weights remain fixed under differentiation. Zero direction count is supported.

`linearize_exposure_mean_radiance` returns the exact reference mean, a log-source intercept and an effective air mass. Holding the last two outputs fixed defines a local approximation away from the reference depth. It reproduces the reference value and first depth derivative. Spectral integration of any output is a separate operation.

## Numerical laws

1. **Objective:** evaluate the weighted extinction mean. **Inputs:** source `S[k]`, vertical depth `tau[k]`, masses `m[i]` and weights `w[i]`. **Outputs:** mean `L[k]`, log mean `ell[k]` and effective mass `m_eff[k]`. **Processing:** omit zero-weight samples and retain scaled log weights `a[i]=log(w[i])-max(log(w))`. Compute `u[i,k]=a[i]-m[i]*tau[k]`, `z[k]=logsumexp_i(u[i,k])`, `ell[k]=log(S[k])+z[k]-logsumexp_i(a[i])`, `L[k]=exp(ell[k])` and `m_eff[k]=sum_i(exp(u[i,k]-z[k])*m[i])`. If all attenuation products overflow at a sample, the transmission and representable derivative are zero; use `m_eff[k]=0`. A zero source gives `ell[k]=-inf` and zero radiance.
2. **Objective:** propagate supplied physical directions. **Inputs:** the preceding mean, mean transmission `T[k]` obtained with unit source, and signed seeds `dS[k,p]`, `dtau[k,p]`. **Outputs:** `dL[k,p]`. **Processing:** calculate `dL[k,p]=T[k]*dS[k,p]-L[k]*m_eff[k]*dtau[k,p]`. This formula remains defined at zero source. Evaluate the two signed products in log magnitude, combine them with signed `logsumexp`, then exponentiate the final magnitude. This retains representable derivatives when transmission alone underflows. Reject nonfinite seeds and derivatives that exceed finite float64 support; near cancellation still has floating-point sensitivity.
3. **Objective:** prepare a local log-tangent model. **Inputs:** the mean and effective mass at reference depth `tau_ref[k]`. **Outputs:** reference radiance, intercept `b[k]` and `m_eff[k]`. **Processing:** set `b[k]=ell[k]+m_eff[k]*tau_ref[k]`. The caller can evaluate the approximation `L_local[k]=exp(b[k]-m_eff[k]*tau[k])`. Reject NaN or positive-infinite intercepts. Negative-infinite intercepts encode zero reference radiance.

The separate mean-air-mass law uses one sample `m_bar=sum(w*m)/sum(w)` with unit weight. In general, `S*exp(-m_bar*tau)` differs from the weighted exposure mean. Tests compare both laws with independent exponential formulas and finite differences; off-reference log-tangent errors remain explicit.

## Migration

The mixed `AbsorptionSlopeRadiance` callback and `coadded_solar_response` API are removed. For the former two-coordinate example, the caller constructs `tau=tau_total+expm1(q[0])*tau_absorber` and `S=(mu/pi)*solar*(1+q[1]*x)`, supplies their physical seeds, then contracts returned spectra with its normalized band response. The caller validates its surface law and chooses its fitted coordinates.

For exposure groups, select masses and averaging weights before calling the spectral law. Equal weights preserve the former logarithmic summation order. The historical Scene band renderer survives only as a labelled numerical reference in Scene tests. Production `DirectSolarScene` evaluates individual physical directions before instrument-owned averaging. No accuracy claim for realistic coaddition, atmospheric spectroscopy or retrieval follows from these software checks.
