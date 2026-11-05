# Gaussian estimation and diagnostics

This reference specifies fitted coordinates, fixed covariance weighting and the conditions for local diagnostics. See the [package introduction](../README.md) for a complete fit.

## Source responsibilities

| Source | Responsibility | Boundary |
| --- | --- | --- |
| `estimation.py` | `solve_map`, callable adaptation and `GaussianEstimate` | Fixed observation/prior covariance, explicit bounds and deterministic prediction/Jacobian callables. |
| `diagnostics.py` | `retrieval_diagnostics` and `RetrievalDiagnostics` | Local undamped Gauss-Newton information; no second fit or constrained posterior. |
| Orsia Uncertainty | Prepare covariance factors, whiten, solve and sample | Consumers import these capabilities directly from their owner. |
| NumPy and SciPy | Arrays, Jacobian SVD and least squares | Physical models and file handling stay with callers. |

Estimation and diagnostics depend on Orsia Uncertainty for covariance weighting and retain their own Jacobian SVD policy. Neither fit nor diagnostics selects a physical forward model. `retrieval_diagnostics` consumes the retained final estimate only when called explicitly.

## Numerical contract

`solve_map(fun, jac, observations, initial_state, *, observation_factor, prior_mean=None, prior_factor=None, lower_bounds=None, upper_bounds=None, method="trf", ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=1000)` minimises `0.5*(F(q)-y).T Se^-1 (F(q)-y) + 0.5*(q-qa).T Sa^-1 (q-qa)`. Omit both prior arguments to remove the second term. The supplied immutable `CovarianceFactor` objects remain fixed during the solve; frozen reference weighting is not a state-dependent Gaussian likelihood. No robust loss, covariance fitting, silent regularisation, pseudoinverse or automatic solver fallback is offered.

- `q` and `initial_state` are float64 `(n,)`; `fun(q)` returns model values `F(q)` as float64 `(m,)`, not residuals; `jac(q)` returns only `dF/dq` as float64 `(m,n)`, not a tuple. `solve_map` constructs the covariance-weighted residuals and prior term internally. Inputs and outputs must be finite. Jacobian columns differentiate the actual dimensionless retrieval coordinates, including caller-supplied affine, log or basis transformations. Analytic Jacobians are the normal route; callers may supply a numerically differentiated callable explicitly.
- `observations` and `initial_state` must be nonempty vectors. Observation and prior factors have sizes `m` and `n`, respectively. Prepare finite float64 variance vectors or dense covariances with `orsia_uncertainty.factor_covariance(C, method=...)`. Inverse operations require minimum/maximum eigenvalue ratio strictly greater than `1e-12`. Diagonal factors retain linear storage. Cholesky and spectral factors share the objective while their whitened components can differ. No dense observation whitening matrix is constructed.
- Array order is the contract: observation covariance rows/columns and Jacobian rows follow `observations`; prior covariance rows/columns, prior mean, bounds and Jacobian columns follow `initial_state`. The caller must supply consistent units: observation covariance entry `(i,j)` has units `unit(y[i])*unit(y[j])`; prior covariance is in squared dimensionless retrieval coordinates. No metadata, unit checking, reordering or conversion is performed.
- The caller defines parameter identities, physical scales and transformations. For `x=xref+Dq`, supply `Kq=Kx D`, transform the prior and bounds consistently, and transport output uncertainty with `D`. A Gaussian prior in `q` need not be Gaussian in physical coordinates. No normalization by the prior is inferred.
- Bounds are float64 `(n,)` arrays. Each side may be infinite and each coordinate independently unbounded, one-sided or two-sided. Omitted sides are infinite. NaNs, equal/reversed bounds and initial states outside bounds are rejected. Remove fixed parameters rather than using equal bounds. `method="lm"` is explicit and requires all bounds infinite and at least `n` augmented residuals. TRF is the default; relative performance has not been established.
- The backend is SciPy `least_squares` with `loss="linear"` and `x_scale=1.0`. All tolerances must be finite and above machine epsilon. `max_nfev` is positive and limits backend residual evaluations, not total physical calls. No iteration callback is exposed in this initial contract.

The callable contract is deterministic and non-mutating within each solve. Inputs passed to callables are independent writable copies, which callables must not modify; outputs must be stable and must not be concurrently mutated or reused by another caller. The adapter snapshots returned arrays and retains only its last exactly equal state, never an approximate cache key or a previous-state Jacobian. Trial states use `fun` without derivatives; `jac` is evaluated only when derivatives are requested and not already cached at that state. Model exceptions propagate; invalid shapes, nonfinite outputs and nonfinite residual cost raise `ValueError`, with no artificial penalty. Derivative correctness remains the caller's responsibility; the adapter does not compare `jac` with numerical derivatives.
`GaussianEstimate` represents the estimated dimensionless state, final prediction/residual/Jacobian, the exact immutable factors used for fitting and prior mean, bounds and numerical termination. Returned floating arrays are writable copies, independent of caller and SciPy work arrays. The integer `active_mask` retains its separate bytes-backed read-only storage. Residual sign is `F(q)-y`; measurement and prior costs include the factor one half. `success`, `status`, `message`, `optimality`, `nfev` and `njev` report the backend outcome, not scientific validity. Budget exhaustion returns `success=False` with the final estimate and linearisation. The existing fields `prediction_calls` and `linearization_calls` count actual adapter calls to `fun` and `jac`, respectively, including final evaluation; they need not equal backend counts and exclude any calls made internally by the supplied functions. Accepted/rejected step counts are `None`, not zero or estimates from evaluation counts. No performance comparison between LM and TRF is implied.

## Check a derivative in fitted coordinates

Run this standalone example in the [development environment](../../../README.md#development-environment). If a physical amplitude is `exp(q)`, the derivative must include that transformation.

```python
import numpy as np


def fun(q):
    return np.array([np.exp(q[0]), 2.0 * np.exp(q[0])])


def jac(q):
    return fun(q)[:, None]


q = np.array([np.log(2.0)])
step = np.array([1e-5])
finite_difference = (fun(q + step) - fun(q - step)) / (2 * step[0])
np.testing.assert_allclose(jac(q)[:, 0], finite_difference, rtol=1e-9)
print(jac(q).ravel().tolist())
```

Expected output: `[2.0, 4.0]`. This checks one synthetic derivative at one state; the solver does not verify derivatives automatically.

## Local diagnostics and source transport

`retrieval_diagnostics(estimate, *, include_gain=False, include_averaging_kernel=False, rank_rtol=1e-12)` uses the retained final Jacobian and reuses the exact covariance factors from the solve without refactorization. `RetrievalDiagnostics` represents local state information and measurement sensitivity. It reports ranks and condition numbers of the measurement-whitened Jacobian and prior-augmented Jacobian separately. These condition numbers are for the Jacobians, not squared normal matrices; rank deficiency gives infinite condition. `rank_rtol` is a finite relative singular-value threshold strictly between zero and one.

The posterior is the undamped Gauss-Newton approximation `(K.T Se^-1 K + Sa^-1)^-1`, computed by SVD of the augmented whitened Jacobian without forming its normal matrix. No trust-region damping is counted as prior information. Gain `G=Spost K.T Se^-1` (shape `(n,m)`) and averaging kernel `A=G K` (shape `(n,n)`) are retained only on request. `degrees_of_freedom=trace(A)` is computed even without retaining `A`. Posterior covariance is a writable float64 `(n,n)` array in state order and squared dimensionless retrieval coordinates; gain is in retrieval-coordinate units per observation unit, and averaging kernels are dimensionless in retrieval coordinates, not automatically vertical kernels. Estimate arrays, remain writable while covariance factors are immutable; keep the final state/Jacobian stable before diagnostics if the original fitted weighting and linearisation are required.

If numerical termination failed, a bound is active according to the backend, or the augmented Jacobian has deficient rank at the selected tolerance, `available=False` and `reason` explains why; posterior, gain, averaging kernel and degrees of freedom are `None`, while ranks and conditions remain available. There is no silent truncated inverse. Backend-active bounds are a numerical indicator, not a distance-based safety guarantee: inspect the retained bounds and use perturbation/ensemble checks near inactive bounds and in strongly nonlinear cases. Numerical decomposition errors propagate instead of producing diagnostics. This API does not estimate a constrained posterior or prove statistical coverage.

The assumed posterior is distinct from propagated measurement error. With a requested gain `G`, compute `G @ actual_covariance @ G.T` for a dense actual error law, `(G * actual_variances) @ G.T` for diagonal variances, and `G @ bias` for signed measurement bias. Keep contributions and cross terms separate when attribution matters; actual covariance need not equal the weighting covariance. A prior-mean error propagates locally with `I-A` under the stated residual/state convention. No actual-error law, cross-correlation or physical bias is inferred.
## Verification scope

Run `devbox run -- just test-inverse` for focused regressions and `devbox run -- just check` for workspace gates. Tests cover independent affine references, dense/diagonal whitening, nonlinear derivative checks and multiple starts, affine coordinate transforms, mixed/active bounds, weak information with/without priors, actual-source and signed-bias transport, seeded perturbation/Monte Carlo cases, invalid models, exhaustion and independent writable ownership. Documentation examples are executable tests. A physical retrieval needs separate model and uncertainty validation. These are bounded synthetic checks, not atmospheric retrieval qualification, a realistic solar/thermal forward coupling, a throughput comparison or concurrent-native safety qualification.
