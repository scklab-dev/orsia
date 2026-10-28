# Orsia Inverse

Start with the [code guide, architecture diagram and progressive examples](src/orsia_inverse/README.md). This page retains the estimation and diagnostic contracts.

Physics-independent, fixed-covariance nonlinear Gaussian estimation for small to moderate dense states. `solve_map` consumes separate model-value (`fun`) and Jacobian (`jac`) callables; `retrieval_diagnostics` computes local information separately. All numerical inputs and outputs use NumPy arrays, without covariance or layout wrappers. Runtime dependencies are only NumPy and SciPy; no other Orsia package, physical model or provider is imported.

## Numerical contract

`solve_map(fun, jac, observations, initial_state, *, observation_covariance, prior_mean=None, prior_covariance=None, lower_bounds=None, upper_bounds=None, method="trf", ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=1000)` minimises `0.5*(F(q)-y).T Se^-1 (F(q)-y) + 0.5*(q-qa).T Sa^-1 (q-qa)`. Omit both prior arguments to remove the second term. These covariances are copied and fixed during the solve; frozen reference weighting is not a state-dependent Gaussian likelihood. No robust loss, covariance fitting, silent regularisation, pseudoinverse or automatic solver fallback is offered.

- `q` and `initial_state` are float64 `(n,)`; `fun(q)` returns model values `F(q)` as float64 `(m,)`, not residuals; `jac(q)` returns only `dF/dq` as float64 `(m,n)`, not a tuple. `solve_map` constructs the covariance-weighted residuals and prior term internally. Inputs and outputs must be finite. Jacobian columns differentiate the actual dimensionless retrieval coordinates, including caller-supplied affine, log or basis transformations. Analytic Jacobians are the normal route; callers may supply a numerically differentiated callable explicitly.
- `observations` and `initial_state` must be nonempty vectors. Observation covariance is a finite float64 array: `(m,)` contains diagonal **variances**, not standard deviations, and `(m,m)` contains a full covariance. Prior covariance follows the same convention with `n`. Covariances must be positive definite with minimum/maximum eigenvalue ratio strictly greater than `1e-12`. Dense symmetry is checked to `1e-12` of the largest absolute entry; accepted roundoff asymmetry is averaged for factorisation. Diagonal weighting retains O(m) storage; dense weighting uses a lower Cholesky factor. No dense observation whitening matrix is constructed.
- Array order is the contract: observation covariance rows/columns and Jacobian rows follow `observations`; prior covariance rows/columns, prior mean, bounds and Jacobian columns follow `initial_state`. The caller must supply consistent units: observation covariance entry `(i,j)` has units `unit(y[i])*unit(y[j])`; prior covariance is in squared dimensionless retrieval coordinates. No metadata, unit checking, reordering or conversion is performed.
- The caller defines parameter identities, physical scales and transformations. For `x=xref+Dq`, supply `Kq=Kx D`, transform the prior and bounds consistently, and transport output uncertainty with `D`. A Gaussian prior in `q` need not be Gaussian in physical coordinates. No normalization by the prior is inferred.
- Bounds are float64 `(n,)` arrays. Each side may be infinite and each coordinate independently unbounded, one-sided or two-sided. Omitted sides are infinite. NaNs, equal/reversed bounds and initial states outside bounds are rejected. Remove fixed parameters rather than using equal bounds. `method="lm"` is explicit and requires all bounds infinite and at least `n` augmented residuals. TRF is the default, not a measured performance winner.
- The backend is SciPy `least_squares` with `loss="linear"` and `x_scale=1.0`. All tolerances must be finite and above machine epsilon. `max_nfev` is positive and limits backend residual evaluations, not total physical calls. No iteration callback is exposed in this initial contract.

The callable contract is deterministic and non-mutating within each solve. Inputs passed to callables are independent writable copies, which callables must not modify; outputs must be stable and must not be concurrently mutated or reused by another caller. The adapter snapshots returned arrays and retains only its last exactly equal state, never an approximate cache key or a previous-state Jacobian. Trial states use `fun` without derivatives; `jac` is evaluated only when derivatives are requested and not already cached at that state. Model exceptions propagate; invalid shapes, nonfinite outputs and nonfinite residual cost raise `ValueError`, with no artificial penalty. Derivative correctness remains the caller's responsibility; the adapter does not compare `jac` with numerical derivatives.

Migration from the former interface: replace `predict=` with `fun=` and `linearize=` with `jac=`, and change the second callable from returning `(prediction, jacobian)` to returning the Jacobian alone. Positional callers must also change that return value. The former keywords and callable type aliases `Prediction` and `Linearization` are no longer provided. There is no longer a second prediction to compare for agreement.

The array-only migration also removes `observation_layout`, `state_layout` and `observation_unit_squared`; these keywords are not accepted. Pass covariance matrices or variance vectors directly and read `posterior_covariance` as an array, without `.matrix`. `GaussianEstimate` no longer has a `state_layout` field. No compatibility wrappers are retained.

`GaussianEstimate` represents the estimated dimensionless state, final prediction/residual/Jacobian, fixed covariances and prior mean, bounds and numerical termination. Returned floating arrays are writable copies, independent of caller and SciPy work arrays. The integer `active_mask` retains its separate bytes-backed read-only storage. Residual sign is `F(q)-y`; measurement and prior costs include the factor one half. `success`, `status`, `message`, `optimality`, `nfev` and `njev` report the backend outcome, not scientific validity. Budget exhaustion returns `success=False` with the final estimate and linearisation. The existing fields `prediction_calls` and `linearization_calls` count actual adapter calls to `fun` and `jac`, respectively, including final evaluation; they need not equal backend counts and exclude any calls made internally by the supplied functions. Accepted/rejected step counts are `None`, not zero or estimates from evaluation counts. This is an explicit limitation relative to the parent target design's step-telemetry requirement. No performance comparison between LM and TRF is implied.

## Local diagnostics and source transport

`retrieval_diagnostics(estimate, *, include_gain=False, include_averaging_kernel=False, rank_rtol=1e-12)` uses the retained final Jacobian and the exact assumed covariances from the solve. `RetrievalDiagnostics` represents local state information and measurement sensitivity, not an additional fit. It reports ranks and condition numbers of the measurement-whitened Jacobian and prior-augmented Jacobian separately. These condition numbers are for the Jacobians, not squared normal matrices; rank deficiency gives infinite condition. `rank_rtol` is a finite relative singular-value threshold strictly between zero and one.

The posterior is the undamped Gauss-Newton approximation `(K.T Se^-1 K + Sa^-1)^-1`, computed by SVD of the augmented whitened Jacobian without forming its normal matrix. No trust-region damping is counted as prior information. Gain `G=Spost K.T Se^-1` (shape `(n,m)`) and averaging kernel `A=G K` (shape `(n,n)`) are retained only on request. `degrees_of_freedom=trace(A)` is computed even without retaining `A`. Posterior covariance is a writable float64 `(n,n)` array in state order and squared dimensionless retrieval coordinates; gain is in retrieval-coordinate units per observation unit, and averaging kernels are dimensionless in retrieval coordinates, not automatically vertical kernels. Estimate arrays, including covariance snapshots, remain writable; do not modify them before diagnostics if the original fitted weighting and linearisation are required.

If numerical termination failed, a bound is active according to the backend, or the augmented Jacobian has deficient rank at the selected tolerance, `available=False` and `reason` explains why; posterior, gain, averaging kernel and degrees of freedom are `None`, while ranks and conditions remain available. There is no silent truncated inverse. Backend-active bounds are a numerical indicator, not a distance-based safety guarantee: inspect the retained bounds and use perturbation/ensemble checks near inactive bounds and in strongly nonlinear cases. Numerical decomposition errors propagate instead of producing diagnostics. This API does not estimate a constrained posterior or prove statistical coverage.

The assumed posterior is distinct from propagated measurement error. With a requested gain `G`, compute `G @ actual_covariance @ G.T` for a dense actual error law, `(G * actual_variances) @ G.T` for diagonal variances, and `G @ bias` for signed measurement bias. Keep contributions and cross terms separate when attribution matters; actual covariance need not equal the weighting covariance. A prior-mean error propagates locally with `I-A` under the stated residual/state convention. No actual-error law, cross-correlation or physical bias is inferred.

## Runnable example

```python
import numpy as np
from orsia_inverse import solve_map, retrieval_diagnostics


def fun(q):
    return np.array([np.exp(q[0]), 2.0 * np.exp(q[0])])


def jac(q):
    return fun(q)[:, None]


estimate = solve_map(
    fun,
    jac,
    np.array([2.0, 4.0]),
    np.zeros(1),
    observation_covariance=np.ones(2),  # diagonal variances in DN^2
)
assert estimate.success
np.testing.assert_allclose(estimate.state, [np.log(2.0)])
diagnostics = retrieval_diagnostics(estimate, include_gain=True, include_averaging_kernel=True)
assert diagnostics.available
np.testing.assert_allclose(diagnostics.posterior_covariance, [[0.05]])
np.testing.assert_allclose(diagnostics.averaging_kernel, [[1.0]])
```

## Verification scope

Run `devbox run -- just test-inverse` for focused regressions, `devbox run -- just check` for workspace gates, and `devbox run -- just clean-install` for independent wheels. Tests cover independent affine references, dense/diagonal whitening, nonlinear derivative checks and multiple starts, affine coordinate transforms, mixed/active bounds, weak information with/without priors, actual-source and signed-bias transport, seeded perturbation/Monte Carlo cases, invalid models, exhaustion and independent writable ownership. Documentation examples are executable tests. These are bounded synthetic checks, not atmospheric retrieval qualification, a realistic solar/thermal forward coupling, a throughput comparison or concurrent-native safety qualification.
