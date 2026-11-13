# Gaussian estimation and diagnostics

This reference specifies fitted coordinates, fixed covariance weighting and the conditions for local diagnostics. See the [package introduction](../README.md) for a complete fit.

## Source responsibilities

| Source | Responsibility | Boundary |
| --- | --- | --- |
| `nonlinear/estimation.py` | Shared model adaptation, `GaussianEstimate`, `FitTermination`, `TrialDomainError` | Immutable terminal state and explicit numerical termination. |
| `nonlinear/map.py`, `nonlinear/damped.py` | Distinct nonlinear algorithms | Common half-squared objective, fixed weighting and optional prior. |
| `priors.py` | `QuadraticPrior` and covariance/precision constructors | Explicit mean and rectangular precision root. |
| `diagnostics.py` | `retrieval_diagnostics` and `RetrievalDiagnostics` | Local undamped Gauss-Newton information; no second fit or constrained posterior. |
| Orsia Uncertainty | Prepare covariance factors, whiten, solve and sample | Consumers import these capabilities directly from their owner. |
| NumPy and SciPy | Arrays, Jacobian SVD and least squares | Physical models and file handling stay with callers. |

Estimation and diagnostics depend on Orsia Uncertainty for covariance weighting and retain their own Jacobian SVD policy. Neither fit nor diagnostics selects a physical forward model. `retrieval_diagnostics` consumes the retained final estimate only when called explicitly.

## Numerical contract

Both nonlinear solvers minimize `0.5 * ||We (F(q)-y)||² + 0.5 * ||Rp (q-qa)||²`. `We` is supplied by an immutable observation covariance factor. `QuadraticPrior` retains `qa` and `Rp`. Its rectangular precision root can regularize selected directions; a rank-deficient penalty need not define a proper probability distribution. The numerical algorithm introduces no hidden state transformation or prior.

Choose exactly one model form:

- `solve_map(fun, jac, observations, initial_state, observation_factor=..., prior=...)` and the matching `solve_damped_gaussian` signature use separate prediction and derivative calls.
- `solve_map(observations=..., initial_state=..., evaluate=value_and_jacobian, observation_factor=..., prior=...)` and the matching damped signature use fused calls.

`q` has shape `(n,)`, observations and predictions `(m,)`, and Jacobians `(m,n)`. Required arrays are finite float64. The caller defines fitted coordinate units and order; observation covariance uses the product units of its rows and columns. Jacobian columns, prior mean/root and bounds follow the same state order. Derivatives include any caller-selected log, affine or basis transformation.

The observation factor must admit Uncertainty's inverse operations: its minimum/maximum eigenvalue ratio must exceed `1e-12`. `prior_from_covariance(mean, factor)` uses the existing whitening operation. `prior_from_precision(mean, precision)` accepts positive-semidefinite diagonal or dense precision using Uncertainty's scale-relative symmetry and PSD tolerance. Both constructors return an owned precision root. Direct `QuadraticPrior(mean, root)` construction supports zero penalty rows and partial regularization.

The per-solve adapter copies observations, initial state, bounds and callback outputs. It caches only its last exactly equal state. Callbacks receive independent writable state copies and must leave them unchanged. Keep output buffers stable until they are snapshotted. Separate prediction calls permit value-only trials; a fused call always computes both outputs. Returned estimate arrays and prior buffers have immutable storage. Use `.copy()` for writable results.

### Numerical methods and failure behavior

`solve_map` uses SciPy `least_squares`, `loss="linear"` and `x_scale=1.0`. TRF supports independent finite or infinite bounds. LM rejects finite bounds and requires at least as many augmented residuals as states. Bounds have shape `(n,)`; NaNs, equal/reversed bounds and infeasible initial states fail. Positive `ftol`, `xtol` and `gtol` must exceed machine epsilon. `max_nfev` limits backend residual calls; total model calls can differ.

`solve_damped_gaussian` supports unbounded states, diagonal damping, decreasing-cost trials and undamped stationary-step checks. `max_iterations` limits attempted outer iterations. Exhaustion and failed trial search retain an unsuccessful estimate. The default `cost_tolerance=5e-11` uses the half-squared objective, matching the former full-square threshold `1e-10`. Numerical damping is excluded from posterior information.

An application adapter may raise `TrialDomainError` for a known unsupported trial. The damped solver rejects that trial and increases damping. Invalid initial states always fail; MAP also propagates trial-domain errors. Other exceptions propagate from either method, including `ValueError` for invalid shapes, nonfinite outputs or cost overflow. Contract errors never become artificial residual penalties.

`GaussianEstimate` stores state, prediction, `F(q)-y` residual, final Jacobian, observation factor, optional prior, bounds/activity and separate half-squared costs. `objective` derives their sum. Its `termination` stores method, success, reason, optional backend status/counts, optimality and actual prediction/linearization call counts. A fused callback increments both actual counts once. Damped termination reports attempted iterations and accepted/rejected steps; unavailable backend counts are `None`. MAP retains backend counts; unavailable step counts are `None`.

## Switch nonlinear methods with one model

This example uses the same fused model, nonzero prior mean and observation factor with both solvers.

```python
import numpy as np
from orsia_inverse import QuadraticPrior, retrieval_diagnostics, solve_damped_gaussian, solve_map
from orsia_uncertainty import factor_covariance

matrix = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 1.0]])


def evaluate(state):
    return matrix @ state, matrix


arguments = dict(
    observations=np.array([1.0, -2.0, 0.5]),
    initial_state=np.zeros(2),
    evaluate=evaluate,
    observation_factor=factor_covariance(np.ones(3), method="cholesky"),
    prior=QuadraticPrior(np.array([0.4, -0.3]), np.array([[0.0, 0.5]])),
)
first = solve_map(**arguments, method="trf")
second = solve_damped_gaussian(**arguments)
assert first.termination.success and second.termination.success
np.testing.assert_allclose(first.state, second.state, atol=1e-8)
assert retrieval_diagnostics(first).available
assert retrieval_diagnostics(second).available
```

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

`retrieval_diagnostics(estimate, include_gain=False, include_averaging_kernel=False, rank_rtol=1e-12, max_bytes=256*1024*1024)` accepts either solver's immutable estimate. `local_gaussian_information(jacobian, observation_factor, prior=..., ...)` exposes the same numerical calculation for a supplied unconstrained local model. It reports ranks and condition numbers of measurement-whitened and prior-augmented Jacobians separately. `rank_rtol` is strictly between zero and one; deficient rank gives infinite condition.

SVD of `[We K; Rp]` yields the undamped posterior `(K.T Se^-1 K + Rp.T Rp)^-1`. The optional gain `G=Spost K.T Se^-1` has shape `(n,m)`; the averaging kernel `A=G K` has shape `(n,n)`. `degrees_of_freedom=trace(A)` is available even when the full kernel is omitted. These returned diagnostic arrays are fresh writable float64 buffers. Their state and observation axes follow the supplied problem.

Failed numerical termination, active bounds or deficient augmented rank leave posterior/gain/kernel unavailable with an explicit reason. Ranks remain available. No truncated inverse or damping replacement is supplied. The direct information function assumes the caller has established applicability of an unconstrained model. `max_bytes` checks a conservative estimate of dense array storage before whitening and SVD; native factorization scratch remains additional storage. An exceeded budget raises `MemoryError`.

`linear_target_uncertainty(design, observation_factor, target, prior=..., ...)` uses this shared information calculation. It returns `LinearTargetUncertainty` with posterior covariance, `target.T @ gain` and `sqrt(target.T @ posterior @ target)`. Supply a unit basis vector for a coordinate or explicit coefficients for a derived linear quantity. Rank failure raises `numpy.linalg.LinAlgError`; discrete channel search can reject such candidates explicitly.

The assumed posterior differs from actual-source sampling covariance. Given local gain `G`, propagate actual error with `G @ C_actual @ G.T` and signed bias with `G @ bias`. Preserve cross terms between correlated sources. A prior-mean error propagates locally with `I-A`. Neither numerical success nor a local Gaussian approximation establishes nonlinear coverage, physical validity or constrained posterior uncertainty.

## Verification scope

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-inverse/tests` for focused regressions and `devbox run -- just check` for workspace gates. Tests cover independent affine references, dense/diagonal whitening, nonlinear derivative checks and multiple starts, affine coordinate transforms, mixed/active bounds, weak information with/without priors, actual-source and signed-bias transport, seeded perturbation/Monte Carlo cases, invalid models, exhaustion and immutable estimate ownership, fused/separate callback parity, partial priors, explicit trial rejection and dense information budgets. Documentation examples are executable tests. A physical retrieval needs separate model and uncertainty validation. These are bounded synthetic checks, not atmospheric retrieval qualification, a realistic solar/thermal forward coupling, a throughput comparison or concurrent-native safety qualification.
