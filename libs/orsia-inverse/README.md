# Orsia Inverse

`orsia-inverse` fits caller-defined nonlinear predictions with fixed Gaussian covariance and optional priors and bounds. A separate diagnostic call computes local ranks, posterior covariance, gain and averaging kernel. `solve_linear` provides a bounded covariance-weighted minimum-norm inverse for explicit matrices, including rank-deficient systems. The package depends on NumPy, SciPy and Orsia Uncertainty; physical models stay with callers.

## Architecture

![The application calls estimation and explicitly passes the estimate to local diagnostics.](docs/diagrams/architecture.svg)

Solid arrows show calls or dependencies of the nonlinear path; the dashed arrow shows the estimate passed by the caller. Diagnostics are a separate operation. The linear path in `orsia_inverse.linear` returns its rank and resolution diagnostics with the estimate. [Full-size SVG](docs/diagrams/architecture.svg) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

## Minimal example

Run the example with this distribution and its dependencies installed, or in the [Orsia development environment](../../README.md#development-environment). Fit a dimensionless log-amplitude to two synthetic DN observations with unit diagonal variances.

```python
import numpy as np
from orsia_inverse import retrieval_diagnostics, solve_map
from orsia_uncertainty import factor_covariance


def fun(q):
    return np.array([np.exp(q[0]), 2.0 * np.exp(q[0])])


def jac(q):
    return fun(q)[:, None]  # (observations, state)


estimate = solve_map(
    fun,
    jac,
    np.array([2.0, 4.0]),
    np.zeros(1),
    observation_factor=factor_covariance(np.ones(2), method="cholesky"),
)
assert estimate.success
np.testing.assert_allclose(estimate.state, [np.log(2.0)])
diagnostics = retrieval_diagnostics(estimate)
assert diagnostics.available, diagnostics.reason
np.testing.assert_allclose(diagnostics.posterior_covariance, [[0.05]])
print(round(estimate.state.item(), 6), round(diagnostics.posterior_covariance.item(), 6))
```

Expected output: `0.693147 0.05`. The variance is the local Gauss-Newton approximation in the fitted log coordinate.

## Fit and diagnostic rules

- `fun(q)` returns model values `(m,)`; `jac(q)` returns `dF/dq` `(m,n)`. Supply derivatives, priors and bounds in the actual fitted coordinates. Residual sign is `F(q)-y`.
- Numerical arrays are finite float64. Prepare observation and optional prior covariances with `orsia_uncertainty.factor_covariance`. The solve retains those immutable factors; coordinate ordering and units remain caller-owned.
- Callables must be deterministic and non-mutating; keep their outputs stable during return/snapshot. Model exceptions propagate, and invalid model values raise `ValueError`.
- Check `success` separately from diagnostic `available` and `reason`. Failed termination, backend-active bounds or deficient augmented rank prevent a posterior; no truncated inverse is supplied.

## Rank-aware linear estimation

`solve_linear(matrix, observations, covariance=..., valid=..., rcond=..., max_bytes=...)` accepts a finite `(read,state)` model and supplied variances or full covariance. Missing observations can be excluded with a Boolean mask; correlations among retained observations are preserved. Covariance whitening uses Orsia Uncertainty. The explicit memory budget is checked before dense factorization.

The returned `LinearEstimate` contains the minimum-norm state, estimator sampling covariance, signed resolution matrix, gain, singular values, rank, condition, usable-read mask and coordinate identifiability. Rank-deficient systems have infinite condition. Null-space components follow the minimum-norm convention, and their true uncertainty is unconstrained. Zero sampling covariance in those directions describes the estimator convention rather than knowledge of the true state. The physical model and state scaling remain caller responsibilities. See the [joint instrument example](../orsia-instruments/docs/spectral-imagers.md#two-beam-separate-detectors-and-a-joint-inverse) for supplied electron covariance and retained native port order.

## Read next

- [Estimation and diagnostics reference](docs/estimation.md): module ownership, derivative checks, conditioning, bounds and uncertainty transport.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-inverse/tests` from the workspace root. Numerical success and synthetic regressions leave physical model validity, nonlinear statistical coverage and atmospheric retrieval performance unqualified.
