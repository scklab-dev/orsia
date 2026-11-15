# Orsia Inverse

`orsia-inverse` provides physics-independent estimation and uncertainty calculations for caller-defined models. It covers nonlinear Gaussian fitting, rank-aware linear inversion, local information and target uncertainty, mean-normalized spectral contrasts, shared spectral offsets, correlated scalar fields and compensated spatial detection. The package depends on NumPy, SciPy and Orsia Uncertainty. Callers supply physical predictions, coordinates, covariance models, masks and decision criteria.

## Architecture

![Inverse capabilities and their shared numerical contracts: nonlinear estimation and diagnostics use QuadraticPrior, the contrast module includes shared-offset estimation, and linear, field and detection methods expose independent entry points.](docs/diagrams/architecture.svg)

The application selects the required capabilities. Within the package, solid arrows point from a consumer to a shared capability or contract; the dashed arrow shows a `GaussianEstimate` passed explicitly by the caller to `retrieval_diagnostics`. Both nonlinear methods and local diagnostics use `QuadraticPrior`. The independent `linear.py` path returns rank and resolution diagnostics with its estimate. The outer arrows show application calls and package-level numerical dependencies. [Full-size SVG](docs/diagrams/architecture.svg) · [PNG](docs/diagrams/architecture.png) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

## Nonlinear estimation

`solve_map` and `solve_damped_gaussian` fit caller-defined predictions with fixed Gaussian covariance and an optional quadratic prior. Both accept either separate prediction/Jacobian callbacks or a fused `evaluate(q)` returning both. They return an immutable `GaussianEstimate` with a nested `FitTermination`; `retrieval_diagnostics` separately computes local ranks, posterior covariance, gain and averaging kernel.

`solve_map` offers SciPy TRF with bounds or LM with unbounded states and sufficient augmented residuals. `solve_damped_gaussian` supports unbounded states, diagonal damping, trial rejection and stationary-step checks. Both methods use the same half-squared Gaussian objective and prior. Numerical damping is excluded from posterior information.

Callers with explicit tangents can implement `jac(q)` as `orsia_numerics.tangents.jacobian(tangent, q, block_size=...)[1]`; the solver expects the derivative array alone. Numerics is an optional caller dependency for this composition. See the [tangent callback contract](../orsia-numerics/docs/tangents.md#python-contract) and the [fit rules](#fit-and-diagnostic-rules) for shapes, ownership and numerical limits.

### Minimal example

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
assert estimate.termination.success
np.testing.assert_allclose(estimate.state, [np.log(2.0)])
diagnostics = retrieval_diagnostics(estimate)
assert diagnostics.available, diagnostics.reason
np.testing.assert_allclose(diagnostics.posterior_covariance, [[0.05]])
print(round(estimate.state.item(), 6), round(diagnostics.posterior_covariance.item(), 6))
```

Expected output: `0.693147 0.05`. The variance is the local Gauss-Newton approximation in the fitted log coordinate.

### Fit and diagnostic rules

- `fun(q)` returns model values `(m,)`; `jac(q)` returns `dF/dq` `(m,n)`. Supply derivatives, priors and bounds in the actual fitted coordinates. Residual sign is `F(q)-y`.
- Numerical arrays are finite float64. Prepare observation weighting with `orsia_uncertainty.factor_covariance`. Supply an optional `QuadraticPrior(mean, precision_root)`, `prior_from_covariance(mean, factor)` or `prior_from_precision(mean, precision)`. Rectangular or rank-deficient roots provide partial regularization. Coordinates and units remain caller-owned.
- Callables must be deterministic and non-mutating; keep outputs stable during return/snapshot. Model exceptions propagate. Only `TrialDomainError` rejects a known unsupported trial in the damped method; malformed outputs raise `ValueError`. All initial-state errors propagate.
- Check `estimate.termination.success` separately from diagnostic `available` and `reason`. Failed termination, backend-active bounds or deficient augmented rank prevent a posterior; no truncated inverse is supplied.

## Rank-aware linear estimation

`solve_linear(matrix, observations, covariance=..., valid=..., rcond=..., max_bytes=...)` accepts a finite `(read,state)` model and supplied variances or full covariance. Missing observations can be excluded with a Boolean mask; correlations among retained observations are preserved. Covariance whitening uses Orsia Uncertainty. The explicit memory budget is checked before dense factorization.

The returned `LinearEstimate` contains the minimum-norm state, estimator sampling covariance, signed resolution matrix, gain, singular values, rank, condition, usable-read mask and coordinate identifiability. Rank-deficient systems have infinite condition. Null-space components follow the minimum-norm convention, and their true uncertainty is unconstrained. Zero sampling covariance in those directions describes the estimator convention rather than knowledge of the true state. The physical model and state scaling remain caller responsibilities. See the [joint instrument example](../orsia-instruments/docs/spectral-imagers.md#two-beam-separate-detectors-and-a-joint-inverse) for supplied electron covariance and retained native port order.

## Local information and target uncertainty

[`orsia_inverse.diagnostics`](src/orsia_inverse/diagnostics.py) exposes `local_gaussian_information(jacobian, observation_factor, prior=..., ...)` for a supplied unconstrained linearization. It shares the undamped information calculation and rank policy with `retrieval_diagnostics`. The caller establishes whether the local unconstrained model is appropriate.

`linear_target_uncertainty(design, observation_factor, target, prior=..., ...)`, in the same module, evaluates a linear quantity `target.T @ state`. It returns a `LinearTargetUncertainty` mapping containing the posterior covariance, measurement weights `target.T @ gain` and standard uncertainty `sqrt(target.T @ posterior @ target)`. A unit basis vector selects one state coordinate. Deficient augmented rank raises `numpy.linalg.LinAlgError`. See the [information and transport reference](docs/estimation.md#local-diagnostics-and-source-transport) for storage limits and the distinction between posterior and actual-source covariance.

Instrument channel selection and `ChannelSearchResult` belong to [`orsia_instruments.channels`](../orsia-instruments/src/orsia_instruments/channels.py). An application can use the target uncertainty in the evaluation callback supplied to `refine_channels`; Inverse operates on the resulting numerical model.

## Mean-normalized channel contrasts

`orsia_inverse.contrast.helmert_basis(channel_count)` returns a writable float64 `(channel, channel-1)` orthonormal zero-mean basis for at least three channels. Column `k-1` has `k` entries `1/sqrt(k*(k+1))`, followed by `-k/sqrt(k*(k+1))` and zeros. `normalized_contrast(signal, covariance=None)` returns contrasts of `signal / mean(signal)`, their derivative with respect to the original signal and optional first-order covariance transport. The signal must be a finite vector with positive mean. A supplied covariance must use the same channel order and squared signal units. The contrasts are dimensionless; the derivative has inverse signal units. Returned arrays are independent writable float64 buffers. The caller keeps borrowed inputs stable during each call.

Mean normalization removes common amplitude within this observable. The covariance transport describes its local linearization. The application retains its parameterization, damping, stopping rules and calibration procedure when using these contrasts in an estimator.

### Shared spectral offsets

The same [`orsia_inverse.contrast`](src/orsia_inverse/contrast.py) module estimates one additive correction in normalized contrasts shared by preselected spectra. `estimate_shared_contrast_offset` receives matching measurements and predictions `(pixel,band)` and conditional covariances `(pixel,band,band)`. It returns a `SharedContrastOffset` mapping with zero-mean relative channel correction, contrast correction, common contrast covariance and convergence diagnostics.

The application owns selection, two-pass processing and variance bookkeeping. The returned covariance represents a common error source across corrected spectra. Too few selected pixels return zero correction, NaN covariance and `converged=False`.

## Correlated scalar fields

[`orsia_inverse.spatial.solve_correlated_field`](src/orsia_inverse/spatial.py) receives selected scalar coordinates, a covariance action and diagonal, and a sparse regularization operator. `masked_second_difference_operator` constructs unscaled mask-contained second differences and selected C-order indices separately. Physical-coordinate conversion, masks and regularization strength remain caller choices.

The solver returns a `CorrelatedFieldEstimate` mapping with the estimated coordinates, data and curvature costs, and nested convergence diagnostics. It uses checked nested preconditioned conjugate gradients; failed inner covariance solves raise `ValueError`, while outer exhaustion returns `converged=False`.

## Compensated spatial detection

[`orsia_inverse.detection.scan_compensated_apertures`](src/orsia_inverse/detection.py) evaluates signed core-minus-annulus contrasts on a scalar field. The caller supplies the field, valid-sample mask and a covariance quadratic-form callback that receives selected flat C-order indices and signed aperture weights. The returned `ApertureContrast` mapping contains score, uncertainty and support maps, plus the strongest supported contrast and its zero-based row and column. Optional variance contributions can retain signed cross terms.

`aperture_scorer` prepares repeated score maps with fixed support and uncertainty. `calibrate_null_threshold` selects an order-statistic threshold from independent null calibration maxima; subsequent detection uses strict exceedance. Callers retain false-alarm targets, calibration/validation population separation and campaign reporting. The field and detection methods use numerical inputs without importing physical models or Instruments.

## Read next

- [Estimation and diagnostics reference](docs/estimation.md): module ownership, derivative checks, conditioning, bounds and uncertainty transport.

## Verification scope

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-inverse/tests` from the workspace root. Numerical success and synthetic regressions leave physical model validity, nonlinear statistical coverage and atmospheric retrieval performance unqualified.
