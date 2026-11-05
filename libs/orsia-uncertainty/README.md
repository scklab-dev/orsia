# Orsia Uncertainty

`orsia-uncertainty` provides covariance algebra, signed-bias transport, explicit source cross-covariances and Gaussian ensembles. NumPy arrays carry the values; callers define coordinates, units, physical error laws and random streams.

## Architecture

![Applications supply semantics to covariance algebra and source/ensemble operations.](docs/diagrams/architecture.svg)

Arrows show calls or dependencies. Either computational module can be used directly. [Full-size SVG](docs/diagrams/architecture.svg) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

## Minimal example

Run the example with this distribution and its dependencies installed, or in the [Orsia development environment](../../README.md#development-environment). Two synthetic DN reads have correlated random errors and a separate signed bias. Their average uses a `(target, source)` Jacobian.

```python
import numpy as np
from orsia_uncertainty import propagate_bias, propagate_covariance

covariance = np.array([[4.0, 1.0], [1.0, 9.0]])  # DN^2
bias = np.array([1.0, -0.5])  # DN
jacobian = np.array([[0.5, 0.5]])
variance = propagate_covariance(covariance, jacobian)
shift = propagate_bias(bias, jacobian)
np.testing.assert_allclose(variance, [[3.75]])
np.testing.assert_allclose(shift, [0.25])
print(variance.item(), shift.item())
```

Expected output: `3.75 0.25`. The output variance is in DN²; the signed shift is in DN. For nonlinear models, transport is local and first order.

## Numerical conventions

- Inputs are float64 arrays. A covariance is a variance vector `(n,)` or a dense matrix `(n,n)`; vector entries are variances. Both covariance axes follow the same caller-defined order.
- Inputs remain unchanged; outputs are independent writable arrays. Reusable `CovarianceFactor` objects own immutable snapshots and require rebuilding after covariance changes.
- Singular positive-semidefinite covariance supports transport and sampling. Solving and whitening require an eigenvalue ratio strictly greater than `1e-12`; no regularisation or pseudoinverse is implicit.
- Source sums require one explicit cross-covariance per source pair. Sampling requires a caller-owned NumPy generator.

## Read next

- [Covariance and ensemble reference](docs/covariance.md): module ownership, prepared-factor reuse, shapes, tolerances and source sums.
- [Orsia Inverse](../orsia-inverse/README.md): fixed-covariance estimation using this package's factors.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-uncertainty/tests` from the workspace root. Synthetic algebra checks leave physical error laws, population coverage and image-scale resource costs application-specific.
