# Covariance algebra and Gaussian ensembles

Use this reference for array shapes, conditioning, prepared factors and explicit source cross-covariances. See the [package introduction](../README.md) for linear transport.

## Source responsibilities

| Source | Responsibility | Boundary |
| --- | --- | --- |
| `covariance.py` | Numerical validation, covariance application/transport, reusable factors, solves, whitening and Gaussian draws | Variance vectors or dense matrices; singular PSD is allowed except for solving/whitening. |
| `operations.py` | Signed-vector transport, source sums, cross terms and empirical covariance | Arrays and ordinary mappings; no source-metadata records or physical laws. |
| NumPy / SciPy | Arrays, linear algebra and explicit numerical factors | Inputs remain unchanged; prepared factors are independent snapshots. |

`operations.py` uses the covariance core for source sums and empirical covariance. Callers can use either module directly; physical source identity and units remain application-owned.

## Array contracts

All numerical inputs are NumPy `float64` arrays. Covariance is either a nonempty variance vector `(n,)`, representing a diagonal matrix, or a dense covariance matrix `(n, n)`. Vector entries are **variances**, not standard deviations. Signed bias is an ordinary vector `(n,)`, kept separate from random covariance. Jacobians have shape `(target, source)`. The caller owns coordinate meanings, vectorisation order, support and unit consistency, including the units carried by Jacobian columns. This package does not check physical homogeneity, convert units, reorder or flatten data.

Functions do not modify inputs and return independent writable arrays. Prepared operations retain independent numerical factors, not the original buffers; rebuild them to incorporate subsequent covariance changes. Keep inputs stable while a calculation or preparation reads them. Calls share no writable work buffers; this does not establish arbitrary native-backend thread safety. Gaussian generators belong to the caller, who also manages RNG sharing.

`validate_covariance(covariance)` performs numerical validation and returns an independent writable snapshot. Variances must be finite and nonnegative; dense matrices must be finite, square, symmetric and positive semidefinite (PSD), with symmetry and PSD tolerance `1e-12 * max(abs(C))`. Accepted dense matrices are symmetrised. Small matrices use NumPy's spectrum; matrices of size at least 128 request only the smallest eigenvalue through SciPy. This is an implementation crossover, not a physical threshold. Numerical checks also occur at public calculation boundaries because an ndarray can be changed after validation; callers do not need to call the validator first.

| Function | Calculation and output |
| --- | --- |
| `covariance_product(C, b)` | `C @ b` for a finite vector `(n,)`, using multiplication for variance-vector C. |
| `solve_covariance(factor, b)` | Solve `C x = b` with an existing factor. |
| `propagate_covariance(C, J)` | Dense `J C J.T`, including all cross terms. |
| `propagate_vector(v, J)` / `propagate_bias(v, J)` | Signed `J @ v`; no covariance is added. |
| `combine_contributions(sources, cross)` | Dense sum of source covariances and declared cross terms. |
| `sample_gaussian(factor, rng, count)` | Centred Gaussian deviations `(count, n)` with realisations in rows. |
| `empirical_covariance(samples)` | Dense covariance of centred rows `(N, n)`, using denominator `N-1`; require `N>=2`, `n>=1`. |

Singular PSD covariances can be applied, transported and sampled. Solving and whitening reject eigenvalue ratios `<=1e-12`, including zero covariance, with `ValueError`; no regularisation or pseudoinverse is implicit. Gaussian factors clip the tiny negative eigenvalues accepted by the PSD tolerance to zero, without adding artificial variance. Invalid numerical dimensions, dtypes, finiteness or covariance properties raise `ValueError`.

## Explicit preparation for repeated use

`factor_covariance(C, method="eigh")` creates a `CovarianceFactor` with independent immutable buffers. Preparation normalizes the finite snapshot before symmetry and spectral checks, including scales such as `1e-300` and `1e300`. Dense spectral preparation computes one eigensystem for PSD validation, conditioning and the square root. `method="cholesky"` checks the spectrum once and retains a normalized lower-triangular root. Variance vectors retain O(n) storage for either method; dense factors use O(n²) storage.

With normalized root `R`, the represented covariance is `scale * R @ R.T`. `whiten(factor, values, transpose=False)` applies `R^-1 / sqrt(scale)` to `(n,)` vectors or `(n, columns)` right-hand sides. Set `transpose=True` to apply the transposed whitening operator for sensitivity propagation. Cholesky and spectral roots can rotate whitened components while preserving the quadratic form `whiten(factor,v) @ whiten(factor,v) = v @ solve_covariance(factor,v)`.

Solving and whitening require the original minimum/maximum eigenvalue ratio strictly greater than `1e-12`. Spectral and diagonal roots also support singular or zero PSD laws for sampling. The spectral root clips only negative eigenvalues within the declared PSD tolerance; the retained spectrum keeps their sign, and inverse operations reject them. Cholesky requires a positive definite dense covariance. Preparation consumes no RNG. Each sampling call consumes `count * n` standard normals in row order, including zero laws, and returns a fresh `(count,n)` array. Seeded components can differ between square-root backends.

Run this standalone example in the [development environment](../../../README.md#development-environment).

```python
import numpy as np
from orsia_uncertainty import factor_covariance, sample_gaussian, solve_covariance, whiten

covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
vector = np.array([7.0, -1.0])
factor = factor_covariance(covariance, method="eigh")
triangle = factor_covariance(covariance, method="cholesky")
np.testing.assert_allclose(solve_covariance(factor, vector), [2.0, -1.0], atol=1e-14)
np.testing.assert_allclose(solve_covariance(triangle, vector), solve_covariance(factor, vector))
weighted = whiten(factor, vector)
np.testing.assert_allclose(weighted @ weighted, vector @ solve_covariance(factor, vector))
samples = sample_gaussian(factor, np.random.default_rng(42), 8)
assert samples.shape == (8, 2)
np.testing.assert_array_equal(samples, sample_gaussian(factor, np.random.default_rng(42), 8))
print(solve_covariance(factor, vector).round(6).tolist(), samples.shape)
```

Expected output: `[2.0, -1.0] (8, 2)`. Rows are realizations and columns are coordinates. Eight seeded samples check replay mechanics; statistical covariance accuracy requires an appropriate ensemble.

## Source sums without source records

`combine_contributions` takes a mapping `{source_id: covariance_array}`. Its insertion order defines pair orientation: for each earlier source `a` and later source `b`, supply exactly one cross matrix `(n,n)` keyed by `(a,b)`. A cross block represents `Cov(error_a, error_b)` and need not be symmetric. An explicit zero block declares zero cross-covariance; a missing pair, reversed key or extra pair is an error. For Gaussian laws, zero cross-covariance also implies independence. No correlation scope, calibration lineage or physical source law is inferred or stored.

The function checks each source covariance and, when any cross block is nonzero, the PSD of the full joint block matrix, not just the covariance of the sum. Even a tiny nonzero cross block triggers that joint check. Exactly zero cross blocks avoid joint allocation/decomposition; the result remains dense even when all sources are diagonal. Source attribution and metadata stay with the caller.

```python
from orsia_uncertainty import combine_contributions

sources = {"noise": np.array([1.0, 2.0]), "calibration": np.array([3.0, 4.0])}
cross = {("noise", "calibration"): np.array([[0.5, 0.0], [0.0, 0.25]])}
np.testing.assert_allclose(combine_contributions(sources, cross), np.diag([5.0, 6.5]))
```
## Validation and cost observations

From the Orsia root, run `devbox run -- just test-uncertainty` or the full `devbox run -- just check`. Numerical tests cover small-matrix references, sequential transport, impossible joint laws, singular draws, conditioning, numerical failures, snapshots and prepared-factor reuse. The instrument-to-retrieval diagnostic compares correlated staged transport with a joint-matrix reference and seeded ensembles; a quadratic example exposes the limits of first-order propagation. These are synthetic software/algebra checks, not qualification of physical error laws, retrieval accuracy or statistical coverage for arbitrary populations.
