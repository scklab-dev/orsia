# Orsia Uncertainty

`orsia-uncertainty` is a numerical core for covariance algebra, signed-vector transport and Gaussian ensembles, not a coordinate, unit or source-metadata framework. Start with the [code guide and progressive examples](src/orsia_uncertainty/README.md).

## Array contracts

All numerical inputs are NumPy `float64` arrays. Covariance is either a nonempty variance vector `(n,)`, representing a diagonal matrix, or a dense covariance matrix `(n, n)`. Vector entries are **variances**, not standard deviations. Signed bias is an ordinary vector `(n,)`, kept separate from random covariance. Jacobians have shape `(target, source)`. The caller owns coordinate meanings, vectorisation order, support and unit consistency, including the units carried by Jacobian columns. This package does not check physical homogeneity, convert units, reorder or flatten data.

Functions do not modify inputs and return independent writable arrays. Prepared operations retain independent numerical factors, not the original buffers; rebuild them to incorporate subsequent covariance changes. Keep inputs stable while a calculation or preparation reads them. Calls share no writable work buffers; this does not establish arbitrary native-backend thread safety. Gaussian generators belong to the caller, who also manages RNG sharing.

`validate_covariance(covariance)` performs numerical validation and returns an independent writable snapshot. Variances must be finite and nonnegative; dense matrices must be finite, square, symmetric and positive semidefinite (PSD), with symmetry and PSD tolerance `1e-12 * max(abs(C))`. Accepted dense matrices are symmetrised. Small matrices use NumPy's spectrum; matrices of size at least 128 request only the smallest eigenvalue through SciPy. This is an implementation crossover, not a physical threshold. Numerical checks also occur at public calculation boundaries because an ndarray can be changed after validation; callers do not need to call the validator first.

| Function | Calculation and output |
| --- | --- |
| `covariance_product(C, b)` | `C @ b` for a finite vector `(n,)`, using multiplication for variance-vector C. |
| `solve_covariance(C, b)` | Solve `C x = b`; dense one-off solves use NumPy LU. |
| `propagate_covariance(C, J)` | Dense `J C J.T`, including all cross terms. |
| `propagate_vector(v, J)` / `propagate_bias(v, J)` | Signed `J @ v`; no covariance is added. |
| `combine_contributions(sources, cross)` | Dense sum of source covariances and declared cross terms. |
| `sample_gaussian(C, rng, count)` | Centred Gaussian deviations `(count, n)` with realisations in rows. |
| `empirical_covariance(samples)` | Dense covariance of centred rows `(N, n)`, using denominator `N-1`; require `N>=2`, `n>=1`. |

Singular PSD covariances can be applied, transported and sampled. Solving and whitening reject eigenvalue ratios `<=1e-12`, including zero covariance, with `ValueError`; no regularisation or pseudoinverse is implicit. Gaussian factors clip the tiny negative eigenvalues accepted by the PSD tolerance to zero, without adding artificial variance. Invalid numerical dimensions, dtypes, finiteness or covariance properties raise `ValueError`.

## Explicit preparation for repeated use

`covariance_solver(C, method="eigh")` returns `solve(b)` for finite vectors `(n,)`. Dense preparation retains one eigensystem; `method="cholesky"` explicitly selects SciPy `cho_factor`/`cho_solve` with an immutable Fortran-order factor. Both methods retain the same spectral conditioning threshold, and diagonal inputs use division for either method. There is no automatic backend choice or cache. Results may differ in rounding from a one-off LU solve; neither backend is promised to be faster for every workload.

`whitener(C)` returns `whiten(values)` for vectors `(n,)` or matrices `(n, columns)`. With `C=L L.T`, it applies `L^-1 values`, not `C^-1 values`, so `whiten(v).T @ whiten(v) = v.T @ C^-1 @ v`. Dense preparation retains a lower Cholesky factor; diagonal preparation retains square-root variances. Matrix columns are separate right-hand sides, not additional coordinate axes. Whitening is dimensionless only when input units match the covariance's base units; that consistency is the caller's responsibility.

`gaussian_sampler(C)` returns `draw(rng, count)` with the same seeded conventions as `sample_gaussian`. Preparation consumes no RNG. Dense factors retain O(n²) storage; diagonal factors retain O(n) storage. Each call requires positive count and allocates fresh writable `(count, n)` output.

```python
import numpy as np
from orsia_uncertainty import gaussian_sampler, covariance_solver, whitener

covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
vector = np.array([7.0, -1.0])
solve = covariance_solver(covariance)
cholesky_solve = covariance_solver(covariance, method="cholesky")
whiten = whitener(covariance)
np.testing.assert_allclose(solve(vector), [2.0, -1.0], atol=1e-14)
np.testing.assert_allclose(cholesky_solve(vector), solve(vector), atol=1e-14)
np.testing.assert_allclose(whiten(vector) @ whiten(vector), vector @ solve(vector))
draw = gaussian_sampler(covariance)
assert draw(np.random.default_rng(42), 8).shape == (8, 2)
```

## Source sums without source records

`combine_contributions` takes a mapping `{source_id: covariance_array}`. Its insertion order defines pair orientation: for each earlier source `a` and later source `b`, supply exactly one cross matrix `(n,n)` keyed by `(a,b)`. A cross block represents `Cov(error_a, error_b)` and need not be symmetric. An explicit zero block declares zero cross-covariance; a missing pair, reversed key or extra pair is an error. For Gaussian laws, zero cross-covariance also implies independence. No correlation scope, calibration lineage or physical source law is inferred or stored.

The function checks each source covariance and, when any cross block is nonzero, the PSD of the full joint block matrix, not just the covariance of the sum. Even a tiny nonzero cross block triggers that joint check. Exactly zero cross blocks avoid joint allocation/decomposition; the result remains dense even when all sources are diagonal. Source attribution and metadata stay with the caller.

```python
from orsia_uncertainty import combine_contributions

sources = {"noise": np.array([1.0, 2.0]), "calibration": np.array([3.0, 4.0])}
cross = {("noise", "calibration"): np.array([[0.5, 0.0], [0.0, 0.25]])}
np.testing.assert_allclose(combine_contributions(sources, cross), np.diag([5.0, 6.5]))
```

## Breaking migration

`ArrayLayout`, `CoordinateKind`, `Sampling`, `Covariance`, `DiagonalCovariance`, `SignedBias`, `Contribution` and `UncertaintySource` are removed, as is `layout.py`. There are no compatibility wrappers. Replace dense records with their matrix arrays, diagonal records with variance arrays, signed-bias records with vectors, and contribution tuples with mappings from source identifiers to covariance arrays. Application metadata belongs outside this numerical core.

Remove layout and unit arguments from `propagate_covariance`, `propagate_vector`, `propagate_bias` and `empirical_covariance`. Results are ndarrays accessed directly, without `.matrix` or `.values`. Replace covariance methods `.apply(b)` and `.solve(b)` with `covariance_product(C, b)` and `solve_covariance(C, b)`. `covariance_solver`, `whitener`, `gaussian_sampler` and `sample_gaussian` take arrays directly. Instrument tolerance APIs also take array biases/covariances and return array components; their read and parameter names remain domain metadata in `orsia-instruments`. The independent [inverse package](../orsia-inverse/README.md) already uses arrays and does not depend on this package.

## Validation and cost observations

From the Orsia root, run `devbox run -- just test-uncertainty` or the full `devbox run -- just check`. Numerical tests cover small-matrix references, sequential transport, impossible joint laws, singular draws, conditioning, numerical failures, snapshots and prepared-factor reuse. The instrument-to-retrieval diagnostic compares correlated staged transport with a joint-matrix reference and seeded ensembles; a quadratic example exposes the limits of first-order propagation. These are synthetic software/algebra checks, not qualification of physical error laws, retrieval accuracy or statistical coverage for arbitrary populations.

The standalone uncertainty cost probe and its report-contract tests are retired. Historical record-based timings do not establish current array-API performance, full-image feasibility or concurrent-native safety. Runtime dependencies remain NumPy and SciPy.
