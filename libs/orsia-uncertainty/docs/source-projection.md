# Source-projected covariance

`ProjectedCovariance` stores error sources and the maps that carry their errors into observation coordinates. Products, transport, selection and marginal variances operate without constructing the observation covariance. The [dense covariance API](covariance.md) retains its array return types and factorization contracts.

## Representation and ownership

Import the record and functions from `orsia_uncertainty` or `orsia_uncertainty.projection`. The constructor takes matching nonempty tuples `projections` and `source_covariances`:

\[
C=\sum_i M_i Q_i M_i^T.
\]

Each M is a finite float64 dense array or SciPy `csr_array` with shape `(observations,sources)`. All maps share the same output dimension and coordinate order. Each Q is a float64 variance vector `(sources,)` or symmetric positive-semidefinite dense matrix `(sources,sources)`, validated with the existing covariance tolerance. Distinct tuple blocks explicitly declare zero cross-covariance. Correlated sources belong in one joint Q, including both cross blocks, or in an equivalent supplied source factorization.

For one concatenated M and joint Q, the model is `M @ Q @ M.T`. Separate read and shared-parameter blocks express `P @ D @ P.T + U @ S @ U.T` only when those groups are uncorrelated. A block name, acquisition ID or shared label never creates or removes covariance.

Construction copies maps and Q arrays. Copies remain writable, but mutation invalidates the representation; construct a new `ProjectedCovariance` before further operations. Products rely on construction invariants and do not revalidate the retained buffers. Transport and selection construct independent snapshots. CSR duplicates are summed and explicit zeros removed before use. Unsupported sparse formats raise `ValueError` for projection maps. Sparse Q and abstract operator inputs are outside this interface; use diagonal/dense source blocks or an explicit supported factorization.

Construction also checks the computed marginal variances of the summed source model. Every negative or nonfinite marginal raises `ValueError`, including tiny negatives from source-space tolerance or roundoff. The source PSD tolerance remains `1e-12` times the largest absolute source entry; it does not authorize clipping or accepting a negative output variance. For example, `Q=diag(1,-1e-13)` passes source validation, but `M=[[0,1e7]]` produces variance -10 and is rejected. Q remains unchanged apart from the existing symmetry normalization. Callers must resolve a rejected source model or its numerical conditioning explicitly.

This conservative check establishes finite nonnegative computed marginals, not a full output PSD certificate. A tolerance-accepted Q can retain a negative direction despite positive output marginals. Transport recomputes marginals in the target coordinates and rejects a negative target variance. Selection also rechecks its new snapshot. Products reuse the construction invariant; materialization additionally checks its computed diagonal for negative entries. Use `validate_covariance` on a bounded materialized matrix when a full output-space spectral check is required.

The added marginal check uses output-vector storage. With N observations and S sources per block, dense maps cost O(N S) for diagonal Q or O(N S²) for dense Q. For CSR maps, the corresponding costs are O(N + nnz) and O(N + sum(k_i²)), where k_i is the nonzero count in row i. Dense source validation retains its existing source-space cost. Construction, transport and selection pay the marginal-check cost once per new representation; covariance products do not repeat it.

## Operations

| Function | Contract |
| --- | --- |
| `projected_covariance_product(covariance, values)` | Compute Cx for a vector or RHS columns through source coordinates. |
| `transport_projected_covariance(covariance, transform)` | Replace every M with T M, using a dense or CSR `(target,observations)` transform. |
| `select_projected_covariance(covariance, indices)` | Select/permutate output rows in supplied int64 order; repetitions retain their full dependence. |
| `projected_variances(covariance)` | Return diagonal entries, computed rowwise without output-square storage. |
| `materialize_covariance(covariance, *, max_bytes)` | Allocate the explicit float64 output only when `8*n*n <= max_bytes`. |
| `solve_diagonal_low_rank(variances, modes, values)` | Solve a strictly positive diagonal plus weighted-mode outer product. |

Source and output units remain caller-defined. Bias remains a signed vector transported by the same map and stored separately from covariance. Shapes and finite float64 support are checked; empty source/output support is rejected. Singular PSD covariance supports products, transport, selection and marginals.

Materialization uses one output matrix and vector workspaces, without an identity matrix. Its byte budget covers the output allocation, excluding retained source storage and temporary source products. Sparse projection transport can increase fill; callers still own representation choice and application memory budgets. Dense Q retains its source-square cost. The sparse test with 100,000 diagonal coordinates checks that the core path avoids an observation-square allocation; it is not an image-scale performance benchmark.

## Independent reads and a common gain mode

```python
import numpy as np
from scipy.sparse import csr_array, diags_array
from orsia_uncertainty import ProjectedCovariance, materialize_covariance, projected_variances, solve_diagonal_low_rank, transport_projected_covariance

# Four radiances with independent variance 1 and a common 2% gain error.
means = np.full(4, 10.0)
mode = (0.02 * means)[:, None]
covariance = ProjectedCovariance(
    (csr_array(diags_array(np.ones(4))), mode),
    (np.ones(4), np.ones(1)),
)
average = transport_projected_covariance(
    covariance, csr_array(np.full((1, 4), 0.25))
)
np.testing.assert_allclose(projected_variances(average), [0.25 + 0.04])
dense = materialize_covariance(covariance, max_bytes=4 * 4 * 8)
rhs = np.array([1.0, 2.0, -1.0, 0.0])
solution = solve_diagonal_low_rank(np.ones(4), mode, rhs)
np.testing.assert_allclose(dense @ solution, rhs, atol=1e-14)
```

The independent variance reduces from 1 to 0.25 in the average; the common gain variance remains 0.04. [Native observation calibration](../../orsia-instruments/docs/observation-cubes.md) supplies the actual DN correction and calibration-parameter derivatives for this error model.

## Specialized solve and numerical limits

`solve_diagonal_low_rank(d, U, rhs)` solves `diag(d) + U @ U.T`. U already includes source standard deviations or a covariance square root. The implementation uses a mode-space Woodbury calculation and the existing conditioned dense Cholesky factorization. A zero-column U gives an ordinary diagonal solve. Diagonal entries must be strictly positive. The small mode-space matrix must satisfy the existing factorization condition rule. A final relative infinity-norm residual greater than `1e-9` raises `ValueError`; severe cancellation therefore fails explicitly. The tolerance is a numerical safeguard for this solver, not a scientific acceptance criterion.

No jitter, singular inverse, arbitrary projected-covariance solve or projected whitening is supplied. General `ProjectedCovariance` products do not imply solvability. For bounded dense systems, materialize explicitly and use `factor_covariance` with `solve_covariance` or `whiten` under their existing conditioning rules.

The [projection tests](../tests/test_projection.py) cover dense/CSR equivalence, source ownership, duplicate sparse entries, singular support, budget rejection, amplified negative directions, numerical overflow and the specialized solve. Scientific error-law validity and application-scale feasibility require separate evidence.
