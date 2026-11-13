# Explicit tangents and dense Jacobians

Numerics provides ordinary functions for explicit directional arithmetic and dense Jacobian assembly. Radiative owns its physical laws and their partial derivatives. Callers own state coordinates, units, parameter transforms and fixed model controls.

## Python contract

Import `jacobian` and `directional_derivative` from `orsia_numerics.tangents`. Both functions accept an explicit tangent callback and return a pair of arrays.

- `jacobian(function, x, *, block_size=None)` returns `(value, derivatives)` with the full dense Jacobian.
- `directional_derivative(function, x, direction)` returns `(value, derivative)` along one vector or a block of directions.

The callback contract is `function(x, directions) -> (F(x), J @ directions)`. State `x` is a finite native float64 vector `(n,)`; directions have shape `(n, k)`. Values can have any array shape, including scalar `()`. Tangents have shape `value.shape + (k,)`. The callback returns finite native float64 arrays and keeps its inputs unchanged.

`jacobian` copies `x` into a read-only snapshot and constructs fresh writable identity seed columns. `block_size` is a positive Python integer or `None`, which selects all columns. With an empty state, the callback runs once with seeds `(0, 0)`. An empty output is supported. Callback values must agree exactly across blocks; a changing value raises `ValueError`.

`directional_derivative` accepts a finite native float64 direction vector `(n,)` or block `(n, k)`. A vector produces derivative shape `value.shape`; a block produces `value.shape + (k,)`. The callback always receives a fresh writable `(n, k)` seed array, with `k=1` for a vector, and a read-only owned state snapshot. It runs once, including for empty states, empty outputs and zero direction columns. Each column specifies state-coordinate perturbations per unit step; the result has output units per step unit. Invalid direction shape, dtype or finite support raises `ValueError`.

Returned value and derivative arrays own fresh writable C-contiguous float64 storage, independent of inputs and callback outputs. The Jacobian shape is `value.shape + (n,)`, in state-coordinate order. Each column has output units per corresponding state-coordinate unit. Invalid state, block size, callback shape, dtype or finite support raises `ValueError`; callback exceptions propagate unchanged. There is no retained tracing state or finite-difference fallback.

```python
import numpy as np
from orsia_numerics.tangents import directional_derivative, jacobian


def square(x, directions):
    return x * x, 2 * x[:, None] * directions


x = np.array([2.0, 3.0])
value, derivatives = jacobian(square, x, block_size=1)
np.testing.assert_array_equal(value, [4.0, 9.0])
np.testing.assert_array_equal(derivatives, [[4.0, 0.0], [0.0, 6.0]])

value, derivative = directional_derivative(square, x, np.array([1.0, -2.0]))
np.testing.assert_array_equal(derivative, [4.0, -12.0])
value, products = directional_derivative(square, x, np.eye(2))
np.testing.assert_array_equal(products, derivatives)
```

## General linear operators

`linear_operator((m, n), jvp, vjp)` adapts a fixed real linear map and its Euclidean transpose to a SciPy `LinearOperator`. The existing callback keywords identify forward (`jvp`) and transpose (`vjp`) products. Imaging uses this adapter with physical response coefficients. The adapter supplies shape, dtype and ownership checks; callers supply both operations.

Forward blocks have shape `(n, k)` and return `(m, k)`; transpose blocks have shape `(m, k)` and return `(n, k)`. Both callbacks accept native float64 arrays, including `k=0`. State and output dimensions must be positive. The products satisfy `sum(w * jvp(v)) = sum(vjp(w) * v)`. Vector products use the block callbacks internally. Units, coordinate scaling, flattening and any physical quadrature weights belong to the caller.

Callback inputs are owned read-only copies; returned products are fresh writable copies. Keep borrowed coefficients stable for the operator lifetime and use call-local accumulation for concurrent calls. Invalid dimensions, shapes, dtypes or finite support raise `ValueError`.

`materialize_operator(operator, max_bytes=..., block_size=16)` assembles a dense float64 matrix from bounded identity-column blocks. Its estimate covers the output matrix, explicit seed block and returned product block. Callback workspace, adapter copies and subsequent diagnostics are outside the estimate. Invalid controls or products raise `ValueError`; an exceeded estimate raises `MemoryError`.

For nonlinear fitting, `orsia_inverse.solve_map` accepts a dense prediction Jacobian. Assemble it with `jacobian(tangent, q, block_size=...)[1]` when a tangent callback is available. Dense assembly, solver workspace and diagnostic allocations require caller resource planning; see [Inverse](../../orsia-inverse/README.md#fit-and-diagnostic-rules).

## Native operations and distribution

`orsia_numerics.get_include()` returns the installed include directory. Native consumers include `<orsia_numerics/tangent.h>`. This header is part of both the source distribution and wheel.

`scale_tangent(count, factor, input, stride, output)` scales a directional vector. `product_tangent(count, a, da, a_stride, b, db, b_stride, output)` computes `b*da + a*db`. These allocation-free functions operate on binary64 values with explicit counts and strides in doubles. Counts can be zero; input strides can be zero or negative. Outputs are contiguous caller-owned storage. Exact in-place aliasing requires unit-stride input; other overlapping storage is unsupported. Inputs must remain stable during a call.

Radiative's absorption, direct-solar and thermal kernels use these shared operations. Physical transmission, Planck, exponential-integral formulas and their derivatives remain in Radiative. The header has no Python dependency, retained state or thread-policy controls. It performs ordinary explicit arithmetic without dual numbers, tracing or runtime compilation.

A downstream source build declares `orsia-numerics==0.1.0` in `[build-system].requires`, imports `get_include` in its build configuration, and supplies that path to the compiler. Build Numerics first and expose its wheel through a local `--find-links` directory when packages are staged independently. Workspace source mappings support local development; isolated archive builds use `--no-sources`. Header changes require rebuilding the consuming native extensions.

Run `devbox run -- just python scripts/validate_tangent_distribution.py` from the Orsia root for archived builds and isolated installed smoke checks. The script builds source distributions, rebuilds wheels with workspace resolution disabled, checks the packaged header, and exercises Numerics plus absorption, solar and thermal imports in a new environment. Native-import GIL checks and numerical checks establish bounded software behavior; application-specific scientific qualification remains separate.
