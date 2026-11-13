# Orsia Numerics

`orsia-numerics` lets applications assemble Jacobians and directional derivatives from compositions of explicit tangent functions. Callers connect each function's value and tangent outputs to the next function, applying the chain rule in their own code. Numerics supplies seeds and assembles the returned derivatives. Independent capabilities include general linear-operator adapters, shared native headers and an optional caller-owned CPU thread pool with native thread limits and available CPU pinning.

## Responsibilities

![The caller composes Square f(x)=x² and Sine g(z)=sin(z), with tangent propagation below. Separate Numerics tools evaluate the composed callback; operator, native-header and execution capabilities have independent caller inputs.](docs/diagrams/architecture.svg)

The solid horizontal arrows show the componentwise composition `h(x) = sin(x²)`. The dashed path below the functions shows tangent propagation. The upward arrow shows Numerics evaluating the complete callback with identity seeds or supplied directions. Separate downward arrows connect caller inputs to the independent operator, native-header and execution capabilities. The execution branch uses Linux affinity when available and warns before running without pinning when either affinity API is absent. [Full-size SVG](docs/diagrams/architecture.svg) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

| Entry point | Capability and boundary |
| --- | --- |
| `orsia_numerics.tangents.jacobian` | Assemble a dense Jacobian from an explicit tangent callback, optionally in blocks of identity columns. |
| `orsia_numerics.tangents.directional_derivative` | Evaluate the same callback along one supplied vector or a block of directions. |
| `orsia_numerics.operators.linear_operator` | Adapt caller-supplied forward and Euclidean-transpose products to a SciPy `LinearOperator`. |
| `orsia_numerics.operators.materialize_operator` | Assemble a dense reference matrix under an explicit output/seed/product allocation budget. |
| `orsia_numerics.get_include` | Locate installed C++ headers for allocation-free tangent scaling and product rules. |
| `orsia_numerics.execution.CpuThreadPoolExecutor` | Schedule independent calls with standard futures, explicit native thread limits and verified worker affinity when available. |

The Python numerical functions use NumPy and SciPy and require no CPU pool or Linux affinity. The package targets CPython 3.14 free-threaded. Scientific kernels remain sequential unless the application schedules them.

## Minimal tangent example

Run the example with this distribution and its dependencies installed, or in the [Orsia development environment](../../README.md#development-environment). Two functions square the dimensionless input components and take their sine: `f(x) = x²`, `g(z) = sin(z)`, and `h(x) = g(f(x)) = sin(x²)`. Sine arguments are in radians. Each function returns its value and a block of directional derivatives. The caller's `sine_of_square` function passes both outputs from `square` to `sine`.

For a seed block `D`, `square` returns `dz = diag(2*x) @ D`. Then `sine` returns `diag(cos(z)) @ dz`, so the composition has Jacobian `diag(2*x*cos(x*x))`. The separate tools `jacobian` and `directional_derivative` evaluate the composed callback. `jacobian` supplies identity-column blocks and assembles the dense result; `directional_derivative` supplies the requested directions.

```python
import numpy as np
from orsia_numerics.tangents import directional_derivative, jacobian


def square(x, directions):
    return x * x, 2.0 * x[:, None] * directions


def sine(z, tangents):
    return np.sin(z), np.cos(z)[:, None] * tangents


def sine_of_square(x, directions):
    z, dz = square(x, directions)
    return sine(z, dz)


x = np.array([2.0, 3.0])
expected_jacobian = np.diag(2.0 * x * np.cos(x * x))
value, derivatives = jacobian(sine_of_square, x, block_size=1)
np.testing.assert_allclose(value, np.sin(x * x), rtol=1e-14, atol=1e-14)
np.testing.assert_allclose(derivatives, expected_jacobian, rtol=1e-14, atol=1e-14)

direction = np.array([1.0, -2.0])
_, derivative = directional_derivative(sine_of_square, x, direction)
np.testing.assert_allclose(derivative, expected_jacobian @ direction, rtol=1e-14, atol=1e-14)
directions = np.array([[1.0, 0.0], [-2.0, 1.0]])
_, products = directional_derivative(sine_of_square, x, directions)
np.testing.assert_allclose(products, expected_jacobian @ directions, rtol=1e-14, atol=1e-14)
print(np.round(derivatives, 6).tolist(), np.round(derivative, 6).tolist())
```

Expected output (rounded to six decimal places): `[[-2.614574, -0.0], [-0.0, -5.466782]] [-2.614574, 10.933563]`.

The callback receives finite native float64 state `(n,)` and seeds `(n,k)`, and returns `(F(x), J @ directions)`. Values can have any array shape; tangents have shape `value.shape + (k,)`. A full Jacobian has shape `value.shape + (n,)`. A vector direction returns a derivative with `value.shape`; a direction block retains the final direction axis.

Callbacks keep inputs unchanged and return the same value across Jacobian blocks. Both functions return independent writable float64 arrays. Derivative formulas come from the callback; there is no tracing or finite-difference fallback. The [tangent contract](docs/tangents.md#python-contract) defines ownership, empty dimensions, validation and units. For nonlinear fitting, pass `jacobian(tangent, q)[1]` from a Jacobian callback to [Inverse's `solve_map`](../orsia-inverse/README.md#fit-and-diagnostic-rules).

## General linear operators and native arithmetic

`linear_operator((m, n), jvp, vjp)` accepts forward products `(n,k)` to `(m,k)` and transpose products `(m,k)` to `(n,k)`. Callers supply both operations, including their physical weights, coordinate scaling and stable retained coefficients. Instruments uses these adapters for fixed physical response coefficients. They provide generic linear maps; Radiative's derivative APIs supply explicit tangents and expose no adjoint interface.

`materialize_operator(operator, max_bytes=..., block_size=16)` builds the dense matrix in column blocks. Its allocation estimate covers the matrix, seeds and returned products; callback workspace, adapter copies and later diagnostics remain outside that estimate. See the [linear-operator contract](docs/tangents.md#general-linear-operators).

Native consumers obtain the include directory with `get_include()` and include `<orsia_numerics/tangent.h>`. The packaged header supplies binary64 scaling and product-rule operations with explicit direction counts and strides. Radiative uses these operations in its absorption, solar and thermal kernels; physical laws and partial derivatives remain in Radiative. See [native operations and distribution](docs/tangents.md#native-operations-and-distribution) for aliasing, build dependencies and installed-header use.

## Optional CPU execution

The executor pins workers when both `os.sched_getaffinity` and `os.sched_setaffinity` are available. If either API is absent, construction emits `RuntimeWarning` and workers run without CPU pinning. Native thread limits remain enforced in both cases. Supported backends are pthreads OpenBLAS, Apple Accelerate BLAS/LAPACK with the public macOS 15+ controls, and one optional OpenMP runtime. Errors from available affinity APIs remain visible. This example uses one worker and bounded submission.

```python
import numpy as np
from orsia_numerics.execution import CpuThreadPoolExecutor


def sum_squares(values):
    return float(values @ values)


blocks = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
with CpuThreadPoolExecutor(max_workers=1) as executor:
    parts = list(executor.map(sum_squares, blocks, buffersize=2))
assert parts == [5.0, 25.0]
print(parts)
```

Expected output: `[5.0, 25.0]`, with an additional warning when an affinity API is unavailable. Supported native thread controls are required. This tiny calculation demonstrates ownership and result ordering; performance depends on task granularity.

### Execution rules

- Select `max_workers` explicitly to leave capacity for other work. The default uses the selected CPU count; separate pools do not reserve CPUs. Without affinity and with `cpus=None`, the count is `os.process_cpu_count() or os.cpu_count() or 1`. Explicit `cpus` then supplies only a count; CPU IDs cannot be applied or verified.
- Load native dependencies before constructing the pool. Supported pthreads OpenBLAS is limited process-wide; supported Accelerate BLAS/LAPACK and OpenMP are serialized inside each worker.
- Keep borrowed inputs stable and writable outputs disjoint. Tasks, initializers and callbacks must preserve affinity and native limits and avoid nested parallel execution.
- Own shutdown explicitly, preferably with `with`. Unsupported controls/runtimes fail during construction; the [executor contract](docs/execution.md) defines worker and cleanup failures.

## Read next

- [Explicit tangents and linear operators](docs/tangents.md): callback shapes, owned outputs, dense Jacobians, forward/transpose products and installed native headers.
- [Executor contract](docs/execution.md): source ownership, independent array snapshots, CPU selection, native limits and lifecycle.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-numerics/tests` from the workspace root. Affinity and a disabled GIL alone do not establish backend thread safety, scientific correctness or speedup.
