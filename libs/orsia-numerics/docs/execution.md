# CPU executor contract

Use this reference when selecting CPUs, sharing a pool or managing native thread limits. Start with the [package introduction](../README.md) for a minimal call.

## Modules and contracts

| Source | Responsibility |
| --- | --- |
| [execution.py](../src/orsia_numerics/execution.py) | Compose standard futures with Linux worker affinity, process-wide OpenBLAS leases and worker-local OpenMP controls. |
| [Executor tests](../tests/test_execution.py) | Standard executor behavior, actual affinity and lifecycle/failure regressions. |

Applications partition tasks and own the pool. Scientific packages use NumPy directly for numerical snapshots and domain checks; they do not schedule through hidden pools. Interval quadrature belongs to `orsia_instruments.quadrature`; domain-specific interpolation and physical checks belong to their consumers.

Domain functions validate finite values and physical ranges with NumPy predicates and comparisons.

When callers submit a scientific kernel to the executor, they keep borrowed inputs stable during the call and use disjoint writable outputs. The executor coordinates process-wide native settings; kernel ownership and thread-safety contracts remain with their respective scientific packages.

## Keep task inputs independent

Run this standalone example in the [development environment](../../../README.md#development-environment). `np.copy` preserves dtype and shape while creating independently writable storage.

```python
import numpy as np

original = np.array([1, 2, 3], dtype=np.int64)
retained = np.copy(original)
original[0] = 99
np.testing.assert_array_equal(retained, [1, 2, 3])
assert retained.dtype == original.dtype and retained.flags.writeable
assert not np.shares_memory(original, retained)
print(retained.tolist())
```

Expected output: `[1, 2, 3]`. Copying alone does not validate finite values or physical ranges.

## Explicit CPU execution

Import `CpuThreadPoolExecutor` from `orsia_numerics.execution`. The application owns and shares the pool with operations that organize batches; no numerical kernel needs an executor argument merely to remain callable sequentially. This resource-owning class implements the public `concurrent.futures.Executor` interface by composing a standard `ThreadPoolExecutor`; it does not use private scheduler internals.

```python
from itertools import batched
from orsia_numerics.execution import CpuThreadPoolExecutor


def square_batch(values):
    return [value * value for value in values]


with CpuThreadPoolExecutor(max_workers=1) as executor:
    results = list(executor.map(square_batch, batched(range(9), 4), buffersize=2))
assert results == [[0, 1, 4, 9], [16, 25, 36, 49], [64]]
```

The constructor is `CpuThreadPoolExecutor(max_workers=None, thread_name_prefix="", initializer=None, initargs=(), *, cpus=None)`. An explicit `cpus` iterable supplies ordered, distinct logical CPU IDs from the constructing thread's allowed affinity set; it must be nonempty. With `cpus=None`, that set is selected in ascending order. `max_workers=None` uses every selected CPU; an explicit positive value uses that many CPUs from the start of the selection and may not exceed its length. CPU IDs are logical; this policy does not infer SMT siblings, NUMA placement, memory capacity or cgroup CPU-time quotas. Separate pools do not reserve CPUs against one another; prefer one shared application pool.

Linux affinity is mandatory: each lazily started worker is pinned to a distinct selected CPU and its affinity is read back before the user initializer runs. The caller's affinity is not changed. Missing affinity APIs raise `NotImplementedError`; invalid selections raise `ValueError`; failure to set or verify worker affinity, or failure of the user initializer, breaks the underlying pool with standard `BrokenThreadPool` errors. As with a standard thread-pool initializer, another worker may already have started work before a failing worker breaks the pool. Tasks and initializers must not change affinity or native thread limits.

`submit(fn, /, *args, **kwargs)` returns a standard `Future`, compatible with `wait` and `as_completed`. `map(fn, *iterables, timeout=None, chunksize=1, buffersize=None)` inherits Python 3.14 ordering, timeout, exception and buffering behavior: `buffersize=None` is unbounded submission and `chunksize` is ignored. Construction does not require context entry before submission. `shutdown(wait=True, *, cancel_futures=False)` follows standard submission and cancellation rules; leaving `with` waits without automatically cancelling pending work when the body raises. Repeated shutdown is supported. As in the standard pool, avoid tasks that wait for other tasks in an exhausted pool or try to synchronously join their own worker.

### Native limits and lifecycle

Construction imports NumPy and `scipy.linalg`, then discovers loaded native controls with threadpoolctl. **Pthreads OpenBLAS is limited to one native thread process-wide; one loaded OpenMP runtime is serialized separately in each pool worker.** OpenMP controls are applied and read back before the user initializer and before every submitted callable, including `map` calls. The executor sets `omp_set_num_threads(1)` and `omp_set_max_active_levels(0)`: zero active parallel levels serializes ordinary, explicitly sized and nested CPU parallel regions, rather than merely disabling nested teams. Worker settings remain in force during callbacks executed by that worker and disappear when it exits; the constructor's and unrelated threads' OpenMP settings are not changed. Callbacks executed by a submitting or cancelling thread are outside the worker-local guarantee.

Missing BLAS controls, missing OpenMP serialization symbols, multiple detected OpenMP runtimes, or detected unsupported runtimes (MKL, BLIS, FlexiBLAS and OpenMP-backed OpenBLAS) raise `RuntimeError` during construction rather than silently allowing uncontrolled parallelism. Failure to enforce the worker limits before initialization breaks the pool with `BrokenThreadPool`; failure before a later callable is reported by its `Future` without running that callable. OpenBLAS limits are also checked before initialization and each callable. OpenMP symbol availability and reported settings do not establish general runtime compatibility or native-library thread safety.

All live Orsia CPU executors in one interpreter share a locked, reference-counted lease over the process-wide OpenBLAS limits. The first records the original counts; the last restores them after its workers exit, including completion callbacks. OpenMP's task-local settings are deliberately excluded from this global restoration: restoring them from a shutdown or joining thread would target the wrong thread. Overlapping lifetimes may close in any order. `shutdown(wait=False)` starts a joining thread and returns without restoring limits prematurely; restoration does not require a second shutdown call. Explicit `shutdown(wait=True)` waits and reports restoration errors, which can be retried; asynchronous cleanup errors are reported by the joining thread. Garbage collection also requests asynchronous cleanup, but is not a substitute for an explicit lifecycle.

The native-CPU contract requires cooperative user code. Load native dependencies before constructing the first executor; do not import native backends, fork, use these executors from multiple subinterpreters, or overlap independently managed native-limit scopes while any executor is alive. A later executor checks for changed library inventory; the implementation does not continuously monitor tasks or intercept dynamic loading. Detection is limited to libraries known to threadpoolctl: unknown libraries, GPU/offload work, Python/process pools and library-specific worker parameters are not controlled. User code must not create another parallel execution layer or change native settings during a callable or callback. Reapplying OpenMP controls between calls cannot prevent code from overriding them inside a call. No environment variables are modified, and no reliance is placed on the development environment's one-thread defaults. User code retains responsibility for backend safety, borrowed-array stability, disjoint writable slices, explicit RNGs and task granularity. Affinity and disabled GIL do not establish scientific correctness or a speedup.

Contract references: Python 3.14 [`concurrent.futures`](https://docs.python.org/3.14/library/concurrent.futures.html), [`os.sched_getaffinity`](https://docs.python.org/3.14/library/os.html#os.sched_getaffinity), the [threadpoolctl implementation and runtime semantics](https://github.com/joblib/threadpoolctl), and OpenMP 5.1 [`omp_set_max_active_levels`](https://www.openmp.org/spec-html/5.1/openmpsu134.html) and [parallel team-size determination](https://www.openmp.org/spec-html/5.1/openmpsu40.html).
Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-numerics/tests` for focused regression and `devbox run -- just check` for the full workspace gates, from the workspace root. Numerics tests cover standard executor behavior, actual Linux worker affinity, native lease lifecycle and bounded NumPy/SciPy sequential/parallel agreement. Radiative consumer tests cover exact dependency equality, dtype/shape and finiteness contracts, independent writable storage on strided, Fortran-order and read-only inputs, one explicit copy per retained numerical field, and removal of repeated finiteness scans in atmosphere, surface and profile constructors. Instrument tests own physical range-validation regressions and polynomial quadrature checks. Injected-backend tests cover thread-local controls, cleanup and rejection paths. A subprocess test compiles a small OpenMP C probe with the Devbox GCC toolchain and measures actual ordinary, explicitly sized and nested team sizes from the initializer, concurrent workers and a worker callback, starting with multithreaded settings and checking synchronous/nonblocking shutdown and overlapping pools. This test skips if GCC is unavailable; it exercises the linked GNU OpenMP runtime, not every OpenMP implementation. These checks do not establish performance, instrument-model accuracy or retrieval performance.
