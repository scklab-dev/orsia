# CPU executor contract

Use this reference when selecting CPUs, sharing a pool or managing native thread limits. Start with the [package introduction](../README.md) for a minimal call.

## Modules and contracts

| Source | Responsibility |
| --- | --- |
| [execution.py](../src/orsia_numerics/execution.py) | Compose standard futures with available worker affinity, process-wide OpenBLAS leases and worker-local Accelerate/OpenMP controls. |
| [Executor tests](../tests/test_execution.py) | Standard executor behavior, actual affinity and lifecycle/failure regressions. |
| [Accelerate tests](../tests/test_execution_accelerate.py) | Worker-local BLAS/LAPACK modes, failure handling and macOS integration. |

Applications partition tasks and own the pool. Scientific packages use NumPy directly for numerical snapshots and domain checks; they do not schedule through hidden pools. Interval quadrature belongs to `orsia_instruments.acquisition.quadrature.quadrature`; domain-specific interpolation and physical checks belong to their consumers.

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

The constructor is `CpuThreadPoolExecutor(max_workers=None, thread_name_prefix="", initializer=None, initargs=(), *, cpus=None)`. Explicit `cpus` must contain distinct, nonnegative logical CPU IDs and must be nonempty. `max_workers=None` uses the selected CPU count; an explicit positive value may not exceed that count. Select `max_workers` explicitly to leave capacity for other work.

When both `os.sched_getaffinity` and `os.sched_setaffinity` are available, explicit CPU IDs must belong to the constructing thread's allowed affinity set. With `cpus=None`, that set is selected in ascending order. Each lazily started worker is pinned to a distinct selected CPU, preserving the explicit CPU order and using the first `max_workers` CPUs. Worker affinity is read back before the user initializer runs. The caller's affinity is unchanged.

If either affinity API is absent, construction emits `RuntimeWarning` and workers run without pinning. With `cpus=None`, the selected count is `os.process_cpu_count() or os.cpu_count() or 1`. Explicit `cpus` supplies only its length: the IDs cannot be applied or checked against the allowed affinity set. Nonempty, distinct, nonnegative ID validation and the worker-count bound still apply. The OS controls placement; workers can share CPUs or migrate between CPUs. Native thread controls, initialization, futures and shutdown retain the same contract.

Invalid selections raise `ValueError`. Errors while reading an available constructor affinity API propagate to the caller. Failure to set or verify worker affinity, or failure of the user initializer, breaks the underlying pool with standard `BrokenThreadPool` errors. Available APIs that fail do not trigger the unpinned fallback. Another worker may already have started work before a failing worker breaks the pool. Tasks and initializers must preserve affinity and native thread limits.

CPU IDs are logical; this policy does not infer SMT siblings, NUMA placement, memory capacity or cgroup CPU-time quotas. Separate pools do not reserve CPUs against one another; prefer one shared application pool.

`submit(fn, /, *args, **kwargs)` returns a standard `Future`, compatible with `wait` and `as_completed`. `map(fn, *iterables, timeout=None, chunksize=1, buffersize=None)` inherits Python 3.14 ordering, timeout, exception and buffering behavior: `buffersize=None` is unbounded submission and `chunksize` is ignored. Construction does not require context entry before submission. `shutdown(wait=True, *, cancel_futures=False)` follows standard submission and cancellation rules; leaving `with` waits without automatically cancelling pending work when the body raises. Repeated shutdown is supported. As in the standard pool, avoid tasks that wait for other tasks in an exhausted pool or try to synchronously join their own worker.

### Native limits and lifecycle

Construction imports NumPy and `scipy.linalg`, then discovers OpenBLAS/OpenMP controls with threadpoolctl. On macOS, `ctypes.util.dllist()` also identifies a loaded Accelerate framework `libBLAS.dylib`. **Pthreads OpenBLAS is limited to one native thread process-wide. Apple Accelerate BLAS/LAPACK and one loaded OpenMP runtime are serialized separately in each pool worker.** Worker-local controls are applied and read back before the user initializer and before every submitted callable, including `map` calls. The executor sets `omp_set_num_threads(1)` and `omp_set_max_active_levels(0)`: zero active parallel levels serializes ordinary, explicitly sized and nested CPU parallel regions, rather than merely disabling nested teams. Worker settings remain in force during callbacks executed by that worker and disappear when it exits; the constructor's and unrelated threads' OpenMP settings are not changed. Callbacks executed by a submitting or cancelling thread are outside the worker-local guarantee.

Accelerate support requires the public `BLASSetThreading` and `BLASGetThreading` APIs introduced in macOS 15. Each worker requests `BLAS_THREADING_SINGLE_THREADED`, checks the setter's success code and reads back the mode. Apple's [threading contract](https://developer.apple.com/documentation/accelerate/blassetthreading(_:)) applies this mode to subsequent BLAS and LAPACK calls in the calling thread. The constructor's and unrelated threads' modes remain unchanged. Worker settings remain active through worker callbacks and disappear with the worker. Other Accelerate subsystems, such as vDSP and BNNS, are outside this BLAS/LAPACK control.

Missing supported BLAS controls, missing Accelerate or OpenMP symbols, multiple detected OpenMP runtimes, or detected unsupported runtimes (MKL, BLIS, FlexiBLAS and OpenMP-backed OpenBLAS) raise `RuntimeError` during construction. Failure to enforce the worker limits before initialization breaks the pool with `BrokenThreadPool`; failure before a later callable is reported by its `Future` without running that callable. OpenBLAS limits are also checked before initialization and each callable. Accelerate/OpenMP symbol availability and reported settings do not establish general runtime compatibility or native-library thread safety.

All live Orsia CPU executors in one interpreter share a locked, reference-counted lease over the process-wide OpenBLAS limits. The first records the original counts; the last restores them after its workers exit, including completion callbacks. Accelerate's thread-local mode and OpenMP's task-local settings are excluded from this global restoration: restoring them from a shutdown or joining thread would target the wrong thread. Overlapping lifetimes may close in any order. `shutdown(wait=False)` starts a joining thread and returns without restoring limits prematurely; restoration does not require a second shutdown call. Explicit `shutdown(wait=True)` waits and reports restoration errors, which can be retried; asynchronous cleanup errors are reported by the joining thread. Garbage collection also requests asynchronous cleanup, but is not a substitute for an explicit lifecycle.

The native-CPU contract requires cooperative user code. Load native dependencies before constructing the first executor; do not import native backends, fork, use these executors from multiple subinterpreters, or overlap independently managed native-limit scopes while any executor is alive. A later executor checks for changed library inventory; the implementation does not continuously monitor tasks or intercept dynamic loading. Detection covers libraries known to threadpoolctl and loaded macOS Accelerate BLAS: unknown libraries, GPU/offload work, Python/process pools and library-specific worker parameters are not controlled. User code must not create another parallel execution layer or change native settings during a callable or callback. Reapplying worker-local controls between calls cannot prevent code from overriding them inside a call. No environment variables are modified, and no reliance is placed on the development environment's one-thread defaults. User code retains responsibility for backend safety, borrowed-array stability, disjoint writable slices, explicit RNGs and task granularity. Affinity and disabled GIL do not establish scientific correctness or a speedup.

Contract references: Python 3.14 [`concurrent.futures`](https://docs.python.org/3.14/library/concurrent.futures.html), [`os.sched_getaffinity`](https://docs.python.org/3.14/library/os.html#os.sched_getaffinity), the [threadpoolctl implementation and runtime semantics](https://github.com/joblib/threadpoolctl), and OpenMP 5.1 [`omp_set_max_active_levels`](https://www.openmp.org/spec-html/5.1/openmpsu134.html) and [parallel team-size determination](https://www.openmp.org/spec-html/5.1/openmpsu40.html).
Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-numerics/tests` for focused regression and `devbox run -- just check` for the full workspace gates, from the workspace root. Numerics tests cover standard executor behavior, actual Linux worker affinity, warned execution with either or both affinity APIs absent, CPU-count fallback, native lease lifecycle and bounded NumPy/SciPy sequential/parallel agreement. Numerical integration and documentation examples also run without affinity APIs; actual pinning tests remain platform-dependent. Radiative consumer tests cover exact dependency equality, dtype/shape and finiteness contracts, independent writable storage on strided, Fortran-order and read-only inputs, one explicit copy per retained numerical field, and removal of repeated finiteness scans in atmosphere, surface and profile constructors. Instrument tests own physical range-validation regressions and polynomial quadrature checks. Injected-backend tests cover thread-local controls, cleanup and rejection paths. The macOS Accelerate test checks the real BLAS/LAPACK mode in the initializer, tasks and a callback, exercises NumPy/SciPy calls, and checks caller-mode isolation with overlapping pools and nonblocking shutdown. The test skips when Accelerate is not loaded. A subprocess test compiles a small OpenMP C probe with the Devbox GCC toolchain and measures actual ordinary, explicitly sized and nested team sizes from the initializer, concurrent workers and a worker callback, starting with multithreaded settings and checking synchronous/nonblocking shutdown and overlapping pools. This test skips if GCC is unavailable; it exercises the linked GNU OpenMP runtime, not every OpenMP implementation. These checks do not establish performance, instrument-model accuracy or retrieval performance.
