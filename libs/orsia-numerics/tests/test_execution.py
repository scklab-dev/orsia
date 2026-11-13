"""Executor contracts, failure cleanup and bounded native interoperability checks."""

from __future__ import annotations

import gc
import inspect
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.thread import BrokenThreadPool
from pathlib import Path
from threading import Barrier, Event, Lock, current_thread, local
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import numpy as np
import pytest
from numpy.typing import NDArray
from orsia_numerics import execution
from orsia_numerics.execution import CpuThreadPoolExecutor
from scipy.linalg import solve
from threadpoolctl import ThreadpoolController

requires_linux_affinity = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"),
    reason="Real pinned execution requires Linux CPU affinity",
)


def affinity(pid: int = 0) -> set[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    assert get_affinity is not None
    return get_affinity(pid)


def test_standard_interface(native: FakeOpenBlas) -> None:
    assert CpuThreadPoolExecutor.map is Executor.map
    assert CpuThreadPoolExecutor.__enter__ is Executor.__enter__
    assert CpuThreadPoolExecutor.__exit__ is Executor.__exit__
    for name in ("submit", "shutdown"):
        actual = inspect.signature(getattr(CpuThreadPoolExecutor, name)).parameters
        expected = inspect.signature(getattr(ThreadPoolExecutor, name)).parameters
        assert [(p.name, p.kind, p.default) for p in actual.values()] == [
            (p.name, p.kind, p.default) for p in expected.values()
        ]
    pool = CpuThreadPoolExecutor(1)
    try:
        future = pool.submit(pow, 2, exp=5)
        assert isinstance(future, Future)
        assert wait([future], timeout=10).done == {future}
        assert [f.result() for f in as_completed([future], timeout=10)] == [32]
    finally:
        pool.shutdown()
    pool.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(int, 1)


@pytest.mark.parametrize("relative_path", ["README.md", "docs/execution.md"])
def test_documented_execution_example(relative_path: str) -> None:
    document = Path(__file__).resolve().parents[1] / relative_path
    examples = re.findall(r"```python\n(.*?)```", document.read_text(), re.DOTALL)
    assert examples
    for example in examples:
        exec(compile(example, "<numerics-documentation-example>", "exec"), {})


@requires_linux_affinity
def test_affinity_precedes_initializer_and_distinct_workers() -> None:
    allowed = affinity()
    cpus = tuple(sorted(allowed, reverse=True)[:2])
    barrier = Barrier(len(cpus))
    initialized = {}
    lock = Lock()

    def initialize(marker: str) -> None:
        with lock:
            initialized[current_thread().ident] = (affinity(), marker)

    def observe() -> tuple[set[int], str]:
        barrier.wait(timeout=10)
        placement, marker = initialized[current_thread().ident]
        assert placement == affinity()
        assert marker == "ready"
        return placement, current_thread().name

    with CpuThreadPoolExecutor(
        thread_name_prefix="science", initializer=initialize, initargs=("ready",), cpus=cpus
    ) as pool:
        results = [f.result(timeout=10) for f in [pool.submit(observe) for _ in cpus]]
    assert {next(iter(affinity)) for affinity, _ in results} == set(cpus)
    assert all(name.startswith("science") for _, name in results)
    assert affinity() == allowed
    with CpuThreadPoolExecutor(1, cpus=cpus) as pool:
        assert pool.submit(affinity).result(timeout=10) == {cpus[0]}


@requires_linux_affinity
def test_automatic_cpus_use_constructor_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = min(affinity())
    original = getattr(os, "sched_getaffinity", None)
    assert original is not None
    main = current_thread()
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {cpu} if current_thread() is main else original(pid)
    )
    with CpuThreadPoolExecutor() as pool:
        assert pool.submit(affinity).result(timeout=10) == {cpu}


@pytest.mark.parametrize("workers", [0, -1, 2])
def test_invalid_worker_count(workers: int, native: FakeOpenBlas) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        CpuThreadPoolExecutor(workers, cpus=[min(affinity())])


@pytest.mark.parametrize("kind", ["empty", "duplicate", "unavailable"])
def test_invalid_cpus(kind: str, native: FakeOpenBlas) -> None:
    allowed = affinity()
    cpu = min(allowed)
    cpus = {"empty": [], "duplicate": [cpu, cpu], "unavailable": [max(allowed) + 1]}[kind]
    with pytest.raises(ValueError, match="cpus"):
        CpuThreadPoolExecutor(cpus=cpus)


@pytest.mark.parametrize(
    "missing",
    [("sched_getaffinity",), ("sched_setaffinity",), ("sched_getaffinity", "sched_setaffinity")],
)
def test_missing_affinity(
    monkeypatch: pytest.MonkeyPatch,
    missing: tuple[str, ...],
    native: FakeOpenBlas,
    openmp: FakeOpenMP,
) -> None:
    for name in ("sched_getaffinity", "sched_setaffinity"):
        if name in missing:
            monkeypatch.delattr(os, name, raising=False)
        else:
            monkeypatch.setattr(os, name, Mock(side_effect=AssertionError("affinity called")))
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    initialized = Event()

    def initialize(marker: str) -> None:
        assert marker == "ready"
        assert native.threads == 1
        assert openmp.counts() == (1, 0)
        initialized.set()

    with pytest.warns(RuntimeWarning, match="without CPU pinning") as warnings:
        pool = CpuThreadPoolExecutor(1, initializer=initialize, initargs=("ready",))
    assert len(warnings) == 1
    assert warnings[0].filename == __file__
    with pool:
        assert pool.submit(openmp.counts).result(10) == (1, 0)
        assert initialized.is_set()
        assert list(pool.map(pow, [2, 3], [3, 2], buffersize=1)) == [8, 9]
        with pytest.raises(ValueError):
            pool.submit(int, "bad").result(10)
    assert native.threads == 4
    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(int)


@pytest.mark.parametrize(
    "process_count, system_count, expected", [(2, 4, 2), (None, 2, 2), (None, None, 1)]
)
def test_unpinned_cpu_count_fallback(
    monkeypatch: pytest.MonkeyPatch,
    native: FakeOpenBlas,
    process_count: int | None,
    system_count: int | None,
    expected: int,
) -> None:
    monkeypatch.delattr(os, "sched_setaffinity")
    monkeypatch.setattr(os, "process_cpu_count", lambda: process_count)
    system = Mock(return_value=system_count)
    monkeypatch.setattr(os, "cpu_count", system)
    barrier = Barrier(expected)

    def observe() -> int | None:
        barrier.wait(10)
        return current_thread().ident

    with pytest.warns(RuntimeWarning, match="without CPU pinning"), CpuThreadPoolExecutor() as pool:
        futures = [pool.submit(observe) for _ in range(expected)]
        assert len({future.result(10) for future in futures}) == expected
    assert system.call_count == (1 if process_count is None else 0)
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="max_workers"):
        CpuThreadPoolExecutor(expected + 1)


@pytest.mark.parametrize("workers, expected", [(None, 2), (1, 1)])
def test_unpinned_explicit_cpus_use_only_count(
    monkeypatch: pytest.MonkeyPatch,
    native: FakeOpenBlas,
    workers: int | None,
    expected: int,
) -> None:
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "process_cpu_count", Mock(side_effect=AssertionError("count called")))
    barrier = Barrier(expected)

    def observe() -> int | None:
        barrier.wait(10)
        return current_thread().ident

    with pytest.warns(RuntimeWarning, match="CPU IDs cannot be applied or verified"):
        pool = CpuThreadPoolExecutor(workers, cpus=iter([100001, 100002]))
    with pool:
        futures = [pool.submit(observe) for _ in range(expected)]
        assert len({future.result(10) for future in futures}) == expected


@pytest.mark.parametrize(
    "cpus, workers", [([], None), ([1, 1], None), ([-1], None), ([1], 0), ([1], 2)]
)
def test_unpinned_selection_validation(
    monkeypatch: pytest.MonkeyPatch,
    native: FakeOpenBlas,
    cpus: list[int],
    workers: int | None,
) -> None:
    monkeypatch.delattr(os, "sched_getaffinity")
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError):
        CpuThreadPoolExecutor(workers, cpus=cpus)
    assert native.threads == 4


def test_constructor_affinity_failure_remains_visible(
    monkeypatch: pytest.MonkeyPatch,
    native: FakeOpenBlas,
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", Mock(side_effect=OSError("affinity denied")))
    with pytest.raises(OSError, match="affinity denied"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


@pytest.mark.parametrize("failure", ["set", "verify", "initializer"])
def test_worker_setup_failure_breaks_pool(
    failure: str, monkeypatch: pytest.MonkeyPatch, native: FakeOpenBlas
) -> None:
    initializer_ran = Event()

    def initialize() -> None:
        initializer_ran.set()
        if failure == "initializer":
            raise ValueError("bad user initialization")

    def fail_set(pid: int, cpus: set[int]) -> None:
        raise OSError("affinity denied")

    if failure == "set":
        monkeypatch.setattr(os, "sched_setaffinity", fail_set)
    elif failure == "verify":
        main = current_thread()
        original = getattr(os, "sched_getaffinity", None)
        assert original is not None
        monkeypatch.setattr(
            os,
            "sched_getaffinity",
            lambda pid: original(pid) if current_thread() is main else set(),
        )
    with CpuThreadPoolExecutor(1, initializer=initialize) as pool:
        future = pool.submit(lambda: pytest.fail("task must not run"))
        with pytest.raises(BrokenThreadPool):
            future.result(timeout=10)
        with pytest.raises(BrokenThreadPool):
            pool.submit(int)
    assert initializer_ran.is_set() == (failure == "initializer")


def test_map_order_multiple_iterables_and_exception(native: FakeOpenBlas) -> None:
    with CpuThreadPoolExecutor(1) as pool:
        assert list(pool.map(pow, [2, 3, 4], [3, 2], chunksize=5, buffersize=1)) == [8, 9]
        results = pool.map(lambda x: 1 / x, [1, 0, 2])
        assert next(results) == 1
        with pytest.raises(ZeroDivisionError):
            next(results)
        assert pool.submit(int, "7").result(timeout=10) == 7
        with pytest.raises(ValueError):
            pool.submit(int, "bad").result(timeout=10)


@pytest.mark.parametrize("buffersize, consumed", [(None, 6), (2, 2)])
def test_map_submission_bound(buffersize: int | None, consumed: int, native: FakeOpenBlas) -> None:
    seen = []
    release = Event()

    def inputs() -> Iterator[int]:
        for i in range(6):
            seen.append(i)
            yield i

    def task(i: int) -> int:
        assert release.wait(10)
        return i

    with CpuThreadPoolExecutor(1) as pool:
        try:
            results = pool.map(task, inputs(), buffersize=buffersize)
            assert len(seen) == consumed
        finally:
            release.set()
        assert list(results) == list(range(6))


def test_map_timeout_and_invalid_buffer(native: FakeOpenBlas) -> None:
    release = Event()
    with CpuThreadPoolExecutor(1) as pool:
        try:
            result = pool.map(lambda _: release.wait(10), [1], timeout=0)
            with pytest.raises(TimeoutError):
                next(result)
            with pytest.raises(ValueError):
                pool.map(int, [1], buffersize=0)
        finally:
            release.set()


def test_context_exception_does_not_cancel_pending_work(native: FakeOpenBlas) -> None:
    completed = Event()
    with pytest.raises(ValueError, match="body"):
        with CpuThreadPoolExecutor(1) as pool:
            pool.submit(completed.set)
            raise ValueError("body")
    assert completed.is_set()


@pytest.mark.parametrize("without_affinity", [False, True])
def test_native_numerical_calls_and_free_threaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
    without_affinity: bool,
) -> None:
    if without_affinity:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.delattr(os, "sched_setaffinity", raising=False)
        monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    controllers = ThreadpoolController().lib_controllers
    accelerate = execution._loaded_accelerate()
    assert controllers or accelerate is not None
    before = [c.get_num_threads() for c in controllers]
    accelerate_before = accelerate.get_threading() if accelerate is not None else None
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    reference = solve(matrix, rhs)

    def compute() -> NDArray[np.float64]:
        assert all(c.get_num_threads() == 1 for c in controllers)
        if accelerate is not None:
            assert accelerate.get_threading() == 1
        assert not sys._is_gil_enabled()
        np.testing.assert_allclose(matrix @ matrix, [[17, 7], [7, 10]])
        return solve(matrix, rhs)

    workers = (
        min(2, len(affinity()))
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
        else 1
    )
    with CpuThreadPoolExecutor(workers) as pool:
        for result in pool.map(lambda _: compute(), range(8), buffersize=2):
            np.testing.assert_allclose(result, reference)
    assert [c.get_num_threads() for c in controllers] == before
    if accelerate is not None:
        assert accelerate.get_threading() == accelerate_before
    assert not sys._is_gil_enabled()


@requires_linux_affinity
def test_real_native_limits_restore_nondefault_counts() -> None:
    controllers = ThreadpoolController().lib_controllers
    before = [c.get_num_threads() for c in controllers]
    assert controllers
    try:
        for controller in controllers:
            controller.set_num_threads(2)
            assert controller.get_num_threads() == 2
        with CpuThreadPoolExecutor(1) as pool:
            counts = pool.submit(lambda: [c.get_num_threads() for c in controllers]).result(10)
            assert counts == [1] * len(controllers)
        assert [c.get_num_threads() for c in controllers] == [2] * len(controllers)
    finally:
        for controller, count in zip(controllers, before, strict=True):
            controller.set_num_threads(count)


class FakeOpenBlas:
    internal_api = "openblas"
    threading_layer = "pthreads"

    def __init__(self, path: str = "fake-blas", threads: int = 4) -> None:
        self.filepath = path
        self.threads = threads
        self.restored = Event()

    def get_num_threads(self) -> int:
        return self.threads

    def set_num_threads(self, threads: int) -> None:
        self.threads = threads
        if threads == 4:
            self.restored.set()


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenBlas]:
    # Unit tests simulate placement only on hosts without the Linux API. Real
    # pinning tests remain Linux-only; unpinned tests explicitly remove these APIs.
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        state = local()
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: getattr(state, "cpus", {0, 1}), raising=False
        )
        monkeypatch.setattr(
            os, "sched_setaffinity", lambda pid, cpus: setattr(state, "cpus", cpus), raising=False
        )
    backend = FakeOpenBlas()
    monkeypatch.setattr(execution, "_loaded_accelerate", lambda: None)
    monkeypatch.setattr(
        execution, "ThreadpoolController", lambda: SimpleNamespace(lib_controllers=[backend])
    )
    yield backend
    with execution._native_lock:
        assert execution._native_users == 0
        assert not execution._native_original
        assert not execution._native_inventory


def test_overlapping_leases_and_non_lifo_shutdown(native: FakeOpenBlas) -> None:
    first = CpuThreadPoolExecutor(1)
    second = CpuThreadPoolExecutor(1)
    try:
        assert native.threads == 1
        first.shutdown()
        assert native.threads == 1
        assert second.submit(pow, 3, 2).result(timeout=10) == 9
    finally:
        first.shutdown()
        second.shutdown()
    assert native.threads == 4


def test_async_shutdown_retains_limits_through_callbacks(native: FakeOpenBlas) -> None:
    entered, release, callback_entered, callback_release = (Event() for _ in range(4))

    def task() -> None:
        entered.set()
        assert release.wait(10)

    def callback(future: Future[None]) -> None:
        callback_entered.set()
        assert callback_release.wait(10)
        assert native.threads == 1

    pool = CpuThreadPoolExecutor(1)
    try:
        active = pool.submit(task)
        assert entered.wait(10)
        active.add_done_callback(callback)
        pending = pool.submit(int, 5)
        pool.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False)
        assert pending.cancelled()
        assert native.threads == 1
        release.set()
        assert callback_entered.wait(10)
        assert active.done()
        assert native.threads == 1
    finally:
        release.set()
        callback_release.set()
        pool.shutdown()
    assert native.restored.wait(10)
    assert native.threads == 4


def test_gc_and_nonblocking_shutdown_restore_without_second_shutdown(native: FakeOpenBlas) -> None:
    pool = CpuThreadPoolExecutor(1)
    assert pool.submit(int, 1).result(timeout=10) == 1
    pool.shutdown(wait=False)
    assert native.restored.wait(10)
    native.restored.clear()
    pool = CpuThreadPoolExecutor(1)
    weak = ref(pool)
    assert pool.submit(int, 2).result(timeout=10) == 2
    del pool
    gc.collect()
    assert weak() is None
    assert native.restored.wait(10)


def test_gc_with_pending_work_does_not_block(native: FakeOpenBlas) -> None:
    release, entered = Event(), Event()

    def task() -> None:
        entered.set()
        assert release.wait(10)
        assert native.threads == 1

    pool = CpuThreadPoolExecutor(1)
    future = pool.submit(task)
    try:
        assert entered.wait(10)
        del pool
        gc.collect()
        assert native.threads == 1
    finally:
        release.set()
    future.result(timeout=10)
    assert native.restored.wait(10)


def test_finalizer_without_cleanup_thread(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = CpuThreadPoolExecutor(1)
    assert pool.submit(int, 2).result(timeout=10) == 2

    def fail_start(self: object) -> None:
        raise RuntimeError("cannot create new thread at interpreter shutdown")

    monkeypatch.setattr(execution.Thread, "start", fail_start)
    del pool
    gc.collect()
    assert native.threads == 4


def test_nonblocking_shutdown_drains_without_cancellation(native: FakeOpenBlas) -> None:
    release, entered = Event(), Event()

    def task() -> None:
        entered.set()
        assert release.wait(10)

    pool = CpuThreadPoolExecutor(1)
    try:
        pool.submit(task)
        assert entered.wait(10)
        pending = pool.submit(int, 5)
        pool.shutdown(wait=False)
        assert not pending.cancelled()
        assert native.threads == 1
    finally:
        release.set()
        pool.shutdown()
    assert pending.result(timeout=10) == 5
    assert native.threads == 4


def test_concurrent_construction_and_shutdown(native: FakeOpenBlas) -> None:
    barrier = Barrier(4)

    def run() -> None:
        with CpuThreadPoolExecutor(1) as pool:
            barrier.wait(timeout=10)
            assert native.threads == 1
            assert pool.submit(int, 3).result(timeout=10) == 3
            barrier.wait(timeout=10)

    with ThreadPoolExecutor(4) as owners:
        for future in [owners.submit(run) for _ in range(4)]:
            future.result(timeout=10)
    assert native.threads == 4


def test_invalid_initializer_and_constructor_rollback(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(TypeError, match="initializer"):
        CpuThreadPoolExecutor(1, initializer=42)  # type: ignore[bad-argument-type]
    assert native.threads == 4

    def fail(*args: object) -> None:
        raise RuntimeError("pool construction failed")

    monkeypatch.setattr(execution, "ThreadPoolExecutor", fail)
    with pytest.raises(RuntimeError, match="construction"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


@pytest.mark.parametrize(
    "api, layer",
    [("mkl", "intel"), ("blis", "pthreads"), ("flexiblas", ""), ("openblas", "openmp")],
)
def test_unsupported_native_backend(native: FakeOpenBlas, api: str, layer: str) -> None:
    native.internal_api, native.threading_layer = api, layer
    with pytest.raises(RuntimeError, match="Unsupported native"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


def test_no_backend(native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execution, "ThreadpoolController", lambda: SimpleNamespace(lib_controllers=[])
    )
    with pytest.raises(RuntimeError, match="No supported"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


def test_changed_library_inventory_or_limits(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    with CpuThreadPoolExecutor(1):
        native.threads = 3
        with pytest.raises(RuntimeError, match="limits changed"):
            CpuThreadPoolExecutor(1)
        native.threads = 1
        monkeypatch.setattr(
            execution,
            "ThreadpoolController",
            lambda: SimpleNamespace(lib_controllers=[native, FakeOpenBlas("new")]),
        )
        with pytest.raises(RuntimeError, match="libraries changed"):
            CpuThreadPoolExecutor(1)
    assert native.threads == 4


def test_partial_native_setup_rollback(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = FakeOpenBlas("second")

    def ignore_limit(threads: int) -> None:
        pass

    monkeypatch.setattr(second, "set_num_threads", ignore_limit)
    monkeypatch.setattr(
        execution, "ThreadpoolController", lambda: SimpleNamespace(lib_controllers=[native, second])
    )
    with pytest.raises(RuntimeError, match="enforce"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


def test_restoration_failure_is_reported_and_retryable(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = CpuThreadPoolExecutor(1)
    setter = native.set_num_threads
    monkeypatch.setattr(native, "set_num_threads", lambda threads: None)
    try:
        with pytest.raises(ExceptionGroup, match="restore"):
            pool.shutdown()
    finally:
        monkeypatch.setattr(native, "set_num_threads", setter)
        pool.shutdown()
    assert native.threads == 4


@pytest.mark.parametrize("shutdown", ["pool.shutdown(wait=False)", ""])
@requires_linux_affinity
def test_interpreter_exit(shutdown: str) -> None:
    code = f"""
from orsia_numerics.execution import CpuThreadPoolExecutor
pool = CpuThreadPoolExecutor(1)
future = pool.submit(pow, 7, 2)
{shutdown}
assert future.result() == 49
print('finished')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "finished"
    assert not result.stderr


class FakeOpenMP:
    """Task-local settings: changing the constructor thread must not fix workers."""

    internal_api = "openmp"

    def __init__(self, path: str = "fake-openmp") -> None:
        self.filepath = path
        self.state = local()
        self.dynlib = SimpleNamespace(
            omp_set_num_threads=Mock(side_effect=lambda n: setattr(self.state, "threads", n)),
            omp_get_max_threads=Mock(side_effect=lambda: getattr(self.state, "threads", 4)),
            omp_set_max_active_levels=Mock(side_effect=lambda n: setattr(self.state, "levels", n)),
            omp_get_max_active_levels=Mock(side_effect=lambda: getattr(self.state, "levels", 3)),
        )

    def counts(self) -> tuple[int, int]:
        return self.dynlib.omp_get_max_threads(), self.dynlib.omp_get_max_active_levels()


@pytest.fixture
def openmp(native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch) -> FakeOpenMP:
    backend = FakeOpenMP()
    monkeypatch.setattr(
        execution,
        "ThreadpoolController",
        lambda: SimpleNamespace(lib_controllers=[native, backend]),
    )
    return backend


def test_openmp_initializer_tasks_callbacks_and_overlapping_pools(openmp: FakeOpenMP) -> None:
    initialized = Event()
    entered, release, callback_done = Event(), Event(), Event()

    def initialize() -> None:
        assert openmp.counts() == (1, 0)
        initialized.set()

    def task() -> tuple[int, int]:
        entered.set()
        assert release.wait(10)
        return openmp.counts()

    observed = []

    def callback(future: Future[tuple[int, int]]) -> None:
        observed.append(openmp.counts())
        callback_done.set()

    first = CpuThreadPoolExecutor(1, initializer=initialize)
    second = CpuThreadPoolExecutor(1)
    try:
        future = first.submit(task)
        assert entered.wait(10)
        assert initialized.is_set()
        future.add_done_callback(callback)
        first.shutdown(wait=False)
        assert openmp.counts() == (4, 3)  # Caller never acquired task-local limits.
        assert second.submit(openmp.counts).result(10) == (1, 0)
        second.shutdown()
        release.set()
        assert future.result(10) == (1, 0)
        assert callback_done.wait(10)
        assert observed == [(1, 0)]
    finally:
        release.set()
        first.shutdown()
        second.shutdown()
    assert openmp.counts() == (4, 3)


def test_openmp_reasserted_before_each_task(openmp: FakeOpenMP) -> None:
    def disturb() -> None:
        openmp.dynlib.omp_set_num_threads(8)
        openmp.dynlib.omp_set_max_active_levels(4)

    # Deliberate contract violation exercises defence between user calls.
    with CpuThreadPoolExecutor(1, initializer=disturb) as pool:
        assert pool.submit(openmp.counts).result(10) == (1, 0)
        pool.submit(disturb).result(10)
        assert pool.submit(openmp.counts).result(10) == (1, 0)


@pytest.mark.parametrize("failure", ["threads", "levels"])
def test_openmp_worker_limit_failure(openmp: FakeOpenMP, failure: str) -> None:
    setter = "omp_set_num_threads" if failure == "threads" else "omp_set_max_active_levels"
    getattr(openmp.dynlib, setter).side_effect = None
    initialized = Event()
    with CpuThreadPoolExecutor(1, initializer=initialized.set) as pool:
        with pytest.raises(BrokenThreadPool):
            pool.submit(lambda: pytest.fail("task must not run")).result(10)
    assert not initialized.is_set()


def test_openmp_task_limit_failure_is_a_future_error(openmp: FakeOpenMP) -> None:
    with CpuThreadPoolExecutor(1) as pool:
        assert pool.submit(openmp.counts).result(10) == (1, 0)
        openmp.dynlib.omp_get_max_active_levels.side_effect = lambda: 2
        with pytest.raises(RuntimeError, match="serial OpenMP"):
            pool.submit(lambda: pytest.fail("task must not run")).result(10)


def test_native_limit_change_prevents_task(native: FakeOpenBlas) -> None:
    with CpuThreadPoolExecutor(1) as pool:
        assert pool.submit(int, 7).result(10) == 7
        native.threads = 4
        with pytest.raises(RuntimeError, match="limits changed"):
            pool.submit(lambda: pytest.fail("task must not run")).result(10)


@pytest.mark.parametrize(
    "symbol",
    [
        "omp_set_num_threads",
        "omp_get_max_threads",
        "omp_set_max_active_levels",
        "omp_get_max_active_levels",
    ],
)
def test_openmp_missing_control_rejected(openmp: FakeOpenMP, symbol: str) -> None:
    delattr(openmp.dynlib, symbol)
    with pytest.raises(RuntimeError, match="Missing OpenMP"):
        CpuThreadPoolExecutor(1)


def test_multiple_openmp_runtimes_rejected(
    native: FakeOpenBlas, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution,
        "ThreadpoolController",
        lambda: SimpleNamespace(lib_controllers=[native, FakeOpenMP(), FakeOpenMP("other")]),
    )
    with pytest.raises(RuntimeError, match="Multiple OpenMP"):
        CpuThreadPoolExecutor(1)
    assert native.threads == 4


def test_openmp_library_inventory_change_rejected(openmp: FakeOpenMP) -> None:
    with CpuThreadPoolExecutor(1):
        openmp.filepath = "new-openmp"
        with pytest.raises(RuntimeError, match="libraries changed"):
            CpuThreadPoolExecutor(1)
