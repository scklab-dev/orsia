"""Run caller-supplied CPU tasks with explicit affinity and native thread limits.

Linux affinity is mandatory. Pthreads OpenBLAS is limited process-wide; OpenMP
parallel regions are serialized in each worker. Load native dependencies before
construction; do not change native limits or affinity from tasks. See the package
README for the supported runtimes and the limits of this cooperative contract.
"""

import os
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from ctypes import c_int
from threading import Lock, Thread
from typing import Any
from weakref import finalize

from threadpoolctl import ThreadpoolController

__all__ = ["CpuThreadPoolExecutor"]


class CpuThreadPoolExecutor(Executor):
    """Schedule futures on distinct pinned CPUs with bounded native parallelism.

    ``cpus=None`` selects the constructing thread's allowed CPUs in ascending
    order. Explicit CPU order is preserved. ``max_workers=None`` uses all selected
    CPUs; otherwise the first ``max_workers`` CPUs are used. CPUs must be distinct,
    available and nonempty. Affinity is set and verified before the user initializer;
    initialization failure breaks the pool as in ``ThreadPoolExecutor``.

    Construction acquires a shared process-wide native-limit lease. Workers start
    lazily. Shutdown releases the lease only after workers (including their future
    callbacks) exit; nonblocking shutdown uses a joining thread. Always prefer an
    explicit context manager or shutdown over garbage-collection cleanup.

    ``map`` and context management are inherited from the public standard-library
    ``Executor`` contract, including Python 3.14 ``buffersize=None`` semantics.
    Task arguments are shared, not copied: callers own mutation, RNGs and partitioning.
    Distinct CPUs are guaranteed within a pool, not reserved across independent pools.

    Args:
        max_workers: Number of selected CPUs to use; None selects all of them.
        thread_name_prefix: Optional name prefix for pool worker threads.
        initializer: Optional callable run in each pinned worker before tasks.
        initargs: Positional arguments passed to the worker initializer.
        cpus: Distinct available logical CPU IDs in preferred worker order; None uses affinity.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        initializer: Callable[..., None] | None = None,
        initargs: tuple[Any, ...] = (),
        *,
        cpus: Iterable[int] | None = None,
    ) -> None:
        """Prepare a pool of pinned workers and acquire the process-wide native lease.

        Args:
            max_workers: Number of selected CPUs to use; None selects all of them.
            thread_name_prefix: Optional name prefix for pool worker threads.
            initializer: Optional callable run in each pinned worker before tasks.
            initargs: Positional arguments passed to the worker initializer.
            cpus: Distinct available logical CPU IDs in preferred worker order; None uses affinity.

        Raises:
            NotImplementedError: Linux CPU affinity is unavailable.
            ValueError: CPU selection or worker count is invalid.
            TypeError: The supplied initializer is not callable.
            RuntimeError: A supported native runtime or thread limit cannot be enforced.
        """
        selected = _select_cpus(cpus, max_workers)
        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        lease = _NativeLease()
        try:
            remaining = iter(selected)
            placement_lock = Lock()

            def initialize() -> None:
                with placement_lock:
                    cpu = next(remaining)
                os.sched_setaffinity(0, {cpu})
                if os.sched_getaffinity(0) != {cpu}:
                    raise RuntimeError(f"Could not enforce worker affinity on CPU {cpu}")
                lease.limit_worker()
                if initializer is not None:
                    initializer(*initargs)

            pool = ThreadPoolExecutor(len(selected), thread_name_prefix, initialize)
            self._resources = _PoolResources(pool, lease)
            self._finalizer = finalize(self, _finalize_pool, self._resources)
        except BaseException:
            lease.release()
            raise

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Schedule work under the native limit and return a standard future.

        Args:
            fn: Callable to run on a pinned worker.
            *args: Positional arguments passed to the callable, shared by reference.
            **kwargs: Keyword arguments passed to the callable, shared by reference.

        Returns:
            Future resolving to the callable's result or raised exception.
        """
        return self._resources.pool.submit(_run_limited, self._resources.lease, fn, args, kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Release pool resources after workers exit and complete their callbacks.

        Args:
            wait: Wait for workers to exit before returning when true.
            cancel_futures: Request cancellation of queued, unstarted tasks.
        """
        self._resources.close(wait, cancel_futures=cancel_futures)
        self._finalizer.detach()


def _run_limited[**P, T](
    lease: _NativeLease, fn: Callable[P, T], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> T:
    lease.limit_worker()
    return fn(*args, **kwargs)


def _select_cpus(cpus: Iterable[int] | None, max_workers: int | None) -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError("CpuThreadPoolExecutor requires Linux CPU affinity")
    allowed = os.sched_getaffinity(0)
    selected = tuple(sorted(allowed) if cpus is None else cpus)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("cpus must be nonempty and contain no duplicates")
    if not set(selected) <= allowed:
        raise ValueError("cpus must belong to the constructing thread's allowed CPU set")
    workers = len(selected) if max_workers is None else max_workers
    if workers <= 0 or workers > len(selected):
        raise ValueError("max_workers must be positive and no greater than the CPU count")
    return selected[:workers]


class _PoolResources:
    """Own the stdlib pool and its lease independently of the public wrapper."""

    def __init__(self, pool: ThreadPoolExecutor, lease: _NativeLease) -> None:
        self.pool = pool
        self.lease = lease
        self.lock = Lock()
        self.reaper: Thread | None = None

    def close(self, wait: bool, *, cancel_futures: bool = False) -> None:
        # Never hold our lock while joining workers: callbacks may call shutdown.
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:
            self.lease.release()
        else:
            with self.lock:
                if self.reaper is None:
                    reaper = Thread(target=self._join, name="orsia-cpu-shutdown", daemon=False)
                    reaper.start()
                    self.reaper = reaper

    def _join(self) -> None:
        self.pool.shutdown(wait=True)
        self.lease.release()


def _finalize_pool(resources: _PoolResources) -> None:
    try:
        resources.close(False)
    except RuntimeError:
        # At interpreter exit Python has already joined stdlib pool workers and
        # forbids creating the cleanup thread. Synchronous release is safe then.
        # This also permits cleanup if thread creation fails during ordinary GC.
        resources.close(True)


_native_lock = Lock()
_native_users = 0
_native_original: list[tuple[Any, int]] = []
_native_inventory: set[str] = set()


class _OpenMPControl:
    """Bind a loaded runtime's task-local controls, without changing the caller."""

    def __init__(self, controller: Any) -> None:
        try:
            self.set_threads = controller.dynlib.omp_set_num_threads
            self.get_threads = controller.dynlib.omp_get_max_threads
            self.set_levels = controller.dynlib.omp_set_max_active_levels
            self.get_levels = controller.dynlib.omp_get_max_active_levels
        except AttributeError as error:
            raise RuntimeError(
                f"Missing OpenMP serialization controls: {controller.filepath}"
            ) from error
        for setter in (self.set_threads, self.set_levels):
            setter.argtypes = [c_int]
            setter.restype = None
        for getter in (self.get_threads, self.get_levels):
            getter.argtypes = []
            getter.restype = c_int

    def limit(self) -> None:
        self.set_threads(1)
        # Zero serializes even regions with an explicit num_threads clause;
        # disabling only nested parallelism (one active level) is insufficient.
        self.set_levels(0)
        if self.get_threads() != 1 or self.get_levels() != 0:
            raise RuntimeError("Could not enforce serial OpenMP execution in worker")


class _NativeLease:
    """Reference-count process-wide settings; never nest restoring limit scopes."""

    def __init__(self) -> None:
        global _native_users, _native_original, _native_inventory
        self.released = False
        # Ensure both declared numerical dependencies' BLAS libraries are loaded
        # before discovering controls. Importing this module alone has no effects.
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401

        with _native_lock:
            controllers = ThreadpoolController().lib_controllers
            openmp = [c for c in controllers if c.internal_api == "openmp"]
            if len(openmp) > 1:
                raise RuntimeError("Multiple OpenMP runtimes are not supported")
            self.openmp = [_OpenMPControl(c) for c in openmp]
            blas = [c for c in controllers if c.internal_api != "openmp"]
            for controller in blas:
                if (
                    controller.internal_api != "openblas"
                    or controller.threading_layer != "pthreads"
                ):
                    raise RuntimeError(
                        "Unsupported native runtime: "
                        f"{controller.internal_api} ({controller.filepath}); "
                        "only pthreads OpenBLAS plus one OpenMP runtime are supported"
                    )
            if not blas:
                raise RuntimeError("No supported native BLAS runtime was detected")
            inventory = {c.filepath for c in controllers}
            if _native_users:
                if inventory != _native_inventory:
                    raise RuntimeError("Native libraries changed while CPU executors are alive")
                if any(c.get_num_threads() != 1 for c in blas):
                    raise RuntimeError("Native thread limits changed while CPU executors are alive")
            else:
                original = [(c, c.get_num_threads()) for c in blas]
                try:
                    for controller, _ in original:
                        controller.set_num_threads(1)
                        if controller.get_num_threads() != 1:
                            raise RuntimeError("Could not enforce one native thread")
                except BaseException:
                    _restore_native(original)
                    raise
                _native_original = original
                _native_inventory = inventory
            self.blas = blas
            _native_users += 1

    def limit_worker(self) -> None:
        # OpenMP ICVs belong to the calling task/thread, not the pool constructor.
        # Keep them for this worker's entire lifetime, including future callbacks.
        for control in self.openmp:
            control.limit()
        if any(c.get_num_threads() != 1 for c in self.blas):
            raise RuntimeError("Native thread limits changed while CPU executors are alive")

    def release(self) -> None:
        global _native_users, _native_original, _native_inventory
        with _native_lock:
            if self.released:
                return
            if _native_users == 1:
                _restore_native(_native_original)
                _native_original = []
                _native_inventory = set()
            _native_users -= 1
            self.released = True


def _restore_native(original: list[tuple[Any, int]]) -> None:
    errors = []
    for controller, threads in reversed(original):
        try:
            controller.set_num_threads(threads)
            if controller.get_num_threads() != threads:
                raise RuntimeError(f"Could not restore native threads for {controller.filepath}")
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Could not restore native thread limits", errors)
