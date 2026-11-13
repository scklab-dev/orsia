"""Accelerate's worker-local BLAS/LAPACK limit and failure contracts."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures.thread import BrokenThreadPool
from ctypes import c_int, c_uint
from threading import Event, local
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from orsia_numerics import execution
from orsia_numerics.execution import CpuThreadPoolExecutor
from scipy.linalg import solve

pytestmark = pytest.mark.filterwarnings("ignore:CPU affinity APIs are unavailable:RuntimeWarning")

ACCELERATE_BLAS = (
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/"
    "vecLib.framework/Versions/A/libBLAS.dylib"
)


class FakeAccelerate:
    def __init__(self) -> None:
        self.state = local()
        self.BLASSetThreading = Mock(side_effect=self.set_threading)
        self.BLASGetThreading = Mock(side_effect=lambda: getattr(self.state, "threading", 0))

    def set_threading(self, mode: int) -> int:
        self.state.threading = mode
        return 0


@pytest.fixture
def accelerate(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAccelerate]:
    library = FakeAccelerate()
    monkeypatch.setattr(execution.sys, "platform", "darwin")
    monkeypatch.setattr(execution, "dllist", lambda: [ACCELERATE_BLAS])
    monkeypatch.setattr(execution, "CDLL", lambda path: library)
    monkeypatch.setattr(
        execution, "ThreadpoolController", lambda: SimpleNamespace(lib_controllers=[])
    )
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    yield library
    assert execution._native_users == 0
    assert not execution._native_original
    assert not execution._native_inventory


def test_accelerate_binding_and_absent_detection(
    accelerate: FakeAccelerate, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = execution._loaded_accelerate()
    assert control is not None
    assert control.filepath == ACCELERATE_BLAS
    assert accelerate.BLASSetThreading.argtypes == [c_uint]
    assert accelerate.BLASSetThreading.restype is c_int
    assert accelerate.BLASGetThreading.argtypes == []
    assert accelerate.BLASGetThreading.restype is c_uint
    monkeypatch.setattr(execution, "dllist", lambda: ["/unrelated/libBLAS.dylib"])
    assert execution._loaded_accelerate() is None
    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.setattr(execution, "dllist", Mock(side_effect=AssertionError("discovery called")))
    assert execution._loaded_accelerate() is None


@pytest.mark.parametrize("symbol", ["BLASSetThreading", "BLASGetThreading"])
def test_accelerate_missing_symbols_fail_construction(
    accelerate: FakeAccelerate, symbol: str
) -> None:
    delattr(accelerate, symbol)
    with pytest.raises(RuntimeError, match="Missing Apple Accelerate threading controls"):
        CpuThreadPoolExecutor(1)


@pytest.mark.parametrize("failure", ["set", "readback"])
def test_accelerate_limit_failure_breaks_pool(accelerate: FakeAccelerate, failure: str) -> None:
    if failure == "set":
        accelerate.BLASSetThreading.side_effect = lambda mode: -1
    else:
        accelerate.BLASGetThreading.side_effect = lambda: 0
    initialized = Event()
    with CpuThreadPoolExecutor(1, initializer=initialized.set) as pool:
        with pytest.raises(BrokenThreadPool):
            pool.submit(lambda: pytest.fail("task must not run")).result(10)
    assert not initialized.is_set()


def test_accelerate_reasserted_and_task_failure_is_visible(accelerate: FakeAccelerate) -> None:
    def disturb() -> None:
        accelerate.BLASSetThreading(0)

    with CpuThreadPoolExecutor(1, initializer=disturb) as pool:
        assert pool.submit(accelerate.BLASGetThreading).result(10) == 1
        pool.submit(disturb).result(10)
        assert pool.submit(accelerate.BLASGetThreading).result(10) == 1
        accelerate.BLASSetThreading.side_effect = lambda mode: -1
        with pytest.raises(RuntimeError, match="single-threaded Apple Accelerate"):
            pool.submit(lambda: pytest.fail("task must not run")).result(10)
    assert accelerate.BLASGetThreading() == 0


def test_accelerate_inventory_change_is_visible(
    accelerate: FakeAccelerate, monkeypatch: pytest.MonkeyPatch
) -> None:
    with CpuThreadPoolExecutor(1):
        monkeypatch.setattr(execution, "dllist", lambda: [])
        with pytest.raises(RuntimeError, match="No supported native BLAS"):
            CpuThreadPoolExecutor(1)


def check_worker_lifecycle(control: execution._AccelerateControl) -> None:
    """Check worker-local limits through real calls, callbacks and overlapping pools."""
    before = control.get_threading()
    entered, release, callback_done = Event(), Event(), Event()
    callback_modes = []
    initializer_modes = []

    def initialize() -> None:
        initializer_modes.append(control.get_threading())

    def task() -> int:
        entered.set()
        assert release.wait(10)
        assert not sys._is_gil_enabled()
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([1.0, 2.0])
        np.testing.assert_allclose(matrix @ solve(matrix, rhs), rhs)
        return control.get_threading()

    def callback(future: Future[int]) -> None:
        callback_modes.append(control.get_threading())
        callback_done.set()

    first = CpuThreadPoolExecutor(1, initializer=initialize)
    second = CpuThreadPoolExecutor(1)
    try:
        future = first.submit(task)
        assert entered.wait(10)
        future.add_done_callback(callback)
        first.shutdown(wait=False)
        assert control.get_threading() == before
        assert second.submit(control.get_threading).result(10) == 1
        second.shutdown()
        release.set()
        assert future.result(10) == 1
        assert callback_done.wait(10)
        assert initializer_modes == [1]
        assert callback_modes == [1]
    finally:
        release.set()
        first.shutdown()
        second.shutdown()
    assert control.get_threading() == before
    assert not sys._is_gil_enabled()


def test_accelerate_worker_lifecycle(accelerate: FakeAccelerate) -> None:
    control = execution._loaded_accelerate()
    assert control is not None
    check_worker_lifecycle(control)


@pytest.mark.skipif(sys.platform != "darwin", reason="Real Accelerate requires macOS")
def test_real_accelerate_worker_lifecycle() -> None:
    control = execution._loaded_accelerate()
    if control is None:
        pytest.skip("Accelerate BLAS is not loaded")
    before = control.get_threading()
    try:
        assert control.set_threading(0) == 0
        check_worker_lifecycle(control)
    finally:
        assert control.set_threading(before) == 0
