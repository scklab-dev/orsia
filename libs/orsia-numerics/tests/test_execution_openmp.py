"""Real OpenMP team-size regression, isolated from the pytest process."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def openmp_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("OpenMP integration requires the Devbox GCC toolchain")
    source = Path(__file__).with_name("native") / "openmp_probe.c"
    library = tmp_path_factory.mktemp("openmp") / "probe.so"
    result = subprocess.run(
        [compiler, "-shared", "-fPIC", "-fopenmp", str(source), "-o", str(library)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return library


@pytest.mark.parametrize("nonblocking", [False, True])
def test_real_openmp_serial_regions(openmp_probe: Path, nonblocking: bool) -> None:
    code = """
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from threadpoolctl import ThreadpoolController
from orsia_numerics.execution import CpuThreadPoolExecutor
import scipy.linalg

probe = ctypes.CDLL(sys.argv[1])
controllers = ThreadpoolController().lib_controllers
omp, = [c for c in controllers if c.internal_api == 'openmp']
omp.dynlib.omp_set_max_active_levels(4)
omp.set_num_threads(4)
blas = [c for c in controllers if c.internal_api == 'openblas']
assert blas
for c in blas:
    c.set_num_threads(3)

def counts():
    return (omp.get_num_threads(), omp.dynlib.omp_get_max_active_levels(),
            probe.default_team_size(), probe.explicit_team_size(), probe.nested_team_size())

thread_limit = omp.dynlib.omp_get_thread_limit()
team_size = min(4, thread_limit)
# Inner teams share the remaining capacity and can reuse it as siblings finish.
# Their total is deterministic when all sixteen threads fit, or only the outer team fits.
nested_min = min(16, thread_limit)
nested_max = team_size * min(4, thread_limit - team_size + 1)

def assert_caller_counts(observed):
    assert observed[:4] == (4, 4, team_size, team_size), observed
    assert nested_min <= observed[4] <= nested_max, observed

assert_caller_counts(counts())
assert not sys._is_gil_enabled()
workers = min(2, len(os.sched_getaffinity(0)))
barrier = Barrier(workers)
entered, release, callback_done = Event(), Event(), Event()
callbacks = []

def initialize():
    assert counts() == (1, 0, 1, 1, 1)
    assert all(c.get_num_threads() == 1 for c in blas)

def task():
    barrier.wait(timeout=20)
    initialize()
    return counts()

def blocked():
    entered.set()
    assert release.wait(20)
    return counts()

def callback(future):
    callbacks.append(counts())
    callback_done.set()

first = CpuThreadPoolExecutor(workers, initializer=initialize)
second = CpuThreadPoolExecutor(1)
try:
    futures = [first.submit(task) for _ in range(workers)]
    assert [f.result(20) for f in futures] == [(1, 0, 1, 1, 1)] * workers
    assert_caller_counts(counts())  # Constructor thread is unchanged.
    with ThreadPoolExecutor(1) as external:
        assert_caller_counts(external.submit(counts).result(20))
    future = first.submit(blocked)
    assert entered.wait(20)
    future.add_done_callback(callback)
    if sys.argv[2] == 'True':
        first.shutdown(wait=False)
    assert second.submit(counts).result(20) == (1, 0, 1, 1, 1)
    second.shutdown()
    assert all(c.get_num_threads() == 1 for c in blas)
    release.set()
    assert future.result(20) == (1, 0, 1, 1, 1)
    assert callback_done.wait(20)
    assert callbacks == [(1, 0, 1, 1, 1)]
finally:
    release.set()
    first.shutdown()
    second.shutdown()
assert_caller_counts(counts())
assert all(c.get_num_threads() == 3 for c in blas)
assert not sys._is_gil_enabled()
print('OpenMP regions serialized; caller and BLAS settings preserved')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(openmp_probe), str(nonblocking)],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "OMP_MAX_ACTIVE_LEVELS": "4",
            "OMP_DYNAMIC": "FALSE",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OpenMP regions serialized" in result.stdout
    assert not result.stderr
