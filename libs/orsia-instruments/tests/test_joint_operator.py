"""Bounded physical photon transpose against the dense all-port reference."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from orsia_instruments.response.joint import spatial_spectral_operator
from test_spectral_chains import grid, joint, port, spectral


def test_all_ports_blocks_mask_transpose_and_concurrency():
    cameras = [port("+"), port("-")]
    grids = [grid(c, order=2) for c in cameras]
    reference = joint(cameras, grids).matrix
    valid = np.arange(reference.shape[0]) % 3 != 0
    saved = valid.copy()
    kwargs = dict(
        x_edges_m=np.array([-1.0, 0.0, 1.0]),
        y_edges_m=np.array([-1.0, 1.0]),
        spectral_edges_cm1=np.array([4500.0, 5000.0, 5500.0]),
        max_bytes=100_000_000,
        read_batch_size=3,
    )
    op = spatial_spectral_operator(cameras, grids, spectral(), valid=valid, **kwargs)
    valid[:] = False
    rng = np.random.default_rng(52)
    v = rng.normal(size=(4, 3))
    w = rng.normal(size=(saved.sum(), 3))
    np.testing.assert_allclose(op @ v, reference[saved] @ v, rtol=5e-15, atol=1e-8)
    np.testing.assert_allclose(op.T @ w, reference[saved].T @ w, rtol=5e-14, atol=1e-8)
    np.testing.assert_allclose(np.sum((op @ v) * w), np.sum(v * (op.T @ w)), rtol=5e-14)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda _: op.T @ w, range(3)):
            np.testing.assert_array_equal(result, op.T @ w)
    assert (op @ np.empty((4, 0))).shape == (saved.sum(), 0)
    kwargs["max_bytes"] = 1
    small = spatial_spectral_operator(cameras, grids, spectral(), **kwargs)
    with pytest.raises(MemoryError):
        small @ v
    with pytest.raises(ValueError):
        spatial_spectral_operator(cameras, grids, spectral(), valid=valid, **kwargs)
