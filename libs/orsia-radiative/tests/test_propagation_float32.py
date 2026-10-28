"""Single-precision formal transport: dtype, limits and double-reference checks."""

import sys
from concurrent.futures import ThreadPoolExecutor

import formal_reference
import numpy as np
import pytest
from orsia_radiative.transfer import formal
from orsia_radiative.transfer.native import propagation_native


@pytest.mark.parametrize("solar_mu", [None, 0.8, 1e-5])
@pytest.mark.parametrize("cells", [0, 1, 19])
@pytest.mark.parametrize("layout", ["strided", "repeated", "wide", "empty"])
def test_float32_preparation(monkeypatch, solar_mu, cells, layout):
    depth = np.repeat(np.resize(np.array([0, 1e-20, 1e-8, 0.3, 1000], dtype=np.float32), cells), 2)[
        ::-2
    ]
    angle = np.float32(0.8)
    mu = np.array([-1, -angle, -np.nextafter(angle, np.float32(0)), 1e-5, angle], dtype=np.float32)[
        ::-1
    ]
    if layout == "repeated":
        mu = np.tile(mu, 17)
    elif layout == "wide":
        mu = np.linspace(1e-5, 1, 257, dtype=np.float32)
    elif layout == "empty":
        mu = mu[:0]
    depth_snapshot, mu_snapshot = depth.copy(), mu.copy()
    calls = []

    def counted(*args):
        calls.append(1)
        return propagation_native.prepare(*args)

    monkeypatch.setattr(formal, "_native_prepare", counted)
    native = formal.propagation_coefficients(depth, mu, solar_mu=solar_mu)
    assert bool(calls) == (layout != "wide")
    monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    fallback = formal.propagation_coefficients(depth, mu, solar_mu=solar_mu)
    double = formal.propagation_coefficients(
        depth.astype(np.float64), mu.astype(np.float64), solar_mu=solar_mu
    )
    for name, value in vars(native).items():
        expected = getattr(fallback, name)
        if value is None:
            assert expected is None
            continue
        assert value.dtype == np.float32 and expected.dtype == np.float32
        assert value.flags.owndata and not value.flags.writeable
        assert not np.shares_memory(value, depth) and not np.shares_memory(value, mu)
        assert np.isfinite(value).all()
        # Two minimum subnormal ULPs permit different expf/NumPy underflow rounding.
        np.testing.assert_allclose(value, expected, rtol=4e-6, atol=3e-45)
        np.testing.assert_allclose(value, getattr(double, name), rtol=8e-6, atol=3e-45)
    np.testing.assert_array_equal(depth, depth_snapshot)
    np.testing.assert_array_equal(mu, mu_snapshot)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("solar", [False, True])
def test_float32_repeated_application_and_exit_only(monkeypatch, native, solar):
    if not native:
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    depth = np.array([0, 1e-20, 1e-8, 0.2, 1000], dtype=np.float32)
    angle = np.float32(0.8)
    mu = np.array([-angle, angle, -np.nextafter(angle, np.float32(0))], dtype=np.float32)
    source = np.linspace(-0.1, 0.7, depth.size * mu.size, dtype=np.float32).reshape(
        depth.size, mu.size
    )
    coefficients = formal.propagation_coefficients(depth, mu, solar_mu=0.8 if solar else None)
    options = dict(solar_source=solar, bottom_radiance=0.2)
    centres, exits = formal.apply_propagation(source, coefficients, **options)
    assert centres.dtype == exits.dtype == np.float32
    for _ in range(3):
        c, e = formal.apply_propagation(source, coefficients, **options)
        np.testing.assert_array_equal(c, centres)
        np.testing.assert_array_equal(e, exits)
        assert not np.shares_memory(c, centres) and not np.shares_memory(e, exits)
        c, e = formal.apply_propagation(source, coefficients, return_centres=False, **options)
        assert c is None
        np.testing.assert_array_equal(e, exits)
    double = formal.propagation_coefficients(
        depth.astype(np.float64), mu.astype(np.float64), solar_mu=0.8 if solar else None
    )
    c64, e64 = formal.apply_propagation(source.astype(np.float64), double, **options)
    np.testing.assert_allclose(centres, c64, rtol=8e-6, atol=1e-8)
    np.testing.assert_allclose(exits, e64, rtol=8e-6, atol=1e-8)


@pytest.mark.parametrize("native", [False, True])
def test_float32_vacuum_thin_and_resonant_limits(monkeypatch, native):
    if not native:
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    mu = np.array([-0.8, 0.8], dtype=np.float32)
    for depth in (0, 1e-20, 1e-8, 1):
        tau = np.array([depth], dtype=np.float32)
        coefficients = formal.propagation_coefficients(tau, mu, solar_mu=0.8)
        assert coefficients.solar_source_weight is not None
        # Downward resonant solar/ray cosines: the integrand is constant exp(-tau/mu).
        np.testing.assert_allclose(
            coefficients.solar_source_weight[0, 0],
            np.exp(-float(tau[0]) / float(mu[0] * -1)) / float(-mu[0]),
            rtol=2e-7,
        )
        if depth < 1e-7:
            np.testing.assert_allclose(coefficients.source_weight, [[1.25, 1.25]], rtol=2e-7)
            np.testing.assert_allclose(
                coefficients.centre_source_weight, [[0.625, 0.625]], rtol=2e-7
            )
    tau = np.zeros(3, dtype=np.float32)
    centres, exits = formal.propagate(np.ones((3, 2), dtype=np.float32), tau, mu, bottom_radiance=2)
    np.testing.assert_array_equal(exits, [3.75, 5.75])
    np.testing.assert_array_equal(centres[:, 1], [2.625, 3.875, 5.125])


@pytest.mark.parametrize(
    "depth_dtype,mu_dtype", [(np.float32, np.float64), (np.float64, np.float32)]
)
@pytest.mark.parametrize("native", [False, True])
def test_mixed_precision_promotes(monkeypatch, depth_dtype, mu_dtype, native):
    if not native:
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
        monkeypatch.setattr(formal, "_native_apply", formal_reference.apply)
    coefficients = formal.propagation_coefficients(
        np.ones(2, dtype=depth_dtype), np.array([-0.8, 0.8], dtype=mu_dtype)
    )
    assert all(a.dtype == np.float64 for a in vars(coefficients).values() if a is not None)
    for source_dtype, coefficient_dtype in ((np.float32, np.float64), (np.float64, np.float32)):
        coefficients = formal.propagation_coefficients(
            np.ones(2, dtype=coefficient_dtype), np.array([-0.8, 0.8], dtype=coefficient_dtype)
        )
        centres, exits = formal.apply_propagation(np.ones((2, 2), dtype=source_dtype), coefficients)
        assert centres.dtype == exits.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_rejects_mixed_output_buffers(dtype):
    depth = np.ones(3, dtype=dtype)
    mu = np.ones(2, dtype=dtype)
    outputs = [np.empty((3, 2), dtype=dtype) for _ in range(6)]
    outputs[-1] = outputs[-1].astype(np.float64 if dtype == np.float32 else np.float32)
    with pytest.raises(ValueError, match="matching"):
        propagation_native.prepare(depth, mu, 0.8, *outputs)


def test_float32_concurrent_application():
    depth = np.array([0, 1e-20, 0.3, 1000], dtype=np.float32)
    mu = np.array([-0.8, 0.8], dtype=np.float32)
    source = np.ones((4, 2), dtype=np.float32)
    coefficients = formal.propagation_coefficients(depth, mu, solar_mu=0.8)
    expected = formal.apply_propagation(source, coefficients, solar_source=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: formal.apply_propagation(source, coefficients, solar_source=True),
                range(40),
            )
        )
    for result in results:
        for value, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(value, reference)
            assert not np.shares_memory(value, reference)
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("native", [False, True])
def test_float32_solar_cosine_must_remain_positive(monkeypatch, native):
    if not native:
        monkeypatch.setattr(formal, "_native_prepare", formal_reference.prepare)
    with pytest.raises(ValueError, match="coefficient precision"):
        formal.propagation_coefficients(
            np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), solar_mu=1e-50
        )
