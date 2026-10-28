"""Independent bounded references, support rejection, photon accounting and physical reads."""

from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import pytest
from orsia_instruments import (
    Acquisition,
    DetectorElectronics,
    DetectorReadings,
    ExpectedElectrons,
    RadianceField,
    cavity_transmission,
    central_difference,
    digitize,
    dispersive_response,
    electron_moments,
    expected_electrons,
    gaussian_filter,
    gaussian_pixel_fraction,
    lvf_centre,
    modulated_pixel_response,
    native_difference,
    rectangular_acquisition,
    sample_radiance,
    sample_readout,
    sampled_optical_response,
    two_beam_ports,
    uniform_radiance_field,
)
from orsia_instruments.quadrature import interval_quadrature
from orsia_radiative import ObservationGeometry, Radiance, SpectralDomain, SpectralGrid
from scipy.constants import c, h
from scipy.integrate import quad
from scipy.special import erf


def acquisition() -> Acquisition:
    return Acquisition(
        ("a",),
        ("p",),
        ("s",),
        ("+",),
        np.array([[0.0, 2.0]]),
        np.array([[0.5, 1.5]]),
        np.zeros((1, 2, 2)),
        np.zeros((1, 2, 2)),
        np.array([[0.1, 0.1]]),
        np.array([[-1.0, 1.0, -1.0, 1.0]]),
        np.array([[-0.1, 0.1, -0.1, 0.1]]),
    )


def field() -> RadianceField:
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.linspace(1000.0, 2000.0, 33),
        SpectralDomain("wavenumber_cm1", ((1000.0, 2000.0),)),
    )
    return RadianceField(
        grid,
        np.array([0.0, 2.0]),
        np.array([-0.1, 0.1]),
        np.array([-0.1, 0.1]),
        np.ones((2, 2, 2, 33)) * grid.points,
    )


def unity(i, wavelength, acq):
    return np.ones((acq.times_s.shape[1], len(wavelength)))


def test_prepared_response_reuses_optics_not_radiance():
    scene, acq = field(), acquisition()
    wave = 0.01 / scene.grid.points
    calls = []

    def response(read, wavelengths, current):
        calls.append((read, len(wavelengths)))
        return np.broadcast_to(wavelengths / wave.max(), (2, len(wavelengths)))

    prepared = sampled_optical_response(acq, response, wave)
    assert calls == [(0, len(wave))]
    calls.clear()
    for scale in (1, 3):
        changed = replace(scene, values=scene.values * scale)
        actual = expected_electrons(changed, acq, prepared, collecting_area_m2=2)
        assert not calls
        reference = expected_electrons(changed, acq, response, collecting_area_m2=2)
        np.testing.assert_allclose(actual.total, reference.total, rtol=3e-15)
        calls.clear()
    np.testing.assert_allclose(prepared(0, wave[::-1], acq), response(0, wave[::-1], acq))
    with pytest.raises(ValueError):
        prepared(0, wave, replace(acq))
    with pytest.raises(ValueError):
        prepared(0, np.array([wave.max() * 2]), acq)
    with pytest.raises(ValueError):
        prepared(0, np.array([wave.min() / 2]), acq)
    with pytest.raises(ValueError):
        prepared(-1, wave, acq)
    with pytest.raises(ValueError):
        prepared(0, np.array([np.nan]), acq)
    assert prepared(0, wave, acq).flags.writeable


@pytest.mark.parametrize(
    "wave",
    [[], [0], [[1]], [1, 1], [np.nan], [np.inf]],
)
def test_prepared_response_rejects_invalid_support(wave):
    with pytest.raises(ValueError):
        sampled_optical_response(acquisition(), unity, wave)


def test_prepared_response_rejects_invalid_fractions():
    for response in (lambda *_: np.ones((1, 2)), lambda *_: np.full((2, 2), 2)):
        with pytest.raises(ValueError):
            sampled_optical_response(acquisition(), response, [1, 2])


def test_prepared_response_preserves_joint_read_node_spectral_dependence():
    original = acquisition()
    parameters = {
        member.name: np.repeat(getattr(original, member.name), 3, axis=0)
        for member in fields(original)
        if member.name not in {"sample_ids", "pixel_ids", "setting_ids", "port_ids"}
    }
    acq = Acquisition(("a", "b", "c"), ("p",) * 3, ("s",) * 3, ("+",) * 3, **parameters)
    scene = field()
    wave = 0.01 / scene.grid.points
    buffer = np.empty((2, len(wave)))

    def response(read, wavelengths, current):
        buffer[:] = (read + 1) * np.array([[0.1], [0.2]]) * wavelengths / wave.max()
        return buffer

    prepared = sampled_optical_response(acq, response, wave)
    buffer[:] = 0
    actual = expected_electrons(scene, acq, prepared, collecting_area_m2=1)
    reference = expected_electrons(scene, acq, response, collecting_area_m2=1)
    np.testing.assert_allclose(actual.total, reference.total, rtol=3e-15)
    np.testing.assert_allclose(actual.total / actual.total[0], [1, 2, 3], rtol=3e-15)


def test_independent_photon_integral_scaling_and_ownership():
    scene, acq = field(), acquisition()
    expected = 0.2 * 1000 / (100 * h * c) * 2 * 0.7 * 0.8
    result = expected_electrons(
        scene,
        acq,
        unity,
        collecting_area_m2=2,
        throughput=0.7,
        quantum_efficiency=0.8,
    )
    np.testing.assert_allclose(result.total, expected, rtol=3e-15)
    for name in ("values", "times_s", "u"):
        assert getattr(scene, name).flags.writeable
    assert acq.times_s.flags.writeable
    assert result.contributions["scene"].flags.writeable
    result.total[:] = 0
    np.testing.assert_allclose(result.total, expected)
    zero = expected_electrons(
        replace(scene, values=scene.values * 0), acq, unity, collecting_area_m2=2
    )
    assert zero.total[0] == 0
    doubled = expected_electrons(
        replace(scene, values=scene.values * 2),
        acq,
        unity,
        collecting_area_m2=2,
        throughput=0.7,
        quantum_efficiency=0.8,
    )
    np.testing.assert_allclose(doubled.total, result.total * 2)
    source = np.array([1.0])
    combined = ExpectedElectrons(
        acq, {"external": source, "internal-after-own-path": np.array([2.0])}
    )
    source[:] = 100
    np.testing.assert_array_equal(combined.total, [3.0])


def test_density_coordinate_reference_and_spectral_refinement():
    # Independent integration in wavelength metres, including density Jacobian.
    scene, acq = field(), acquisition()

    def response(i, wavelength, acq):
        return np.broadcast_to(
            gaussian_filter(wavelength, 7e-6, 1e-6), (acq.times_s.shape[1], len(wavelength))
        )

    reference = (
        quad(
            lambda lam: (
                (0.01 / lam)
                * (0.01 / lam**2)
                * lam
                / (h * c)
                * np.exp(-0.5 * ((lam - 7e-6) / 1e-6) ** 2)
            ),
            5e-6,
            1e-5,
            epsabs=0,
            epsrel=1e-12,
        )[0]
        * 0.2
    )
    errors = []
    for size in (33, 129, 513):
        points = np.linspace(1000.0, 2000.0, size)
        refined = replace(
            scene,
            grid=replace(scene.grid, points=points),
            values=np.broadcast_to(points, (2, 2, 2, size)),
        )
        value = expected_electrons(refined, acq, response, collecting_area_m2=1).total[0]
        errors.append(abs(value / reference - 1))
    assert errors[2] < errors[1] / 10 < errors[0] / 100
    assert errors[-1] < 2e-6


def test_scene_interpolation_and_explicit_uniform_extension():
    scene = field()
    values = (
        (1 + scene.times_s[:, None, None, None])
        * (1 + scene.u[None, :, None, None])
        * (1 + scene.v[None, None, :, None])
        * scene.grid.points
    )
    scene = replace(scene, values=values)
    result = sample_radiance(scene, np.array([0.5]), np.array([[0.025, -0.02]]), slice(0, 2))
    np.testing.assert_allclose(result[0], 1.5 * 1.025 * 0.98 * scene.grid.points[:2])
    radiance = Radiance(
        replace(scene.grid, domain=None), scene.values[0, 0, 0], ObservationGeometry(0)
    )
    uniform = uniform_radiance_field(
        radiance,
        time_bounds_s=(0, 2),
        u_bounds=(-0.1, 0.1),
        v_bounds=(-0.1, 0.1),
        spectral_bounds_cm1=(1000, 2000),
    )
    np.testing.assert_array_equal(uniform.values[1, 1, 1], radiance.values)
    declared = replace(radiance, grid=scene.grid)
    with pytest.raises(ValueError, match="override"):
        uniform_radiance_field(
            declared,
            time_bounds_s=(0, 2),
            u_bounds=(-0.1, 0.1),
            v_bounds=(-0.1, 0.1),
            spectral_bounds_cm1=(900, 2000),
        )
    for times, directions in (
        (np.array([-1.0]), np.array([[0.0, 0.0]])),
        (np.array([1.0]), np.array([[0.2, 0.0]])),
        (np.array([np.nan]), np.array([[0.0, 0.0]])),
        (np.array([[1.0]]), np.array([[0.0, 0.0]])),
    ):
        with pytest.raises(ValueError):
            sample_radiance(scene, times, directions, slice(None))


@pytest.mark.parametrize(
    "change",
    [
        {"times_s": np.array([0.0, 0.0])},
        {"u": np.array([1.0, 2.0])},
        {"values": np.zeros((2, 2, 2, 1))},
        {"values": -np.ones((2, 2, 2, 33))},
        {"v": np.array([np.nan, 0.1])},
        {"grid": SpectralGrid("wavelength_nm", np.array([1.0, 2.0]))},
        {"grid": SpectralGrid("wavenumber_cm1", np.array([1000.0, 2000.0]))},
    ],
)
def test_scene_rejects_invalid_contract(change):
    with pytest.raises(ValueError):
        replace(field(), **change)


@pytest.mark.parametrize(
    "change",
    [
        {"sample_ids": ()},
        {"pixel_ids": ("",)},
        {"times_s": np.zeros((1, 0))},
        {"directions_uv": np.zeros((1, 1, 2))},
        {"exposure_bounds_s": np.array([[2.0, 1.0]])},
        {"times_s": np.array([[-1.0, 1.0]])},
        {"directions_uv": np.ones((1, 2, 2))},
        {"weights_sr_s": np.array([[0.0, 0.0]])},
        {"weights_sr_s": np.array([[-1.0, 2.0]])},
        {"pixel_bounds_m": np.array([[1.0, -1.0, 0.0, 1.0]])},
    ],
)
def test_acquisition_rejects_invalid_contract(change):
    with pytest.raises(ValueError):
        replace(acquisition(), **change)


def test_rectangular_quadrature_geometry_and_motion():
    kwargs = dict(
        sample_ids=("r",),
        pixel_ids=("p",),
        setting_ids=("s",),
        port_ids=("p",),
        exposure_bounds_s=np.array([[0.0, 2.0]]),
        pixel_bounds_m=np.array([[-1.0, 1.0, -1.0, 1.0]]),
        direction_bounds_uv=(-0.1, 0.1, -0.1, 0.1),
        pointing=lambda t: np.column_stack((t / 10, t * 0)),
        focal_length_m=2.0,
    )
    acq = rectangular_acquisition(**kwargs, time_order=3, angular_order=8)
    reference = (
        quad(lambda u: quad(lambda v: 1 / np.sqrt(1 - u * u - v * v), -0.1, 0.1)[0], -0.1, 0.1)[0]
        * 2
    )
    np.testing.assert_allclose(acq.weights_sr_s.sum(), reference, rtol=2e-14)
    np.testing.assert_allclose(
        acq.image_xy_m[0, :, 0], 2 * (acq.directions_uv[0, :, 0] - acq.times_s[0] / 10)
    )
    for changes in (
        {"focal_length_m": -1},
        {"direction_bounds_uv": (0, 1, 0, 1)},
        {"pointing": lambda t: np.zeros(len(t))},
    ):
        with pytest.raises(ValueError):
            rectangular_acquisition(**cast(Any, kwargs | changes))
    for bounds in ((0, 0, 1), (0, 1, 0), (np.nan, 1, 2)):
        with pytest.raises(ValueError):
            interval_quadrature(*bounds)


def test_disperse_gaussian_normalization_loss_and_finite_pixels():
    centres = np.array([[0.0, 0.0], [0.3, -0.2]])
    bound = np.array([-1.0, 1.0, -1.0, 1.0])
    fraction = gaussian_pixel_fraction(bound, centres, 0.4)
    reference = np.prod(
        (erf((1 - centres) / (0.4 * np.sqrt(2))) - erf((-1 - centres) / (0.4 * np.sqrt(2)))) / 2,
        axis=1,
    )
    np.testing.assert_allclose(fraction, reference, rtol=1e-14)
    total = sum(
        gaussian_pixel_fraction([x, x + 0.5, -5, 5], centres, 0.4)
        for x in np.arange(-5.0, 5.0, 0.5)
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-14)
    assert np.all(fraction < 1)
    assert gaussian_pixel_fraction([8, 9, -1, 1], np.zeros((1, 2)), 1)[0] > 0
    wave = np.array([1.0, 2.0, 3.0])
    response = dispersive_response(
        wave, bound, centres, sigma_m=0.4, reference_m=2, dispersion=1, transmission=0.7
    )
    np.testing.assert_allclose(response[:, 1], 0.7 * fraction)
    assert response[0, 0] < response[0, 1]
    for b, xy, sigma in (
        ([0, 0, 0, 1], centres, 1),
        (bound, np.ones(3), 1),
        (bound, centres, -1),
        (bound, centres * np.nan, 1),
    ):
        with pytest.raises(ValueError):
            gaussian_pixel_fraction(b, xy, sigma)


def test_filter_lvf_width_peak_and_joint_pixel_integral():
    wave = np.linspace(1.0, 3.0, 2001)
    narrow = gaussian_filter(wave, 2, 0.05, 0.8)
    wide = gaussian_filter(wave, 2, 0.1, 0.8)
    assert max(narrow) == max(wide) == 0.8
    np.testing.assert_allclose(np.trapezoid(wide, wave) / np.trapezoid(narrow, wave), 2)
    np.testing.assert_allclose(
        lvf_centre(
            0.5, reference_m=2, gradient=0.1, temperature_delta_k=2, thermal_shift_m_per_k=0.01
        ),
        2.07,
    )
    assert lvf_centre(0, reference_m=2, gradient=0, angle_rad=0.3) < 2
    centres = np.array([[0.0, 0.0], [0.2, 0.0]])
    bounds = [-1, 1, -1, 1]
    result = modulated_pixel_response(
        np.array([2.0]),
        bounds,
        centres,
        sigma_m=0.5,
        transmission=lambda x, y, w: np.array([0.7]),
        pixel_order=24,
    )
    np.testing.assert_allclose(
        result[:, 0], 0.7 * gaussian_pixel_fraction(bounds, centres, 0.5), rtol=1e-13
    )

    def local(x, y, wave):
        return gaussian_filter(wave, 2 + x * 0.2, 0.1)

    coupled = modulated_pixel_response(
        np.array([2.1]), bounds, centres, sigma_m=0.5, transmission=local, pixel_order=24
    )
    # Independent double integral, not the pixel-centre separable approximation.
    reference = quad(
        lambda x: quad(
            lambda y: (
                np.exp(-(x * x + y * y) / 0.5)
                / (0.5 * np.pi)
                * np.exp(-0.5 * ((2.1 - 2 - x * 0.2) / 0.1) ** 2)
            ),
            -1,
            1,
        )[0],
        -1,
        1,
    )[0]
    np.testing.assert_allclose(coupled[0, 0], reference, rtol=2e-12)
    separated = gaussian_pixel_fraction(bounds, centres, 0.5) * local(0, 0, np.array([2.1]))
    assert abs(coupled[0, 0] / separated[0] - 1) > 0.02
    with pytest.raises(ValueError):
        modulated_pixel_response(
            np.array([2.0]),
            bounds,
            centres,
            sigma_m=1e-3,
            transmission=lambda x, y, w: np.ones(1),
            pixel_order=1,
        )
    with pytest.raises(ValueError):
        modulated_pixel_response(
            np.array([2.0]),
            bounds,
            np.zeros((1, 1, 2)),
            sigma_m=1,
            transmission=lambda x, y, w: np.ones(1),
        )


def test_interference_ports_and_cavity_independent_references():
    plus, minus = two_beam_ports(np.array([1.0, 2.0, 3.0]), 0, transmission=0.8)
    np.testing.assert_array_equal(plus, 0.8)
    np.testing.assert_array_equal(minus, 0)
    plus, minus = two_beam_ports(
        np.linspace(1, 3, 100), 4.2, phase_rad=0.3, visibility=0.7, transmission=0.8
    )
    np.testing.assert_allclose(plus + minus, 0.8, rtol=1e-15)
    assert min(plus) >= 0 and min(minus) >= 0
    rho = 0.8
    spacing = 10.0
    sigma = np.array([1.0, 1.0 + 1 / (4 * spacing), 1.0 + 1 / (2 * spacing)])
    actual = cavity_transmission(1 / sigma, spacing, reflectivity=rho, order_transmission=1)
    np.testing.assert_allclose(actual, [1.0, (1 - rho) ** 2 / (1 + rho) ** 2, 1.0], rtol=1e-13)
    phase = 4 * np.pi * spacing * sigma
    # Independent geometric-series complex amplitude for equal lossless mirrors.
    reference = abs((1 - rho) / (1 - rho * np.exp(1j * phase))) ** 2
    np.testing.assert_allclose(actual, reference, rtol=1e-13)
    half_phase = 2 * np.arcsin((1 - rho) / (2 * np.sqrt(rho)))
    half_sigma = 1 + half_phase / (4 * np.pi * spacing)
    np.testing.assert_allclose(
        cavity_transmission(1 / half_sigma, spacing, reflectivity=rho, order_transmission=1),
        0.5,
        rtol=1e-12,
    )
    assert cavity_transmission(1, spacing, order_transmission=0) == 0


def test_sampling_can_miss_resonances_and_alias_fringes():
    # An entire Airy free spectral range has independent mean (1-R)/(1+R).
    rho = 0.98
    reference = (1 - rho) / (1 + rho)
    errors = []
    for count in (9, 129, 8193):
        sigma = np.linspace(1.0, 1.05, count)
        transmission = cavity_transmission(
            1 / sigma, 10, reflectivity=rho, coating_phase_rad=0.43, order_transmission=1
        )
        mean = np.trapezoid(transmission, sigma) / 0.05
        errors.append(abs(mean / reference - 1))
    assert errors[0] > 0.25
    assert errors[2] < 1e-10 and errors[2] < errors[1]
    # 32 cycles sampled at only 16 intervals alias to a constant bright port.
    means = []
    for count in (17, 1025):
        sigma = np.linspace(1.0, 2.0, count)
        plus, _ = two_beam_ports(1 / sigma, 32)
        means.append(np.trapezoid(plus, sigma))
    assert means[0] > 0.99
    np.testing.assert_allclose(means[1], 0.5, atol=1e-14)


def test_two_beam_finite_exposure_independent_sinc():
    time, weights = interval_quadrature(0.0, 1.0, 24)
    opd = 0.2 + 1.3 * time
    plus, minus = two_beam_ports(1.0, opd, visibility=0.7)
    expected = 0.5 * (1 + 0.7 * np.cos(2 * np.pi * 0.85) * np.sinc(1.3))
    np.testing.assert_allclose(weights @ plus, expected, rtol=1e-13)
    np.testing.assert_allclose(weights @ (plus + minus), 1.0, atol=1e-14)


@pytest.mark.parametrize(
    "call",
    [
        lambda: gaussian_filter(0, 1, 1),
        lambda: gaussian_filter(1, 1, 1, 2),
        lambda: lvf_centre(0, reference_m=1, gradient=1, effective_index=0.5),
        lambda: lvf_centre(0, reference_m=1, gradient=1, angle_rad=2),
        lambda: two_beam_ports(1, np.nan),
        lambda: two_beam_ports(1, 0, visibility=-1),
        lambda: cavity_transmission(1, 1, reflectivity=1, order_transmission=1),
        lambda: cavity_transmission(1, 1, coating_phase_rad=np.nan, order_transmission=1),
        lambda: cavity_transmission(1, 1, order_transmission=-1),
    ],
)
def test_response_physical_rejection(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "change",
    [
        {"collecting_area_m2": -1},
        {"throughput": 2},
        {"quantum_efficiency": np.nan},
        {"source_name": ""},
    ],
)
def test_integration_rejection(change):
    with pytest.raises(ValueError):
        expected_electrons(field(), acquisition(), unity, **({"collecting_area_m2": 1} | change))


def test_bad_response_support_and_source_rejection():
    for changed in (
        replace(acquisition(), exposure_bounds_s=np.array([[-0.1, 2.0]])),
        replace(acquisition(), direction_bounds_uv=np.array([[-0.11, 0.1, -0.1, 0.1]])),
    ):
        with pytest.raises(ValueError, match="entire acquisition"):
            expected_electrons(field(), changed, unity, collecting_area_m2=1)
    for bounds in (np.array([[0.1, -0.1, -0.1, 0.1]]), np.array([[0.01, 0.1, -0.1, 0.1]])):
        with pytest.raises(ValueError):
            replace(acquisition(), direction_bounds_uv=bounds)
    for response in (lambda i, w, a: np.ones(len(w)), lambda i, w, a: unity(i, w, a) * -1):
        with pytest.raises(ValueError):
            expected_electrons(field(), acquisition(), response, collecting_area_m2=1)
    for sources in (
        {},
        {"x": np.array([-1.0])},
        {"x": np.array([1.0, 2.0])},
        {"x": np.array([np.nan])},
    ):
        with pytest.raises(ValueError):
            ExpectedElectrons(acquisition(), sources)


def test_detector_analytic_ensemble_reproducibility_and_port_noise():
    signal = ExpectedElectrons(acquisition(), {"scene": np.array([100.0])})
    detector = DetectorElectronics(
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        full_well_electrons=1e6,
        electrons_per_dn=0.01,
        bias_dn=10000,
        adc_bits=32,
    )
    mean, variance = electron_moments(signal, detector)
    np.testing.assert_array_equal(mean, [104.0])
    np.testing.assert_array_equal(variance, [113.0])
    readings = sample_readout(signal, detector, np.random.default_rng(817), realisations=100000)
    values = (readings.dn[:, 0] - detector.bias_dn) * detector.electrons_per_dn
    # Predeclared six-sigma mean and 3% variance tolerance; rounding variance <= g²/4.
    assert abs(values.mean() - mean[0]) < 6 * np.sqrt(variance[0] / len(values))
    assert abs(values.var() / variance[0] - 1) < 0.03
    assert not readings.full_well_saturated.any() and not readings.adc_clipped.any()
    again = sample_readout(signal, detector, np.random.default_rng(817), realisations=100000)
    np.testing.assert_array_equal(again.dn, readings.dn)
    other = sample_readout(signal, detector, np.random.default_rng(819), realisations=100000)
    difference = (readings.dn[:, 0] - other.dn[:, 0]) * 0.01
    assert abs(difference.var() / (2 * variance[0]) - 1) < 0.03
    with pytest.raises(ValueError):
        readings.dn.setflags(write=True)


def test_digitization_order_flags_rounding_and_zero_counts():
    detector = DetectorElectronics(
        full_well_electrons=10, electrons_per_dn=2, bias_dn=1, adc_bits=3
    )
    charge = np.array([[0.0], [1.0], [3.0], [10.0], [100.0], [100.0]])
    noise = np.array([[0.0], [0.0], [0.0], [0.0], [8.0], [-20.0]])
    reads = digitize(acquisition(), charge, noise, detector)
    np.testing.assert_array_equal(reads.dn[:, 0], [1, 2, 2, 6, 7, 0])
    np.testing.assert_array_equal(reads.full_well_saturated[:, 0], [False] * 3 + [True] * 3)
    np.testing.assert_array_equal(reads.adc_clipped[:, 0], [False] * 4 + [True] * 2)
    signal = ExpectedElectrons(acquisition(), {"scene": np.array([0.0])})
    reads = sample_readout(signal, DetectorElectronics(), np.random.default_rng(1))
    np.testing.assert_array_equal(reads.dn, 0)
    assert not reads.full_well_saturated.any()


@pytest.mark.parametrize("mean", [0.05, 1e6])
def test_low_and_high_poisson_charge(mean):
    signal = ExpectedElectrons(acquisition(), {"scene": np.array([mean])})
    detector = DetectorElectronics(full_well_electrons=1e9, adc_bits=32)
    readings = sample_readout(signal, detector, np.random.default_rng(911), realisations=100000)
    values = readings.dn[:, 0]
    assert abs(values.mean() - mean) < 6 * np.sqrt(mean / len(values))
    assert abs(values.var() / mean - 1) < 0.08
    assert not readings.full_well_saturated.any() and not readings.adc_clipped.any()


def test_detector_parameter_perturbations_keep_native_mean_and_dispersion_separate():
    signal = ExpectedElectrons(acquisition(), {"scene": np.array([100.0])})
    detector = DetectorElectronics(dark_electrons_per_s=2, read_noise_electrons=3)
    mean, variance = electron_moments(signal, detector)
    changed_mean, changed_variance = electron_moments(
        signal, replace(detector, dark_electrons_per_s=2.1)
    )
    np.testing.assert_allclose(changed_mean - mean, 0.2)
    np.testing.assert_allclose(changed_variance - variance, 0.2)
    changed_mean, changed_variance = electron_moments(
        signal, replace(detector, read_noise_electrons=3.1)
    )
    np.testing.assert_array_equal(changed_mean, mean)
    np.testing.assert_allclose(changed_variance - variance, 3.1**2 - 3**2)
    saturated = sample_readout(
        signal,
        replace(detector, full_well_electrons=1, read_noise_electrons=0),
        np.random.default_rng(91),
        realisations=20,
    )
    np.testing.assert_array_equal(saturated.dn, 1)
    assert saturated.full_well_saturated.all()


@pytest.mark.parametrize(
    "change",
    [
        {"dark_electrons_per_s": -1},
        {"read_noise_electrons": -1},
        {"full_well_electrons": 0},
        {"electrons_per_dn": 0},
        {"bias_dn": np.nan},
        {"adc_bits": 0},
        {"adc_bits": 33},
    ],
)
def test_detector_rejects_invalid_parameters(change):
    with pytest.raises(ValueError):
        DetectorElectronics(**change)


def test_detector_rejects_invalid_readings_and_draw_count():
    acq = acquisition()
    signal = ExpectedElectrons(acq, {"x": np.ones(1)})
    with pytest.raises(ValueError):
        sample_readout(signal, DetectorElectronics(), np.random.default_rng(1), realisations=0)
    for charge, noise in (
        (np.ones(1), np.ones(1)),
        (-np.ones((1, 1)), np.ones((1, 1))),
        (np.ones((1, 1)), np.array([[np.nan]])),
    ):
        with pytest.raises(ValueError):
            digitize(acq, charge, noise, DetectorElectronics())
    for dn, full, adc in (
        (np.ones(1, dtype=np.int64), np.zeros(1, bool), np.zeros(1, bool)),
        (np.ones((1, 1)), np.zeros((1, 1), bool), np.zeros((1, 1), bool)),
        (-np.ones((1, 1), dtype=np.int64), np.zeros((1, 1), bool), np.zeros((1, 1), bool)),
    ):
        with pytest.raises(ValueError):
            DetectorReadings(acq, cast(Any, dn), full, adc)


def test_signed_perturbations_identity_and_step_refinement():
    acq = acquisition()

    def evaluate(x):
        return ExpectedElectrons(acq, {"x": np.array([np.exp(x)])})

    baseline = evaluate(1.0)
    np.testing.assert_array_equal(native_difference(baseline, baseline), 0)
    errors = [abs(central_difference(evaluate, 1, step)[0] / np.e - 1) for step in (0.1, 0.05)]
    assert errors[1] < errors[0] / 3.9
    assert native_difference(baseline, evaluate(1.1))[0] > 0
    for step in (0, np.nan, 1e-30):
        with pytest.raises(ValueError):
            central_difference(evaluate, 1, step)
    with pytest.raises(ValueError):
        native_difference(baseline, replace(baseline, acquisition=replace(acq, port_ids=("-",))))
    with pytest.raises(ValueError):
        native_difference(baseline, ExpectedElectrons(acq, {"y": np.ones(1)}))
