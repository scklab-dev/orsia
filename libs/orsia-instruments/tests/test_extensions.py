"""Bounded synthetic references for tabulated paths, detector maps and tolerancing."""

from dataclasses import replace

import numpy as np
import pytest
from orsia_instruments import (
    DetectorElectronics,
    DetectorOperatingPoints,
    ExpectedElectrons,
    analogue_dn_budget,
    combine_source_paths,
    digitize,
    electron_covariance,
    electron_moments,
    expected_electrons,
    monte_carlo_dn,
    monte_carlo_native,
    native_sensitivities,
    native_tolerance_budget,
    pixel_quantum_efficiency,
    sample_readout,
    tabulated_pixel_response,
    tabulated_transmission,
)
from orsia_uncertainty import validate_covariance
from scipy.integrate import quad
from test_instruments import acquisition, field, unity  # pyrefly: ignore[missing-import]


def two_reads():
    a = acquisition()
    return replace(
        a,
        sample_ids=("a", "b"),
        pixel_ids=("p", "q"),
        setting_ids=("s", "s"),
        port_ids=("+", "+"),
        exposure_bounds_s=np.repeat(a.exposure_bounds_s, 2, axis=0),
        times_s=np.repeat(a.times_s, 2, axis=0),
        directions_uv=np.repeat(a.directions_uv, 2, axis=0),
        image_xy_m=np.repeat(a.image_xy_m, 2, axis=0),
        weights_sr_s=np.repeat(a.weights_sr_s, 2, axis=0),
        pixel_bounds_m=np.repeat(a.pixel_bounds_m, 2, axis=0),
        direction_bounds_uv=np.repeat(a.direction_bounds_uv, 2, axis=0),
    )


def points(n=2):
    return DetectorOperatingPoints(
        np.zeros(n), np.zeros(n), np.ones(n) * 10000, np.ones(n), np.zeros(n), np.zeros(n)
    )


def test_tabulated_passband_leaks_support_and_photon_path():
    wavelengths = np.linspace(1e-6, 10e-6, 301)
    nodes = np.array([1e-6, 2e-6, 3e-6, 8e-6, 9e-6, 10e-6])
    response = np.array([0, 1, 0, 0, 0.05, 0])
    transmission = tabulated_transmission(wavelengths, nodes, response)
    np.testing.assert_allclose(transmission[wavelengths == 2e-6], 1)
    np.testing.assert_allclose(transmission[wavelengths == 9e-6], 0.05)
    np.testing.assert_allclose(np.trapezoid(response, nodes), 1.05e-6, rtol=1e-14)
    np.testing.assert_allclose(np.trapezoid(transmission, wavelengths), 1.05e-6, rtol=2e-4)
    with pytest.raises(ValueError, match="support"):
        tabulated_transmission([0.9e-6], nodes, response)
    np.testing.assert_array_equal(
        tabulated_transmission([0.9e-6], nodes, response, outside="zero"), [0]
    )
    with pytest.raises(ValueError):
        tabulated_transmission([2e-6], nodes[::-1], response)
    with pytest.raises(ValueError):
        tabulated_transmission([2e-6], nodes, np.ones(len(nodes)) * 2)

    scene = field()
    path_scene = expected_electrons(
        scene, acquisition(), unity, collecting_area_m2=1e-16, source_name="external"
    )
    internal = expected_electrons(
        replace(scene, values=scene.values * 3),
        acquisition(),
        lambda i, wl, acq: 0.2 * unity(i, wl, acq),
        collecting_area_m2=1e-16,
        source_name="internal",
    )
    combined = combine_source_paths(path_scene, internal)
    np.testing.assert_allclose(combined.total, path_scene.total * 1.6)
    np.testing.assert_allclose(
        electron_moments(combined, DetectorElectronics())[0], path_scene.total * 1.6
    )
    with pytest.raises(ValueError, match="unique"):
        combine_source_paths(path_scene, path_scene)
    with pytest.raises(ValueError, match="identities"):
        combine_source_paths(
            path_scene, replace(internal, acquisition=replace(acquisition(), sample_ids=("other",)))
        )
    with pytest.raises(ValueError, match="intervals"):
        combine_source_paths(
            path_scene,
            replace(
                internal,
                acquisition=replace(acquisition(), exposure_bounds_s=np.array([[0.0, 3.0]])),
            ),
        )


def test_tilted_table_mass_finite_pixel_and_convergence():
    # Normalized tilted density (1 + x*y/2)/4 on [-1,1]^2; its off-axis tilt
    # and finite-pixel integral have an independent elementary reference.
    x = np.linspace(-1, 1, 25)
    y = np.linspace(-1, 1, 25)
    density = (1 + x[:, None] * y[None, :] / 2) / 4

    def response(*, pixel_order=12, **overrides):
        return tabulated_pixel_response(
            wavelength_m=np.array([1e-6, 2e-6]),
            bounds_m=overrides.get("bounds_m", np.array([0.0, 0.5, 0.0, 0.5])),
            image_xy_m=overrides.get("image_xy_m", np.zeros((2, 2))),
            x_offsets_m=overrides.get("x_offsets_m", x),
            y_offsets_m=overrides.get("y_offsets_m", y),
            density_per_m2=overrides.get("density_per_m2", density),
            transmission=lambda xp, yp, wl: np.ones(2),
            pixel_order=pixel_order,
        )

    expected = (0.5**2 + (0.5**2 / 2) ** 2 / 2) / 4
    coarse = response(pixel_order=2)
    fine = response(pixel_order=12)
    np.testing.assert_allclose(coarse, expected, atol=1e-14)
    np.testing.assert_allclose(fine, expected, atol=1e-14)
    # Shifted centroid in wavelength: one wavelength sees a lower captured mass.
    centres = np.zeros((2, 2, 2))
    centres[:, 1, 0] = 0.2
    shifted = response(image_xy_m=centres, pixel_order=20)
    assert not np.allclose(shifted[:, 0], shifted[:, 1])
    with pytest.raises(ValueError, match="unit physical mass"):
        response(density_per_m2=2 * density)
    with pytest.raises(ValueError, match="under-resolved"):
        response(
            bounds_m=[-1.0, 1.0, -1.0, 1.0],
            x_offsets_m=[-1.0, 0.0, 1.0],
            y_offsets_m=[-1.0, 0.0, 1.0],
            density_per_m2=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            pixel_order=1,
        )
    # Supplied table with smooth separable density converges to independent integral.
    xx = np.linspace(-2, 2, 121)
    d = np.exp(-(xx[:, None] ** 2) - xx[None, :] ** 2) / np.pi
    w = tabulated_pixel_response(
        np.array([1e-6]),
        [0.0, 0.6, -0.5, 0.4],
        np.zeros((1, 2)),
        xx,
        xx,
        d,
        transmission=lambda px, py, wl: np.ones(1),
        pixel_order=32,
    )
    reference = (
        quad(lambda z: np.exp(-z * z), 0, 0.6)[0]
        * quad(lambda z: np.exp(-z * z), -0.5, 0.4)[0]
        / np.pi
    )
    np.testing.assert_allclose(w, reference, rtol=4e-4)


def test_nonuniform_operating_point_qe_noise_covariance_nonlinearity():
    acq = two_reads()
    qe = pixel_quantum_efficiency(acq, {"p": 0.5, "q": 1.0})
    np.testing.assert_array_equal(qe, [[0.5], [1.0]])
    with pytest.raises(ValueError, match="exactly"):
        pixel_quantum_efficiency(acq, {"p": 0.5})
    fixed = DetectorOperatingPoints(
        np.array([2.0, 3.0]),
        np.array([1.0, 2.0]),
        np.array([100.0, 200.0]),
        np.array([2.0, 4.0]),
        np.array([10.0, 20.0]),
        np.array([0.25, -0.25]),
        12,
    )
    signal = ExpectedElectrons(acq, {"scene": np.array([9.0, 19.0])})
    mean, variance = electron_moments(signal, fixed)
    np.testing.assert_allclose(mean, [13, 25])
    np.testing.assert_allclose(variance, [14, 29])
    inputs = np.array([[20.0, 50.0], [100.0, 200.0]])
    readings = digitize(acq, inputs, np.zeros_like(inputs), fixed)
    np.testing.assert_array_equal(readings.dn, [[20, 32], [72, 58]])
    assert readings.full_well_saturated[1].all()
    assert not readings.adc_clipped.any()
    np.testing.assert_allclose(
        electron_covariance(signal, fixed, row_ids=("r", "r"), row_noise_electrons=3),
        [[23, 9], [9, 38]],
    )
    with pytest.raises(ValueError, match="requires one nonempty group"):
        sample_readout(signal, fixed, np.random.default_rng(1), row_noise_electrons=1)
    with pytest.raises(ValueError, match="must match physical reads"):
        sample_readout(signal, points(1), np.random.default_rng(1))
    with pytest.raises(ValueError, match="nonlinearity"):
        replace(fixed, nonlinearity=np.array([0.0, 0.6]))

    homogeneous = DetectorElectronics(
        dark_electrons_per_s=2, read_noise_electrons=3, full_well_electrons=1000, bias_dn=100
    )
    same = DetectorOperatingPoints(
        np.full(2, 2.0),
        np.full(2, 3.0),
        np.full(2, 1000.0),
        np.ones(2),
        np.full(2, 100.0),
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        electron_covariance(signal, homogeneous), electron_covariance(signal, same)
    )
    np.testing.assert_array_equal(
        digitize(acq, inputs, np.zeros_like(inputs), homogeneous).dn,
        digitize(acq, inputs, np.zeros_like(inputs), same).dn,
    )
    np.testing.assert_array_equal(
        sample_readout(signal, homogeneous, np.random.default_rng(4), realisations=10).dn,
        sample_readout(signal, same, np.random.default_rng(4), realisations=10).dn,
    )


def test_common_noise_ensemble_and_reproducibility():
    acq = two_reads()
    signal = ExpectedElectrons(acq, {"scene": np.full(2, 1000.0)})
    electronic = DetectorElectronics(full_well_electrons=1e7, adc_bits=32)
    groups = dict(row_ids=("same", "same"), row_noise_electrons=10.0)
    cov = electron_covariance(signal, electronic, **groups)
    samples = sample_readout(
        signal, electronic, np.random.default_rng(17), realisations=25000, **groups
    ).dn
    np.testing.assert_allclose(samples.mean(axis=0), [1000.0, 1000.0], atol=3)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), cov, rtol=0.07, atol=15)
    np.testing.assert_array_equal(
        samples[:5],
        sample_readout(
            signal, electronic, np.random.default_rng(17), realisations=25000, **groups
        ).dn[:5],
    )


def test_tolerance_linear_analytic_monte_carlo_and_identity_rejection():
    acq = two_reads()

    def evaluate(values):
        x, y = values["band"], values["pointing"]
        return ExpectedElectrons(acq, {"scene": np.array([100 + 2 * x + 3 * y, 200 - x + 4 * y])})

    nominal, j = native_sensitivities(
        evaluate, {"band": 0.0, "pointing": 0.0}, {"band": 1e-3, "pointing": 1e-3}
    )
    np.testing.assert_allclose(j, [[2, 3], [-1, 4]], atol=1e-10)

    covariance = validate_covariance(np.array([[0.01, 0.005], [0.005, 0.02]]))
    bias = np.copy(np.array([0.1, -0.2]))
    budget = native_tolerance_budget(
        nominal,
        ("band", "pointing"),
        j,
        bias,
        covariance,
        electron_covariance(nominal, DetectorElectronics()),
    )
    np.testing.assert_allclose(budget.signed_bias, [-0.4, -0.9], atol=1e-11)
    np.testing.assert_allclose(budget.instrument_covariance, j @ covariance @ j.T)
    assert budget.jacobian.flags.writeable
    mc_bias, mc_cov = monte_carlo_native(
        evaluate, {"band": 0.0, "pointing": 0.0}, bias, covariance, np.random.default_rng(99), 12000
    )
    np.testing.assert_allclose(mc_bias, budget.signed_bias, atol=0.015)
    np.testing.assert_allclose(mc_cov, budget.instrument_covariance, atol=0.03)
    baseline, dn_budget = analogue_dn_budget(
        budget,
        nominal,
        DetectorElectronics(full_well_electrons=1000, electrons_per_dn=2, bias_dn=10),
    )
    np.testing.assert_array_equal(baseline, [60, 110])
    np.testing.assert_allclose(dn_budget.jacobian, j / 2)
    np.testing.assert_allclose(dn_budget.instrument_covariance, budget.instrument_covariance / 4)
    with pytest.raises(ValueError, match="unsaturated"):
        analogue_dn_budget(budget, nominal, DetectorElectronics(full_well_electrons=100))
    with pytest.raises(ValueError, match="nonlinear"):
        analogue_dn_budget(budget, nominal, replace(points(), nonlinearity=np.array([0.1, 0.0])))
    draws = monte_carlo_dn(
        evaluate,
        {"band": 0.0, "pointing": 0.0},
        bias,
        covariance,
        DetectorElectronics(full_well_electrons=1000, electrons_per_dn=2, bias_dn=10),
        np.random.default_rng(99),
        100,
    )
    assert draws.dn.shape == (100, 2)
    assert draws.acquisition.sample_ids == ("a", "b")
    assert not draws.full_well_saturated.any()
    with pytest.raises(ValueError, match="axes"):
        native_tolerance_budget(nominal, ("only",), j, bias, covariance, np.eye(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        native_tolerance_budget(
            nominal, ("band", "pointing"), j, bias, covariance, np.array([[1.0, 2.0], [2.0, 1.0]])
        )
    with pytest.raises(ValueError, match="identities"):
        native_sensitivities(
            lambda values: (
                replace(evaluate(values), acquisition=replace(acq, sample_ids=("other", "b")))
                if values["band"] != 0
                else evaluate(values)
            ),
            {"band": 0.0, "pointing": 0.0},
            {"band": 0.01, "pointing": 0.01},
        )


def test_nonlinear_linearisation_limit_and_dn_sampling_saturation():
    acq = acquisition()

    def nonlinear(parameters):
        p = parameters["offset"]
        return ExpectedElectrons(acq, {"scene": np.array([80 + 4 * p + 3 * p**2])})

    nominal, coarse = native_sensitivities(nonlinear, {"offset": 1.0}, {"offset": 0.1})
    _, fine = native_sensitivities(nonlinear, {"offset": 1.0}, {"offset": 0.01})
    np.testing.assert_allclose(coarse, fine, atol=1e-10)

    bias = np.copy(np.array([0.0]))
    covariance = validate_covariance(np.array([[0.5]]))
    budget = native_tolerance_budget(
        nominal,
        ("offset",),
        fine,
        bias,
        covariance,
        electron_covariance(nominal, DetectorElectronics()),
    )
    mean, mc_cov = monte_carlo_native(
        nonlinear, {"offset": 1.0}, bias, covariance, np.random.default_rng(71), 3000
    )
    assert mean[0] > 1.0  # positive quadratic mean shift absent from the Jacobian
    assert mc_cov[0, 0] > budget.instrument_covariance[0, 0]
    readings = monte_carlo_dn(
        nonlinear,
        {"offset": 1.0},
        bias,
        covariance,
        DetectorElectronics(full_well_electrons=90, adc_bits=8),
        np.random.default_rng(71),
        500,
    )
    assert readings.full_well_saturated.any()
    assert not readings.full_well_saturated.all()
    with pytest.raises(ValueError, match="unsaturated"):
        analogue_dn_budget(budget, nominal, DetectorElectronics(full_well_electrons=80, adc_bits=8))


def test_array_tolerance_budget_diagonal_equivalence_and_independent_snapshots():
    nominal = ExpectedElectrons(two_reads(), {"scene": np.array([100.0, 200.0])})
    jacobian = np.array([[2.0, 3.0], [-1.0, 4.0]])
    bias = np.array([0.1, -0.2])
    variances = np.array([0.01, 0.02])
    noise = np.diag([100.0, 200.0])
    diagonal = native_tolerance_budget(nominal, ("x", "y"), jacobian, bias, variances, noise)
    dense = native_tolerance_budget(nominal, ("x", "y"), jacobian, bias, np.diag(variances), noise)
    np.testing.assert_allclose(diagonal.instrument_covariance, dense.instrument_covariance)
    np.testing.assert_allclose(diagonal.signed_bias, [-0.4, -0.9])
    for result in (
        diagonal.jacobian,
        diagonal.signed_bias,
        diagonal.instrument_covariance,
        diagonal.detector_covariance,
    ):
        assert result.flags.writeable
        for supplied in (jacobian, bias, variances, noise):
            assert not np.shares_memory(result, supplied)
    jacobian.fill(0)
    bias.fill(0)
    variances.fill(0)
    noise.fill(0)
    np.testing.assert_array_equal(diagonal.jacobian, [[2.0, 3.0], [-1.0, 4.0]])
    np.testing.assert_allclose(diagonal.signed_bias, [-0.4, -0.9])
    np.testing.assert_array_equal(diagonal.detector_covariance, np.diag([100.0, 200.0]))
    with pytest.raises(ValueError, match="axes"):
        replace(diagonal, signed_bias=np.zeros(3))
    with pytest.raises(ValueError, match="axes"):
        replace(diagonal, detector_covariance=np.eye(3))
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(diagonal, instrument_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_array_parameter_laws_reject_invalid_bias_and_covariance_shapes():
    def evaluate(parameters):
        return ExpectedElectrons(acquisition(), {"scene": np.array([100.0 + parameters["offset"]])})

    for bias, covariance in (
        (np.zeros(2), np.ones(1)),
        (np.zeros(1), np.eye(2)),
        (np.full(1, np.nan), np.ones(1)),
    ):
        with pytest.raises(ValueError, match="parameter dimensions"):
            monte_carlo_native(
                evaluate, {"offset": 0.0}, bias, covariance, np.random.default_rng(1), 2
            )
        with pytest.raises(ValueError, match="parameter dimensions"):
            monte_carlo_dn(
                evaluate,
                {"offset": 0.0},
                bias,
                covariance,
                DetectorElectronics(),
                np.random.default_rng(1),
                2,
            )
