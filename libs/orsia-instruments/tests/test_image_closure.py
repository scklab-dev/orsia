"""Integrated small-image closure cases; synthetic numerical budgets, not mission limits.

Axes are exposure,row,column. Radiance is W m-2 sr-1 (cm-1)-1; lengths
are metres and pointing offsets are direction cosines. No providers or workers.
"""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from orsia_instruments import (
    DetectorElectronics,
    DetectorMaps,
    RadianceField,
    cavity_imager,
    combine_image_source_paths,
    dispersive_imager,
    electron_covariance,
    exposure_sequence,
    filter_imager,
    gaussian_filter,
    image_photoelectrons,
    image_readout,
    interferometric_imager,
    lvf_imager,
    monte_carlo_native,
    native_sensitivities,
    native_tolerance_budget,
    regular_pixel_bounds,
    sample_readout,
    simulate_images,
)
from orsia_radiative import SpectralDomain, SpectralGrid
from orsia_uncertainty import (
    propagate_covariance,
    sample_gaussian,
    validate_covariance,
)
from scipy.constants import c, h
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr, roots_legendre

FAMILIES = ("filter", "lvf", "dispersive", "two-beam", "cavity")
# Predeclared for these smooth, bounded cases; no real instrument implied.
REFINEMENT_BUDGET = 2e-4
BLOCK_BUDGET = 5e-13


def case(family, *, points=257, pixel_order=8, pointing=0.0, band_shift=0.0, throughput=1.0):
    nu = np.linspace(4900.0, 5100.0, points)
    grid = SpectralGrid("wavenumber_cm1", nu, SpectralDomain("wavenumber_cm1", ((4900, 5100),)))
    t = np.array([-0.1, 0.2])
    u = v = np.array([-0.003, 0.003])
    spectrum = 1e-3 * (1 + 0.2 * np.cos((nu - 5000) / 30))
    values = (
        (1 + 2 * t[:, None, None, None])
        * (1 + 30 * u[None, :, None, None] - 20 * v[None, None, :, None])
        * spectrum
    )
    field = RadianceField(grid, t, u, v, values)
    common = dict(
        focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5, throughput=throughput
    )
    if family == "filter":
        optics = filter_imager(
            **common, filter=partial(gaussian_filter, centre_m=2e-6 + band_shift, width_m=2e-8)
        )
    elif family == "lvf":
        optics = lvf_imager(
            **common,
            reference_m=2e-6 + band_shift,
            gradient=3e-4,
            sigma_m=2e-8,
            pixel_order=pixel_order,
        )
    elif family == "dispersive":
        optics = dispersive_imager(**common, reference_m=2e-6 + band_shift, dispersion=40.0)
    elif family == "two-beam":
        optics = interferometric_imager(**common, opd_m=[0, 2e-5], port="+")
    else:
        optics = cavity_imager(
            **common, spacing_m=[1e-5, 1.01e-5], reflectivity=0.3, order_transmission=0.8
        )
    inputs = dict(
        pixel_bounds_m=regular_pixel_bounds((1, 2), pitch_m=6e-5),
        exposures=exposure_sequence(
            start_s=[0, 0.05],
            duration_s=0.02,
            row_delay_s=0.001,
            direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
            pointing_uv=lambda times: np.column_stack((0.006 * times + pointing, -0.003 * times)),
        ),
        optics=optics,
    )
    return field, inputs


def signal(family, *, time_order=5, angular_order=9, **kwargs):
    field, inputs = case(family, **kwargs)
    return simulate_images(
        field,
        **inputs,
        time_order=time_order,
        angular_order=angular_order,
        detector=DetectorMaps(quantum_efficiency=0.7),
    ).photoelectrons


@pytest.mark.parametrize("family", FAMILIES)
def test_independent_axis_refinement_and_under_sampling(family):
    baseline = signal(family)
    for change in (
        dict(points=513),
        dict(time_order=9),
        dict(angular_order=15),
        dict(pixel_order=12),
    ):
        refined = signal(family, **change)
        assert np.max(abs(refined - baseline)) / np.max(refined) < REFINEMENT_BUDGET
    # Not a claim that refinement can repair an under-resolved scene representation.
    coarse = signal(family, points=3, time_order=1, angular_order=1, pixel_order=1)
    assert np.max(abs(coarse - baseline)) / np.max(baseline) > REFINEMENT_BUDGET


@pytest.mark.parametrize("family", FAMILIES)
def test_rolling_rows_with_evolving_scene_and_moving_pointing(family):
    field, inputs = case(family)
    inputs["pixel_bounds_m"] = regular_pixel_bounds((2, 2), pitch_m=6e-5)
    rolled = simulate_images(field, **inputs, time_order=5, angular_order=9)
    for row in range(2):
        exposures = exposure_sequence(
            start_s=np.array([0, 0.05]) + row * 0.001,
            duration_s=0.02,
            direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
            pointing_uv=lambda times: np.column_stack((0.006 * times, -0.003 * times)),
        )
        reference = simulate_images(
            field,
            pixel_bounds_m=inputs["pixel_bounds_m"][row : row + 1],
            optics=inputs["optics"],
            exposures=exposures,
            time_order=5,
            angular_order=9,
        )
        np.testing.assert_allclose(
            rolled.photoelectrons[:, row : row + 1], reference.photoelectrons, rtol=BLOCK_BUDGET
        )
    assert not np.array_equal(rolled.photoelectrons[0], rolled.photoelectrons[1])


@pytest.mark.parametrize("family", FAMILIES)
def test_source_composition_matches_integrated_path(family):
    field, inputs = case(family)
    detector = DetectorMaps(quantum_efficiency=0.7)
    integrated = simulate_images(field, **inputs, detector=detector, time_order=5, angular_order=9)
    _, background_inputs = case(family, throughput=0.4)
    background = RadianceField(field.grid, field.times_s, field.u, field.v, 0.2 * field.values)
    paths = [
        image_photoelectrons(
            field,
            **inputs,
            detector=detector,
            source_name="scene",
            time_order=5,
            angular_order=9,
        ),
        image_photoelectrons(
            background,
            **background_inputs,
            detector=detector,
            source_name="supplied_background",
            time_order=5,
            angular_order=9,
        ),
    ]
    composed = image_readout(combine_image_source_paths(*paths))
    np.testing.assert_allclose(
        composed.photoelectrons, 1.08 * integrated.photoelectrons, rtol=BLOCK_BUDGET
    )
    np.testing.assert_allclose(
        composed.photoelectrons, sum(composed.contributions.values()), rtol=BLOCK_BUDGET
    )


def test_independent_continuous_reference_and_finite_detector_loss():
    # Flat spectrum, static uniform field: independently integrate Gaussian pixel
    # probabilities with adaptive QUADPACK, not Orsia quadratures/response helpers.
    field, inputs = case("filter", points=513)
    field = RadianceField(
        field.grid, field.times_s, field.u, field.v, np.full_like(field.values, 1e-3)
    )
    inputs["exposures"] = exposure_sequence(
        start_s=[0], duration_s=0.02, direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003)
    )
    inputs["optics"] = filter_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        filter=lambda wave: np.ones_like(wave),
    )
    actual = simulate_images(field, **inputs, angular_order=15).photoelectrons[0].ravel()
    spectral = 1e-3 * np.log(5100 / 4900) / (100 * h * c)
    references = []
    for xmin, xmax, ymin, ymax in inputs["pixel_bounds_m"].reshape(-1, 4):

        def integrand(v, u, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
            px = ndtr((xmax - 0.02 * u) / 3e-5) - ndtr((xmin - 0.02 * u) / 3e-5)
            py = ndtr((ymax - 0.02 * v) / 3e-5) - ndtr((ymin - 0.02 * v) / 3e-5)
            return px * py / np.sqrt(1 - u * u - v * v)

        angular = quad(
            lambda u: quad(lambda v: integrand(v, u), -0.003, 0.003, epsabs=1e-12)[0],
            -0.003,
            0.003,
            epsabs=1e-12,
        )[0]
        references.append(spectral * angular * 0.02 * 1e-8)
    np.testing.assert_allclose(actual, references, rtol=2e-8)
    incident = (
        spectral
        * 0.02
        * 1e-8
        * quad(
            lambda u: quad(lambda v: 1 / np.sqrt(1 - u * u - v * v), -0.003, 0.003)[0],
            -0.003,
            0.003,
        )[0]
    )
    assert 0 < actual.sum() < incident  # no renormalisation onto active pixels


@pytest.mark.parametrize("family", FAMILIES)
def test_family_image_against_independent_static_reference(family):
    field, inputs = case(family, points=1025, pixel_order=12)
    field = RadianceField(
        field.grid, field.times_s, field.u, field.v, np.full_like(field.values, 1e-3)
    )
    inputs["exposures"] = exposure_sequence(
        start_s=[0, 0.05],
        duration_s=0.02,
        direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
    )
    actual = simulate_images(field, **inputs, angular_order=15).photoelectrons

    # Closed-form truncated Gaussian product for LVF replaces pixel quadrature;
    # adaptive spectral integration replaces Orsia's sampled trapezoids. Spatial
    # nodes use SciPy rather than Orsia's quadrature implementation, and no Orsia
    # response, photon-conversion or scene interpolation helpers are called.
    def reference(order):
        nodes, weights = roots_legendre(order)
        u, v = np.meshgrid(nodes * 0.003, nodes * 0.003, indexing="ij")
        measure = weights[:, None] * weights * 0.003**2 / np.sqrt(1 - u * u - v * v)
        values = np.empty((2, 1, 2))
        for exposure in range(2):
            for column, bounds in enumerate(inputs["pixel_bounds_m"][0]):
                xmin, xmax, ymin, ymax = bounds

                def spectrum(nu, exposure=exposure, bounds=bounds):
                    xmin, xmax, ymin, ymax = bounds
                    wave = 0.01 / nu
                    mx, my = 0.02 * u, 0.02 * v
                    if family == "dispersive":
                        mx = mx + 40 * (wave - 2e-6)
                    py = ndtr((ymax - my) / 3e-5) - ndtr((ymin - my) / 3e-5)
                    if family == "lvf":
                        variance = (2e-8) ** 2 + (3e-4 * 3e-5) ** 2
                        mean = (mx * (2e-8) ** 2 + 3e-4 * (3e-5) ** 2 * (wave - 2e-6)) / variance
                        sigma = 3e-5 * 2e-8 / np.sqrt(variance)
                        px = ndtr((xmax - mean) / sigma) - ndtr((xmin - mean) / sigma)
                        band = (
                            2e-8
                            / np.sqrt(variance)
                            * np.exp(-((wave - 2e-6 - 3e-4 * mx) ** 2) / (2 * variance))
                        )
                    else:
                        px = ndtr((xmax - mx) / 3e-5) - ndtr((xmin - mx) / 3e-5)
                        if family == "filter":
                            band = np.exp(-0.5 * ((wave - 2e-6) / 2e-8) ** 2)
                        elif family == "two-beam":
                            band = (1 + np.cos(2 * np.pi * (exposure * 2e-5) / wave)) / 2
                        elif family == "cavity":
                            band = 0.8 / (
                                1
                                + 4
                                * 0.3
                                / (1 - 0.3) ** 2
                                * np.sin(2 * np.pi * (1e-5 + exposure * 1e-7) / wave) ** 2
                            )
                        else:
                            band = 1.0
                    return np.sum(measure * px * py * band) * 1e-3 / (100 * h * c * nu)

                integral, error = quad_vec(spectrum, 4900, 5100, epsrel=1e-10)
                assert error < abs(integral) * 1e-8
                values[exposure, 0, column] = integral * 0.02 * 1e-8
        return values

    independent = reference(24)
    np.testing.assert_allclose(independent, reference(32), rtol=1e-10)
    np.testing.assert_allclose(actual, independent, rtol=2e-5)


def evaluate_parameters(parameters):
    """Rebuild geometry, optical law and QE; retain first exposure's two reads.

    Parameter coordinates are dimensionless: one pointing unit is 1e-5,
    one band unit is 1e-9 m, and one QE unit is a fractional change.
    The supplied second path is synthetic, not an inferred thermal background.
    """
    field, inputs = case(
        "filter",
        pointing=parameters["pointing"] * 1e-5,
        band_shift=parameters["band"] * 1e-9,
    )
    inputs["exposures"] = replace(inputs["exposures"], bounds_s=inputs["exposures"].bounds_s[:1])
    detector = DetectorMaps(quantum_efficiency=0.7 * (1 + parameters["qe"]))
    paths = [
        image_photoelectrons(
            field,
            **inputs,
            detector=detector,
            source_name=name,
            time_order=5,
            angular_order=9,
        )
        for name in ("scene", "supplied_background")
    ]
    return combine_image_source_paths(*paths).signal


def test_composed_image_parameter_bias_covariance_against_monte_carlo():
    parameters = dict(pointing=0.0, band=0.0, qe=0.0)
    steps = dict(pointing=0.01, band=0.01, qe=0.001)
    nominal, jacobian = native_sensitivities(evaluate_parameters, parameters, steps)
    _, refined = native_sensitivities(
        evaluate_parameters, parameters, {name: value / 2 for name, value in steps.items()}
    )
    np.testing.assert_allclose(jacobian, refined, rtol=2e-4)

    scales = np.array([0.1, 0.1, 0.002])
    correlation = np.array([[1, 0.3, 0], [0.3, 1, 0.2], [0, 0.2, 1]])
    covariance = validate_covariance(correlation * scales[:, None] * scales)
    bias = np.copy(np.array([0.02, -0.03, 0.001]))
    budget = native_tolerance_budget(
        nominal,
        tuple(parameters),
        jacobian,
        bias,
        covariance,
        electron_covariance(nominal, DetectorElectronics(read_noise_electrons=3)),
    )
    mc_bias, mc_covariance = monte_carlo_native(
        evaluate_parameters, parameters, bias, covariance, np.random.default_rng(921), 1600
    )
    # Six standard errors for two declared outputs; covariance relative norm
    # tolerance accommodates finite-population sampling, not a tuned fit.
    standard_error = np.sqrt(np.diag(budget.instrument_covariance) / 1600)
    assert np.all(abs(mc_bias - budget.signed_bias) < 6 * standard_error)
    assert np.linalg.norm(mc_covariance - budget.instrument_covariance) < (
        0.15 * np.linalg.norm(budget.instrument_covariance)
    )
    assert np.trace(budget.detector_covariance) > 0


def test_large_pointing_perturbations_expose_non_linear_bias():
    parameters = dict(pointing=0.0, band=0.0, qe=0.0)
    nominal = evaluate_parameters(parameters).total
    # Symmetric paired draws have exactly zero first-order mean by construction.
    # Large geometry changes lose finite-detector flux and create second-order bias.
    offsets = np.random.default_rng(511).normal(0, 100, 100)
    differences = [
        evaluate_parameters({**parameters, "pointing": sign * offset}).total - nominal
        for offset in offsets
        for sign in (-1, 1)
    ]
    nonlinear_bias = np.mean(differences, axis=0)
    assert np.linalg.norm(nonlinear_bias) / np.linalg.norm(nominal) > 0.01


def test_joint_optical_qe_and_gain_uncertainty_with_conditional_readout():
    parameters = dict(pointing=0.0, band=0.0, qe=0.0, gain=0.0)
    nominal, electron_jacobian = native_sensitivities(
        evaluate_parameters, parameters, dict(pointing=0.01, band=0.01, qe=0.001, gain=0.001)
    )
    # Gain coordinate is fractional electrons/DN change, not optical throughput.
    gain = 0.01
    dn_jacobian = electron_jacobian / gain
    dn_jacobian[:, -1] = -nominal.total / gain

    scales = np.array([0.1, 0.1, 0.002, 0.01])
    correlation = np.eye(4)
    correlation[0, 1] = correlation[1, 0] = 0.3
    correlation[2, 3] = correlation[3, 2] = 0.3
    covariance = validate_covariance(correlation * scales[:, None] * scales)
    parameter_only = propagate_covariance(covariance, dn_jacobian)
    conditional = np.diag((nominal.total + 3**2) / gain**2)
    expected_covariance = parameter_only + conditional
    rng = np.random.default_rng(371)
    count = 1800
    draws = sample_gaussian(covariance, rng, count)
    readings = np.empty((count, 2))
    for index, draw in enumerate(draws):
        perturbed = dict(zip(parameters, draw, strict=True))
        signal = evaluate_parameters(perturbed)
        electronics = DetectorElectronics(
            electrons_per_dn=gain * (1 + perturbed["gain"]),
            bias_dn=10000,
            read_noise_electrons=3,
            full_well_electrons=1e9,
            adc_bits=32,
        )
        readout = sample_readout(signal, electronics, rng)
        assert not readout.full_well_saturated.any() and not readout.adc_clipped.any()
        readings[index] = readout.dn[0]
    expected_mean = nominal.total / gain + 10000
    standard_error = np.sqrt(np.diag(expected_covariance) / count)
    assert np.all(abs(readings.mean(axis=0) - expected_mean) < 6 * standard_error)
    empirical = np.cov(readings, rowvar=False)
    assert (
        np.linalg.norm(empirical - expected_covariance) / np.linalg.norm(expected_covariance) < 0.15
    )


def test_composed_readout_common_noise_and_saturation_boundaries():
    field, inputs = case("filter")
    detector = DetectorMaps(
        quantum_efficiency=0.7,
        electrons_per_dn=0.01,
        bias_dn=10000,
        adc_bits=32,
        full_well_electrons=1e9,
    )

    def paths():
        return combine_image_source_paths(
            *[
                image_photoelectrons(
                    field,
                    **inputs,
                    detector=detector,
                    source_name=name,
                    time_order=5,
                    angular_order=9,
                )
                for name in ("scene", "supplied_background")
            ]
        )

    images = image_readout(
        paths(),
        detector=detector,
        rng=np.random.default_rng(617),
        realisations=12000,
        exposure_noise_electrons=100,
    )
    assert images.dn is not None
    counts = ((images.dn - 10000) * 0.01).reshape(12000, 4)
    covariance = np.cov(counts, rowvar=False)
    reference = np.diag(images.photoelectrons.ravel())
    reference[:2, :2] += 100**2
    reference[2:, 2:] += 100**2
    assert np.linalg.norm(covariance - reference) / np.linalg.norm(reference) < 0.06
    # Gain perturbations are recomputed by readout, not folded into optical QE.
    for gain in (0.01, 0.02):
        trial = DetectorMaps(
            quantum_efficiency=0.7,
            electrons_per_dn=gain,
            bias_dn=10000,
            adc_bits=32,
            full_well_electrons=1e9,
        )
        draws = image_readout(
            paths(),
            detector=trial,
            rng=np.random.default_rng(811),
            realisations=2000,
        )
        assert draws.dn is not None
        empirical = (draws.dn.mean(axis=0) - 10000) * gain
        assert np.all(
            abs(empirical - draws.photoelectrons) < 6 * np.sqrt(draws.photoelectrons / 2000) + gain
        )
    threshold = float(images.photoelectrons[0, 0, 0])
    clipping = DetectorMaps(
        quantum_efficiency=0.7,
        full_well_electrons=threshold,
        electrons_per_dn=1,
        adc_bits=32,
    )
    clipped = image_readout(
        paths(),
        detector=clipping,
        rng=np.random.default_rng(919),
        realisations=6000,
    )
    assert clipped.full_well_saturated is not None and clipped.dn is not None
    frequency = clipped.full_well_saturated[:, 0, 0, 0].mean()
    assert 0.4 < frequency < 0.6
    assert clipped.dn[:, 0, 0, 0].mean() < threshold - 0.1 * np.sqrt(threshold)
