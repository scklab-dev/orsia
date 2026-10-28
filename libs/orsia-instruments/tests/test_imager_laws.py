"""Interchangeable physical laws, joint integration and backwards compatibility."""

from functools import partial
from math import erf
from pathlib import Path

import numpy as np
import pytest
from orsia_instruments import (
    RadianceField,
    cavity_imager,
    cavity_transmission,
    dispersive_imager,
    dispersive_response,
    exposure_sequence,
    filter_imager,
    gaussian_filter,
    gaussian_pixel_fraction,
    image_photoelectrons,
    interferometric_imager,
    lorentzian_filter,
    lvf_imager,
    polynomial_image_map,
    rectangular_acquisition,
    regular_pixel_bounds,
    simulate_images,
    spatial_filter_imager,
    super_gaussian_filter,
    tabulated_transmission,
    two_beam_ports,
)
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.integrate import quad

COMMON = dict(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5)
WAVE = np.array([2.01e-6, 1.98e-6, 2e-6])


def test_documented_factory_example(acquisition):
    guide = Path(__file__).resolve().parents[1] / "README.md"
    code = (
        guide.read_text()
        .split("### Interchangeable factory example\n", 1)[1]
        .split("```python\n", 1)[1]
        .split("```", 1)[0]
    )
    namespace = {}
    exec(compile(code, str(guide), "exec"), namespace)
    for optics in namespace["optics_by_name"].values():
        response = optics.response(0, 0, WAVE, acquisition)
        assert response.shape == (len(acquisition.times_s[0]), len(WAVE))
        assert np.all((response > 0) & (response <= 1))


@pytest.fixture
def acquisition():
    return rectangular_acquisition(
        sample_ids=("r",),
        pixel_ids=("p",),
        setting_ids=("s",),
        port_ids=("direct",),
        exposure_bounds_s=np.array([[0.0, 0.01]]),
        pixel_bounds_m=np.array([[-2e-5, 2e-5, -2e-5, 2e-5]]),
        direction_bounds_uv=(-0.001, 0.001, -0.001, 0.001),
        pointing=lambda t: np.zeros((len(t), 2)),
        focal_length_m=0.02,
        time_order=1,
        angular_order=2,
    )


def test_lvf_profile_matches_existing_gaussian_and_spatial_filter(acquisition):
    profile = partial(gaussian_filter, width_m=2e-8)
    options = dict(
        reference_m=[2e-6, 2.01e-6],
        gradient=4e-4,
        curvature_per_m=0.02,
        peak=0.7,
        axis="y",
        reference_position_m=1e-6,
        pixel_order=12,
    )
    legacy = lvf_imager(**COMMON, **options, sigma_m=2e-8)
    supplied = lvf_imager(**COMMON, **options, profile=profile)
    for exposure in (0, 1):
        expected = legacy.response(exposure, 0, WAVE, acquisition)
        np.testing.assert_array_equal(supplied.response(exposure, 0, WAVE, acquisition), expected)
    assert "sigma_m" not in supplied.settings
    np.testing.assert_array_equal(supplied.settings["reference_m"], [2e-6, 2.01e-6])
    generic = spatial_filter_imager(
        **COMMON,
        filter=lambda x, y, w: 0.7 * profile(w, 2e-6 + 4e-4 * (y - 1e-6) + 0.02 * (y - 1e-6) ** 2),
        pixel_order=12,
    )
    np.testing.assert_array_equal(
        generic.response(0, 0, WAVE, acquisition), supplied.response(0, 0, WAVE, acquisition)
    )


def test_uniform_spatial_filter_matches_analytic_path(acquisition):
    law = partial(lorentzian_filter, centre_m=2e-6, fwhm_m=2e-8, peak=0.7)
    uniform = filter_imager(**COMMON, filter=law)
    spatial = spatial_filter_imager(**COMMON, filter=lambda x, y, w: law(w), pixel_order=12)
    np.testing.assert_allclose(
        spatial.response(0, 0, WAVE, acquisition),
        uniform.response(0, 0, WAVE, acquisition),
        rtol=2e-14,
    )


@pytest.mark.parametrize(
    "factory,params,offset",
    [
        (interferometric_imager, dict(opd_m=[0, 1e-5], port="-", visibility=0.7), "opd_offset_m"),
        (cavity_imager, dict(spacing_m=[1e-5, 1.1e-5], order_transmission=0.7), "spacing_offset_m"),
    ],
)
def test_zero_offsets_match_analytic_path(acquisition, factory, params, offset):
    legacy = factory(**COMMON, **params)
    spatial = factory(**COMMON, **params, **{offset: lambda x, y: 0.0}, pixel_order=12)
    for exposure in (0, 1):
        np.testing.assert_allclose(
            spatial.response(exposure, 0, WAVE, acquisition),
            legacy.response(exposure, 0, WAVE, acquisition),
            rtol=2e-14,
        )


@pytest.mark.parametrize("kind", ["lvf", "spatial", "interference", "cavity"])
def test_joint_integral_against_independent_adaptive_reference(acquisition, kind):
    # Integrate only x numerically; the y integral is an independent erf expression.
    # Local laws vary within each pixel, not just between pixel centres.
    def transmission(x, w):
        if kind in ("lvf", "spatial"):
            return lorentzian_filter(w, 2e-6 + 0.001 * x, 2e-8)
        if kind == "interference":
            return two_beam_ports(w, 1e-5 + 0.1 * x, visibility=0.8)[0]
        return cavity_transmission(w, 1e-5 + 0.005 * x, reflectivity=0.5, order_transmission=0.7)

    def optics(order):
        if kind == "lvf":
            return lvf_imager(
                **COMMON,
                reference_m=2e-6,
                gradient=0.001,
                profile=partial(lorentzian_filter, fwhm_m=2e-8),
                pixel_order=order,
            )
        if kind == "spatial":
            return spatial_filter_imager(
                **COMMON, filter=lambda x, y, w: transmission(x, w), pixel_order=order
            )
        if kind == "interference":
            return interferometric_imager(
                **COMMON,
                opd_m=1e-5,
                port="+",
                visibility=0.8,
                opd_offset_m=lambda x, y: 0.1 * x,
                pixel_order=order,
            )
        return cavity_imager(
            **COMMON,
            spacing_m=1e-5,
            reflectivity=0.5,
            order_transmission=0.7,
            spacing_offset_m=lambda x, y: 0.005 * x,
            pixel_order=order,
        )

    sigma = 3e-5
    reference = np.empty((len(acquisition.times_s[0]), len(WAVE)))
    for i, (cx, cy) in enumerate(acquisition.image_xy_m[0]):
        mass_y = (
            erf((2e-5 - cy) / (sigma * np.sqrt(2))) - erf((-2e-5 - cy) / (sigma * np.sqrt(2)))
        ) / 2
        for j, w in enumerate(WAVE):
            reference[i, j] = (
                mass_y
                * quad(
                    lambda x, cx=cx, w=w: (
                        np.exp(-0.5 * ((x - cx) / sigma) ** 2)
                        / (np.sqrt(2 * np.pi) * sigma)
                        * float(transmission(x, w))
                    ),
                    -2e-5,
                    2e-5,
                    epsabs=1e-13,
                    epsrel=1e-13,
                )[0]
            )
    coarse = optics(8).response(0, 0, WAVE, acquisition)
    fine = optics(32).response(0, 0, WAVE, acquisition)
    np.testing.assert_allclose(fine, reference, rtol=2e-11, atol=1e-14)
    assert np.max(np.abs(fine - reference)) <= np.max(np.abs(coarse - reference)) + 1e-15
    # Sampling transmission only at the centre gives a measurably different result.
    centre_only = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], acquisition.image_xy_m[0], sigma
    )[:, None] * transmission(0, WAVE)
    assert np.max(np.abs(fine - centre_only)) > 1e-4


def test_spectral_interferometer_parameters_and_complementary_ports(acquisition):
    def phase(w):
        return 0.3 + (w - 2e-6) / 1e-7

    def visibility(w):
        return 0.7 + (w - 2e-6) / 1e-7

    def transmission(w):
        return np.full_like(w, 0.6)

    opts = dict(
        opd_m=[0.0, 1e-5], phase_rad=phase, visibility=visibility, transmission=transmission
    )
    plus = interferometric_imager(**COMMON, **opts, port="+")
    minus = interferometric_imager(**COMMON, **opts, port="-")
    fraction = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], acquisition.image_xy_m[0], 3e-5
    )[:, None]
    for index, opd in enumerate([0.0, 1e-5]):
        p = plus.response(index, 0, WAVE, acquisition)
        m = minus.response(index, 0, WAVE, acquisition)
        np.testing.assert_allclose(p + m, fraction * transmission(WAVE))
        reference = two_beam_ports(
            WAVE,
            opd,
            phase_rad=phase(WAVE),
            visibility=visibility(WAVE),
            transmission=transmission(WAVE),
        )
        np.testing.assert_array_equal(p, fraction * reference[0])


def test_spectral_cavity_parameters_legacy_order_and_callback_support(acquisition):
    calls = []

    def index(w):
        calls.append(w.copy())
        assert np.all((w > 1.9e-6) & (w < 2.1e-6))
        return 1.4 + (w - 2e-6) / 1e-6

    opts = dict(
        spacing_m=1e-5,
        refractive_index=index,
        reflectivity=lambda w: np.full_like(w, 0.5),
        coating_phase_rad=lambda w: (w - 2e-6) / 1e-6,
        order_transmission=lambda w: np.array(0.7),
    )
    optics = cavity_imager(**COMMON, **opts)
    assert not calls  # No eager evaluation at an invented wavelength.
    actual = optics.response(0, 0, WAVE, acquisition)
    fraction = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], acquisition.image_xy_m[0], 3e-5
    )[:, None]
    np.testing.assert_allclose(
        actual,
        fraction
        * cavity_transmission(
            WAVE,
            1e-5,
            refractive_index=index(WAVE),
            reflectivity=0.5,
            coating_phase_rad=(WAVE - 2e-6) / 1e-6,
            order_transmission=0.7,
        ),
    )


def test_polynomial_image_map_formula_and_no_mutation():
    image = np.array([[1e-5, 2e-5], [-1e-5, 3e-5]])
    before = image.copy()
    wave_before = WAVE.copy()
    result = polynomial_image_map(
        image, WAVE, reference_m=2e-6, dispersion=3, quadratic_per_m=4e5, smile=0.2, keystone=0.1
    )
    delta = WAVE - 2e-6
    np.testing.assert_allclose(
        result[..., 0], image[:, 0, None] + 3 * delta + 4e5 * delta**2 + 0.2 * image[:, 1, None]
    )
    np.testing.assert_allclose(result[..., 1], image[:, 1, None] + 0.1 * delta)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(WAVE, wave_before)
    assert result.flags.writeable and not np.shares_memory(result, image)


def test_supplied_image_map_and_spectral_efficiency(acquisition):
    law = partial(
        polynomial_image_map,
        reference_m=2e-6,
        dispersion=4,
        quadratic_per_m=1e4,
        smile=0.1,
        keystone=0.2,
    )
    opts = dict(reference_m=2e-6, dispersion=4, quadratic_per_m=1e4, smile=0.1, keystone=0.2)
    builtin = dispersive_imager(**COMMON, **opts, transmission=0.7)
    supplied = dispersive_imager(
        **COMMON, image_map=law, transmission=lambda w: np.full_like(w, 0.7)
    )
    np.testing.assert_array_equal(
        supplied.response(0, 0, WAVE, acquisition), builtin.response(0, 0, WAVE, acquisition)
    )
    np.testing.assert_array_equal(
        builtin.response(0, 0, WAVE, acquisition),
        dispersive_response(
            WAVE,
            acquisition.pixel_bounds_m[0],
            acquisition.image_xy_m[0],
            sigma_m=3e-5,
            **opts,
            transmission=0.7,
        ),
    )
    assert not supplied.settings

    def nonlinear(image, wave):
        result = np.broadcast_to(image[:, None, :], (len(image), len(wave), 2)).copy()
        result[..., 0] += 1e-5 * np.sin((wave - 2e-6) / 1e-8)
        return result

    supplied = dispersive_imager(
        **COMMON, image_map=nonlinear, transmission=lambda w: lorentzian_filter(w, 2e-6, 2e-8)
    )
    fraction = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], nonlinear(acquisition.image_xy_m[0], WAVE), 3e-5
    )
    np.testing.assert_array_equal(
        supplied.response(0, 0, WAVE, acquisition), fraction * lorentzian_filter(WAVE, 2e-6, 2e-8)
    )


@pytest.mark.parametrize(
    "factory,opts",
    [
        (lvf_imager, dict(reference_m=2e-6, gradient=0)),
        (
            lvf_imager,
            dict(
                reference_m=2e-6,
                gradient=0,
                sigma_m=2e-8,
                profile=partial(lorentzian_filter, fwhm_m=2e-8),
            ),
        ),
        (dispersive_imager, {}),
        (dispersive_imager, dict(reference_m=2e-6)),
        (dispersive_imager, dict(image_map=polynomial_image_map, reference_m=2e-6)),
        (dispersive_imager, dict(image_map=polynomial_image_map, dispersion=0)),
        (dispersive_imager, dict(image_map=polynomial_image_map, smile=0.1)),
        (spatial_filter_imager, dict(filter=lambda x, y, w: w, pixel_order=0)),
        (interferometric_imager, dict(opd_m=0, port="+", pixel_order=0)),
        (cavity_imager, dict(spacing_m=1e-5, order_transmission=1, pixel_order=0)),
    ],
)
def test_incompatible_or_missing_parameters(factory, opts):
    with pytest.raises(ValueError):
        factory(**COMMON, **opts)


@pytest.mark.parametrize("value", [-0.1, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("family", ["lvf", "spatial", "visibility", "reflectivity", "transmission"])
def test_invalid_callback_values_before_acceptance(acquisition, value, family):
    def law(w):
        return np.full_like(w, value)

    if family == "lvf":
        optics = lvf_imager(
            **COMMON, reference_m=2e-6, gradient=0, peak=0, profile=lambda w, c: law(w)
        )
    elif family == "spatial":
        optics = spatial_filter_imager(**COMMON, filter=lambda x, y, w: law(w))
    elif family == "visibility":
        optics = interferometric_imager(**COMMON, opd_m=0, port="+", visibility=law)
    elif family == "reflectivity":
        optics = cavity_imager(**COMMON, spacing_m=1e-5, order_transmission=0, reflectivity=law)
    else:
        optics = dispersive_imager(**COMMON, reference_m=2e-6, dispersion=0, transmission=law)
    with pytest.raises(ValueError):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize("shape", [(), (1,), (1, 3), (3, 1)])
@pytest.mark.parametrize("family", ["lvf", "spatial", "phase", "index", "transmission"])
def test_callback_shapes_are_not_implicitly_broadcast(acquisition, shape, family):
    def law(w):
        return np.full(shape, 0.5)

    if family == "lvf":
        optics = lvf_imager(**COMMON, reference_m=2e-6, gradient=0, profile=lambda w, c: law(w))
    elif family == "spatial":
        optics = spatial_filter_imager(**COMMON, filter=lambda x, y, w: law(w))
    elif family == "phase":
        optics = interferometric_imager(**COMMON, opd_m=0, port="+", phase_rad=law)
    elif family == "index":
        optics = cavity_imager(**COMMON, spacing_m=1e-5, order_transmission=1, refractive_index=law)
    else:
        optics = dispersive_imager(**COMMON, reference_m=2e-6, dispersion=0, transmission=law)
    with pytest.raises(ValueError, match="same shape"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", ["opd", "spacing"])
def test_invalid_spatial_offsets(acquisition, value, kind):
    if kind == "opd":
        optics = interferometric_imager(
            **COMMON, opd_m=0, port="+", opd_offset_m=lambda x, y: value
        )
    else:
        optics = cavity_imager(
            **COMMON, spacing_m=1e-5, order_transmission=1, spacing_offset_m=lambda x, y: value
        )
    with pytest.raises(ValueError, match="finite scalar"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize(
    "params",
    [
        {"refractive_index": lambda w: np.zeros_like(w)},
        {"reflectivity": lambda w: np.ones_like(w)},
        {"coating_phase_rad": lambda w: np.full_like(w, np.nan)},
    ],
)
def test_cavity_physical_domains(acquisition, params):
    optics = cavity_imager(
        **COMMON,
        spacing_m=1e-5,
        order_transmission=1,
        **params,
    )
    with pytest.raises(ValueError):
        optics.response(0, 0, WAVE, acquisition)


def test_cavity_rejects_nonpositive_total_spacing(acquisition):
    optics = cavity_imager(
        **COMMON, spacing_m=1e-5, order_transmission=1, spacing_offset_m=lambda x, y: -1e-5
    )
    with pytest.raises(ValueError, match="spacing"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize(
    "bad",
    [
        lambda p, w: p,
        lambda p, w: np.zeros((len(p), len(w), 1)),
        lambda p, w: np.full((len(p), len(w), 2), np.nan),
    ],
)
def test_invalid_image_map(acquisition, bad):
    optics = dispersive_imager(**COMMON, image_map=bad)
    with pytest.raises(ValueError, match="image_map"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize(
    "image,wave,kwargs",
    [
        (np.zeros((2, 3)), WAVE, {}),
        (np.array([[np.nan, 0]]), WAVE, {}),
        (np.zeros((2, 2)), WAVE[:, None], {}),
        (np.zeros((2, 2)), WAVE, {"smile": np.nan}),
        (np.full((2, 2), 1e308), WAVE, {"smile": 10}),
    ],
)
def test_invalid_polynomial_map(image, wave, kwargs):
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        polynomial_image_map(image, wave, reference_m=2e-6, dispersion=1, **kwargs)


def test_tabulated_spatial_support_errors_are_preserved(acquisition):
    optics = spatial_filter_imager(
        **COMMON, filter=lambda x, y, w: tabulated_transmission(w, [1.99e-6, 2.01e-6], [0.5, 0.8])
    )
    with pytest.raises(ValueError, match="support"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize("kind", ["lvf", "spatial", "interference", "cavity", "dispersive"])
def test_dense_optical_charge_and_exposure_settings(kind):
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.linspace(4900.0, 5100.0, 17),
        SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),)),
    )
    scene = RadianceField(
        grid,
        np.array([0.0, 0.1]),
        np.array([-0.002, 0.002]),
        np.array([-0.002, 0.002]),
        np.full((2, 2, 2, 17), 1e-3),
    )
    law = partial(super_gaussian_filter, centre_m=2e-6, fwhm_m=3e-8)
    if kind == "lvf":
        optics = lvf_imager(
            **COMMON,
            reference_m=[2e-6, 2.01e-6],
            gradient=1e-4,
            profile=partial(lorentzian_filter, fwhm_m=2e-8),
        )
    elif kind == "spatial":
        optics = spatial_filter_imager(**COMMON, filter=lambda x, y, w: law(w) * (0.7 + 100 * x))
    elif kind == "interference":
        optics = interferometric_imager(
            **COMMON,
            opd_m=[0, 1e-5],
            port="+",
            transmission=law,
            opd_offset_m=lambda x, y: 0.01 * x,
        )
    elif kind == "cavity":
        optics = cavity_imager(
            **COMMON,
            spacing_m=[1e-5, 1.1e-5],
            order_transmission=law,
            refractive_index=lambda w: 1.4 + (w - 2e-6) / 1e-6,
            spacing_offset_m=lambda x, y: 0.0001 * x,
        )
    else:
        optics = dispersive_imager(
            **COMMON,
            image_map=partial(polynomial_image_map, reference_m=2e-6, dispersion=4),
            transmission=law,
        )
    inputs = dict(
        pixel_bounds_m=regular_pixel_bounds((1, 2), pitch_m=3e-5),
        optics=optics,
        exposures=exposure_sequence(
            start_s=[0.0, 0.05], duration_s=0.01, direction_bounds_uv=(-0.002, 0.002, -0.002, 0.002)
        ),
        angular_order=3,
        time_order=1,
    )
    dense = simulate_images(scene, **inputs)
    charge = image_photoelectrons(scene, **inputs)
    np.testing.assert_allclose(
        charge.signal.total.reshape(dense.photoelectrons.shape), dense.photoelectrons, rtol=5e-13
    )
    assert np.all(dense.photoelectrons > 0)
    for name, setting in optics.settings.items():
        np.testing.assert_array_equal(dense.settings[name], setting)
