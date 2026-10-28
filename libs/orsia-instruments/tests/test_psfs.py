"""Callable PSFs: normalization, coordinate context, joint integrals and compatibility."""

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from orsia_instruments import (
    RadianceField,
    cavity_imager,
    dispersive_imager,
    exposure_sequence,
    filter_imager,
    gaussian_pixel_fraction,
    gaussian_psf,
    image_photoelectrons,
    interferometric_imager,
    lvf_imager,
    rectangular_acquisition,
    regular_pixel_bounds,
    simulate_images,
    spatial_filter_imager,
    tabulated_pixel_response,
    tabulated_psf,
)
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.integrate import quad

COMMON = dict(focal_length_m=0.02, collecting_area_m2=1e-8)
WAVE = np.array([2.01e-6, 1.98e-6, 2e-6])
SIGMA = (3e-5, 2e-5)
FACTORIES = [
    (filter_imager, dict(filter=lambda w: np.full_like(w, 0.7))),
    (spatial_filter_imager, dict(filter=lambda x, y, w: np.full_like(w, 0.5 + 1e4 * x))),
    (lvf_imager, dict(reference_m=2e-6, gradient=0.001, sigma_m=2e-8)),
    (interferometric_imager, dict(opd_m=[0, 1e-5], port="+")),
    (interferometric_imager, dict(opd_m=1e-5, port="-", opd_offset_m=lambda x, y: 0.1 * x)),
    (cavity_imager, dict(spacing_m=1e-5, order_transmission=0.7)),
    (
        cavity_imager,
        dict(spacing_m=1e-5, order_transmission=0.7, spacing_offset_m=lambda x, y: 0.001 * y),
    ),
    (dispersive_imager, dict(reference_m=2e-6, dispersion=1.0, transmission=0.7)),
]


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


def wrapped_gaussian(offsets, wave, field, exposure):
    return gaussian_psf(offsets, wave, field, exposure, sigma_m=SIGMA)


@pytest.mark.parametrize("factory,kwargs", FACTORIES)
def test_gaussian_binding_and_generic_quadrature_match_legacy(acquisition, factory, kwargs):
    legacy = factory(**COMMON, psf_sigma_m=SIGMA, pixel_order=20, **kwargs)
    bound = factory(**COMMON, psf=partial(gaussian_psf, sigma_m=SIGMA), pixel_order=20, **kwargs)
    generic = factory(**COMMON, psf=wrapped_gaussian, pixel_order=20, **kwargs)
    for exposure in (0, 1):
        expected = legacy.response(exposure, 0, WAVE, acquisition)
        np.testing.assert_array_equal(bound.response(exposure, 0, WAVE, acquisition), expected)
        np.testing.assert_allclose(
            generic.response(exposure, 0, WAVE, acquisition), expected, rtol=3e-14
        )


@pytest.mark.parametrize("factory,kwargs", FACTORIES)
def test_psf_selection_and_order_validation(factory, kwargs):
    for selection in ({}, dict(psf=wrapped_gaussian, psf_sigma_m=SIGMA)):
        with pytest.raises(ValueError, match="exactly one"):
            factory(**COMMON, **kwargs, **selection)
    with pytest.raises(ValueError, match="order"):
        factory(**COMMON, **kwargs, psf=wrapped_gaussian, pixel_order=0)


def test_analytic_path_and_owned_sigma(acquisition):
    sigma = np.array(SIGMA)
    optics = filter_imager(
        **COMMON, filter=np.ones_like, psf=partial(gaussian_psf, sigma_m=sigma), pixel_order=1
    )
    sigma[:] = 1e-8
    expected = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], acquisition.image_xy_m[0], SIGMA
    )
    np.testing.assert_array_equal(
        optics.response(0, 0, WAVE, acquisition), np.broadcast_to(expected[:, None], (4, 3))
    )


def test_catalogue_gaussian_density_and_normalization():
    offsets = np.array([[[0.0, 0.0], [SIGMA[0], SIGMA[1]], [0.0, 0.0]]])
    saved = offsets.copy()
    values = gaussian_psf(offsets, WAVE, np.zeros((1, 2)), 2, sigma_m=SIGMA)
    np.testing.assert_allclose(
        values[0, :2], np.array([1.0, np.exp(-1)]) / (2 * np.pi * np.prod(SIGMA))
    )
    np.testing.assert_array_equal(offsets, saved)
    # Independent whole-plane integral in sigma-scaled coordinates.
    mass = quad(
        lambda u: (
            gaussian_psf(
                np.array([[[u * SIGMA[0], 0.0]]]), WAVE[:1], np.zeros((1, 2)), 0, sigma_m=SIGMA
            )[0, 0]
            * SIGMA[0]
            * np.sqrt(2 * np.pi)
            * SIGMA[1]
        ),
        -10,
        10,
    )[0]
    assert mass == pytest.approx(1, abs=1e-12)
    scalar = gaussian_psf(offsets, WAVE, np.zeros((1, 2)), 0, sigma_m=3e-5)
    assert scalar[0, 0] == pytest.approx(1 / (2 * np.pi * (3e-5) ** 2))


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.ones(3)])
def test_invalid_gaussian_sigma(sigma):
    with pytest.raises(ValueError):
        gaussian_psf(np.zeros((1, 3, 2)), WAVE, np.zeros((1, 2)), 0, sigma_m=sigma)


@pytest.mark.parametrize(
    "offsets,wave,field",
    [
        (np.zeros((1, 3)), WAVE, np.zeros((1, 2))),
        (np.zeros((1, 3, 2)), WAVE[:, None], np.zeros((1, 2))),
        (np.zeros((1, 3, 2)), WAVE, np.zeros(2)),
        (np.full((1, 3, 2), np.nan), WAVE, np.zeros((1, 2))),
        (np.zeros((1, 3, 2)), WAVE, np.full((1, 2), np.inf)),
    ],
)
def test_catalogue_coordinate_contract(offsets, wave, field):
    with pytest.raises(ValueError, match="PSF needs"):
        gaussian_psf(offsets, wave, field, 0, sigma_m=1)


def test_chromatic_field_exposure_context_and_mapped_offsets(acquisition):
    original = acquisition.image_xy_m[0].copy()

    def widths(w, field, exposure):
        return 3e-5 * (w[None, :] / 2e-6) * (1 + field[:, 0, None] / 1e-4) * (1 + 0.2 * exposure)

    def psf(offsets, w, field, exposure):
        np.testing.assert_array_equal(field, original)
        sigma = widths(w, field, exposure)
        return np.exp(-0.5 * np.sum((offsets / sigma[..., None]) ** 2, axis=-1)) / (
            2 * np.pi * sigma**2
        )

    def mapping(field, w):
        result = np.broadcast_to(field[:, None, :], (len(field), len(w), 2)).copy()
        result[..., 0] += (w - 2e-6) * 400
        return result

    optics = dispersive_imager(**COMMON, psf=psf, image_map=mapping, pixel_order=16)
    for exposure in (0, 1):
        expected = gaussian_pixel_fraction(
            acquisition.pixel_bounds_m[0],
            mapping(original, WAVE),
            widths(WAVE, original, exposure)[..., None],
        )
        actual = optics.response(exposure, 0, WAVE, acquisition)
        np.testing.assert_allclose(actual, expected, rtol=2e-14)
        blocked = np.concatenate(
            [optics.response(exposure, 0, WAVE[i : i + 1], acquisition) for i in range(3)], axis=1
        )
        np.testing.assert_array_equal(blocked, actual)
    np.testing.assert_array_equal(acquisition.image_xy_m[0], original)


def test_table_bilinear_axes_support_and_explicit_normalization():
    x, y = np.array([-2.0, 0.0, 2.0]), np.array([-1.0, 1.0])
    table = (1 + x[:, None] / 4 + y[None, :] / 4) / 8
    saved = table.copy()
    offsets = np.array([[[-2.0, -1.0], [1.0, 0.5], [2.0, 1.0], [3.0, 0.0], [0.0, -2.0]]])
    wave = np.ones(5)
    field = np.zeros((1, 2))
    law = partial(tabulated_psf, x_offsets_m=x, y_offsets_m=y)
    actual = law(offsets, wave, field, 0, density_per_m2=table)
    np.testing.assert_allclose(actual, [[0.25 / 8, 1.375 / 8, 1.75 / 8, 0, 0]])
    np.testing.assert_array_equal(table, saved)
    with pytest.raises(ValueError, match="integral must be one"):
        law(offsets, wave, field, 0, density_per_m2=table * 2)
    np.testing.assert_allclose(
        law(offsets, wave, field, 0, density_per_m2=table * 2, normalize=True), actual
    )


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("pixel", [False, True])
def test_tabulated_interpolation_irregular_grid_support_and_ownership(readonly, pixel):
    x = np.array([-2.0, -0.5, 3.0])
    y = np.array([-1.0, 0.25, 2.0])
    table = 4 + x[:, None] + y[None, :] / 2 + x[:, None] * y[None, :] / 8
    mass = np.trapezoid(np.trapezoid(table, y, axis=1), x)
    table /= mass
    px, py = np.meshgrid([-3.0, -2.0, -0.25, 3.0, 4.0], [-2.0, -1.0, 0.5, 2.0, 3.0], indexing="ij")
    # Strided coordinate storage exercises the consumers' (..., x/y) axis order.
    storage = np.zeros((*px.shape, 4))
    offsets = storage[..., ::2]
    offsets[..., 0], offsets[..., 1] = px, py
    before = [a.copy() for a in (x, y, table, offsets)]
    if readonly:
        for array in (x, y, table, offsets):
            array.setflags(write=False)
    expected = (4 + px + py / 2 + px * py / 8) / mass
    expected[(px < x[0]) | (px > x[-1]) | (py < y[0]) | (py > y[-1])] = 0
    wave = np.ones(px.shape[1])
    if pixel:
        # One midpoint quadrature node at (0,0): offsets equal minus the centroids.
        actual = tabulated_pixel_response(
            wave,
            [-0.1, 0.1, -0.1, 0.1],
            -offsets,
            x,
            y,
            table,
            transmission=lambda qx, qy, w: np.ones_like(w),
            pixel_order=1,
        )
        expected *= 0.04
    else:
        actual = tabulated_psf(
            offsets,
            wave,
            np.zeros((len(px), 2)),
            0,
            x_offsets_m=x,
            y_offsets_m=y,
            density_per_m2=table,
        )
    assert actual.shape == expected.shape and actual.flags.writeable
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)
    for array, saved in zip((x, y, table, offsets), before, strict=True):
        np.testing.assert_array_equal(array, saved)
        assert not np.shares_memory(actual, array)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_tabulated_psf_empty_coordinates(shape):
    actual = tabulated_psf(
        np.empty((*shape, 2)),
        np.ones(shape[1]),
        np.zeros((shape[0], 2)),
        0,
        x_offsets_m=[0, 1],
        y_offsets_m=[0, 1],
        density_per_m2=np.ones((2, 2)),
    )
    assert actual.shape == shape and actual.flags.writeable


@pytest.mark.parametrize(
    "overrides",
    [
        dict(x_offsets_m=[0]),
        dict(x_offsets_m=[0, 0]),
        dict(y_offsets_m=[1, 0]),
        dict(y_offsets_m=[0, np.nan]),
        dict(x_offsets_m=[[0, 1]]),
        dict(density_per_m2=np.ones((2, 3))),
        dict(density_per_m2=-np.ones((2, 2))),
        dict(density_per_m2=np.full((2, 2), np.nan)),
        dict(density_per_m2=np.zeros((2, 2))),
    ],
)
def test_invalid_tables(overrides):
    kwargs = (
        dict(x_offsets_m=[0, 1], y_offsets_m=[0, 1], density_per_m2=np.ones((2, 2))) | overrides
    )
    with pytest.raises(ValueError):
        tabulated_psf(np.zeros((1, 1, 2)), np.ones(1), np.zeros((1, 2)), 0, **kwargs)


def test_joint_table_integral_not_centre_sample(acquisition):
    # Linear density on an asymmetric, non-square x/y table, unit integral.
    x, y = np.array([-1e-4, 0, 1e-4]), np.array([-8e-5, 8e-5])
    density = np.broadcast_to((1 + x[:, None] / 2e-4) / (2e-4 * 1.6e-4), (3, 2)).copy()
    psf = partial(tabulated_psf, x_offsets_m=x, y_offsets_m=y, density_per_m2=density)

    def law(x, y, w):
        return np.full_like(w, 0.5 + 1e4 * x)

    optics = spatial_filter_imager(**COMMON, psf=psf, filter=law, pixel_order=2)
    actual = optics.response(0, 0, WAVE, acquisition)
    # Exact polynomial integral: E[x]=0 and E[x^2]=a^2/3 over [-a,a].
    centres = acquisition.image_xy_m[0, :, 0]
    expected = (
        (4e-5) ** 2
        / (2e-4 * 1.6e-4)
        * (0.5 * (1 - centres / 2e-4) + (1e4 / 2e-4) * (2e-5) ** 2 / 3)
    )
    np.testing.assert_allclose(actual, np.broadcast_to(expected[:, None], actual.shape), rtol=1e-14)
    assert np.all(actual < 1)  # Detector truncation is retained, not normalized away.
    legacy = tabulated_pixel_response(
        WAVE,
        acquisition.pixel_bounds_m[0],
        acquisition.image_xy_m[0],
        x,
        y,
        density,
        transmission=law,
        pixel_order=2,
    )
    np.testing.assert_allclose(actual, legacy, rtol=1e-14)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_callback_density_even_with_zero_transmission(acquisition, value):
    optics = filter_imager(
        **COMMON, filter=np.zeros_like, psf=lambda d, w, f, e: np.full(d.shape[:2], value)
    )
    with pytest.raises(ValueError, match="PSF density"):
        optics.response(0, 0, WAVE, acquisition)


@pytest.mark.parametrize("shape", [(), (3,), (1, 3), (4, 3, 1)])
def test_callback_exact_shape(acquisition, shape):
    optics = filter_imager(**COMMON, filter=np.ones_like, psf=lambda d, w, f, e: np.ones(shape))
    with pytest.raises(ValueError, match="PSF density must have shape"):
        optics.response(0, 0, WAVE, acquisition)


def test_underresolved_or_unnormalized_callback_is_not_clipped(acquisition):
    optics = filter_imager(
        **COMMON, filter=np.ones_like, psf=lambda d, w, f, e: np.full(d.shape[:2], 1e12)
    )
    with pytest.raises(ValueError, match="fraction exceeds one"):
        optics.response(0, 0, WAVE, acquisition)
    narrow = filter_imager(
        **COMMON,
        filter=np.ones_like,
        pixel_order=2,
        psf=lambda d, w, f, e: gaussian_psf(d, w, f, e, sigma_m=1e-5),
    )
    refined = filter_imager(
        **COMMON,
        filter=np.ones_like,
        pixel_order=20,
        psf=lambda d, w, f, e: gaussian_psf(d, w, f, e, sigma_m=1e-5),
    )
    exact = filter_imager(**COMMON, filter=np.ones_like, psf_sigma_m=1e-5)
    reference = exact.response(0, 0, WAVE, acquisition)
    fine = refined.response(0, 0, WAVE, acquisition)
    np.testing.assert_allclose(fine, reference, rtol=1e-13)
    assert np.max(abs(narrow.response(0, 0, WAVE, acquisition) - reference)) > 1e-3


def test_callback_exceptions_propagate_and_not_called_at_construction(acquisition):
    def law(d, w, f, e):
        raise RuntimeError("calibration unavailable")

    optics = filter_imager(**COMMON, filter=np.ones_like, psf=law)
    with pytest.raises(RuntimeError, match="calibration unavailable"):
        optics.response(0, 0, WAVE, acquisition)


def test_dense_psf_exposure_dependence():
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.linspace(4900.0, 5100.0, 7),
        SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),)),
    )
    scene = RadianceField(
        grid,
        np.array([0.0, 0.1]),
        np.array([-0.002, 0.002]),
        np.array([-0.002, 0.002]),
        np.full((2, 2, 2, 7), 1e-3),
    )

    def law(d, w, f, e):
        return gaussian_psf(d, w, f, e, sigma_m=3e-5 * (1 + 0.2 * e))

    optics = filter_imager(**COMMON, filter=np.ones_like, psf=law, pixel_order=8)
    kwargs = dict(
        pixel_bounds_m=regular_pixel_bounds((2, 3), pitch_m=2e-5),
        optics=optics,
        exposures=exposure_sequence(
            start_s=[0.0, 0.05], duration_s=0.01, direction_bounds_uv=(-0.001, 0.001, -0.001, 0.001)
        ),
        time_order=1,
        angular_order=2,
    )
    dense = simulate_images(scene, **kwargs)
    charge = image_photoelectrons(scene, **kwargs)
    assembled = charge.signal.total.reshape(dense.photoelectrons.shape)
    np.testing.assert_allclose(assembled, dense.photoelectrons, rtol=2e-14)
    assert not np.allclose(assembled[0], assembled[1])


def test_documented_psf_examples(acquisition):
    guide = Path(__file__).resolve().parents[1] / "README.md"
    code = (
        guide.read_text()
        .split("### PSF callback contract\n", 1)[1]
        .split("```python\n", 1)[1]
        .split("```", 1)[0]
    )
    namespace = {}
    exec(compile(code, str(guide), "exec"), namespace)
    for name in ("gaussian_optics", "tabulated_optics"):
        values = namespace[name].response(0, 0, WAVE, acquisition)
        assert np.all((values > 0) & (values < 1))


def test_unit_mass_roundoff_and_zero_outside_detector(acquisition):
    half_width = 2e-5
    psf = partial(
        tabulated_psf,
        x_offsets_m=[-half_width, half_width],
        y_offsets_m=[-half_width, half_width],
        density_per_m2=np.full((2, 2), (1 + 2e-12) / (2 * half_width) ** 2),
    )

    def centred(field, wave):
        return np.zeros((len(field), len(wave), 2))

    optics = dispersive_imager(**COMMON, psf=psf, image_map=centred, pixel_order=8)
    np.testing.assert_array_equal(optics.response(0, 0, WAVE, acquisition), np.ones((4, 3)))

    def outside(field, wave):
        return np.full((len(field), len(wave), 2), 1.0)

    optics = dispersive_imager(**COMMON, psf=psf, image_map=outside)
    np.testing.assert_array_equal(optics.response(0, 0, WAVE, acquisition), np.zeros((4, 3)))


def test_rotated_two_component_psf_against_adaptive_reference(acquisition):
    # Nonseparable, bimodal profile; both components have unit whole-plane mass.
    scale = 2e-5
    angle = 0.6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    widths = np.array([0.9, 0.4])
    covariance = rotation @ np.diag(widths**2) @ rotation.T
    precision = np.linalg.inv(covariance)

    def psf(offsets, wave, field, exposure):
        result = np.zeros(offsets.shape[:2])
        for fraction, shift in ((0.7, -0.4), (0.3, 0.6)):
            local = (offsets / scale - [shift, 0]) @ rotation
            result += fraction * np.exp(-0.5 * np.sum((local / widths) ** 2, axis=-1))
        return result / (2 * np.pi * np.prod(widths) * scale**2)

    def transmission(x, y, w):
        return np.full_like(w, 0.5 + 0.2 * x / scale)

    def integral(cx, cy):
        def at_x(x):
            def at_y(y):
                density = 0.0
                for fraction, shift in ((0.7, -0.4), (0.3, 0.6)):
                    delta = np.array([x - cx - shift, y - cy])
                    density += fraction * np.exp(-0.5 * delta @ precision @ delta)
                return density * (0.5 + 0.2 * x) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))

            return quad(at_y, -1, 1, epsabs=1e-12, epsrel=1e-12)[0]

        return quad(at_x, -1, 1, epsabs=1e-12, epsrel=1e-12)[0]

    reference = np.array([integral(*centre) for centre in acquisition.image_xy_m[0] / scale])
    fine = spatial_filter_imager(**COMMON, psf=psf, filter=transmission, pixel_order=24)
    coarse = spatial_filter_imager(**COMMON, psf=psf, filter=transmission, pixel_order=3)
    np.testing.assert_allclose(fine.response(0, 0, WAVE, acquisition)[:, 0], reference, rtol=1e-12)
    assert np.max(abs(coarse.response(0, 0, WAVE, acquisition)[:, 0] - reference)) > 1e-4


def test_partial_subclass_does_not_bypass_custom_density(acquisition):
    class WiderGaussian(partial):
        def __call__(self, offsets, wave, field, exposure):
            return gaussian_psf(offsets, wave, field, exposure, sigma_m=6e-5)

    psf = WiderGaussian(gaussian_psf, sigma_m=3e-5)
    optics = filter_imager(**COMMON, filter=np.ones_like, psf=psf, pixel_order=12)
    reference = filter_imager(**COMMON, filter=np.ones_like, psf_sigma_m=6e-5)
    np.testing.assert_allclose(
        optics.response(0, 0, WAVE, acquisition),
        reference.response(0, 0, WAVE, acquisition),
        rtol=1e-14,
    )
