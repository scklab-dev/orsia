"""Physical image API contracts, kernel equivalence and grouped noise."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from orsia_instruments import (
    DetectorMaps,
    DetectorOperatingPoints,
    ImagingOptics,
    RadianceField,
    cavity_imager,
    cavity_transmission,
    dispersive_imager,
    dispersive_response,
    expected_electrons,
    exposure_sequence,
    filter_imager,
    gaussian_filter,
    gaussian_pixel_fraction,
    image_acquisition,
    image_photoelectrons,
    interferometric_imager,
    lvf_imager,
    regular_pixel_bounds,
    sample_readout,
    simulate_images,
    tabulated_transmission,
)
from orsia_radiative import SpectralDomain, SpectralGrid


@pytest.fixture
def case():
    points = np.linspace(4900.0, 5100.0, 33)
    grid = SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),))
    )
    scene = RadianceField(
        grid,
        np.array([0.0, 0.2]),
        np.array([-0.003, 0.003]),
        np.array([-0.003, 0.003]),
        np.full((2, 2, 2, 33), 1e-3),
    )
    inputs = dict(
        pixel_bounds_m=regular_pixel_bounds((2, 3), pitch_m=(4e-5, 5e-5)),
        optics=filter_imager(
            focal_length_m=0.02,
            collecting_area_m2=1e-8,
            psf_sigma_m=3e-5,
            filter=partial(gaussian_filter, centre_m=2e-6, width_m=2e-8),
        ),
        exposures=exposure_sequence(
            start_s=[0.0, 0.05],
            duration_s=0.01,
            direction_bounds_uv=(-0.003, 0.003, -0.003, 0.003),
            row_delay_s=0.002,
        ),
    )
    return scene, inputs


def test_geometry_pitch_active_position_and_ownership():
    pixels = regular_pixel_bounds(
        (2, 3), pitch_m=(4.0, 6.0), active_size_m=(2.0, 4.0), centre_xy_m=(10.0, 20.0)
    )
    np.testing.assert_array_equal(pixels[0, 0], [5.0, 7.0, 15.0, 19.0])
    np.testing.assert_array_equal(pixels[1, 2], [13.0, 15.0, 21.0, 25.0])
    assert pixels.flags.writeable
    pixels[0, 0] = 0
    assert regular_pixel_bounds((1, 1), pitch_m=2).tolist() == [[[-1.0, 1.0, -1.0, 1.0]]]


@pytest.mark.parametrize(
    "options",
    [
        dict(shape=(0, 1), pitch_m=1),
        dict(shape=(1, 1), pitch_m=0),
        dict(shape=(1, 1), pitch_m=1, active_size_m=2),
        dict(shape=(1, 1), pitch_m=[1, 2, 3]),
        dict(shape=(1, 1), pitch_m=1, centre_xy_m=0),
    ],
)
def test_bad_geometry(options):
    with pytest.raises(ValueError):
        regular_pixel_bounds(**options)


def test_dense_maps_timing_and_low_level_reference(case):
    scene, inputs = case
    qe = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    detector = DetectorMaps(
        quantum_efficiency=qe,
        bias_dn=np.arange(6).reshape(2, 3) + 100,
        electrons_per_dn=2,
        dark_electrons_per_s=3,
    )
    qe[:] = 0  # construction snapshots maps
    dense = simulate_images(
        scene,
        **inputs,
        detector=detector,
        rng=np.random.default_rng(2),
        realisations=3,
    )
    assert dense.photoelectrons.shape == (2, 2, 3)
    assert dense.dn is not None
    assert dense.dn.shape == (3, 2, 2, 3)
    np.testing.assert_allclose(dense.exposure_bounds_s[:, 1], [[0.002, 0.012], [0.052, 0.062]])
    assert dense.photoelectrons.flags.writeable
    assert dense.dn.flags.writeable
    assert not dense.settings
    charge = image_photoelectrons(scene, **inputs, detector=detector)
    np.testing.assert_allclose(
        charge.signal.total.reshape(dense.photoelectrons.shape), dense.photoelectrons
    )
    means = simulate_images(scene, **inputs, detector=detector)
    np.testing.assert_allclose(means.photoelectrons, dense.photoelectrons, rtol=5e-13)
    assert means.dn is None and means.adc_clipped is None and means.full_well_saturated is None

    exposures = inputs["exposures"]
    acquisition = image_acquisition(
        exposure_bounds_s=exposures.bounds_s,
        setting_ids=("arbitrary-a", "arbitrary-b"),
        pixel_bounds_m=inputs["pixel_bounds_m"],
        row_offsets_s=np.array([0.0, 0.002]),
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=0.02,
    )

    def response(read, wave, acq):
        return gaussian_pixel_fraction(acq.pixel_bounds_m[read], acq.image_xy_m[read], 3e-5)[
            :, None
        ] * gaussian_filter(wave, 2e-6, 2e-8)

    reference = expected_electrons(
        scene,
        acquisition,
        response,
        collecting_area_m2=1e-8,
        quantum_efficiency=np.tile(np.asarray(detector.quantum_efficiency).ravel(), 2)[:, None],
    )
    np.testing.assert_allclose(dense.photoelectrons.ravel(), reference.total, rtol=5e-13)


def test_bias_gain_full_well_and_adc_maps(case):
    scene, inputs = case
    dark = np.array([[0, 0, 0], [0, 0, 0.0]])
    bias = np.array([[12, 23, 999], [45, 56, -1.0]])
    detector = DetectorMaps(
        quantum_efficiency=0,
        dark_electrons_per_s=dark,
        electrons_per_dn=np.full((2, 3), 2.0),
        bias_dn=bias,
        adc_bits=8,
    )
    images = simulate_images(scene, **inputs, detector=detector, rng=np.random.default_rng(3))
    assert images.dn is not None and images.adc_clipped is not None
    np.testing.assert_array_equal(images.dn[0, 0], [[12, 23, 255], [45, 56, 0]])
    np.testing.assert_array_equal(
        images.adc_clipped[0, 0], [[False, False, True], [False, False, True]]
    )
    saturated = simulate_images(
        scene,
        **inputs,
        detector=DetectorMaps(full_well_electrons=1e-9),
        rng=np.random.default_rng(3),
    )
    assert saturated.full_well_saturated is not None
    assert saturated.full_well_saturated.all()


@pytest.mark.parametrize("kind", ["row", "column", "exposure"])
def test_common_noise_renews_across_exposures(case, kind):
    scene, inputs = case
    images = simulate_images(
        scene,
        **inputs,
        realisations=3000,
        detector=DetectorMaps(quantum_efficiency=0, bias_dn=1000),
        rng=np.random.default_rng(12),
        row_noise_electrons=20 if kind == "row" else 0,
        column_noise_electrons=20 if kind == "column" else 0,
        exposure_noise_electrons=20 if kind == "exposure" else 0,
    )
    assert images.dn is not None
    noise = images.dn.astype(float) - 1000
    if kind == "row":
        np.testing.assert_array_equal(noise[:, :, :, 0], noise[:, :, :, 2])
        a, b = noise[:, 0, 0, 0], noise[:, 0, 1, 0]
    elif kind == "column":
        np.testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])
        a, b = noise[:, 0, 0, 0], noise[:, 0, 0, 1]
    else:
        np.testing.assert_array_equal(noise[:, :, 0, 0], noise[:, :, 1, 2])
        a, b = noise[:, 0, 0, 0], noise[:, 1, 0, 0]
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.06
    assert np.var(a) == pytest.approx(400, rel=0.08)
    assert abs(np.corrcoef(noise[:, 0, 0, 0], noise[:, 1, 0, 0])[0, 1]) < 0.06


def test_support_callback_acquisition_and_scene_bounds(case):
    scene, inputs = case
    seen = []

    def provider(acq):
        seen.extend(acq.sample_ids)
        return scene

    image = simulate_images(provider, **inputs)
    assert len(seen) == len(set(seen)) == 12
    assert seen[:4] == ["s0-p0", "s0-p1", "s0-p2", "s0-p3"]
    assert image.photoelectrons.shape == (2, 2, 3)
    bad = replace(inputs["exposures"], bounds_s=np.array([[0.19, 0.21], [0.05, 0.06]]))
    inputs["exposures"] = bad
    with pytest.raises(ValueError, match="entire acquisition support"):
        simulate_images(scene, **inputs)


def test_constant_and_variable_pointing_and_exposure_snapshots(case):
    scene, inputs = case
    starts = np.array([0.0, 0.05])
    pointing = np.array([0.001, 0.0])
    sequence = exposure_sequence(
        start_s=starts,
        duration_s=0.01,
        direction_bounds_uv=inputs["exposures"].direction_bounds_uv,
        pointing_uv=pointing,
    )
    starts[:] = 9
    pointing[:] = 9
    np.testing.assert_array_equal(sequence.pointing(np.array([0.0])), [[0.001, 0.0]])
    assert sequence.bounds_s[0, 0] == 0
    callback = exposure_sequence(
        start_s=[0.0, 0.05],
        duration_s=0.01,
        direction_bounds_uv=sequence.direction_bounds_uv,
        pointing_uv=lambda t: np.broadcast_to([0.001, 0.0], (len(t), 2)),
    )
    inputs["exposures"] = sequence
    a = simulate_images(scene, **inputs)
    inputs["exposures"] = callback
    b = simulate_images(scene, **inputs)
    np.testing.assert_array_equal(a.photoelectrons, b.photoelectrons)


@pytest.mark.parametrize(
    "factory,params",
    [
        (filter_imager, dict(filter=partial(gaussian_filter, centre_m=2e-6, width_m=2e-8))),
        (lvf_imager, dict(reference_m=2e-6, gradient=0.0, sigma_m=2e-8, pixel_order=12)),
        (interferometric_imager, dict(opd_m=[0.0, 2e-5], port="+")),
        (interferometric_imager, dict(opd_m=[0.0, 2e-5], port="-")),
        (cavity_imager, dict(spacing_m=[1e-5, 1.1e-5], order_transmission=0.8)),
        (dispersive_imager, dict(reference_m=2e-6, dispersion=40.0)),
    ],
)
def test_optical_families_physical_sequences_and_charge(case, factory, params):
    scene, inputs = case
    optics = factory(
        focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=(3e-5, 4e-5), **params
    )
    inputs["optics"] = optics
    dense = simulate_images(scene, **inputs)
    charge = image_photoelectrons(scene, **inputs)
    np.testing.assert_allclose(dense.photoelectrons.ravel(), charge.signal.total, rtol=5e-13)
    assert dense.port == params.get("port", "direct")
    assert np.all(dense.photoelectrons >= 0)
    assert np.sum(dense.photoelectrons) > 0


def test_supplied_filter_gaussian_equivalence(case):
    scene, inputs = case
    common = dict(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=(3e-5, 4e-5))

    def response(exposure, read, wave, acquisition):
        fraction = gaussian_pixel_fraction(
            acquisition.pixel_bounds_m[read], acquisition.image_xy_m[read], (3e-5, 4e-5)
        )[:, None]
        return fraction * (0.7 * np.exp(-0.5 * ((wave - 2e-6) / 2e-8) ** 2))

    inputs["optics"] = ImagingOptics(0.02, 1e-8, response, {}, throughput=0.6)
    reference = simulate_images(scene, **inputs)
    inputs["optics"] = filter_imager(
        **common,
        filter=partial(gaussian_filter, centre_m=2e-6, width_m=2e-8, peak=0.7),
        throughput=0.6,
    )
    actual = simulate_images(scene, **inputs)
    np.testing.assert_allclose(actual.photoelectrons, reference.photoelectrons, rtol=5e-13)
    assert not actual.settings
    assert actual.port == "direct"


def test_supplied_tabulated_filter_and_support(case):
    scene, inputs = case
    law = partial(
        tabulated_transmission,
        nodes_m=np.array([1.95e-6, 2e-6, 2.05e-6]),
        fractions=np.array([0.1, 0.8, 0.4]),
    )
    inputs["optics"] = filter_imager(
        focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5, filter=law
    )
    dense = simulate_images(scene, **inputs)
    rebuilt = image_photoelectrons(scene, **inputs).signal.total.reshape(dense.photoelectrons.shape)
    np.testing.assert_allclose(rebuilt, dense.photoelectrons, rtol=5e-13)

    acquisition = image_acquisition(
        exposure_bounds_s=inputs["exposures"].bounds_s,
        setting_ids=("a", "b"),
        pixel_bounds_m=inputs["pixel_bounds_m"],
        row_offsets_s=np.zeros(2),
        direction_bounds_uv=inputs["exposures"].direction_bounds_uv,
        pointing=inputs["exposures"].pointing,
        focal_length_m=0.02,
    )
    wave = np.array([2.025e-6, 1.975e-6])
    original = wave.copy()
    fraction = gaussian_pixel_fraction(
        acquisition.pixel_bounds_m[0], acquisition.image_xy_m[0], 3e-5
    )[:, None]
    np.testing.assert_allclose(
        inputs["optics"].response(0, 0, wave, acquisition), fraction * [0.6, 0.45]
    )
    np.testing.assert_array_equal(wave, original)
    with pytest.raises(ValueError, match="support"):
        inputs["optics"].response(0, 0, np.array([2.1e-6]), acquisition)
    blocking = filter_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        filter=partial(law, outside="zero"),
    )
    np.testing.assert_array_equal(blocking.response(0, 0, np.array([2.1e-6]), acquisition), 0)


@pytest.mark.parametrize("value", [-0.1, 1.01, np.nan, np.inf])
def test_supplied_filter_invalid_transmission(case, value):
    scene, inputs = case
    inputs["optics"] = filter_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        filter=lambda wave: np.full_like(wave, value),
    )
    with pytest.raises(ValueError, match="filter transmission"):
        simulate_images(scene, **inputs)


@pytest.mark.parametrize("shape", [(), (1,), (1, 33), (33, 1)])
def test_supplied_filter_rejects_implicit_broadcast(case, shape):
    scene, inputs = case
    inputs["optics"] = filter_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        filter=lambda wave: np.full(shape, 0.5),
    )
    with pytest.raises(ValueError, match="same shape"):
        simulate_images(scene, **inputs)


def test_lvf_zero_gradient_and_two_ports_conservation(case):
    scene, inputs = case
    common = dict(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5)
    gaussian = filter_imager(**common, filter=partial(gaussian_filter, centre_m=2e-6, width_m=2e-8))
    lvf = lvf_imager(**common, reference_m=2e-6, gradient=0, sigma_m=2e-8, axis="y", pixel_order=12)
    inputs["optics"] = gaussian
    a = simulate_images(scene, **inputs)
    inputs["optics"] = lvf
    b = simulate_images(scene, **inputs)
    np.testing.assert_allclose(a.photoelectrons, b.photoelectrons, rtol=1e-12)
    plus = interferometric_imager(**common, opd_m=[0, 1e-5], port="+")
    minus = interferometric_imager(**common, opd_m=[0, 1e-5], port="-")
    inputs["optics"] = plus
    p = simulate_images(scene, **inputs)
    inputs["optics"] = minus
    m = simulate_images(scene, **inputs)
    np.testing.assert_allclose(
        p.photoelectrons[0], (p.photoelectrons + m.photoelectrons)[1], rtol=5e-13
    )
    assert not m.photoelectrons[0].any()
    np.testing.assert_array_equal(p.settings["opd_m"], [0, 1e-5])


def test_local_lvf_refinement_cavity_callback_and_dispersion_reference(case):
    scene, inputs = case
    common = dict(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5)
    opts = dict(
        reference_m=2e-6,
        gradient=5e-4,
        sigma_m=1e-8,
        curvature_per_m=0.01,
        reference_position_m=1e-5,
    )
    inputs["optics"] = lvf_imager(**common, **opts, pixel_order=8)
    coarse = simulate_images(scene, **inputs)
    inputs["optics"] = lvf_imager(**common, **opts, pixel_order=12)
    fine = simulate_images(scene, **inputs)
    np.testing.assert_allclose(coarse.photoelectrons, fine.photoelectrons, rtol=1e-8)
    acq = image_acquisition(
        exposure_bounds_s=np.array([[0.0, 0.01]]),
        setting_ids=("unused",),
        pixel_bounds_m=inputs["pixel_bounds_m"],
        row_offsets_s=np.zeros(2),
        direction_bounds_uv=inputs["exposures"].direction_bounds_uv,
        pointing=inputs["exposures"].pointing,
        focal_length_m=0.02,
    )
    wave = 0.01 / scene.grid.points
    cavity = cavity_imager(**common, spacing_m=1e-5, order_transmission=lambda wave: wave * 0 + 0.6)
    fraction = gaussian_pixel_fraction(acq.pixel_bounds_m[0], acq.image_xy_m[0], 3e-5)[:, None]
    np.testing.assert_allclose(
        cavity.response(0, 0, wave, acq),
        fraction * cavity_transmission(wave, 1e-5, order_transmission=0.6),
    )
    disperser = dispersive_imager(**common, reference_m=2e-6, dispersion=3)
    np.testing.assert_array_equal(
        disperser.response(0, 0, wave, acq),
        dispersive_response(
            wave,
            acq.pixel_bounds_m[0],
            acq.image_xy_m[0],
            sigma_m=3e-5,
            reference_m=2e-6,
            dispersion=3,
        ),
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(quantum_efficiency=1.1),
        dict(electrons_per_dn=0),
        dict(read_noise_electrons=-1),
        dict(nonlinearity=1),
        dict(bias_dn=np.nan),
        dict(adc_bits=0),
        dict(quantum_efficiency=np.ones(2)),
        dict(quantum_efficiency=np.empty((0, 2))),
    ],
)
def test_invalid_detector_maps(options):
    with pytest.raises(ValueError):
        DetectorMaps(**options)


@pytest.mark.parametrize(
    "change",
    [
        dict(time_order=0),
        dict(angular_order=0),
        dict(realisations=0),
        dict(row_noise_electrons=-1),
        dict(column_noise_electrons=np.nan),
        dict(detector=DetectorMaps(quantum_efficiency=np.ones((3, 2)))),
        dict(pixel_bounds_m=np.zeros((2, 3, 4))),
    ],
)
def test_invalid_simulation_inputs_eager(case, change):
    scene, inputs = case
    with pytest.raises(ValueError):
        simulate_images(scene, **{**inputs, **change})


def test_optical_settings_mismatch_and_snapshot(case):
    scene, inputs = case
    centre = np.array([2e-6, 2.01e-6])
    optics = lvf_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        reference_m=centre,
        gradient=0,
        sigma_m=2e-8,
    )
    centre[:] = 4e-6
    np.testing.assert_array_equal(optics.settings["reference_m"], [2e-6, 2.01e-6])
    assert np.asarray(optics.settings["reference_m"]).flags.writeable
    inputs["optics"] = optics
    images = simulate_images(scene, **inputs)
    assert not np.array_equal(images.photoelectrons[0], images.photoelectrons[1])
    with pytest.raises(ValueError, match="rebuild the optical model"):
        # Deliberately exercise the runtime guard as well as the static prohibition.
        replace(optics, settings={"reference_m": [2e-6]})  # pyrefly: ignore[unexpected-keyword]
    inputs["optics"] = lvf_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        reference_m=[2e-6],
        gradient=0,
        sigma_m=2e-8,
    )
    with pytest.raises(ValueError, match="match exposures"):
        simulate_images(scene, **inputs)


@pytest.mark.parametrize(
    "start,duration,pointing",
    [
        ([], 0.01, (0, 0)),
        ([[0]], 0.01, (0, 0)),
        ([0], 0, (0, 0)),
        ([0], 0.01, [0]),
        ([np.nan], 0.01, (0, 0)),
        ([0], 0.01, (np.inf, 0)),
    ],
)
def test_invalid_exposure_inputs(start, duration, pointing):
    with pytest.raises(ValueError):
        exposure_sequence(
            start_s=start,
            duration_s=duration,
            pointing_uv=pointing,
            direction_bounds_uv=(-0.01, 0.01, -0.01, 0.01),
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(row_delay_s=np.nan),
        dict(bounds_s=np.array([[1.0, 0.0]])),
        dict(bounds_s=np.empty((0, 2))),
        dict(direction_bounds_uv=(-1.0, 1.0, -1.0, 1.0)),
        dict(direction_bounds_uv=(0.01, -0.01, -0.01, 0.01)),
        dict(direction_bounds_uv=(np.nan, 0.01, -0.01, 0.01)),
    ],
)
def test_invalid_exposure_record(case, change):
    _, inputs = case
    with pytest.raises(ValueError):
        replace(inputs["exposures"], **change)


@pytest.mark.parametrize(
    "factory,params",
    [
        (lvf_imager, dict(reference_m=[], gradient=0, sigma_m=1e-8)),
        (lvf_imager, dict(reference_m=[[2e-6]], gradient=0, sigma_m=1e-8)),
        (lvf_imager, dict(reference_m=2e-6, gradient=0, sigma_m=-1)),
        (lvf_imager, dict(reference_m=2e-6, gradient=1, sigma_m=1e-8, axis="z")),
        (lvf_imager, dict(reference_m=2e-6, gradient=np.nan, sigma_m=1e-8)),
        (lvf_imager, dict(reference_m=2e-6, gradient=1, sigma_m=1e-8, pixel_order=0)),
        (interferometric_imager, dict(opd_m=0, port="difference")),
        (cavity_imager, dict(spacing_m=1e-5, order_transmission=1, reflectivity=1)),
        (dispersive_imager, dict(reference_m=2e-6, dispersion=np.inf)),
    ],
)
def test_invalid_optical_models(factory, params):
    with pytest.raises(ValueError):
        factory(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5, **params)


def test_bad_psf_and_port():
    with pytest.raises(ValueError, match="PSF sigma"):
        filter_imager(
            focal_length_m=0.02,
            collecting_area_m2=1e-8,
            psf_sigma_m=[1, 2, 3],
            filter=partial(gaussian_filter, centre_m=2e-6, width_m=1e-8),
        )
    with pytest.raises(ValueError, match="port"):
        ImagingOptics(0.02, 1e-8, lambda e, r, w, a: w, {}, port="")


def test_extra_read_noise_validation_before_rng_advance(case):
    scene, inputs = case
    acq = image_acquisition(
        exposure_bounds_s=np.array([[0.0, 0.01]]),
        setting_ids=("unused",),
        pixel_bounds_m=inputs["pixel_bounds_m"],
        row_offsets_s=np.zeros(2),
        direction_bounds_uv=inputs["exposures"].direction_bounds_uv,
        pointing=inputs["exposures"].pointing,
        focal_length_m=0.02,
    )
    signal = expected_electrons(
        scene, acq, lambda r, w, a: np.zeros((a.times_s.shape[1], len(w))), collecting_area_m2=1e-8
    )
    zeros, ones = np.zeros(len(acq.sample_ids)), np.ones(len(acq.sample_ids))
    electronics = DetectorOperatingPoints(zeros, zeros, ones * 1e6, ones, zeros + 100, zeros)
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    for noise in (np.zeros(1), np.full((1, len(acq.sample_ids)), np.nan)):
        with pytest.raises(ValueError, match="extra read noise"):
            sample_readout(signal, electronics, rng, extra_read_noise_electrons=noise)
    assert state == rng.bit_generator.state
    result = sample_readout(
        signal, electronics, rng, extra_read_noise_electrons=np.full((1, len(acq.sample_ids)), 7.0)
    )
    assert result.dn[0, 0] == 107
