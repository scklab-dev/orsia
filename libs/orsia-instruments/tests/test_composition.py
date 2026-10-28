"""Composable optical paths, external charge and bounded image detector contracts."""

from dataclasses import FrozenInstanceError, replace
from functools import partial

import numpy as np
import pytest
from orsia_instruments import (
    DetectorElectronics,
    DetectorMaps,
    ExpectedElectrons,
    ImagingOptics,
    RadianceField,
    ReadSchedule,
    combine_image_source_paths,
    combine_source_paths,
    digitize,
    electron_covariance,
    electron_moments,
    expected_electrons,
    exposure_sequence,
    filter_imager,
    gaussian_filter,
    image_acquisition,
    image_charge,
    image_photoelectrons,
    image_readout,
    lvf_imager,
    native_sensitivities,
    read_schedule,
    rectangular_acquisition,
    regular_pixel_bounds,
    sample_readout,
    simulate_images,
)
from orsia_radiative import SpectralDomain, SpectralGrid


@pytest.fixture
def optical_case():
    points = np.linspace(4900.0, 5100.0, 17)
    scene = RadianceField(
        SpectralGrid(
            "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),))
        ),
        np.array([0.0, 1.0]),
        np.array([-0.002, 0.002]),
        np.array([-0.002, 0.002]),
        np.full((2, 2, 2, 17), 1e-3),
    )
    inputs = dict(
        pixel_bounds_m=regular_pixel_bounds((2, 3), pitch_m=3e-5),
        optics=filter_imager(
            focal_length_m=0.02,
            collecting_area_m2=1e-8,
            psf_sigma_m=3e-5,
            filter=partial(gaussian_filter, centre_m=2e-6, width_m=3e-8),
        ),
        exposures=exposure_sequence(
            start_s=[0.0, 0.1],
            duration_s=0.01,
            row_delay_s=0.001,
            direction_bounds_uv=(-0.002, 0.002, -0.002, 0.002),
        ),
        time_order=1,
        angular_order=2,
    )
    return scene, inputs


def external(values=None, **kwargs):
    if values is None:
        values = {"external": np.zeros((2, 2, 3))}
    return image_charge(
        values,
        exposure_bounds_s=np.array([[0.0, 0.01], [0.1, 0.11]]),
        row_delay_s=0.001,
        **kwargs,
    )


def test_detector_without_geometry_and_legacy_identity():
    bounds = np.array([[0.0, 1.0], [2.0, 4.0]])
    reads = ReadSchedule(("a", "b"), ("p", "q"), ("s", "s"), ("+", "+"), bounds)
    bounds[:] = 9
    assert read_schedule(reads) is reads
    assert not hasattr(reads, "weights_sr_s")
    assert reads.exposure_bounds_s.flags.writeable
    signal = ExpectedElectrons(reads, {"scene": np.array([10.0, 20.0])})
    combined = combine_source_paths(
        signal, ExpectedElectrons(reads, {"warm": np.array([2.0, 3.0])})
    )
    electronics = DetectorElectronics(dark_electrons_per_s=2, read_noise_electrons=3)
    mean, variance = electron_moments(combined, electronics)
    np.testing.assert_array_equal(mean, [14.0, 27.0])
    np.testing.assert_array_equal(variance, [23.0, 36.0])
    np.testing.assert_array_equal(electron_covariance(combined, electronics), np.diag(variance))
    result = digitize(reads, mean[None], np.zeros((1, 2)), electronics)
    assert result.acquisition is reads and result.reads is reads
    np.testing.assert_array_equal(result.dn, [[14, 27]])
    assert sample_readout(combined, electronics, np.random.default_rng(3)).reads is reads
    nominal, derivative = native_sensitivities(
        lambda p: ExpectedElectrons(reads, {"s": np.full(2, 10 + p["offset"])}),
        {"offset": 0.0},
        {"offset": 0.1},
    )
    assert nominal.reads is reads
    np.testing.assert_allclose(derivative, 1.0)
    acquisition = rectangular_acquisition(
        sample_ids=reads.sample_ids,
        pixel_ids=reads.pixel_ids,
        setting_ids=reads.setting_ids,
        port_ids=reads.port_ids,
        exposure_bounds_s=reads.exposure_bounds_s,
        pixel_bounds_m=np.tile([-0.1, 0.1, -0.1, 0.1], (2, 1)),
        direction_bounds_uv=(-0.01, 0.01, -0.01, 0.01),
        pointing=lambda t: np.zeros((len(t), 2)),
        focal_length_m=1,
    )
    old = ExpectedElectrons(acquisition, {"a": np.zeros(2)})
    assert old.acquisition is acquisition
    np.testing.assert_array_equal(old.reads.exposure_bounds_s, reads.exposure_bounds_s)
    readout = sample_readout(old, electronics, np.random.default_rng(1))
    assert readout.acquisition is acquisition
    assert readout.reads.sample_ids == reads.sample_ids
    assert combine_source_paths(old, combined).acquisition is acquisition
    with pytest.raises(ValueError, match="identities"):
        combine_source_paths(
            signal, replace(signal, acquisition=replace(reads, pixel_ids=("q", "p")))
        )
    with pytest.raises(ValueError, match="intervals"):
        combine_source_paths(
            signal,
            replace(
                signal, acquisition=replace(reads, exposure_bounds_s=reads.exposure_bounds_s + 1)
            ),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"sample_ids": ()},
        {"sample_ids": ("a", "a")},
        {"port_ids": ("", "")},
        {"exposure_bounds_s": np.zeros((2, 2))},
        {"exposure_bounds_s": np.ones((2, 3))},
        {"exposure_bounds_s": np.full((2, 2), np.nan)},
    ],
)
def test_invalid_read_schedule(change):
    args = dict(
        sample_ids=("a", "b"),
        pixel_ids=("p", "q"),
        setting_ids=("s", "s"),
        port_ids=("+", "+"),
        exposure_bounds_s=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    with pytest.raises(ValueError):
        ReadSchedule(**(args | change))


def test_explicit_assembly_equals_shortcut_and_keeps_sources(optical_case):
    scene, inputs = optical_case
    detector = DetectorMaps(
        quantum_efficiency=0.7,
        bias_dn=np.arange(6).reshape(2, 3) + 100,
        dark_electrons_per_s=2,
        read_noise_electrons=1,
    )
    noise = dict(
        realisations=3, row_noise_electrons=2, column_noise_electrons=3, exposure_noise_electrons=4
    )
    dense = simulate_images(
        scene, **inputs, detector=detector, rng=np.random.default_rng(5), **noise
    )
    explicit = image_readout(
        image_photoelectrons(scene, **inputs, detector=detector),
        detector=detector,
        rng=np.random.default_rng(5),
        **noise,
    )
    for name in ("photoelectrons", "dn", "full_well_saturated", "adc_clipped", "exposure_bounds_s"):
        np.testing.assert_array_equal(getattr(explicit, name), getattr(dense, name))
    np.testing.assert_array_equal(dense.contributions["scene"], dense.photoelectrons)
    assert dense.contributions["scene"].flags.writeable
    with pytest.raises(TypeError):
        dense.contributions["new"] = dense.photoelectrons  # pyrefly: ignore[unsupported-operation]


def test_spectral_qe_replaces_map_on_full_image(optical_case):
    scene, inputs = optical_case
    calls = []

    def law(indices, wave):
        assert len(wave) == len(scene.grid.points)
        assert len(indices[0]) == 12
        assert wave.flags.writeable and indices[0].flags.writeable
        calls.append(tuple(a.copy() for a in indices))
        return (0.2 + 0.03 * indices[1][:, None] + 0.04 * indices[2][:, None]) * (wave[None] / 2e-6)

    charge = image_photoelectrons(
        scene, **inputs, detector=DetectorMaps(quantum_efficiency=0), quantum_efficiency=law
    )
    dense = image_readout(charge)
    assert dense.photoelectrons.min() > 0
    assert len(calls) == 1
    exposure, row, column = calls[0]
    np.testing.assert_array_equal(exposure, np.repeat([0, 1], 6))
    values = (0.2 + 0.03 * row[:, None] + 0.04 * column[:, None]) * (
        (0.01 / scene.grid.points)[None] / 2e-6
    )

    def response(read, wave, acq):
        return inputs["optics"].response(int(exposure[read]), read, wave, acq)

    exposures = inputs["exposures"]
    acquisition = image_acquisition(
        exposure_bounds_s=exposures.bounds_s,
        setting_ids=("0", "1"),
        pixel_bounds_m=inputs["pixel_bounds_m"],
        row_offsets_s=np.arange(2) * exposures.row_delay_s,
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=inputs["optics"].focal_length_m,
        time_order=inputs["time_order"],
        angular_order=inputs["angular_order"],
    )
    reference = expected_electrons(
        scene,
        acquisition,
        response,
        collecting_area_m2=1e-8,
        quantum_efficiency=values,
    )
    np.testing.assert_allclose(charge.signal.total, reference.total, rtol=1e-14)


@pytest.mark.parametrize(
    "law",
    [
        lambda i, w: np.ones(len(w)),
        lambda i, w: np.full((len(i[0]), len(w)), 1.1),
        lambda i, w: np.full((len(i[0]), len(w)), np.nan),
    ],
)
def test_bad_spectral_qe(optical_case, law):
    scene, inputs = optical_case
    with pytest.raises(ValueError):
        image_photoelectrons(scene, **inputs, quantum_efficiency=law)


def test_qe_support_and_ignored_map_shape(optical_case):
    scene, inputs = optical_case
    detector = DetectorMaps(quantum_efficiency=np.zeros((9, 8)))

    def law(indices, wave):
        return np.full((len(indices[0]), len(wave)), 0.5)

    result = simulate_images(scene, **inputs, detector=detector, quantum_efficiency=law)
    assert result.photoelectrons.min() > 0
    np.testing.assert_array_equal(
        image_readout(external(), detector=detector).photoelectrons,
        0,
    )
    with pytest.raises(ValueError, match="map"):
        image_photoelectrons(scene, **inputs, detector=detector)

    def unsupported(indices, wave):
        raise ValueError("outside supplied QE support")

    with pytest.raises(ValueError, match="outside supplied QE support"):
        image_photoelectrons(scene, **inputs, quantum_efficiency=unsupported)
    with pytest.raises(ValueError, match="source name"):
        image_photoelectrons(scene, **inputs, source_name="")


def test_different_source_grids_quadratures_and_external_background(optical_case):
    scene, inputs = optical_case
    points = np.linspace(4800.0, 5200.0, 23)
    warm = replace(
        scene,
        grid=SpectralGrid(
            "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4800.0, 5200.0),))
        ),
        values=np.full((2, 2, 2, 23), 2e-4),
    )
    a = image_photoelectrons(scene, **inputs)
    b = image_photoelectrons(
        warm, **(inputs | {"time_order": 2, "angular_order": 3}), source_name="warm"
    )
    c = external({"other": np.full((2, 2, 3), 5.0)})
    combined = combine_image_source_paths(a, b, c)
    result = image_readout(combined, detector=DetectorMaps(quantum_efficiency=0))
    assert set(result.contributions) == {"scene", "warm", "other"}
    np.testing.assert_array_equal(result.contributions["other"], 5.0)
    np.testing.assert_allclose(
        result.photoelectrons, sum(result.contributions.values()), rtol=1e-15
    )
    with pytest.raises(ValueError, match="unique"):
        combine_image_source_paths(a, a)
    with pytest.raises(ValueError):
        combine_image_source_paths(a, replace(b, image_shape=(1, 2, 3)))
    with pytest.raises(ValueError, match="identities"):
        reordered = replace(b.signal.reads, sample_ids=tuple(reversed(b.signal.reads.sample_ids)))
        combine_image_source_paths(
            a, replace(b, signal=ExpectedElectrons(reordered, b.signal.contributions))
        )
    with pytest.raises(ValueError):
        combine_image_source_paths()


@pytest.mark.parametrize("kind", ["row", "column", "exposure"])
def test_external_noise_groups_on_full_image(kind):
    options = {f"{kind}_noise_electrons": 20.0}
    charge = image_charge(
        {"zero": np.zeros((2, 2, 3))},
        exposure_bounds_s=np.array([[0.0, 1.0], [1.0, 2.0]]),
    )
    result = image_readout(
        charge,
        detector=DetectorMaps(bias_dn=1000),
        rng=np.random.default_rng(6),
        realisations=2000,
        **options,
    )
    assert result.dn is not None
    noise = result.dn - 1000
    if kind == "row":
        np.testing.assert_array_equal(noise[:, :, :, 0], noise[:, :, :, 2])
    elif kind == "column":
        np.testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])
    else:
        np.testing.assert_array_equal(noise[:, :, 0, 0], noise[:, :, 1, 2])
    assert np.var(noise[:, 0, 0, 0]) == pytest.approx(400, rel=0.1)
    assert abs(np.corrcoef(noise[:, 0, 0, 0], noise[:, 1, 0, 0])[0, 1]) < 0.08


def test_external_snapshots_and_dense_readout_failures():
    values = np.full((2, 2, 3), 12.0)
    charge = external({"source": values})
    values[:] = 100
    np.testing.assert_array_equal(charge.signal.total, 12.0)
    changed = replace(
        charge.signal.reads, exposure_bounds_s=charge.signal.reads.exposure_bounds_s.copy()
    )
    changed.exposure_bounds_s[1] += 0.1
    with pytest.raises(ValueError, match="columns"):
        image_readout(
            replace(charge, signal=ExpectedElectrons(changed, charge.signal.contributions))
        )
    with pytest.raises(ValueError, match="map"):
        image_readout(charge, detector=DetectorMaps(bias_dn=np.zeros((3, 2))))
    with pytest.raises(ValueError):
        image_readout(charge, realisations=0)
    with pytest.raises(ValueError):
        image_readout(charge, row_noise_electrons=-1)
    with pytest.raises(ValueError, match="port"):
        replace(charge, port="+")
    with pytest.raises(ValueError, match="settings"):
        replace(charge, settings={"opd_m": [0]})
    measured = image_readout(charge)
    noisy = image_readout(charge, rng=np.random.default_rng(5))
    assert measured.dn is None and noisy.dn is not None


@pytest.mark.parametrize(
    "options",
    [
        {"row_delay_s": np.inf},
        {"port": ""},
        {"exposure_bounds_s": [[0, 1]]},
        {"exposure_bounds_s": [[0, 0], [1, 2]]},
        {"exposure_bounds_s": [[0, np.inf], [1, 2]]},
        {"contributions": {}},
        {"contributions": {"s": np.full((2, 2, 3), -1.0)}},
        {"contributions": {"s": np.full((2, 2, 3), np.nan)}},
        {"contributions": {"s": np.zeros((2, 3, 2)), "t": np.zeros((2, 2, 3))}},
    ],
)
def test_external_invalid_inputs(options):
    args = dict(
        contributions={"s": np.zeros((2, 2, 3))},
        exposure_bounds_s=np.array([[0, 1], [1, 2]]),
    )
    with pytest.raises(ValueError):
        image_charge(**(args | options))


def test_full_image_charge_shape_and_readout_ownership():
    charge = external()
    for shape in ((0, 2, 3), (1, 2, 3), (2, 2)):
        with pytest.raises(ValueError):
            replace(charge, image_shape=shape)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(ValueError, match="signal"):
        replace(
            charge,
            signal=ExpectedElectrons(charge.signal.reads, {"bad": np.ones(12)}),
            image_shape=(1, 2, 3),
        )
    result = image_readout(charge, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(result.photoelectrons, 0)
    assert result.dn is not None and result.dn.shape == (1, 2, 2, 3)
    result.photoelectrons[:] = 1
    np.testing.assert_array_equal(charge.signal.total, 0)


def test_optical_model_description_cannot_drift(optical_case):
    scene, inputs = optical_case
    optics = lvf_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        reference_m=[2e-6, 2.01e-6],
        gradient=0,
        sigma_m=3e-8,
    )
    for changes in (
        {"settings": {"reference_m": [4e-6, 4e-6]}},
        {"focal_length_m": 0.03},
        {"collecting_area_m2": 2e-8},
        {"throughput": 1},
        {"port": "direct"},
        {"response": lambda e, r, w, a: np.zeros((1, len(w)))},
    ):
        with pytest.raises(ValueError, match="rebuild"):
            replace(optics, **changes)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(FrozenInstanceError):
        optics.throughput = 0  # pyrefly: ignore[read-only]
    copied = replace(optics)
    assert copied.response is optics.response and copied.settings is optics.settings
    with pytest.raises(TypeError):
        ImagingOptics()
    custom = ImagingOptics(0.02, 1e-8, optics.response, optics.settings)
    np.testing.assert_array_equal(
        simulate_images(scene, **(inputs | {"optics": custom})).photoelectrons,
        simulate_images(scene, **(inputs | {"optics": optics})).photoelectrons,
    )
