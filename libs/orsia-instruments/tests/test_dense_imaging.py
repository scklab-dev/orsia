"""Dense image/integration contracts and physical response regression checks."""

import numpy as np
import pytest
from orsia_instruments import (
    DetectorMaps,
    cavity_imager,
    combine_image_source_paths,
    detector_response,
    dispersive_imager,
    expected_electrons,
    exposure_sequence,
    filter_imager,
    image_acquisition,
    image_charge,
    image_photoelectrons,
    image_readout,
    interferometric_imager,
    lvf_imager,
    photoelectron_response,
    regular_pixel_bounds,
    response_electrons,
    simulate_images,
)
from orsia_instruments.scene import RadianceField
from orsia_radiative import SpectralDomain, SpectralGrid
from scipy.constants import c, h


def case():
    grid = SpectralGrid(
        "wavenumber_cm1",
        np.array([4000.0, 5000.0, 6000.0]),
        SpectralDomain("wavenumber_cm1", ((4000.0, 6000.0),)),
    )
    scene = RadianceField(
        grid,
        np.array([0.0, 1.0]),
        np.array([-0.01, 0.01]),
        np.array([-0.01, 0.01]),
        np.ones((2, 2, 2, 3)),
    )
    exposures = exposure_sequence(
        start_s=[0.0, 0.5],
        duration_s=0.1,
        direction_bounds_uv=(-0.01, 0.01, -0.01, 0.01),
        row_delay_s=0.05,
    )
    pixels = regular_pixel_bounds((2, 2), pitch_m=1e-4)
    optics = filter_imager(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        filter=lambda wave: np.ones_like(wave),
    )
    return scene, exposures, pixels, optics


def test_full_spectral_coefficients_and_qe_callback():
    scene, exposures, pixels, _ = case()
    acq = image_acquisition(
        exposure_bounds_s=exposures.bounds_s,
        setting_ids=("0", "1"),
        pixel_bounds_m=pixels,
        row_offsets_s=np.array([0.0, 0.05]),
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=0.02,
        time_order=2,
        angular_order=2,
    )
    observed = []

    def qe(indices, wave):
        observed.append((indices.shape, wave.shape))
        return np.full((len(indices), len(wave)), 0.5)

    def response(read, wave, acquisition):
        assert wave.shape == (3,)
        return np.full((acquisition.times_s.shape[1], len(wave)), 0.25)

    coeff = photoelectron_response(
        scene.grid,
        acq,
        response,
        collecting_area_m2=2.0,
        quantum_efficiency=qe,
    )
    weights = np.array([500.0, 1000.0, 500.0]) / (h * c * 100 * scene.grid.points)
    np.testing.assert_allclose(
        coeff,
        acq.weights_sr_s[:, :, None] * (2 * 0.5 * 0.25 * weights),
        rtol=1e-15,
    )
    assert observed == [((8,), (3,))]
    expected = expected_electrons(
        scene,
        acq,
        response,
        collecting_area_m2=2.0,
        quantum_efficiency=qe,
    )
    np.testing.assert_allclose(expected.total, coeff.sum(axis=(1, 2)))
    assert coeff.flags.writeable
    with pytest.raises(ValueError, match="QE callback"):
        photoelectron_response(
            scene.grid,
            acq,
            response,
            collecting_area_m2=2.0,
            quantum_efficiency=lambda indices, wave: np.ones((1, len(wave))),
        )


def test_full_image_charge_and_readout():
    scene, exposures, pixels, optics = case()
    images = simulate_images(
        scene,
        pixel_bounds_m=pixels,
        optics=optics,
        exposures=exposures,
        detector=DetectorMaps(quantum_efficiency=0),
        rng=np.random.default_rng(4),
        realisations=2,
        row_noise_electrons=1,
    )
    assert images.photoelectrons.shape == (2, 2, 2)
    assert images.dn is not None and images.dn.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(images.photoelectrons, 0)
    np.testing.assert_allclose(images.exposure_bounds_s[:, 1], [[0.05, 0.15], [0.55, 0.65]])
    assert images.dn.flags.writeable
    seen = []
    charge = image_photoelectrons(
        lambda acq: (seen.append(len(acq.sample_ids)), scene)[1],
        pixel_bounds_m=pixels,
        optics=optics,
        exposures=exposures,
    )
    assert seen == [8]
    external = image_charge(
        {"external": np.ones((2, 2, 2))},
        exposure_bounds_s=exposures.bounds_s,
        row_delay_s=exposures.row_delay_s,
    )
    combined = combine_image_source_paths(charge, external)
    means = image_readout(combined)
    np.testing.assert_allclose(means.photoelectrons, charge.signal.total.reshape(2, 2, 2) + 1)
    assert set(means.contributions) == {"scene", "external"}
    assert means.dn is None


@pytest.mark.parametrize(
    "factory,settings",
    [
        (filter_imager, {"filter": lambda wave: np.ones_like(wave)}),
        (lvf_imager, {"reference_m": 2e-6, "gradient": 0.0, "sigma_m": 2e-8}),
        (interferometric_imager, {"opd_m": [0.0, 1e-5], "port": "+"}),
        (cavity_imager, {"spacing_m": [1e-5, 1.1e-5], "order_transmission": 0.7}),
        (dispersive_imager, {"reference_m": 2e-6, "dispersion": 1.0}),
    ],
)
def test_full_image_optical_families(factory, settings):
    scene, exposures, pixels, _ = case()
    optics = factory(
        focal_length_m=0.02,
        collecting_area_m2=1e-8,
        psf_sigma_m=3e-5,
        **settings,
    )
    image = simulate_images(scene, pixel_bounds_m=pixels, optics=optics, exposures=exposures)
    assert image.photoelectrons.shape == (2, 2, 2)
    assert np.all(np.isfinite(image.photoelectrons))
    assert np.all(image.photoelectrons >= 0)
    assert image.photoelectrons.sum() > 0


def test_correlated_read_noise_groups():
    scene, exposures, pixels, optics = case()
    images = simulate_images(
        scene,
        pixel_bounds_m=pixels,
        optics=optics,
        exposures=exposures,
        detector=DetectorMaps(quantum_efficiency=0, bias_dn=1000),
        rng=np.random.default_rng(7),
        realisations=2000,
        row_noise_electrons=20,
    )
    assert images.dn is not None
    np.testing.assert_array_equal(images.dn[:, :, :, 0], images.dn[:, :, :, 1])
    assert abs(np.corrcoef(images.dn[:, 0, 0, 0], images.dn[:, 0, 1, 0])[0, 1]) < 0.1


def test_dense_projected_response_and_direct_correlation():
    scene, exposures, pixels, _ = case()
    acquisition = image_acquisition(
        exposure_bounds_s=exposures.bounds_s[:1],
        setting_ids=("0",),
        pixel_bounds_m=pixels[:1, :1],
        row_offsets_s=np.array([0.0]),
        direction_bounds_uv=exposures.direction_bounds_uv,
        pointing=exposures.pointing,
        focal_length_m=0.02,
        time_order=1,
        angular_order=2,
    )
    xy = np.zeros((*acquisition.times_s.shape, 2))
    axis = np.array([-1.0, 0.0, 1.0])
    physical = detector_response(
        scene.grid,
        acquisition,
        lambda read, wave, current: np.ones((current.times_s.shape[1], len(wave))),
        ground_xy_m=xy,
        along_m=axis,
        cross_m=axis,
        support_bounds_m=np.array([[-1.0, 1.0, -1.0, 1.0]]),
        collecting_area_m2=1,
    )
    weights = photoelectron_response(
        scene.grid,
        acquisition,
        lambda read, wave, current: np.ones((current.times_s.shape[1], len(wave))),
        collecting_area_m2=1,
    )
    np.testing.assert_allclose(physical.coefficients[0, 1, 1], weights.sum(axis=1)[0])
    scene_values = np.ones((3, 3, 3))
    actual = response_electrons(physical, scene_values)
    np.testing.assert_allclose(actual.total, weights.sum(axis=(1, 2)))
    from orsia_instruments import ReadSchedule

    rows = np.array([2, 3], dtype=np.int64)
    cols = np.array([2, 3], dtype=np.int64)
    translated = ReadSchedule(
        tuple(f"p{i}" for i in range(4)),
        ("p",) * 4,
        ("0",) * 4,
        ("direct",) * 4,
        np.tile(acquisition.exposure_bounds_s, (4, 1)),
    )
    image = np.ones((6, 6, 3))
    kwargs = dict(
        scene_along_m=np.arange(6.0) - 2,
        scene_cross_m=np.arange(6.0) - 2,
        target_indices=(rows, cols),
        reads=translated,
    )
    direct = response_electrons(physical, image, method="direct", **kwargs)
    correlated = response_electrons(physical, image, method="correlation", **kwargs)
    np.testing.assert_allclose(direct.total, correlated.total)
    np.testing.assert_allclose(direct.total, np.repeat(actual.total, 4))
