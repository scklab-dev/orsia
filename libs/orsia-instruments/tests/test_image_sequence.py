"""Whole-image acquisition identities, scene support and scan refinement."""

import numpy as np
import pytest
from orsia_instruments import (
    DetectorMaps,
    ImagingOptics,
    RadianceField,
    expected_electrons,
    exposure_sequence,
    gaussian_filter,
    gaussian_pixel_fraction,
    image_acquisition,
    lvf_centre,
    rectangular_acquisition,
    simulate_images,
)
from orsia_radiative import SpectralDomain, SpectralGrid


@pytest.fixture
def inputs():
    points = np.linspace(4900.0, 5100.0, 65)
    grid = SpectralGrid(
        "wavenumber_cm1", points, SpectralDomain("wavenumber_cm1", ((4900.0, 5100.0),))
    )
    pixels = np.array(
        [
            [[-8e-5, 0, -4e-5, 0], [0, 8e-5, -4e-5, 0]],
            [[-8e-5, 0, 0, 4e-5], [0, 8e-5, 0, 4e-5]],
        ]
    )
    exposures = np.array([[0.0, 0.02], [0.05, 0.07]])
    offsets = np.array([0.0, 0.006])
    patch = (-0.003, 0.003, -0.003, 0.003)

    def pointing(t):
        return np.column_stack((0.002 * t + 3e-4 * np.sin(100 * t), np.zeros(len(t))))

    def field_for_acquisition(acq):
        t = np.array([acq.exposure_bounds_s.min(), acq.exposure_bounds_s.max()])
        u = v = np.array([-0.003, 0.003])
        spectrum = 1e-3 * (1 - 0.2 * np.exp(-(((points - 5000) / 15) ** 2)))
        values = (
            1
            + 2 * t[:, None, None, None]
            + 30 * u[None, :, None, None]
            - 10 * v[None, None, :, None]
        ) * spectrum
        return RadianceField(grid, t, u, v, values)

    def response(i, wave, acq):
        # Supplied LVF varies across pixels, with wavelength centre tied to x.
        x = acq.pixel_bounds_m[i, :2].mean()
        centre = lvf_centre(x, reference_m=2e-6, gradient=5e-4)
        return gaussian_pixel_fraction(acq.pixel_bounds_m[i], acq.image_xy_m[i], 3e-5)[
            :, None
        ] * gaussian_filter(wave, centre, 2e-8)

    return exposures, pixels, offsets, patch, pointing, field_for_acquisition, response


def scan(inputs, *, time_order=3, angular_order=5):
    exposures, pixels, offsets, patch, pointing, _, _ = inputs
    return image_acquisition(
        exposure_bounds_s=exposures,
        setting_ids=("first", "second"),
        pixel_bounds_m=pixels,
        row_offsets_s=offsets,
        direction_bounds_uv=patch,
        pointing=pointing,
        focal_length_m=0.02,
        time_order=time_order,
        angular_order=angular_order,
    )


def test_dense_image_matches_independent_acquisition_and_radiometry(inputs):
    exposures, pixels, offsets, patch, pointing, field, response = inputs
    acquisition = scan(inputs)
    assert len(acquisition.sample_ids) == 8
    dense = rectangular_acquisition(
        sample_ids=tuple(
            f"s{s}-p{r * 2 + c}" for s in range(2) for r in range(2) for c in range(2)
        ),
        pixel_ids=tuple(f"r{r}-c{c}" for _ in range(2) for r in range(2) for c in range(2)),
        setting_ids=tuple(k for k in ("first", "second") for _ in range(4)),
        port_ids=("direct",) * 8,
        exposure_bounds_s=np.array(
            [exposures[s] + offsets[r] for s in range(2) for r in range(2) for _ in range(2)]
        ),
        pixel_bounds_m=np.array(
            [pixels[r, c] for _ in range(2) for r in range(2) for c in range(2)]
        ),
        direction_bounds_uv=patch,
        pointing=pointing,
        focal_length_m=0.02,
    )
    assert acquisition.sample_ids == dense.sample_ids
    assert acquisition.pixel_ids == dense.pixel_ids
    assert acquisition.setting_ids == dense.setting_ids
    np.testing.assert_array_equal(acquisition.exposure_bounds_s, dense.exposure_bounds_s)
    np.testing.assert_allclose(acquisition.image_xy_m, dense.image_xy_m, rtol=0, atol=0)
    full = expected_electrons(field(dense), dense, response, collecting_area_m2=1e-8)
    images = simulate_images(
        field,
        pixel_bounds_m=pixels,
        optics=ImagingOptics(0.02, 1e-8, lambda s, r, w, a: response(r, w, a), {}),
        exposures=exposure_sequence(
            start_s=exposures[:, 0],
            duration_s=0.02,
            direction_bounds_uv=patch,
            pointing_uv=pointing,
            row_delay_s=offsets[1],
        ),
        detector=DetectorMaps(read_noise_electrons=0, full_well_electrons=1e9),
        rng=np.random.default_rng(4),
    )
    np.testing.assert_allclose(images.photoelectrons.ravel(), full.total, rtol=5e-13)
    assert images.dn is not None and images.dn.shape == (1, 2, 2, 2)
    assert images.full_well_saturated is not None and not images.full_well_saturated.any()


def test_motion_jitter_rolling_and_static_uniform_limit(inputs):
    acquisition = scan(inputs)
    assert acquisition.sample_ids[:3] == ("s0-p0", "s0-p1", "s0-p2")
    assert acquisition.exposure_bounds_s[2, 0] == pytest.approx(0.006)
    assert not np.array_equal(acquisition.image_xy_m[0, :5], acquisition.image_xy_m[0, -5:])
    exposures, pixels, _, patch, _, field, response = inputs
    acq = image_acquisition(
        exposure_bounds_s=exposures[:1],
        setting_ids=("first",),
        pixel_bounds_m=pixels,
        row_offsets_s=np.zeros(2),
        direction_bounds_uv=patch,
        pointing=lambda t: np.zeros((len(t), 2)),
        focal_length_m=0.02,
    )
    uniform = field(acq)
    values = np.broadcast_to(uniform.values[0, 0, 0], uniform.values.shape)
    constant = RadianceField(uniform.grid, uniform.times_s, uniform.u, uniform.v, values)
    a = expected_electrons(constant, acq, response, collecting_area_m2=1e-8)
    assert a.total[0] == pytest.approx(a.total[2], rel=2e-13)
    assert a.total[1] == pytest.approx(a.total[3], rel=2e-13)


def test_time_refinement_and_scene_coverage(inputs):
    _, _, _, _, _, field, response = inputs
    values = []
    for order in (2, 5, 9):
        acq = scan(inputs, time_order=order)
        values.append(
            expected_electrons(field(acq), acq, response, collecting_area_m2=1e-8).total[0]
        )
    assert abs(values[1] - values[2]) < abs(values[0] - values[2])
    acq = scan(inputs)
    resident = field(acq)
    short = RadianceField(
        resident.grid, acq.times_s[0, [0, -1]], resident.u, resident.v, resident.values
    )
    with pytest.raises(ValueError, match="entire acquisition support"):
        expected_electrons(short, acq, response, collecting_area_m2=1e-8)


@pytest.mark.parametrize(
    "change", ["exposures", "pixels", "offsets", "settings", "negative", "ordering"]
)
def test_bad_image_acquisition_inputs(inputs, change):
    exposures, pixels, offsets, patch, pointing, _, _ = inputs
    kw = dict(
        exposure_bounds_s=exposures.copy(),
        setting_ids=("first", "second"),
        pixel_bounds_m=pixels.copy(),
        row_offsets_s=offsets.copy(),
        direction_bounds_uv=patch,
        pointing=pointing,
        focal_length_m=0.02,
    )
    if change == "exposures":
        kw["exposure_bounds_s"] = np.zeros((1, 3))
    elif change == "pixels":
        kw["pixel_bounds_m"][0, 0, 0] = np.nan
    elif change == "offsets":
        kw["row_offsets_s"] = np.ones(3)
    elif change == "settings":
        kw["setting_ids"] = ("first", "")
    elif change == "negative":
        kw["row_offsets_s"][0] = np.inf
    else:
        kw["exposure_bounds_s"][0] = [0.02, 0.0]
    with pytest.raises(ValueError):
        image_acquisition(**kw)


def test_snapshot_and_callback_support(inputs):
    exposures, pixels, offsets, patch, pointing, field, response = inputs
    acquisition = scan(inputs)
    exposures[:] = 100
    pixels[:] = 100
    offsets[:] = 100
    assert acquisition.exposure_bounds_s[0, 0] == 0
    assert acquisition.pixel_bounds_m[0, 0] == -8e-5
    assert acquisition.sample_ids[:2] == ("s0-p0", "s0-p1")
    # Invalid callback result is checked during dense acquisition construction.
    with pytest.raises(ValueError):
        image_acquisition(
            exposure_bounds_s=np.array([[0.0, 0.01]]),
            setting_ids=("s",),
            pixel_bounds_m=np.array([[[-1e-5, 1e-5, -1e-5, 1e-5]]]),
            row_offsets_s=np.array([0.0]),
            direction_bounds_uv=patch,
            pointing=lambda t: np.full((len(t), 2), np.nan),
            focal_length_m=0.02,
        )
    # Already built acquisition remains usable after the caller mutates its inputs.
    assert (
        expected_electrons(
            field(acquisition), acquisition, response, collecting_area_m2=1e-8
        ).total.size
        == 8
    )


def test_scene_callback_must_cover_full_acquisition(inputs):
    _, _, _, _, _, field, response = inputs

    def incomplete(acquisition):
        resident = field(acquisition)
        return RadianceField(
            resident.grid,
            np.array([0.0, 0.01]),
            resident.u,
            resident.v,
            np.broadcast_to(resident.values[:1], (2, *resident.values.shape[1:])),
        )

    _, pixels, offsets, patch, pointing, _, _ = inputs
    with pytest.raises(ValueError, match="entire acquisition support"):
        simulate_images(
            incomplete,
            pixel_bounds_m=pixels,
            exposures=exposure_sequence(
                start_s=[0.0, 0.05],
                duration_s=0.02,
                direction_bounds_uv=patch,
                pointing_uv=pointing,
                row_delay_s=offsets[1],
            ),
            optics=ImagingOptics(0.02, 1e-8, lambda s, r, w, a: response(r, w, a), {}),
        )
