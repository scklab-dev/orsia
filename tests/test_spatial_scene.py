"""Connected physical-scene acquisition and fixed-response Jacobian reference cases."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from orsia_dispersion import puff_concentration, puff_layer_columns, puff_path_columns
from orsia_geometry import affine_surface_points, segment_layer_paths
from orsia_instruments import FocalPlaneImager, ImagingOptics, simulate_frames
from orsia_instruments.optics.camera import camera_read_bands
from orsia_instruments.photodetection.frames import frame_response_coefficients
from orsia_radiative import SpectralDomain, clip_atmosphere_height, wavenumber_grid
from scipy.constants import Avogadro
from scipy.integrate import quad

from notebooks.support.spatial_scene import (
    SyntheticRadianceModel,
    chart_rotation,
    detector_prediction,
    make_acquisition,
    plume_excluded_mass_bound,
    synthetic_cross_section,
    synthetic_inputs,
)


def spectral_grid(count=9, interval=(4000.0, 4100.0)):
    return wavenumber_grid(
        np.linspace(4000, 4100, count), domain=SpectralDomain("wavenumber_cm1", (interval,))
    )


@pytest.mark.parametrize("mode", ["independent_column", "heterogeneous_direct_solar"])
def test_direct_simulation_is_independent_of_tangent_computation(monkeypatch, mode):
    import orsia_scene.radiance as composition

    from notebooks.support import spatial_scene

    inputs = synthetic_inputs()
    camera, grid, coverage = make_acquisition(inputs, angular_order=2)
    model = SyntheticRadianceModel(
        mode=mode, inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    radiance = model.radiance
    evaluate = model.radiance_tangent
    spectral = spectral_grid()
    args = (
        grid.xy_m.reshape(-1, 2),
        grid.view_local.reshape(-1, 3),
        grid.rays.acquisition.times_s.ravel(),
        spectral,
    )
    old_value, _ = evaluate(*args)
    np.testing.assert_array_equal(radiance(*args), old_value)
    original_columns = spatial_scene.puff_molecular_columns

    def amounts_only(*args, **kwargs):
        assert kwargs.get("scale_tangent") is None
        return original_columns(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("direct simulation must not compute tangents")

    monkeypatch.setattr(spatial_scene, "puff_molecular_columns", amounts_only)
    monkeypatch.setattr(spatial_scene, "surface_radiance_tangent", forbidden)
    monkeypatch.setattr(composition, "direct_surface_radiance_tangent", forbidden)
    monkeypatch.setattr(composition, "integrate_path_optical_depth_tangent", forbidden)
    model = SyntheticRadianceModel(
        mode=mode, inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    direct = model.radiance
    np.testing.assert_array_equal(direct(*args), old_value)
    result = simulate_frames(camera, grid, spectral, direct, coverage)
    assert (result.images.photoelectrons > 0).all()


@pytest.mark.parametrize(
    "index,kind",
    [
        (0, "shape"),
        (0, "nan"),
        (1, "shape"),
        (1, "nan"),
        (1, "norm"),
        (2, "shape"),
        (2, "nan"),
        (2, "support"),
        (3, "shape"),
        (3, "nan"),
        (3, "support"),
        (3, "reverse"),
        (3, "empty"),
    ],
)
@pytest.mark.parametrize("tangent", [False, True])
def test_flat_physical_query_validation(index, kind, tangent):
    inputs = synthetic_inputs()
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    callbacks = (model.radiance, model.radiance_tangent)
    args: list[Any] = [
        np.zeros((2, 2)),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.array([0.0, 0.1]),
        spectral_grid(),
    ]
    bad = args[index].points.copy() if index == 3 else args[index].copy()
    if kind == "shape":
        bad = bad[None]
    elif kind == "nan":
        bad.flat[0] = np.nan
    elif kind == "norm":
        bad *= 2
    elif kind == "support":
        bad += 1000
    elif kind == "reverse":
        bad = bad[::-1]
    else:
        bad = bad[:0]
    with pytest.raises(ValueError):
        args[index] = wavenumber_grid(bad) if index == 3 else bad
        callbacks[int(tangent)](*args)


@pytest.mark.parametrize(
    "mode,sloped",
    [
        ("independent_column", False),
        ("heterogeneous_direct_solar", False),
        ("heterogeneous_direct_solar", True),
    ],
)
def test_scene_extraction_preserves_pre_extraction_electrons_and_sensitivities(mode, sloped):
    # Saved before extraction, with angular_order=3, spectral_count=9 and default
    # time/batch orders. This is a behavior regression, not an independent truth.
    baseline = json.loads(
        (Path(__file__).parent / "fixtures/spatial-scene-baseline.json").read_text()
    )
    inputs = synthetic_inputs()
    if sloped:
        inputs["slopes"] = np.array([0.02, -0.01])
    value, tangent = detector_prediction(
        mode=mode, inputs=inputs, angular_order=3, spectral_count=9
    )
    key = mode + ("_sloped" if sloped else "")
    assert_allclose(value, baseline[key], rtol=2e-14, atol=0)
    assert_allclose(tangent, baseline[key + "_tangent"], rtol=2e-14, atol=0)


def test_gas_conversion_partial_layer_and_independent_path_quadrature():
    inputs = synthetic_inputs()
    column = clip_atmosphere_height(inputs["background"], 250.0)
    xy = np.array([100.0, 0.0])
    layer = puff_layer_columns(inputs["plume"], xy, column.height_boundaries_m, boundary="free")
    conversion = Avogadro / inputs["molar_mass_kg_mol"]
    for i, (lo, hi) in enumerate(
        zip(column.height_boundaries_m[:-1], column.height_boundaries_m[1:], strict=True)
    ):
        reference = quad(
            lambda z: float(
                puff_concentration(inputs["plume"], np.array([*xy, z]), boundary="free")
            ),
            lo,
            hi,
            epsabs=1e-13,
        )[0]
        assert_allclose(layer[i] * conversion, reference * conversion, rtol=2e-11)
    point = np.array([[*xy, 250.0]])
    amounts = []
    for direction in (np.array([[0.6, 0.0, 0.8]]), np.array([[-0.6, 0.0, 0.8]])):
        starts, ends, _, _ = segment_layer_paths(
            point, direction, inputs["background"].height_boundaries_m, minimum_upward_cosine=0.2
        )
        mass = puff_path_columns(inputs["plume"], starts, ends, boundary="free")
        for a, b, value in zip(starts, ends, mass, strict=True):
            length = np.linalg.norm(b - a)
            ref = (
                quad(
                    lambda s, a=a, b=b: float(
                        puff_concentration(inputs["plume"], a + s * (b - a), boundary="free")
                    ),
                    0,
                    1,
                    epsabs=1e-13,
                )[0]
                * length
            )
            assert_allclose(value * conversion, ref * conversion, rtol=2e-10)
        amounts.append(mass.sum())
    assert amounts[0] > 2 * amounts[1]
    # Finite model atmosphere intentionally excludes the Gaussian tails. Here the
    # prescribed field retains >99.99% of global mass above flat terrain and below top.
    from scipy.special import ndtr

    centre, sigma = 800.0, 120.0
    retained = ndtr((1500 - centre) / sigma) - ndtr((250 - centre) / sigma)
    assert 1 - retained < 3e-6
    assert plume_excluded_mass_bound(inputs) / inputs["plume"].masses_kg.sum() < 3e-6


@pytest.mark.parametrize("mode", ["independent_column", "heterogeneous_direct_solar"])
def test_complete_electron_jacobian_and_batch_invariance(mode):
    inputs = synthetic_inputs()
    prediction, tangent = detector_prediction(
        mode=mode, inputs=inputs, angular_order=3, spectral_count=9, batch_size=1
    )
    other, other_tangent = detector_prediction(
        mode=mode, inputs=inputs, angular_order=3, spectral_count=9, batch_size=8
    )
    assert_allclose(other, prediction, rtol=2e-15)
    assert_allclose(other_tangent, tangent, rtol=2e-15)
    assert prediction.shape == (2, 2, 2) and tangent.shape == (2, 2, 2, 2)
    for k, name in enumerate(("plume_amplitude", "reflectance_offset")):
        centre = 1.0 if k == 0 else 0.0
        for step in (1e-2, 1e-3, 1e-4):
            options = dict(mode=mode, inputs=inputs, angular_order=3, spectral_count=9)
            plus, _ = detector_prediction(**options, **{name: centre + step})
            minus, _ = detector_prediction(**options, **{name: centre - step})
            assert_allclose((plus - minus) / (2 * step), tangent[..., k], rtol=2e-7, atol=0.02)
    assert (tangent[..., 0] < 0).all() and (tangent[..., 1] > 0).all()


def test_zero_plume_homogeneous_column_path_and_uniform_acquisition():
    inputs = synthetic_inputs()
    inputs["reflectance_contrast"] = 0.0
    column, jcol = detector_prediction(
        mode="independent_column",
        inputs=inputs,
        plume_amplitude=0,
        angular_order=3,
        spectral_count=9,
    )
    path, jpath = detector_prediction(
        mode="heterogeneous_direct_solar",
        inputs=inputs,
        plume_amplitude=0,
        angular_order=3,
        spectral_count=9,
    )
    assert_allclose(column, path, rtol=1e-14)
    assert_allclose(jcol[..., 1], jpath[..., 1], rtol=1e-14)
    # Remove all puffs: amplitude one must recover amplitude-zero background.
    inputs["plume"] = replace(
        inputs["plume"],
        release_times_s=np.empty(0),
        masses_kg=np.empty(0),
        centres_m=np.empty((0, 3)),
        covariances_m2=np.empty((0, 3, 3)),
    )
    empty, _ = detector_prediction(
        mode="heterogeneous_direct_solar", inputs=inputs, angular_order=3, spectral_count=9
    )
    assert_allclose(empty, path, rtol=1e-14)
    camera, grid, coverage = make_acquisition(inputs, angular_order=3)
    spectral = spectral_grid()
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=0, reflectance_offset=0
    )
    radiance = model.radiance

    # Independent homogeneous plane-parallel expression at actual ray zeniths.
    def homogeneous(xy, directions, times, query_spectral):
        points = query_spectral.points
        local = directions
        amount = 2e25 * 2e-6 * (1500 - 250)
        depth = amount * synthetic_cross_section(points)
        return 0.25 * 0.02 * 0.8 / np.pi * np.exp(-depth * (1 / 0.8 + 1 / local[..., 2, None]))

    physical = simulate_frames(camera, grid, spectral, radiance, coverage)
    reference = simulate_frames(camera, grid, spectral, homogeneous, coverage)
    assert_allclose(physical.images.photoelectrons, reference.images.photoelectrons, rtol=1e-14)
    assert physical.images.dn is None


def test_query_permutation_frame_and_direction_dependence():
    inputs = synthetic_inputs()
    xy = np.array([[100.0, 0.0], [0.0, 10.0], [150.0, -20.0]])
    times = np.array([0.0, 0.1, 0.2])
    local = np.array([[-0.5, 0.0, np.sqrt(0.75)]] * 3)
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    evaluate = model.radiance_tangent
    points = spectral_grid()
    values, tangent = evaluate(xy, local, times, points)
    order = [2, 0, 1]
    shuffled, dshuffled = evaluate(xy[order], local[order], times[order], points)
    assert_allclose(shuffled, values[order], rtol=1e-14)
    assert_allclose(dshuffled, tangent[order], rtol=1e-14)
    repeated_xy = np.broadcast_to(xy[:1], xy.shape)
    frozen, _ = evaluate(repeated_xy, local, times, points)
    assert_allclose(frozen, np.broadcast_to(frozen[:1], frozen.shape), rtol=1e-14)
    alternative = local.copy()
    alternative[:, 0] *= -1
    changed, _ = evaluate(xy, alternative, times, points)
    assert np.max(np.abs(changed - values)) > 1e-7


def test_coverage_rejects_unproven_continuous_support_before_radiance():
    inputs = synthetic_inputs()
    camera, grid, coverage = make_acquisition(inputs, angular_order=2, time_order=1)
    spectral = spectral_grid()
    assert coverage(grid, spectral).all()
    assert not coverage(replace(grid), spectral).any()  # Unknown law/certificate.
    assert not coverage(grid, spectral_grid(interval=(3999.0, 4101.0))).any()
    inputs["time_interval"] = (0.0, float(grid.rays.acquisition.times_s.max()) + 0.001)
    assert not coverage(grid, spectral).all()  # All quadrature nodes still covered.
    inputs["time_interval"] = (0.0, 0.3)
    bound = float(np.max(np.abs(grid.xy_m))) + 10
    inputs["horizontal_interval"] = (-bound, bound)
    assert not coverage(grid, spectral).all()  # Surface nodes covered, displaced paths not.
    called = []

    def forbidden(*args):
        called.append(True)
        raise AssertionError("coverage must reject before radiance")

    with pytest.raises(ValueError, match="continuous"):
        simulate_frames(camera, grid, spectral, forbidden, coverage)
    assert not called


@pytest.mark.parametrize(
    "edges,height",
    [([-20000.0, -19500.0, -18500.0], -19750.0), ([0.0, 1000.0, 2500.0], 2000.0)],
)
def test_coverage_rejects_height_domain_outside_fixed_angular_bound(edges, height):
    inputs = synthetic_inputs()
    # Configure before binding: the certificate must reject unsupported laws even
    # when every sampled ray hits and the entire declared chart is wide enough.
    inputs["background"] = replace(inputs["background"], height_boundaries_m=np.array(edges))
    inputs["height_m"] = height
    inputs["horizontal_interval"] = (-100000.0, 100000.0)
    camera, grid, coverage = make_acquisition(inputs, angular_order=2, time_order=1)
    if height < 0:
        assert (5000 - height) / np.cos(0.3) ** 2 > 20000
    spectral = spectral_grid()
    assert not coverage(grid, spectral).any()

    def forbidden(*args):
        raise AssertionError("unsupported height domain must fail before radiance")

    with pytest.raises(ValueError, match="continuous"):
        simulate_frames(camera, grid, spectral, forbidden, coverage)


def test_projection_slope_and_unsupported_modes_queries():
    inputs = synthetic_inputs()
    inputs["slopes"] = np.array([0.02, -0.01])
    _, grid, coverage = make_acquisition(inputs, angular_order=2)
    assert coverage(grid, spectral_grid()).all()
    local_origins = np.einsum(
        "...ji,...j->...i",
        chart_rotation(grid.rays.acquisition.times_s, inputs["angular_rate_rad_s"]),
        grid.rays.origins_gcrs_m - np.array([6.4e6, 0.0, 0.0]),
    )
    local_directions = np.einsum(
        "...ji,...j->...i",
        chart_rotation(grid.rays.acquisition.times_s, inputs["angular_rate_rad_s"]),
        grid.rays.directions_gcrs,
    )
    points, _ = affine_surface_points(
        grid.xy_m, height_m=inputs["height_m"], slopes=inputs["slopes"]
    )
    displacement = points - local_origins
    assert_allclose(np.cross(displacement, local_directions), 0, atol=2e-12)
    assert_allclose(grid.view_local, -local_directions, atol=0, rtol=0)
    image, _ = detector_prediction(
        mode="heterogeneous_direct_solar", inputs=inputs, angular_order=2, spectral_count=9
    )
    assert (image > 0).all()
    for mode in ("independent_column", "heterogeneous_scattering", "thermal"):
        with pytest.raises(ValueError):
            SyntheticRadianceModel(
                mode=mode,  # pyrefly: ignore[bad-argument-type] - invalid mode rejection
                inputs=inputs,
                plume_amplitude=1,
                reflectance_offset=0,
            )
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    evaluate = model.radiance_tangent
    args: list[Any] = [
        grid.xy_m[:1].reshape(-1, 2),
        grid.view_local[:1].reshape(-1, 3),
        grid.rays.acquisition.times_s[:1].reshape(-1),
        spectral_grid(),
    ]
    for index, bad in (
        (0, args[0] + 1e6),
        (1, -args[1]),
        (2, args[2] + 1),
        (3, wavenumber_grid(args[3].points + 1000)),
    ):
        query = args.copy()
        query[index] = bad
        with pytest.raises(ValueError):
            evaluate(*query)


def test_signed_response_uses_joint_weights_and_refinements():
    inputs = synthetic_inputs()
    camera, grid, coverage = make_acquisition(inputs, angular_order=3)
    spectral = spectral_grid()
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    evaluate = model.radiance_tangent
    _, derivative = evaluate(
        grid.xy_m.reshape(-1, 2),
        grid.view_local.reshape(-1, 3),
        grid.rays.acquisition.times_s.reshape(-1),
        spectral,
    )
    derivative = derivative.reshape((*grid.rays.acquisition.times_s.shape, len(spectral.points), 2))
    jacobian = np.zeros((8, 2))
    for selected, spectral_slice, weights in frame_response_coefficients(
        camera, grid, spectral, batch_size=3
    ):
        jacobian[selected] += np.einsum(
            "nqs,nqsk->nk", weights, derivative[selected, :, spectral_slice]
        )
    _, reference = detector_prediction(
        mode="heterogeneous_direct_solar", inputs=inputs, angular_order=3, spectral_count=9
    )
    assert_allclose(jacobian, reference.reshape(8, 2), rtol=1e-14)
    # Response weights vary by band; derivative varies jointly over space and spectrum.
    assert np.ptp(derivative[0, ..., 0], axis=0).max() > 1e-8
    spatial = []
    for order in (13, 21, 29):
        value, _ = detector_prediction(
            mode="heterogeneous_direct_solar", inputs=inputs, angular_order=order, spectral_count=33
        )
        spatial.append(value)
    assert np.max(np.abs(spatial[2] - spatial[1]) / spatial[2]) < 2e-6
    assert np.linalg.norm(spatial[2] - spatial[1]) < np.linalg.norm(spatial[1] - spatial[0])
    fine, _ = detector_prediction(
        mode="heterogeneous_direct_solar",
        inputs=inputs,
        angular_order=5,
        spectral_count=65,
        time_order=4,
    )
    coarse, _ = detector_prediction(
        mode="heterogeneous_direct_solar",
        inputs=inputs,
        angular_order=5,
        spectral_count=33,
        time_order=2,
    )
    assert_allclose(coarse, fine, rtol=2e-5)


def test_nonseparable_physical_response_signed_electron_finite_difference():
    inputs = synthetic_inputs()
    camera, grid, coverage = make_acquisition(inputs, angular_order=3)
    original = camera.optics.response

    def response(exposure, read, wave, acquisition):
        # Explicit additional physical field-dependent spectral transmission in [0,1].
        factor = 0.8 + 0.1 * np.tanh(acquisition.image_xy_m[read, :, 0, None] / 0.002) * (
            (0.01 / wave - 4050) / 50
        )
        return original(exposure, read, wave, acquisition) * factor

    optics = ImagingOptics(
        camera.optics.focal_length_m,
        camera.optics.collecting_area_m2,
        response,
        camera.optics.settings,
        camera.optics.throughput,
        camera.optics.port,
    )
    camera = FocalPlaneImager(camera.detector, optics, camera.support_radius_m)
    strip_grid, strip_coverage = grid, coverage
    # Only channel labels change; rays and continuous-support proof stay identical.
    grid = replace(grid, read_bands=camera_read_bands(camera, grid.native_indices))

    def coverage(current, spectral):
        if current is grid:
            return strip_coverage(strip_grid, spectral)
        return np.zeros(len(current.read_bands), dtype=bool)

    spectral = spectral_grid()
    model = SyntheticRadianceModel(
        mode="heterogeneous_direct_solar", inputs=inputs, plume_amplitude=1, reflectance_offset=0
    )
    radiance = model.radiance
    evaluate = model.radiance_tangent
    derivative = evaluate(
        grid.xy_m.reshape(-1, 2),
        grid.view_local.reshape(-1, 3),
        grid.rays.acquisition.times_s.reshape(-1),
        spectral,
    )[1]
    derivative = derivative.reshape((*grid.rays.acquisition.times_s.shape, len(spectral.points), 2))
    jacobian = np.zeros((8, 2))
    for selected, spectral_slice, coefficients in frame_response_coefficients(
        camera, grid, spectral, batch_size=3
    ):
        jacobian[selected] += np.einsum(
            "nqs,nqsk->nk", coefficients, derivative[selected, :, spectral_slice]
        )
        normalized = coefficients[0] / coefficients[0].sum(axis=0)
        assert not np.allclose(normalized[:, 0], normalized[:, -1], rtol=1e-6, atol=1e-10)
    value = simulate_frames(camera, grid, spectral, radiance, coverage).images.photoelectrons
    assert value.shape == (2, 2, 2)
    for k in range(2):
        values = []
        for sign in (1, -1):
            model = SyntheticRadianceModel(
                mode="heterogeneous_direct_solar",
                inputs=inputs,
                plume_amplitude=1 + (sign * 1e-3 if k == 0 else 0),
                reflectance_offset=sign * 1e-3 if k == 1 else 0,
            )
            source = model.radiance
            values.append(
                simulate_frames(camera, grid, spectral, source, coverage).images.photoelectrons
            )
        assert_allclose(((values[0] - values[1]) / 2e-3).ravel(), jacobian[:, k], rtol=2e-7)


def test_path_splitting_and_independent_spectral_temporal_refinement():
    inputs = synthetic_inputs()
    start = np.array([[100.0, 0.0, 250.0]])
    direction = np.array([[0.6, 0.0, 0.8]])
    amounts = []
    for edges in (np.array([0.0, 500.0, 1500.0]), np.linspace(0, 1500, 31)):
        a, b, _, _ = segment_layer_paths(start, direction, edges, minimum_upward_cosine=0.2)
        amounts.append(puff_path_columns(inputs["plume"], a, b, boundary="free").sum())
    assert_allclose(amounts[0], amounts[1], rtol=1e-13)
    common = dict(mode="heterogeneous_direct_solar", inputs=inputs, angular_order=5)
    spectral_coarse, _ = detector_prediction(**common, spectral_count=33, time_order=4)
    reference, _ = detector_prediction(**common, spectral_count=65, time_order=4)
    temporal_coarse, _ = detector_prediction(**common, spectral_count=65, time_order=2)
    assert_allclose(spectral_coarse, reference, rtol=2e-5)
    assert_allclose(temporal_coarse, reference, rtol=2e-5)


def test_radiance_model_keeps_one_parameter_binding_and_mapping_snapshot():
    inputs = synthetic_inputs()
    model = SyntheticRadianceModel("heterogeneous_direct_solar", inputs, 1.0, 0.0)
    args = (
        np.array([[100.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.1]),
        wavenumber_grid(np.linspace(4000.0, 4100.0, 5)),
    )
    expected = model.radiance(*args)
    inputs["reflectance_contrast"] = 0.0
    assert_allclose(model.radiance(*args), expected, rtol=0, atol=0)
    model = SyntheticRadianceModel(
        mode=model.mode,
        inputs=model.inputs,
        plume_amplitude=1.0,
        reflectance_offset=0.0,
    )
    direct = model.radiance
    tangent = model.radiance_tangent
    assert_allclose(direct(*args), expected, rtol=0, atol=0)
    value, derivative = tangent(*args)
    assert_allclose(value, expected, rtol=2e-15)
    for k, field in enumerate(("plume_amplitude", "reflectance_offset")):
        step = 1e-4
        plus = replace(model, **{field: getattr(model, field) + step})
        minus = replace(model, **{field: getattr(model, field) - step})
        assert_allclose(
            (plus.radiance(*args) - minus.radiance(*args)) / (2 * step),
            derivative[..., k],
            rtol=2e-7,
        )
    with pytest.raises(ValueError, match="plume amplitude"):
        replace(model, plume_amplitude=-1)
