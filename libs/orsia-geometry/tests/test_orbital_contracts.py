"""Adapter shape, support, ownership and derivative contracts with an analytic backend double."""

import sys
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from orsia_geometry import (
    SGP4MeanElements,
    axis_rotations,
    compute_lines_of_sight,
    ellipsoid_intersections,
    frame_rotations,
    numerical_propagator,
    sgp4_propagator,
    transform_rays,
    transform_states,
    wgs84_surface_projection,
    wgs84_surface_projection_from_ray,
)
from orsia_geometry.constants import WGS84_SEMIAXES_M

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
ELEMENTS = np.array([7e6, 0.001, 0.2, 0.3, 0.4, 0.5])
X = np.array([1.0, 0.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


class Instant:
    def __init__(self, value):
        self.value = value

    @classmethod
    def from_datetime(cls, value):
        return cls((value - EPOCH).total_seconds())

    def __add__(self, other):
        return Instant(self.value + other.seconds)


@pytest.fixture
def backend(monkeypatch):
    calls: list[Any] = []
    rotation_calls = []

    def rotate(source, dest, time):
        rotation_calls.append((source, dest, time.value))
        sign = 1 if source == "GCRF" else -1
        matrix = axis_rotations(Z, np.array(sign * 0.01 * time.value))
        return SimpleNamespace(to_rotation_matrix=lambda: matrix)

    def initial(*elements):
        calls.append(elements)
        return SimpleNamespace(to_pv=lambda: (2 * X, X))

    def interpolate(instants):
        assert all(a.value <= b.value for a, b in zip(instants, instants[1:], strict=False))
        return np.array([[2 + t.value, 0.0, 0.0, 1.0, 0.0, 0.0] for t in instants]).reshape(-1, 6)

    result = SimpleNamespace(interp=interpolate)

    def propagate(**kwargs):
        calls.append(kwargs)
        return result

    def sgp4(native, instants, **kwargs):
        calls.append((native, kwargs))
        return (
            np.tile(2 * X, (len(instants), 1)),
            np.tile(X, (len(instants), 1)),
            np.zeros(len(instants), dtype=np.int32),
        )

    fake = SimpleNamespace(
        time=Instant,
        duration=lambda **kwargs: SimpleNamespace(**kwargs),
        frame=SimpleNamespace(GCRF="GCRF", TEME="TEME", ITRF="ITRF"),
        frametransform=SimpleNamespace(eop_status=lambda t: "observed", rotation=rotate),
        propsettings=lambda **kwargs: SimpleNamespace(**kwargs),
        integrator=SimpleNamespace(rkv98="RKV98"),
        gravmodel=SimpleNamespace(egm96="EGM96"),
        tidemodel=SimpleNamespace(none="none"),
        kepler=initial,
        propagate=propagate,
        sgp4=sgp4,
        sgp4_gravconst=SimpleNamespace(wgs72="WGS72"),
        sgp4_opsmode=SimpleNamespace(afspc="AFSPC"),
        TLE=SimpleNamespace(from_omm=lambda mapping: dict(mapping)),
        result=result,
        calls=calls,
        rotation_calls=rotation_calls,
    )
    monkeypatch.setitem(sys.modules, "satkit", fake)
    return fake


def test_time_conversion_and_frame_arrays(backend):
    times = np.array([1.0, 0.0, 1.0])
    actual = frame_rotations(times, epoch=EPOCH, validity_s=(0, 1))
    assert actual.flags.owndata and actual.flags.writeable
    np.testing.assert_allclose(actual, axis_rotations(Z, times * 0.01))
    same = frame_rotations(times, epoch=EPOCH, validity_s=(0, 1), to_frame="GCRS")
    assert same.flags.owndata and same.flags.writeable
    np.testing.assert_array_equal(same, np.tile(np.eye(3), (3, 1, 1)))
    empty = frame_rotations(np.array([]), epoch=EPOCH, validity_s=(0, 1))
    assert empty.shape == (0, 3, 3) and empty.flags.owndata and empty.flags.writeable


@pytest.mark.parametrize("output_frame", ["ITRF", "TEME"])
@pytest.mark.parametrize("times", [[2.0, 0.0, 1.0, 1.0], [], [0.0]])
@pytest.mark.parametrize("ray_count", [0, 1, 3])
def test_lines_of_sight_output_frames_shapes_times_and_ownership(
    backend, output_frame, times, ray_count
):
    times = np.array(times)
    positions = np.column_stack((2 + times, 3 - times, times * 0.5))
    velocities = np.tile(X, (len(times), 1))
    platform = axis_rotations(Z, 0.3 + times * 0.2)
    relative = axis_rotations(X, 0.1 + times * 0.1)
    sensor = np.eye(3)[:ray_count].copy()
    offset = np.array([0.1, 0.2, -0.1])
    inputs = [times, positions, velocities, platform, relative, sensor, offset]
    snapshots = [array.copy() for array in inputs]
    received = []

    def state(t):
        received.append(t.copy())
        return positions, velocities

    def platform_law(t):
        received.append(t.copy())
        return platform

    def instrument_law(t):
        received.append(t.copy())
        return relative

    origins, directions = compute_lines_of_sight(
        times,
        epoch=EPOCH,
        validity_s=(0, 2),
        trajectory=state,
        platform_attitude=platform_law,
        instrument_attitude=instrument_law,
        directions_i=sensor,
        mount_offset_p_m=offset,
        output_frame=output_frame,
    )
    assert origins.shape == directions.shape == (len(times), ray_count, 3)
    # Check each vector independently, including a noncommuting platform/mount pair.
    for t in range(len(times)):
        rotation = axis_rotations(Z, np.array(0.01 * times[t]))
        for r in range(ray_count):
            np.testing.assert_allclose(
                origins[t, r], rotation @ (positions[t] + platform[t] @ offset)
            )
            np.testing.assert_allclose(
                directions[t, r], rotation @ platform[t] @ relative[t] @ sensor[r]
            )
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-15)
    assert backend.rotation_calls == [("GCRF", output_frame, t) for t in times]
    # Exact requested instants only: ray transformation needs no derivative padding.
    assert len(received) == 3
    for t in received:
        np.testing.assert_array_equal(t, times)
    for array in (origins, directions):
        assert array.dtype == np.float64 and array.flags.owndata and array.flags.writeable
        assert not any(np.shares_memory(array, source) for source in inputs)
        array[:] = 0
    for source, snapshot in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(source, snapshot)


@pytest.mark.parametrize("output_frame", ["ITRF", "TEME"])
def test_lines_of_sight_backend_errors_propagate(backend, monkeypatch, output_frame):
    options = dict(
        epoch=EPOCH,
        validity_s=(0, 0),
        trajectory=lambda t: (np.tile(2 * X, (len(t), 1)), np.tile(X, (len(t), 1))),
        platform_attitude=lambda t: np.tile(np.eye(3), (len(t), 1, 1)),
        instrument_attitude=lambda t: np.tile(np.eye(3), (len(t), 1, 1)),
        directions_i=Z[None],
        mount_offset_p_m=np.zeros(3),
        output_frame=output_frame,
    )
    backend.frametransform.eop_status = lambda t: "extrapolated"
    with pytest.raises(RuntimeError, match="coverage"):
        compute_lines_of_sight(np.array([0.0]), **options)
    assert backend.rotation_calls == []
    # A broken required installation still propagates its import error unchanged.
    monkeypatch.setitem(sys.modules, "satkit", None)
    with pytest.raises(ImportError):
        compute_lines_of_sight(np.array([0.0]), **options)


def test_rotation_derivative_and_velocity_refinement(backend):
    times = np.array([0.0, 0.5, 1.0])
    positions = np.tile(2 * X, (3, 1))
    velocity = np.tile(X, (3, 1))
    errors = []
    for step in (0.5, 0.25, 0.125):
        p, v = transform_states(
            times, positions, velocity, epoch=EPOCH, validity_s=(0, 1), derivative_step_s=step
        )
        expected = axis_rotations(Z, times * 0.01)
        r = np.einsum("tij,tj->ti", expected, positions)
        exact = np.einsum("tij,tj->ti", expected, velocity) + np.cross(0.01 * Z, r)
        np.testing.assert_allclose(p, r, atol=1e-15)
        errors.append(np.max(np.abs(v - exact)))
        assert p.flags.owndata and v.flags.owndata
    assert errors[2] < errors[1] / 3 and errors[1] < errors[0] / 3
    np.testing.assert_array_equal(positions, np.tile(2 * X, (3, 1)))


def test_time_frame_and_coverage_failures(backend):
    for epoch in (EPOCH.replace(tzinfo=None), EPOCH.astimezone(timezone(timedelta(hours=1)))):
        with pytest.raises(ValueError, match="UTC"):
            frame_rotations(np.array([0.0]), epoch=epoch, validity_s=(0, 1))
    for times, validity in (
        (np.array([2.0]), (0.0, 1.0)),
        (np.array([np.nan]), (0.0, 1.0)),
        (np.array([[0.0]]), (0.0, 1.0)),
        (np.array([0.0]), (1.0, 0.0)),
    ):
        with pytest.raises(ValueError, match="validity"):
            frame_rotations(times, epoch=EPOCH, validity_s=validity)
    with pytest.raises(ValueError, match="frames"):
        frame_rotations(
            np.array([0.0]), epoch=EPOCH, validity_s=(0, 1), from_frame=cast(Any, "unknown")
        )
    backend.frametransform.eop_status = lambda t: "extrapolated"
    with pytest.raises(RuntimeError, match="coverage"):
        frame_rotations(np.array([0.0]), epoch=EPOCH, validity_s=(0, 1))
    with pytest.raises(ValueError, match="matching"):
        transform_states(
            np.array([0.0]), np.ones((1, 2)), np.ones((1, 3)), epoch=EPOCH, validity_s=(0, 1)
        )


def test_numerical_constructor_retains_owned_dense_result(backend):
    elements = ELEMENTS.copy()
    trajectory = numerical_propagator(elements, epoch=EPOCH, validity_s=(0, 2))
    np.testing.assert_array_equal(elements, ELEMENTS)
    elements[:] = np.nan
    times = np.array([2.0, 0.0, 1.0, 1.0, 2.0])
    times.flags.writeable = False
    positions, velocity = trajectory(times)
    np.testing.assert_array_equal(positions[:, 0], [4.0, 2.0, 3.0, 3.0, 4.0])
    np.testing.assert_array_equal(velocity, np.tile(X, (5, 1)))
    for value in (positions, velocity):
        assert value.dtype == np.float64 and value.flags.owndata and value.flags.writeable
        assert not np.shares_memory(value, times)
    assert not np.shares_memory(positions, velocity)
    np.testing.assert_array_equal(times, [2, 0, 1, 1, 2])
    positions[:] = 0
    assert trajectory(np.array([0.0]))[0][0, 0] == 2
    assert trajectory(np.array([]))[0].shape == (0, 3)
    with pytest.raises(ValueError, match="validity"):
        trajectory(np.array([-1.0]))
    backend.result.interp = lambda instants: np.full((len(instants), 6), np.nan)
    with pytest.raises(RuntimeError, match="dense"):
        trajectory(np.array([0.0]))


@pytest.mark.parametrize(
    "elements",
    [
        np.ones(5),
        np.array([-1.0, 0, 0, 0, 0, 0]),
        np.array([1.0, 1, 0, 0, 0, 0]),
        np.array([1.0, 0, -1, 0, 0, 0]),
    ],
)
def test_invalid_classical_elements(elements, backend):
    with pytest.raises(ValueError, match="elements"):
        numerical_propagator(elements, epoch=EPOCH, validity_s=(0, 1))


def test_numerical_support_validation(backend):
    for validity in ((-1.0, 1.0), (0.0, 0.0)):
        with pytest.raises(ValueError, match="forward"):
            numerical_propagator(ELEMENTS, epoch=EPOCH, validity_s=validity)
    assert backend.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"gravity_degree": 0},
        {"gravity_degree": 70},
        {"gravity_order": 2},
        {"gravity_degree": 70, "gravity_order": 70},
        {"sun_gravity": True},
        {"moon_gravity": True},
        {"absolute_tolerance": 1e-7},
        {"relative_tolerance": 1e-13},
    ],
)
def test_scalar_controls_keep_native_model_explicit_and_settings_fresh(backend, changes):
    numerical_propagator(ELEMENTS, epoch=EPOCH, validity_s=(1, 2), **changes)
    call = backend.calls[-1]
    settings = call["propsettings"]
    assert vars(settings) == {
        "gravity_degree": changes.get("gravity_degree", 2),
        "gravity_order": changes.get("gravity_order", 0),
        "use_sun_gravity": changes.get("sun_gravity", False),
        "use_moon_gravity": changes.get("moon_gravity", False),
        "abs_error": changes.get("absolute_tolerance", 1e-4),
        "rel_error": changes.get("relative_tolerance", 1e-11),
        "gravity_model": "EGM96",
        "integrator": "RKV98",
        "use_spaceweather": False,
        "tide_model": "none",
        "use_relativistic_correction": False,
        "require_eop_coverage": True,
        "enable_interp": True,
    }
    assert call["begin"].value == 0 and call["end"].value == 2
    assert call["satproperties"] is None and call["output_phi"] is False
    numerical_propagator(ELEMENTS, epoch=EPOCH, validity_s=(0, 2))
    assert backend.calls[-1]["propsettings"] is not settings


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"gravity_degree": -1}, "gravity"),
        ({"gravity_degree": 71}, "gravity"),
        ({"gravity_order": -1}, "gravity"),
        ({"gravity_order": 3}, "gravity"),
        ({"gravity_degree": 0, "gravity_order": 1}, "gravity"),
        *[
            ({name: value}, "tolerances")
            for name in ("absolute_tolerance", "relative_tolerance")
            for value in (0.0, -1.0, float("inf"), float("-inf"), float("nan"))
        ],
    ],
)
def test_numerical_scalar_domain_errors_precede_backend_use(backend, changes, message):
    with pytest.raises(ValueError, match=message):
        numerical_propagator(ELEMENTS, epoch=EPOCH, validity_s=(0, 1), **changes)
    assert backend.calls == []


def test_removed_native_settings_keyword_is_rejected(backend):
    with pytest.raises(TypeError, match="settings"):
        cast(Any, numerical_propagator)(ELEMENTS, epoch=EPOCH, validity_s=(0, 1), settings=None)
    assert backend.calls == []


def mean_elements():
    return SGP4MeanElements(
        norad_id=25544,
        epoch=EPOCH,
        eccentricity=0.001,
        inclination_rad=0.2,
        raan_rad=0.3,
        argument_of_perigee_rad=0.4,
        mean_anomaly_rad=0.5,
        mean_motion_rad_s=0.001,
        bstar_per_earth_radius=1e-5,
        object_name="original",
    )


def test_sgp4_scalar_contract_and_errors(backend):
    elements = mean_elements()
    trajectory = sgp4_propagator(elements, epoch=EPOCH, validity_s=(0, 1))
    with pytest.raises(FrozenInstanceError):
        cast(Any, elements).object_name = "changed"
    p, v = trajectory(np.array([0.0, 1.0]))
    assert p.shape == v.shape == (2, 3)
    native, options = backend.calls[-1]
    assert native["NORAD_CAT_ID"] == elements.norad_id
    assert native["INCLINATION"] == pytest.approx(np.rad2deg(elements.inclination_rad))
    assert options["gravconst"] == "WGS72" and options["opsmode"] == "AFSPC"
    backend.sgp4 = lambda *args, **kwargs: (np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1))
    with pytest.raises(ValueError, match="SGP4"):
        trajectory(np.array([0.0]))
    with pytest.raises(ValueError, match="derivative"):
        sgp4_propagator(elements, epoch=EPOCH, validity_s=(0, 1), derivative_step_s=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"norad_id": 0},
        {"norad_id": 2147483648},
        {"epoch": EPOCH.replace(tzinfo=None)},
        {"epoch": EPOCH.astimezone(timezone(timedelta(hours=1)))},
        {"eccentricity": 1},
        {"inclination_rad": -1},
        {"raan_rad": 2 * np.pi},
        {"argument_of_perigee_rad": -1},
        {"mean_anomaly_rad": float("nan")},
        {"mean_motion_rad_s": 0},
        {"bstar_per_earth_radius": float("inf")},
        {"mean_motion_dot_rad_s2": float("nan")},
        {"mean_motion_ddot_rad_s3": float("inf")},
    ],
)
def test_invalid_scalar_elements(changes):
    with pytest.raises(ValueError):
        replace(mean_elements(), **changes)


def test_broken_satkit_installation_import_error_propagates(monkeypatch):
    monkeypatch.setitem(sys.modules, "satkit", None)
    with pytest.raises(ImportError):
        frame_rotations(np.array([0.0]), epoch=EPOCH, validity_s=(0, 1))


@pytest.mark.parametrize("leading", [(), (0,), (2, 0, 4), (2, 3, 4)])
@pytest.mark.parametrize("frames", [("GCRS", "ITRF"), ("TEME", "GCRS"), ("ITRF", "ITRF")])
def test_transform_rays_shapes_dates_rotations_and_ownership(backend, leading, frames):
    times = (
        (np.arange(np.prod(leading, dtype=int)).reshape(leading) % 3 / 2)[..., ::-1].copy()
        if leading
        else np.array(0.5)
    )
    # Strided read-only source arrays exercise arbitrary matching leading axes.
    origins = np.broadcast_to([7e6, 2e6, -1e6], (*leading, 3))[..., ::-1]
    directions = np.broadcast_to([0.0, 0.6, 0.8], (*leading, 3))[..., ::-1]
    times.flags.writeable = False
    snapshots = [a.copy() for a in (times, origins, directions)]
    source, destination = frames
    actual = transform_rays(
        times,
        origins,
        directions,
        epoch=EPOCH,
        validity_s=(0, 1),
        from_frame=source,
        to_frame=destination,
    )
    angle = times * (0 if source == destination else (0.01 if source == "GCRS" else -0.01))
    for output, value in zip(actual, (origins, directions), strict=True):
        expected = np.stack(
            (
                np.cos(angle) * value[..., 0] - np.sin(angle) * value[..., 1],
                np.sin(angle) * value[..., 0] + np.cos(angle) * value[..., 1],
                value[..., 2],
            ),
            axis=-1,
        )
        assert output.shape == (*leading, 3)
        np.testing.assert_allclose(output, expected, atol=2e-9, rtol=0)
        assert output.dtype == np.float64 and output.flags.owndata and output.flags.writeable
        assert not any(np.shares_memory(output, a) for a in (times, origins, directions))
    np.testing.assert_allclose(np.linalg.norm(actual[1], axis=-1), 1, atol=1e-15, rtol=0)
    assert len(backend.rotation_calls) == (0 if source == destination else len(np.unique(times)))
    for output in actual:
        output[...] = 0
    for value, snapshot in zip((times, origins, directions), snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)


@pytest.mark.parametrize(
    "times,origins,directions",
    [
        (np.array(0.0), np.array(1.0), X),
        (np.zeros(2), np.ones((2, 2)), np.ones((2, 2))),
        (np.zeros(2), np.zeros((2, 3)), X),
        (np.zeros((2, 1)), np.zeros((2, 3)), np.tile(X, (2, 1))),
        (np.array(0.0), np.array([np.nan, 0, 0]), X),
        (np.array(0.0), X, np.array([np.inf, 0, 0])),
        (np.array(0.0), X, 2 * X),
        (np.array(0.0), X, np.zeros(3)),
    ],
)
def test_transform_rays_invalid_inputs(backend, times, origins, directions):
    with pytest.raises(ValueError, match="rays"):
        transform_rays(times, origins, directions, epoch=EPOCH, validity_s=(0, 1))
    assert not backend.rotation_calls


def test_transform_rays_time_and_backend_failures(backend, monkeypatch):
    for times, validity in (
        (np.array(np.nan), (0, 1)),
        (np.array(2.0), (0, 1)),
        (np.array(0.0), (1, 0)),
        (np.array(0.0), (0, np.inf)),
    ):
        with pytest.raises(ValueError, match="validity"):
            transform_rays(times, X, X, epoch=EPOCH, validity_s=validity)
    with pytest.raises(ValueError, match="UTC"):
        transform_rays(np.array(0.0), X, X, epoch=EPOCH.replace(tzinfo=None), validity_s=(0, 1))
    with pytest.raises(ValueError, match="frames"):
        transform_rays(
            np.array(0.0), X, X, epoch=EPOCH, validity_s=(0, 1), to_frame=cast(Any, "unknown")
        )
    backend.frametransform.eop_status = lambda t: "extrapolated"
    with pytest.raises(RuntimeError, match="coverage"):
        transform_rays(np.array(0.0), X, X, epoch=EPOCH, validity_s=(0, 1))
    monkeypatch.setitem(sys.modules, "satkit", None)
    with pytest.raises(ImportError):
        transform_rays(np.array(0.0), X, X, epoch=EPOCH, validity_s=(0, 1))


@pytest.mark.parametrize("leading", [(), (0,), (2, 0, 4), (2, 2, 3)])
@pytest.mark.parametrize("from_ray", [False, True])
def test_wgs84_projection_callback_chart_snapshot_hits_misses_and_axes(backend, leading, from_ray):
    a = WGS84_SEMIAXES_M[0]
    origin = np.array([a, 0, 0])
    surface = (
        wgs84_surface_projection_from_ray(
            origin_gcrs_m=origin,
            direction_gcrs=-X,
            time_s=0.0,
            epoch=EPOCH,
            validity_s=(0, 1),
        )
        if from_ray
        else wgs84_surface_projection(epoch=EPOCH, validity_s=(0, 1), chart_origin_itrf_m=origin)
    )
    origin[:] = np.nan  # The retained chart is a private snapshot.
    count = int(np.prod(leading))
    times = (np.arange(count).reshape(leading) % 3) / 2
    # ITRF rays cycle through exterior inward hit, exterior outward miss, inside exit.
    mode = np.arange(count).reshape(leading) % 3
    radii = np.where(mode == 2, 0.0, 2 * a)
    itrf_origins = radii[..., None] * X
    itrf_directions = np.where((mode == 0)[..., None], -X, X)
    inverse_rotation = axis_rotations(Z, -0.01 * times)
    origins = np.einsum("...ij,...j->...i", inverse_rotation, itrf_origins)
    directions = np.einsum("...ij,...j->...i", inverse_rotation, itrf_directions)
    snapshots = [value.copy() for value in (origins, directions, times)]
    xy, hits = surface(origins, directions, times)
    assert xy.shape == (*leading, 2) and hits.shape == leading
    np.testing.assert_array_equal(hits, mode != 1)
    np.testing.assert_allclose(xy[hits], 0, atol=1e-8, rtol=0)
    assert np.isnan(xy[~hits]).all()
    for output in (xy, hits):
        assert output.flags.owndata and output.flags.writeable
        assert not any(np.shares_memory(output, value) for value in (origins, directions, times))
        output[...] = 0
    for value, snapshot in zip((origins, directions, times), snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)
    if not leading:
        with pytest.raises(ValueError, match="validity"):
            surface(origins, directions, np.array(2.0))
        backend.frametransform.eop_status = lambda t: "extrapolated"
        with pytest.raises(RuntimeError, match="coverage"):
            surface(origins, directions, times)


@pytest.mark.parametrize("reference_time", [0.0, 0.5, 1.0])
def test_wgs84_projection_from_ray_matches_composition_and_snapshots(backend, reference_time):
    reference_origin = np.array([12.0, -34.0, 7e6])[::-1]
    reference_direction = np.array([0.0, 0.0, -1.0])[::-1]
    reference_origin.flags.writeable = reference_direction.flags.writeable = False
    snapshots = [a.copy() for a in (reference_origin, reference_direction)]
    validity = np.array([0.0, 1.0])
    options = dict(epoch=EPOCH, validity_s=cast(Any, validity))
    surface = wgs84_surface_projection_from_ray(
        origin_gcrs_m=reference_origin,
        direction_gcrs=reference_direction,
        time_s=reference_time,
        **options,
    )
    origins_itrf, directions_itrf = transform_rays(
        np.array(reference_time), reference_origin, reference_direction, **options
    )
    hit, chart_origin = ellipsoid_intersections(origins_itrf, directions_itrf)
    assert hit
    expected_surface = wgs84_surface_projection(chart_origin_itrf_m=chart_origin, **options)
    for value, snapshot in zip((reference_origin, reference_direction), snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)
        value.flags.writeable = True
        value[:] = np.nan
    validity[:] = np.nan
    times = np.array([1.0, 0.0, 0.5, 0.5])
    origins = np.broadcast_to([7e6, 123.0, -89.0], (4, 3))
    directions = np.array([[-1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [-1.0, 0, 0]])
    actual_xy, actual_hits = surface(origins, directions, times)
    expected_xy, expected_hits = expected_surface(origins, directions, times)
    np.testing.assert_allclose(actual_xy, expected_xy, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(actual_hits, expected_hits)
    assert actual_hits.tolist() == [True, False, True, True]
    assert np.isnan(actual_xy[1]).all()


@pytest.mark.parametrize(
    "origin,direction,time",
    [
        (np.ones((1, 3)), -X, 0.0),
        (np.ones(2), -X, 0.0),
        (7e6 * X, np.array([-X]), 0.0),
        (7e6 * X, -X, np.array([0.0])),
        (7e6 * X, -X, np.empty(0)),
        (np.array([np.nan, 0, 0]), -X, 0.0),
        (np.array([np.inf, 0, 0]), -X, 0.0),
        (7e6 * X, np.array([-1.0, np.nan, 0.0]), 0.0),
        (7e6 * X, -2 * X, 0.0),
        (7e6 * X, np.zeros(3), 0.0),
        (7e6 * X, -X, np.nan),
        (7e6 * X, -X, np.inf),
        (7e6 * X, -X, -0.01),
        (7e6 * X, -X, 1.01),
    ],
)
def test_wgs84_projection_from_ray_rejects_invalid_single_ray(backend, origin, direction, time):
    with pytest.raises(ValueError):
        wgs84_surface_projection_from_ray(
            origin_gcrs_m=origin,
            direction_gcrs=direction,
            time_s=time,
            epoch=EPOCH,
            validity_s=(0, 1),
        )
    assert not backend.rotation_calls


def test_wgs84_projection_from_ray_misses_poles_and_backend_errors(backend, monkeypatch):
    options = dict(epoch=EPOCH, validity_s=(0, 1), time_s=0.0)
    with pytest.raises(ValueError, match="reference ray misses WGS84"):
        wgs84_surface_projection_from_ray(origin_gcrs_m=7e6 * X, direction_gcrs=X, **options)
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="pole"):
            wgs84_surface_projection_from_ray(
                origin_gcrs_m=sign * 7e6 * Z, direction_gcrs=-sign * Z, **options
            )
    backend.frametransform.eop_status = lambda t: "extrapolated"
    with pytest.raises(RuntimeError, match="coverage"):
        wgs84_surface_projection_from_ray(origin_gcrs_m=7e6 * X, direction_gcrs=-X, **options)
    monkeypatch.setitem(sys.modules, "satkit", None)
    with pytest.raises(ImportError):
        wgs84_surface_projection_from_ray(origin_gcrs_m=7e6 * X, direction_gcrs=-X, **options)


@pytest.mark.parametrize(
    "epoch,validity",
    [
        (EPOCH.replace(tzinfo=None), (0, 1)),
        (EPOCH.astimezone(timezone(timedelta(hours=1))), (0, 1)),
        (EPOCH, (1, 0)),
        (EPOCH, (0, np.inf)),
        (EPOCH, (np.nan, 1)),
        (EPOCH, (0,)),
    ],
)
def test_wgs84_projection_from_ray_invalid_epoch_and_interval(backend, epoch, validity):
    with pytest.raises(ValueError):
        wgs84_surface_projection_from_ray(
            origin_gcrs_m=7e6 * X,
            direction_gcrs=-X,
            time_s=0.0,
            epoch=epoch,
            validity_s=validity,
        )
    assert not backend.rotation_calls
