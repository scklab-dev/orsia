"""Fixed WGS84 chart coordinates against independent geodetic east/north axes."""

import builtins
from datetime import UTC, datetime

import numpy as np
import pytest
from orsia_geometry import project_local_east_north, wgs84_surface_projection
from orsia_geometry.constants import WGS84_FLATTENING, WGS84_SEMIAXES_M


def test_east_north_geodetic_basis_displacements_and_misses():
    latitude, longitude = np.deg2rad([47.0, -123.0])
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    prime_vertical = WGS84_SEMIAXES_M[0] / np.sqrt(1 - eccentricity_squared * sin_lat**2)
    origin = prime_vertical * np.array(
        [cos_lat * cos_lon, cos_lat * sin_lon, (1 - eccentricity_squared) * sin_lat]
    )
    # Independent geodetic basis: a radial normal would produce the wrong north/up split.
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    displacements = np.array([[0, 0, 0], [13, -7, 29], [-81, 123, -3], [0, 0, 1e4]])
    points = origin + displacements @ np.array([east, north, up])
    points = np.vstack((points, [np.nan] * 3))
    points.flags.writeable = origin.flags.writeable = False
    point_snapshot, origin_snapshot = points.copy(), origin.copy()
    actual = project_local_east_north(points, origin_itrf_m=origin)
    np.testing.assert_allclose(actual[:-1], displacements[:, :2], atol=1e-9, rtol=0)
    assert np.isnan(actual[-1]).all()
    assert actual.dtype == np.float64 and actual.flags.owndata and actual.flags.writeable
    assert not np.shares_memory(actual, points) and not np.shares_memory(actual, origin)
    actual[:] = 0
    np.testing.assert_array_equal(points, point_snapshot)
    np.testing.assert_array_equal(origin, origin_snapshot)


@pytest.mark.parametrize("leading", [(), (0,), (2, 0, 4), (2, 3, 4)])
def test_east_north_leading_axes_and_equatorial_signs(leading):
    origin = np.array([WGS84_SEMIAXES_M[0], 0.0, 0.0])
    points = np.broadcast_to(origin + [999.0, 23.0, -71.0], (*leading, 3))
    actual = project_local_east_north(points, origin_itrf_m=origin)
    assert actual.shape == (*leading, 2)
    np.testing.assert_array_equal(actual, np.broadcast_to([23, -71], actual.shape))
    assert actual.flags.owndata and actual.flags.writeable


@pytest.mark.parametrize(
    "origin",
    [
        np.ones((1, 3)),
        np.ones(2),
        np.array([np.nan, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0]),
        np.zeros(3),
        np.array([WGS84_SEMIAXES_M[0] + 0.01, 0.0, 0.0]),
        np.array([0.0, 0.0, WGS84_SEMIAXES_M[2]]),
        np.array([0.0, 0.0, -WGS84_SEMIAXES_M[2]]),
    ],
)
def test_invalid_chart_origins_fail_for_projection_and_factory(origin):
    with pytest.raises(ValueError, match="origin"):
        project_local_east_north(np.empty((0, 3)), origin_itrf_m=origin)
    with pytest.raises(ValueError, match="origin"):
        wgs84_surface_projection(
            epoch=datetime(2023, 1, 1, tzinfo=UTC), validity_s=(0, 1), chart_origin_itrf_m=origin
        )


@pytest.mark.parametrize(
    "points",
    [
        np.array(0.0),
        np.ones((2, 2)),
        np.array([np.nan, 0.0, np.nan]),
        np.array([np.inf, np.inf, np.inf]),
        np.array([[np.nan, np.nan, np.nan], [0.0, 0.0, -np.inf]]),
    ],
)
def test_invalid_chart_points(points):
    with pytest.raises(ValueError, match="points"):
        project_local_east_north(points, origin_itrf_m=np.array([WGS84_SEMIAXES_M[0], 0, 0]))


def test_chart_surface_tolerance_and_near_polar_support():
    a, _, b = WGS84_SEMIAXES_M
    # Either side of the documented dimensionless 1e-10 surface tolerance.
    accepted = np.array([a * np.sqrt(1 + 0.5e-10), 0, 0])
    np.testing.assert_array_equal(
        project_local_east_north(accepted, origin_itrf_m=accepted), [0, 0]
    )
    with pytest.raises(ValueError, match="height-zero"):
        project_local_east_north(accepted, origin_itrf_m=np.array([a * np.sqrt(1 + 2e-10), 0, 0]))
    near_pole = np.array([1e-6, 0, b * np.sqrt(1 - (1e-6 / a) ** 2)])
    np.testing.assert_allclose(
        project_local_east_north(near_pole + [0, 1, 0], origin_itrf_m=near_pole), [1, 0], atol=1e-15
    )


def test_surface_factory_is_lazy_and_validates_time_domain(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "satkit" or name.startswith("satkit."):
            raise AssertionError("native import boundary reached")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    options = dict(
        epoch=datetime(2023, 1, 1, tzinfo=UTC),
        validity_s=(0.0, 1.0),
        chart_origin_itrf_m=np.array([WGS84_SEMIAXES_M[0], 0, 0]),
    )
    surface = wgs84_surface_projection(**options)
    with pytest.raises(AssertionError, match="native import boundary reached"):
        surface(np.array([7e6, 0, 0]), np.array([-1.0, 0, 0]), np.array(0.0))
    with pytest.raises(ValueError, match="UTC"):
        wgs84_surface_projection(
            epoch=datetime(2023, 1, 1),
            validity_s=(0, 1),
            chart_origin_itrf_m=np.array([WGS84_SEMIAXES_M[0], 0, 0]),
        )


@pytest.mark.parametrize("interval", [(1, 0), (0, np.inf), (np.nan, 1), (0,)])
def test_surface_factory_invalid_time_domain(interval):
    with pytest.raises(ValueError, match="interval"):
        wgs84_surface_projection(
            epoch=datetime(2023, 1, 1, tzinfo=UTC),
            validity_s=interval,
            chart_origin_itrf_m=np.array([WGS84_SEMIAXES_M[0], 0, 0]),
        )
