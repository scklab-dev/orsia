"""Connect the physical Scene callable directly to native instrument integration."""

from datetime import UTC, datetime

import numpy as np
from numpy.testing import assert_allclose
from orsia_geometry import LayeredCells, LocalCartesianFrame, PointLattice, RectangularCells
from orsia_instruments import frame_electron_tangents, simulate_frames
from orsia_radiative import LayerCrossSections, LevelAtmosphere, SpectralDomain, wavenumber_grid
from orsia_scene import DirectSolarScene, SpatialOpticalDepth

from notebooks.support.spatial_scene import make_acquisition, synthetic_inputs


def test_direct_solar_scene_needs_no_raw_array_callback_adapter():
    cam, grid, _ = make_acquisition(synthetic_inputs(), angular_order=2)
    spectral = wavenumber_grid(
        np.linspace(4000.0, 4100.0, 5), domain=SpectralDomain("wavenumber_cm1", ((4000.0, 4100.0),))
    )
    # The synthetic camera surface callback defines this local chart and horizontal plane.
    frame = LocalCartesianFrame("test camera surface chart", "synthetic altitude")
    points = PointLattice.regular((1, 1), (10000.0, 10000.0), (0.0, 0.0), frame)
    cells = LayeredCells(
        RectangularCells.around_points(points, (10000.0, 10000.0)),
        np.array([0.0, 20.0]),
        np.zeros((1, 1)),
    )
    atmosphere = LevelAtmosphere(
        cells.height_edges_m,
        np.array([1e5, 9e4]),
        np.full(2, 280.0),
        np.zeros(2),
        ("CH4",),
        np.full((2, 1), 2e-6),
    )
    table = LayerCrossSections(
        "CH4",
        "all",
        "cross_section",
        spectral,
        np.array([280.0]),
        np.array([9.5e4]),
        np.full((1, 5), 1e-23),
        height_edges_m=cells.height_edges_m,
    )
    optics = SpatialOpticalDepth(atmosphere, (table,), spectral=spectral, support=cells)
    scene = DirectSolarScene(
        optics,
        anchor_wavelengths_nm=(2000.0, 2500.0),
        surface_reflectance=np.full((1, 1, 2), 0.3),
        beam_normal_irradiance=np.full(5, 0.03),
        solar_direction=np.array([0.0, 0.0, 1.0]),
        epoch=datetime(2023, 5, 16, tzinfo=UTC),
        validity_s=(0.0, 10.0),
        method="independent_column",
        continuation="finite",
        visibility="assume_clear",
    )

    def coverage(current, requested):
        # Known fixed horizontal projection; check complete declared bounds, not only nodes.
        assert current.rays.epoch == scene.epoch
        assert requested.same_points(spectral)
        bounds = current.bounds_m
        times = current.schedule.reads.exposure_bounds_s
        return (
            np.all(bounds[:, [0, 2]] >= -5000, axis=1)
            & np.all(bounds[:, [1, 3]] < 5000, axis=1)
            & (times[:, 0] >= 0)
            & (times[:, 1] <= 10)
        )

    baseline = simulate_frames(cam, grid, spectral, scene.radiance, coverage)
    blocked = simulate_frames(
        cam, grid, spectral, scene.radiance, coverage, read_batch_size=1, spectral_block_size=1
    )
    assert_allclose(blocked.images.photoelectrons, baseline.images.photoelectrons, rtol=5e-15)
    tangent_scene = scene.with_tangents(np.empty((0, 1)), np.ones((1, 1, 2, 1)))
    signal, derivative = frame_electron_tangents(
        cam, grid, spectral, tangent_scene.radiance_tangent, coverage, spectral_block_size=2
    )
    assert_allclose(signal.total, baseline.images.photoelectrons.ravel(), rtol=5e-15)
    assert_allclose(derivative[:, 0], signal.total / 0.3, rtol=5e-15)
