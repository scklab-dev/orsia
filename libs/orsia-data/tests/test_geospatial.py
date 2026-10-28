"""Synthetic native rasters and analytic spatial contracts, never provider truth."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from orsia_data import (
    RasterBand,
    RasterGrid,
    crop_raster,
    iter_raster_windows,
    mosaic_rasters,
    read_raster_band,
    read_raster_grid,
    reproject_raster,
    sample_raster,
    window_from_bounds,
    window_grid,
)

rio = pytest.importorskip("rasterio")
CRS = pytest.importorskip("pyproj").CRS


def field(*, categorical=False):
    grid = RasterGrid(CRS.from_epsg(4326), (1, 0, 0, 0, -1, 2), (2, 2))
    return RasterBand(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones((2, 2), dtype=bool), grid, "1", categorical
    )


def test_windows_and_ownership():
    source = field()
    selected = crop_raster(source, (1, 1, 1, 1))
    assert selected.grid.transform == (1, 0, 1, 0, -1, 1)
    assert selected.values.item() == 4
    selected.values[:] = 0
    assert source.values[-1, -1] == 4
    for window in ((-1, 0, 1, 1), (0, 0, 0, 1), (1, 1, 2, 1)):
        with pytest.raises(ValueError):
            window_grid(source.grid, window)
    assert window_from_bounds(source.grid, (0.2, 0.2, 1.8, 1.8), clip=False) == (0, 0, 2, 2)
    assert window_from_bounds(source.grid, (-2, -2, 1, 1), clip=True) == (1, 0, 1, 1)
    for bounds in ((-2, -2, 1, 1), (0, 0, 0, 0), (3, 3, 4, 4), (0, 0, np.nan, 1)):
        with pytest.raises(ValueError):
            window_from_bounds(source.grid, bounds, clip=False)
    with pytest.raises(ValueError):
        window_from_bounds(source.grid, (3, 3, 4, 4), clip=True)
    assert list(iter_raster_windows(source.grid, (1, 3))) == [(0, 0, 1, 2), (1, 0, 1, 2)]
    with pytest.raises(ValueError):
        list(iter_raster_windows(source.grid, (0, 1)))


def test_validation():
    source = field()
    with pytest.raises(ValueError, match="horizontal CRS"):
        replace(source.grid, crs=CRS.from_epsg(4979))
    for kwargs in (
        {"transform": (0, 0, 0, 0, 1, 0)},
        {"shape": (0, 1)},
        {"transform": (np.nan, 0, 0, 0, 1, 0)},
    ):
        with pytest.raises(ValueError):
            replace(source.grid, **kwargs)  # type: ignore[bad-argument-type]
    for kwargs in (
        {"values": np.ones((1, 2))},
        {"valid": np.ones((2, 2))},
        {"values": np.full((2, 2), np.nan)},
        {"units": ""},
        {"values": np.full((2, 2), 1.5), "categorical": True},
    ):
        with pytest.raises(ValueError):
            replace(source, **kwargs)  # type: ignore[bad-argument-type]


def test_local_native_window(tmp_path: Path):
    path = tmp_path / "test.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int16",
        nodata=-999,
        crs="EPSG:4326",
        transform=rio.Affine(1, 0, 0, 0, -1, 2),
    ) as dst:
        dst.write(np.array([[2, -999], [4, 6]], dtype="int16"), 1)
    result = read_raster_band(path, units="K", window=(0, 1, 2, 1))
    assert read_raster_grid(path).shape == (2, 2)
    np.testing.assert_array_equal(result.valid, [[False], [True]])
    assert result.values[-1, 0] == 6
    assert result.grid.transform == (1, 0, 1, 0, -1, 2)
    with pytest.raises(ValueError, match="band"):
        read_raster_band(path, units="K", band=2)
    with pytest.raises((ValueError, IsADirectoryError)):
        read_raster_band(tmp_path, units="K")
    with pytest.raises(FileNotFoundError):
        read_raster_band(tmp_path / "missing", units="K")
    (tmp_path / "not_raster").write_text("not a raster")
    with pytest.raises(ValueError, match="GeoTIFF"):
        read_raster_grid(tmp_path / "not_raster")
    with rio.open(
        tmp_path / "no_crs.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        transform=rio.Affine(1, 0, 0, 0, -1, 2),
    ) as dst:
        dst.write(np.ones((2, 2), dtype="uint8"), 1)
    with pytest.raises(ValueError, match="CRS"):
        read_raster_band(tmp_path / "no_crs.tif", units="K")


def test_sampling():
    source = field()
    x, y = np.array([0.5, 1.0, 1.5, -1, np.nan, 1e300]), np.array([1.5, 1.0, 0.5, 0, 0, 0])
    values, valid = sample_raster(source, x, y, method="bilinear")
    np.testing.assert_allclose(values[:3], [1, 2.5, 4])
    np.testing.assert_array_equal(valid, [True, True, True, False, False, False])
    values, valid = sample_raster(
        source, np.array([0.0, 1.99, 2.0]), np.array([2.0, 0.01, 0.0]), method="nearest"
    )
    np.testing.assert_allclose(values[:2], [1, 4])
    assert not valid[-1]
    source.valid[0, 1] = False
    assert not sample_raster(source, np.array([1.0]), np.array([1.0]), method="bilinear")[1][0]
    # Exact centre does not depend on zero-weight missing neighbours.
    assert sample_raster(source, np.array([0.5]), np.array([1.5]), method="bilinear")[1][0]
    with pytest.raises(ValueError):
        sample_raster(source, np.zeros(1), np.zeros(2), method="nearest")
    with pytest.raises(ValueError):
        sample_raster(field(categorical=True), x, y, method="bilinear")
    with pytest.raises(ValueError):
        sample_raster(source, x, y, method="cubic")  # type: ignore


def test_rotated_sampling_and_crop():
    source = field()
    source = replace(source, grid=replace(source.grid, transform=(2, 1, 10, 1, -2, 20)))
    value, valid = sample_raster(source, np.array([14.5]), np.array([18.5]), method="nearest")
    assert value[0] == 4 and valid[0]
    assert crop_raster(source, (1, 1, 1, 1)).grid.transform == (2, 1, 13, 1, -2, 19)


def test_reprojection_and_mosaic():
    source = field()
    warped, coverage = reproject_raster(source, source.grid, method="nearest", minimum_coverage=1)
    np.testing.assert_array_equal(warped.values, source.values)
    np.testing.assert_array_equal(coverage, 1)
    assert not np.shares_memory(warped.values, source.values)
    # A changed CRS is not treated as a relabelled array.
    target = RasterGrid(CRS.from_epsg(3857), (111319.49, 0, 0, 0, -111342.10, 222684.2), (2, 2))
    projected, _ = reproject_raster(source, target, method="nearest", minimum_coverage=1)
    np.testing.assert_array_equal(projected.values, source.values)
    outside = replace(source.grid, shape=(2, 3))
    projected, coverage = reproject_raster(source, outside, method="nearest", minimum_coverage=1)
    assert not projected.valid[:, -1].any() and not coverage[:, -1].any()
    other = replace(source, values=source.values + 10, valid=source.valid.copy())
    for overlap, expected in (("first", source.values), ("last", other.values)):
        mosaic = mosaic_rasters(
            (source, other), source.grid, method="nearest", minimum_coverage=1, overlap=overlap
        )
        np.testing.assert_array_equal(mosaic.values, expected)
    source.valid[0, 0] = False
    mosaic = mosaic_rasters(
        (source, other), source.grid, method="nearest", minimum_coverage=1, overlap="first"
    )
    assert mosaic.values[0, 0] == 11
    with pytest.raises(ValueError):
        mosaic_rasters((), source.grid, method="nearest", minimum_coverage=1, overlap="first")
    with pytest.raises(ValueError):
        mosaic_rasters(
            (source, replace(other, units="K")),
            source.grid,
            method="nearest",
            minimum_coverage=1,
            overlap="first",
        )
    with pytest.raises(ValueError):
        reproject_raster(source, source.grid, method="nearest", minimum_coverage=0)


def test_partial_bilinear_coverage():
    source = field()
    source.valid[0, 0] = False
    target = RasterGrid(source.grid.crs, (1, 0, 0.5, 0, -1, 1.5), (1, 1))
    full, weights = reproject_raster(source, target, method="bilinear", minimum_coverage=1)
    assert not full.valid.item()
    partial, _ = reproject_raster(source, target, method="bilinear", minimum_coverage=0.5)
    assert partial.valid.item()
    assert weights.item() == pytest.approx(0.75)
    assert partial.values.item() == pytest.approx(3)
