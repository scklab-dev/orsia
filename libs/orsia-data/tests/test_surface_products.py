"""Native-format synthetic surface fixtures; not authentic-product qualification."""

from pathlib import Path

import numpy as np
import pytest
from orsia_data import (
    read_aster_ged,
    read_copernicus_dem,
    read_ecostress_l2t,
    read_sentinel2_l2a,
    read_sentinel2_metadata,
)

rio = pytest.importorskip("rasterio")
h5py = pytest.importorskip("h5py")


def write_raster(path, values, *, crs="EPSG:32631", jp2=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"REVERSIBLE": "YES", "QUALITY": "100", "RESOLUTIONS": 1} if jp2 else {}
    with rio.open(
        path,
        "w",
        driver="JP2OpenJPEG" if jp2 else "GTiff",
        count=1,
        height=values.shape[0],
        width=values.shape[1],
        dtype=values.dtype,
        crs=crs,
        transform=rio.Affine(20, 0, 500000, 0, -20, 5000000),
        **options,
    ) as dst:
        dst.write(values, 1)


def safe_fixture(tmp_path, *, baseline="04.00", offset="-1000", quantification="10000"):
    safe = tmp_path / "fixture.SAFE"
    safe.mkdir()
    paths = ["GRANULE/G/IMG_DATA/R20m/T_B11_20m", "GRANULE/G/IMG_DATA/R20m/T_SCL_20m"]
    for name, values in zip(
        paths,
        (
            np.array([[0, 500], [1000, 11000]], dtype="uint16"),
            np.array([[0, 4], [8, 11]], dtype="uint8"),
        ),
        strict=True,
    ):
        write_raster(safe / (name + ".jp2"), values, jp2=True)
    offsets = "" if offset is None else f'<BOA_ADD_OFFSET band_id="11">{offset}</BOA_ADD_OFFSET>'
    metadata = f"""<n:Level xmlns:n="urn:sentinel:test">
    <PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE><PRODUCT_URI>fixture.SAFE</PRODUCT_URI>
    <PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>
    <PRODUCT_START_TIME>2024-01-01T12:00:00Z</PRODUCT_START_TIME>
    <PRODUCT_STOP_TIME>2024-01-01T12:00:01Z</PRODUCT_STOP_TIME>
    <BOA_QUANTIFICATION_VALUE>{quantification}</BOA_QUANTIFICATION_VALUE>{offsets}
    {"".join(f"<IMAGE_FILE>{name}</IMAGE_FILE>" for name in paths)}</n:Level>"""
    (safe / "MTD_MSIL2A.xml").write_text(metadata)
    return safe


def test_sentinel_native_decode_and_window(tmp_path):
    safe = safe_fixture(tmp_path)
    product = read_sentinel2_metadata(safe)
    assert product["PROCESSING_BASELINE"] == "04.00"
    result = read_sentinel2_l2a(safe, granule="G", bands=("B11", "SCL"), resolution_m=20)
    np.testing.assert_allclose(result["B11"].values, [[np.nan, -0.05], [0, 1]], equal_nan=True)
    np.testing.assert_array_equal(result["SCL"].valid, [[False, True], [True, True]])
    assert result["SCL"].categorical and result["SCL"].values[1, 0] == 8
    window = read_sentinel2_l2a(
        safe, granule="G", bands=("B11",), resolution_m=20, window=(1, 1, 1, 1)
    )["B11"]
    assert window.values.item() == 1
    result["B11"].values[:] = -100
    assert window.values.item() == 1
    with pytest.raises(ValueError, match="missing"):
        read_sentinel2_l2a(safe, granule="G", bands=("B11",), resolution_m=60)


def test_sentinel_old_baseline(tmp_path):
    safe = safe_fixture(tmp_path, baseline="03.00", offset=None)
    result = read_sentinel2_l2a(safe, granule="G", bands=("B11",), resolution_m=20)
    assert result["B11"].values[-1, -1] == 1.1


@pytest.mark.parametrize(
    "changes", [{"offset": None}, {"offset": "nan"}, {"quantification": "0"}, {"baseline": "nan"}]
)
def test_sentinel_bad_scales(tmp_path, changes):
    safe = safe_fixture(tmp_path, **changes)
    with pytest.raises(ValueError):
        read_sentinel2_l2a(safe, granule="G", bands=("B11",), resolution_m=20)


@pytest.mark.parametrize(
    "changes",
    [
        {"bands": ()},
        {"bands": ("B10",)},
        {"bands": ("B11", "B11")},
        {"granule": "../G"},
        {"resolution_m": 15},
    ],
)
def test_sentinel_bad_selection(tmp_path, changes):
    with pytest.raises(ValueError):
        read_sentinel2_l2a(
            tmp_path, **({"granule": "G", "bands": ("B11",), "resolution_m": 20} | changes)
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ("S2MSI2A", "S2MSI1C"),
        ("<PRODUCT_TYPE>S2MSI2A</PRODUCT_TYPE>", ""),
        ('band_id="11"', 'band_id="99"'),
        (
            '<BOA_ADD_OFFSET band_id="11">-1000</BOA_ADD_OFFSET>',
            '<BOA_ADD_OFFSET band_id="11">-1000</BOA_ADD_OFFSET>' * 2,
        ),
    ],
)
def test_sentinel_bad_metadata(tmp_path, old, new):
    safe = safe_fixture(tmp_path)
    path = safe / "MTD_MSIL2A.xml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_sentinel2_l2a(safe, granule="G", bands=("B11",), resolution_m=20)


def test_sentinel_rejects_entities_and_escape(tmp_path):
    safe = safe_fixture(tmp_path)
    path = safe / "MTD_MSIL2A.xml"
    original = path.read_text()
    path.write_text('<!DOCTYPE x [<!ENTITY data "text">]>' + original)
    with pytest.raises(ValueError, match="entities"):
        read_sentinel2_metadata(safe)
    path.write_text(original)
    image = safe / "GRANULE/G/IMG_DATA/R20m/T_B11_20m.jp2"
    outside = tmp_path / "outside.jp2"
    image.rename(outside)
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        read_sentinel2_l2a(safe, granule="G", bands=("B11",), resolution_m=20)


@pytest.mark.parametrize(
    "layer,values,units,categorical",
    [
        ("LST", [[300, np.nan], [0, 290]], "K", False),
        ("LST_err", [[0, 1], [2, 3]], "K", False),
        ("EmisWB", [[0.9, 1.1], [0, 1]], "1", False),
        ("height", [[-10, 100], [200, 0]], "m", False),
        ("view_zenith", [[0, 10], [20, 30]], "degree", False),
        ("QC", [[0, 65535], [4, 8]], "1", True),
        ("water", [[0, 1], [0, 1]], "1", True),
        ("cloud", [[0, 1], [1, 0]], "1", True),
    ],
)
def test_ecostress_native_physical(tmp_path, layer, values, units, categorical):
    path = tmp_path / f"ECOv002_L2T_LSTE_00000_001_31TCJ_20240101T120000_0700_01_{layer}.tif"
    write_raster(path, np.array(values, dtype="uint16" if categorical else "float32"))
    band = read_ecostress_l2t(path, layer=layer)
    assert band.units == units and band.categorical == categorical
    assert band.values[0, 0] == pytest.approx(values[0][0])
    if layer == "LST":
        assert not band.valid[0, 1] and not band.valid[1, 0]
    if layer == "EmisWB":
        assert not band.valid[0, 1]
    if layer == "QC":
        assert band.valid.all() and band.values[0, 1] == 65535


def test_ecostress_rejects_product_and_layer(tmp_path):
    with pytest.raises(ValueError):
        read_ecostress_l2t(tmp_path / "ECOv001_L2T_LSTE_a_LST.tif", layer="LST")
    with pytest.raises(ValueError):
        read_ecostress_l2t(tmp_path / "ECOv002_L2T_LSTE_a_unknown.tif", layer="unknown")


def aster_fixture(tmp_path: Path):
    path = tmp_path / "AG100.v003.40.-005.0001.h5"
    with h5py.File(path, "w") as source:
        for name, value, shape in (
            ("Emissivity/Mean", 950, (5, 1000, 1000)),
            ("Emissivity/SDev", 25, (5, 1000, 1000)),
            ("Temperature/Mean", 30000, (1000, 1000)),
            ("Temperature/SDev", 150, (1000, 1000)),
        ):
            source.create_dataset(name, shape=shape, dtype="int32", fillvalue=value, chunks=True)
        for name, values in (
            (
                "Latitude",
                np.broadcast_to(40 - (np.arange(1000)[:, None] + 0.5) * 0.001, (1000, 1000)),
            ),
            (
                "Longitude",
                np.broadcast_to(-5 + (np.arange(1000)[None, :] + 0.5) * 0.001, (1000, 1000)),
            ),
        ):
            source.create_dataset(
                f"Geolocation/{name}", data=values.astype("float32"), compression="gzip"
            )
        source["Emissivity/Mean"][0, 0, 0] = -9999
    return path


def test_aster_scales_and_spatial_support(tmp_path):
    path = aster_fixture(tmp_path)
    result = read_aster_ged(
        path,
        fields=(
            "emissivity_10",
            "emissivity_14",
            "emissivity_sdev_11",
            "temperature",
            "temperature_sdev",
        ),
        window=(0, 0, 2, 2),
    )
    assert result["emissivity_10"].values[1, 1] == pytest.approx(0.95)
    assert not result["emissivity_10"].valid[0, 0]
    assert result["emissivity_14"].valid[0, 0]
    assert result["emissivity_sdev_11"].values[0, 0] == pytest.approx(0.0025)
    assert result["temperature"].values[0, 0] == 300
    assert result["temperature_sdev"].values[0, 0] == 1.5
    assert result["temperature"].grid.transform == (0.001, 0, -5, 0, -0.001, 40)
    for fields in ((), ("other",), ("temperature", "temperature")):
        with pytest.raises(ValueError):
            read_aster_ged(path, fields=fields, window=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        read_aster_ged(tmp_path / "AG100.v002.40.-005.0001.h5", fields=("temperature",))
    with pytest.raises(ValueError):
        read_aster_ged(tmp_path / "AG100.v003.91.-005.0001.h5", fields=("temperature",))


def test_aster_invalid_support(tmp_path):
    path = aster_fixture(tmp_path)
    with h5py.File(path, "r+") as source:
        source["Geolocation/Latitude"][0, 0] = 0
    with pytest.raises(ValueError, match="geolocation"):
        read_aster_ged(path, fields=("temperature",), window=(0, 0, 1, 1))
    with h5py.File(path, "r+") as source:
        source["Geolocation/Latitude"][0, 0] = 39.9995
        del source["Temperature/Mean"]
        source.create_dataset("Temperature/Mean", data=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="dimensions"):
        read_aster_ged(path, fields=("temperature",), window=(0, 0, 1, 1))


def test_dem_semantics(tmp_path):
    path = tmp_path / "dem.tif"
    write_raster(path, np.array([[0.0, -10.0], [100.0, np.nan]], dtype="float32"), crs="EPSG:4326")
    band = read_copernicus_dem(path)
    assert band.valid[0, 0] and band.values[0, 1] == -10 and not band.valid[1, 1]
    write_raster(path, np.zeros((2, 2), dtype="float32"))
    with pytest.raises(ValueError, match="WGS84"):
        read_copernicus_dem(path)
