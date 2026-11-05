"""Encoded SAFE metadata retains ordinary XML and rejects DTDs before expansion."""

import codecs
from xml.etree import ElementTree as ET

import pytest
from orsia_data import read_sentinel2_metadata


@pytest.mark.parametrize(
    "encoding,bom",
    [
        ("utf-8", b""),
        ("utf-8", codecs.BOM_UTF8),
        ("utf-16-le", b""),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", b""),
        ("utf-16-be", codecs.BOM_UTF16_BE),
    ],
)
@pytest.mark.parametrize("declaration", ["none", "doctype", "internal", "external"])
def test_sentinel_metadata_encoding_and_dtd_rejection(tmp_path, encoding, bom, declaration):
    xml_encoding = "UTF-8" if encoding == "utf-8" else "UTF-16"
    doctype = {
        "none": "",
        "doctype": "<!DOCTYPE Level>",
        "internal": '<!DOCTYPE Level [<!ENTITY product "S2MSI2A">]>',
        "external": '<!DOCTYPE Level SYSTEM "https://example.invalid/metadata.dtd">',
    }[declaration]
    product = "&product;" if declaration == "internal" else "S2MSI2A"
    metadata = f'''<?xml version="1.0" encoding="{xml_encoding}"?>{doctype}
    <n:Level xmlns:n="urn:test">
      <PRODUCT_TYPE>{product}</PRODUCT_TYPE><PRODUCT_URI>fixture&amp;test.SAFE</PRODUCT_URI>
      <PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>
      <PRODUCT_START_TIME>2024-01-01T12:00:00Z</PRODUCT_START_TIME>
      <PRODUCT_STOP_TIME>2024-01-01T12:00:01Z</PRODUCT_STOP_TIME>
    </n:Level>'''
    path = tmp_path / "MTD_MSIL2A.xml"
    payload = bom + metadata.encode(encoding)
    path.write_bytes(payload)
    if declaration == "none":
        result = read_sentinel2_metadata(tmp_path)
        assert result["PRODUCT_TYPE"] == "S2MSI2A"
        assert result["PRODUCT_URI"] == "fixture&test.SAFE"
    else:
        with pytest.raises(ValueError, match="DTD|entities"):
            read_sentinel2_metadata(tmp_path)
    assert path.read_bytes() == payload


def test_sentinel_metadata_retains_parse_errors_and_size_bound(tmp_path):
    path = tmp_path / "MTD_MSIL2A.xml"
    path.write_text("<Level>")
    with pytest.raises(ET.ParseError, match="no element found"):
        read_sentinel2_metadata(tmp_path)
    path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="bound"):
        read_sentinel2_metadata(tmp_path)
