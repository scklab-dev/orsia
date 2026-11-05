"""Explicit spectroscopy acquisition/readers, without numerical materialisation."""

from ..services.hitran import acquire_hitran_references, fetch_hitran_lines
from .continuum import download_mt_ckd_water, fetch_mt_ckd_water, read_mt_ckd_water
from .cross_sections import read_cross_section_table
from .lines import read_aer_lines, read_hitran_lines
from .partition_sums import read_partition_table

__all__ = [
    "acquire_hitran_references",
    "download_mt_ckd_water",
    "fetch_mt_ckd_water",
    "fetch_hitran_lines",
    "read_aer_lines",
    "read_hitran_lines",
    "read_partition_table",
    "read_cross_section_table",
    "read_mt_ckd_water",
]
