"""Explicit solar acquisition/readers; numerical contracts live in orsia-radiative."""

from ..services.lisird import fetch_solar_spectrum
from .irradiance import read_solar_spectrum_csv

__all__ = ["read_solar_spectrum_csv", "fetch_solar_spectrum"]
