"""Explicit solar acquisition/readers; numerical contracts live in orsia-radiative."""

from .irradiance import read_solar_spectrum_csv
from .lisird import fetch_solar_spectrum

__all__ = ["read_solar_spectrum_csv", "fetch_solar_spectrum"]
