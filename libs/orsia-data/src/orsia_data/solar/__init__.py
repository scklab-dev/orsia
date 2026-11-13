"""Explicit solar acquisition/readers; numerical contracts live in orsia-radiative."""

from orsia_data.services.lisird import fetch_solar_spectrum
from orsia_data.solar.irradiance import read_solar_spectrum_csv

__all__ = ["read_solar_spectrum_csv", "fetch_solar_spectrum"]
