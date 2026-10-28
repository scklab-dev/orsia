# Notebook support

This package provides notebook-specific input preparation, figures, reference caching and local archive handling. The two notebooks select their scientific inputs and illustrative instrument layouts explicitly. The public Orsia libraries own reusable radiative and instrument calculations.

`five_band_gases.py` handles trace-gas projection and UV cross sections. `notebook_five_band_support.py` checks the local input manifest and produces comparison displays. `notebook_sasktran2_cache.py` launches the isolated `notebook_sasktran2_layers.py` worker and stores local references under ignored `outputs/`. `notebook_instrument_responses.py` selects atmospheric cells and extracts embedded figures. `notebook_instrument_support.py` handles instrument displays, archive receipts and checks; `notebook_ground_pixel.py` provides independent offline reference calculations.

See the [notebook guide](../README.md) for preparation, execution and replay recipes.
