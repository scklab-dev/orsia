"""Explicit NumPy reference adapters for the native formal buffer signatures."""

import numpy as np
from numpy.typing import NDArray
from orsia_radiative.transfer import formal


def prepare[Real: np.float32 | np.float64](
    extinction: NDArray[Real],
    mu: NDArray[Real],
    solar_mu: float,
    transmission: NDArray[Real],
    centre_transmission: NDArray[Real],
    source_weight: NDArray[Real],
    centre_source_weight: NDArray[Real],
    solar_source_weight: NDArray[Real] | None,
    solar_centre_source_weight: NDArray[Real] | None,
) -> None:
    """Fill supplied buffers using the NumPy coefficient builder."""
    formal._prepare_numpy(
        extinction,
        mu,
        solar_mu if solar_source_weight is not None else None,
        transmission,
        centre_transmission,
        source_weight,
        centre_source_weight,
        solar_source_weight,
        solar_centre_source_weight,
    )


def apply[Real: np.float32 | np.float64](
    mu: NDArray[Real],
    source: NDArray[Real],
    transmission: NDArray[Real],
    weight: NDArray[Real],
    centre_transmission: NDArray[Real] | None,
    centre_weight: NDArray[Real] | None,
    bottom_radiance: float,
    centres: NDArray[Real] | None,
    exits: NDArray[Real],
) -> None:
    """Copy NumPy recurrence outputs into the supplied native-style buffers."""
    # Exit-only evaluation leaves the centre coefficients unused.
    half_transmission = transmission if centre_transmission is None else centre_transmission
    half_weight = weight if centre_weight is None else centre_weight
    coefficients = formal.PropagationCoefficients(
        mu, transmission, half_transmission, weight, half_weight, None, None
    )
    expected_centres, expected_exits = formal._apply_numpy(
        source, coefficients, weight, half_weight, bottom_radiance, centres is not None
    )
    if centres is not None:
        assert expected_centres is not None
        centres[:] = expected_centres
    exits[:] = expected_exits
