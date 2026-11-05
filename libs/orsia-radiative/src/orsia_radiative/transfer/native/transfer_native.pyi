"""Private disjoint-output buffer ABI for serial preparation and recurrence."""

from collections.abc import Buffer, Callable
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

def bind_thermal_expn(
    owner: ModuleType,
    second_radiation: float,
    rayleigh_jeans: float,
    /,
) -> tuple[
    Callable[
        [Buffer, Buffer, Buffer, Buffer | None, Buffer | None, Buffer, Buffer, float, float, bool],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ],
    Callable[[Buffer, Buffer], NDArray[np.float64]],
]: ...
def spectral_orders(
    optical: tuple[Buffer | None, ...],
    irradiance: Buffer,
    reflectance: Buffer,
    directions: Buffer,
    weights: Buffer,
    layers: int,
    subdivisions: int,
    max_order: int,
    moments: int,
    zenith: float,
    view_zenith: float,
    azimuth: float,
    /,
) -> NDArray[np.float64]: ...
def bind_spectral_blas(
    owner: ModuleType,
    /,
) -> Callable[
    [
        tuple[Buffer | None, ...],
        Buffer,
        Buffer,
        Buffer,
        Buffer,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
    ],
    NDArray[np.float64],
]: ...
def prepare(
    extinction: Buffer,
    mu: Buffer,
    solar_mu: float,
    transmission: Buffer,
    centre_transmission: Buffer,
    source_weight: Buffer,
    centre_source_weight: Buffer,
    solar_source_weight: Buffer | None,
    solar_centre_source_weight: Buffer | None,
    /,
) -> None: ...
def rayleigh_orders(
    depth: Buffer,
    factors: tuple[Buffer, Buffer],
    beam_source: Buffer,
    coefficients: tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer | None, Buffer | None],
    boundary_centres: Buffer,
    boundary_exits: Buffer,
    weights: Buffer,
    direct_boundary: float,
    reflectance: float,
    max_order: int,
    /,
) -> NDArray[np.float64]: ...
def bind_rayleigh_blas(
    owner: ModuleType,
    /,
) -> Callable[
    [
        Buffer,
        tuple[Buffer, Buffer],
        Buffer,
        tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer | None, Buffer | None],
        Buffer,
        Buffer,
        Buffer,
        float,
        float,
        int,
    ],
    NDArray[np.float64],
]: ...
def apply(
    mu: Buffer,
    source: Buffer,
    transmission: Buffer,
    weight: Buffer,
    half_transmission: Buffer | None,
    half_weight: Buffer | None,
    bottom: float,
    centres: Buffer | None,
    exits: Buffer,
    /,
) -> None: ...
