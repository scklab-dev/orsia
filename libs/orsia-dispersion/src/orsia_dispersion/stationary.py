"""Describe an eastward stationary Gaussian plume and evaluate its physical profile."""

from dataclasses import dataclass

import numpy as np

from orsia_dispersion.forcing import FloatArray


@dataclass(frozen=True, kw_only=True)
class StationaryGaussianPlume:
    """Describe an eastward class-D plume above a flat reflecting plane.

    Attributes:
        emission_rate_kg_s: Nonnegative continuous mass emission, kg/s.
        wind_speed_m_s: Positive eastward wind speed, m/s.
        source_regularization_m: Positive minimum dispersion distance, m.
        stack_height_m: Release height above the reflecting plane, m, nonnegative.
        stack_inside_diameter_m: Positive stack diameter and minimum width scale, m.
        stack_exit_velocity_m_s: Positive exit speed, m/s.
        stack_exit_temperature_k: Positive exit temperature, K.
        ambient_temperature_k: Positive ambient temperature, K.
        source_east_m: Source east coordinate in the caller's local chart, m.
        source_north_m: Source north coordinate in the same chart, m.
        ground_altitude_m: Flat reflection-plane altitude in that chart, m.
        removal_timescale_s: Optional positive first-order removal timescale, s.
            All fields are immutable scalars. Terrain clips this distribution;
            it does not alter its flat-ground reflection physics.
    """

    emission_rate_kg_s: float
    wind_speed_m_s: float
    source_regularization_m: float
    stack_height_m: float
    stack_inside_diameter_m: float
    stack_exit_velocity_m_s: float
    stack_exit_temperature_k: float
    ambient_temperature_k: float
    source_east_m: float = 0.0
    source_north_m: float = 0.0
    ground_altitude_m: float = 0.0
    removal_timescale_s: float | None = None

    def __post_init__(self) -> None:
        """Validate finite physical parameters.

        Raises:
            ValueError: Rates, coordinates or positive physical scales are invalid.
        """
        positive = (
            self.wind_speed_m_s,
            self.source_regularization_m,
            self.stack_inside_diameter_m,
            self.stack_exit_velocity_m_s,
            self.stack_exit_temperature_k,
            self.ambient_temperature_k,
        )
        other = (
            self.emission_rate_kg_s,
            self.stack_height_m,
            self.source_east_m,
            self.source_north_m,
            self.ground_altitude_m,
        )
        if (
            not np.isfinite((*positive, *other)).all()
            or min(positive) <= 0
            or self.emission_rate_kg_s < 0
            or self.stack_height_m < 0
        ):
            raise ValueError(
                "finite coordinates, nonnegative emission/height and positive scales required"
            )
        if self.removal_timescale_s is not None and (
            not np.isfinite(self.removal_timescale_s) or self.removal_timescale_s <= 0
        ):
            raise ValueError("removal_timescale_s must be positive and finite")


def stationary_profile(
    model: StationaryGaussianPlume, east_m: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate reflected-plume Gaussian parameters on downwind coordinates.

    Args:
        model: Validated stationary physical declaration.
        east_m: Finite absolute east coordinates, m, of any shape.

    Returns:
        Centre altitude, crosswind sigma, vertical sigma (m), and mass per
        downwind length (kg/m), with the input shape. Upwind mass is zero.
        The profile uses Briggs rise and class-D widths with source regularization.
    """
    x = east_m - model.source_east_m
    d = np.maximum(x, model.source_regularization_m)
    buoyancy = (
        9.80665
        * model.stack_exit_velocity_m_s
        * model.stack_inside_diameter_m**2
        * max(model.stack_exit_temperature_k - model.ambient_temperature_k, 0)
        / (4 * model.stack_exit_temperature_k)
    )
    final = (
        21.425 * buoyancy**0.75 if buoyancy < 55 else 38.71 * buoyancy**0.6
    ) / model.wind_speed_m_s
    rise = np.minimum(1.6 * buoyancy ** (1 / 3) * d ** (2 / 3) / model.wind_speed_m_s, final)
    centre = model.ground_altitude_m + model.stack_height_m + rise
    sy = 0.08 * d / np.sqrt(1 + 0.0001 * d)
    sz = np.maximum(0.06 * d / np.sqrt(1 + 0.0015 * d), model.stack_inside_diameter_m / 2)
    amount = np.where(x >= 0, model.emission_rate_kg_s / model.wind_speed_m_s, 0.0)
    if model.removal_timescale_s is not None:
        amount *= np.exp(
            -np.maximum(x - model.source_regularization_m, 0)
            / (model.wind_speed_m_s * model.removal_timescale_s)
        )
    return centre, sy, sz, amount
