"""Synthetic thermal spectral/band Jacobians and bounded serial timings."""

import json
import platform
import sys
from dataclasses import replace
from statistics import median
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from orsia_numerics.tangents import directional_derivative, jacobian
from orsia_radiative import Absorption, AtmosphericOptics, LayerAtmosphere, ObservationGeometry, ColumnScene, SurfaceState, wavenumber_grid
from orsia_radiative.transfer.thermal import thermal_radiance, thermal_radiance_tangent
from threadpoolctl import threadpool_info


def model(layers: int, points: int, method: Literal["expn", "quadrature"]):
    """Construct synthetic fixed optics and an explicit four-coordinate model."""
    grid = wavenumber_grid(np.linspace(700.0, 1300.0, points))
    boundaries = np.linspace(1e5, 1e4, layers + 1)
    atmosphere = LayerAtmosphere(
        np.linspace(0.0, 20000.0, layers + 1),
        boundaries,
        (boundaries[1:] + boundaries[:-1]) / 2,
        np.linspace(285.0, 220.0, layers),
        np.full(layers, 2e28),
        np.full(layers, 1.9e28),
        np.full(layers, 1e27),
    )
    base_depth = np.broadcast_to(
        0.02 + 0.1 * np.exp(-(((grid.points - 1000.0) / 90.0) ** 2)), (layers, points)
    ).copy()

    def inputs(x: NDArray[np.float64]):
        state = replace(atmosphere, temperature_k=atmosphere.temperature_k + x[1])
        scene = ColumnScene(
            state,
            SurfaceState(grid, temperature_k=float(x[2]), emissivity=np.full(points, x[3])),
            ObservationGeometry(30.0),
        )
        optics = AtmosphericOptics(grid, Absorption(grid, base_depth * x[0]), None, None)
        return scene, optics

    def evaluate(x: NDArray[np.float64], v: NDArray[np.float64]):
        scene, optics = inputs(x)
        value, tangent = thermal_radiance_tangent(
            scene,
            optics=optics,
            optical_depth_tangent=base_depth[..., None] * v[0],
            layer_temperature_tangent=np.broadcast_to(v[1], (layers, v.shape[1])),
            surface_temperature_tangent=v[2],
            emissivity_tangent=np.broadcast_to(v[3], (points, v.shape[1])),
            hemispheric_method=method,
        )
        return value.values, tangent

    def value_only(x: NDArray[np.float64]):
        scene, optics = inputs(x)
        return thermal_radiance(scene, optics=optics, hemispheric_method=method).values

    return grid.points, evaluate, value_only


def main() -> None:
    x = np.array([1.0, 0.0, 300.0, 0.8])
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "gil_enabled_after_imports": sys._is_gil_enabled(),
        "native_pools": threadpool_info(),
        "state_coordinates": [
            "absorption_scale",
            "layer_temperature_offset_K",
            "surface_temperature_K",
            "emissivity",
        ],
        "layers": 16,
        "points": 512,
        "repetitions": 7,
        "timing_scope": (
            "serial full Python model composition and native transfer; "
            "one warmup; median wall seconds"
        ),
        "methods": {},
    }
    assert not sys._is_gil_enabled()
    for method in ("expn", "quadrature"):
        grid, evaluate, value_only = model(16, 512, method)
        value, derivatives = jacobian(evaluate, x)
        np.testing.assert_array_equal(value, value_only(x))
        direction = np.array([0.1, 1.0, -2.0, 0.01])
        direction_value, derivative = directional_derivative(evaluate, x, direction)
        np.testing.assert_array_equal(direction_value, value)
        np.testing.assert_allclose(derivative, derivatives @ direction, rtol=2e-13, atol=1e-15)
        band_value = np.trapezoid(value, grid)
        band_jacobian = np.trapezoid(derivatives, grid, axis=0)
        timings = {}
        for k in (0, 1, 3, 4):
            directions = np.eye(4)[:, :k]
            evaluate(x, directions)
            durations = []
            for _ in range(7):
                start = perf_counter()
                evaluate(x, directions)
                durations.append(perf_counter() - start)
            timings[str(k)] = median(durations)
        durations = []
        value_only(x)
        for _ in range(7):
            start = perf_counter()
            value_only(x)
            durations.append(perf_counter() - start)
        timings["value_only"] = median(durations)
        report["methods"][method] = {
            "spectral_shape": list(value.shape),
            "jacobian_shape": list(derivatives.shape),
            "spectral_samples_W_m2_sr_cm1": value[[0, 256, -1]].tolist(),
            "spectral_jacobian_samples": derivatives[[0, 256, -1]].tolist(),
            "band_radiance_W_m2_sr": float(band_value),
            "band_jacobian": band_jacobian.tolist(),
            "direction": direction.tolist(),
            "band_directional_derivative": float(np.trapezoid(derivative, grid)),
            "median_seconds": timings,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
