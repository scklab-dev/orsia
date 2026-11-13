"""Run a small DN-to-native-cube example; optionally save its support/error figure.

From the repository root:
    UV_NO_SYNC=1 MPLBACKEND=Agg devbox run -- just python examples/native_observation_cube.py
    Add --figure outputs/native-observation-cube.png to save the plot.

All inputs are synthetic. Coordinates use a local plane, with two offset filter
rows and two exposures. Spectral coefficients are supplied calibration examples,
not an optical instrument performance prediction. No network or data files are used.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from orsia_instruments import (
    DetectorProperties,
    GaussianPSF,
    ImageMeasurements,
    NativeBandResponse,
    RadiometricParameterMaps,
    calibrate_observations,
    camera_acquisition_grid,
    filter_camera,
    filter_strip,
    form_observation_cube,
    raster_read_schedule,
    regular_pixel_bounds,
)
from orsia_radiative import SpectralDomain, wavenumber_grid
from orsia_uncertainty import materialize_covariance, projected_variances
from scipy.sparse import csr_array


def example_inputs():
    """Build explicit synthetic DN, calibration, native geometry, mask and weights.

    Returns:
        Images, response, detector, geometry, derivative maps, mask and sparse weights.
    """
    pixels = regular_pixel_bounds((2, 2), pitch_m=20e-6)
    detector = DetectorProperties(
        pixel_bounds_m=pixels,
        pitch_m=None,
        quantum_efficiency=1,
        dark_electrons_per_s=0,
        read_noise_electrons=2,
        full_well_electrons=1e6,
        electrons_per_dn=2,
        bias_dn=10,
        nonlinearity=0,
        adc_bits=16,
    )
    camera = filter_camera(
        detector,
        [
            filter_strip(0, 1, band="a", transmission=np.ones_like),
            filter_strip(1, 2, band="b", transmission=np.ones_like),
        ],
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(2e-6),
    )
    schedule = raster_read_schedule(np.array([[0.0, 0.01], [0.05, 0.06]]), (2, 2))

    def trajectory(times):
        position = np.column_stack((times * 0, times * 0.2, times * 0 + 300))
        return position, np.broadcast_to([0.0, 0.2, 0.0], position.shape)

    def platform(times):
        return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times), 3, 3))

    def identity(times):
        return np.broadcast_to(np.eye(3), (len(times), 3, 3))

    def surface(origins, directions, times):
        distance = -origins[..., 2] / directions[..., 2]
        return (origins + distance[..., None] * directions)[..., :2], -directions, distance > 0

    geometry = camera_acquisition_grid(
        camera,
        schedule,
        epoch=datetime(2023, 5, 16, tzinfo=UTC),
        validity_s=(0.0, 1.0),
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=identity,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301,
        time_order=1,
        angular_order=2,
    )
    spectral = wavenumber_grid(
        np.array([4000.0, 5000.0, 6000.0]),
        domain=SpectralDomain("wavenumber_cm1", ((4000.0, 6000.0),)),
    )
    # Supplied achieved spectral sensitivity: sum=2 for a and sum=4 for b.
    coefficients = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [0.0, 2.0, 2.0]] * 2
    )
    acquisition = geometry.rays.acquisition
    spatial = np.full(acquisition.times_s.shape, 1 / acquisition.times_s.shape[1])
    response = NativeBandResponse(
        schedule,
        spectral,
        geometry.read_bands,
        coefficients,
        spatial,
        acquisition.times_s,
        acquisition.directions_uv,
    )
    dn = np.array([20, 30, 40, 50, 20, 30, 40, 50], dtype=np.int64).reshape(1, 2, 2, 2)
    flags = np.zeros_like(dn, dtype=bool)
    images = ImageMeasurements(np.zeros((2, 2, 2)), schedule, dn, flags, flags, {})
    # One dimensionless common fractional gain error: dg/dtheta = nominal gain.
    parameters = RadiometricParameterMaps(
        ("common fractional gain",),
        ("1",),
        np.full((8, 1), 2.0),
        np.zeros((8, 1)),
        np.zeros((8, 1)),
    )
    valid = np.ones(8, dtype=bool)
    valid[[3, 5, 7]] = False
    # Logical C order: (sample0,a), (sample0,b), (sample1,a), (sample1,b).
    # Masked a read is removed explicitly and its remaining partner gets weight 1.
    weights = csr_array(
        (
            np.array([0.5, 0.5, 0.5, 0.5, 1.0]),
            (np.array([0, 0, 1, 1, 2]), np.array([0, 4, 2, 6, 1])),
        ),
        shape=(4, 8),
    )
    return images, response, detector, geometry, parameters, valid, weights


def run_example(figure: Path | None = None):
    """Assert dense/structured agreement and optionally plot native footprints and errors.

    Args:
        figure: Optional destination PNG, whose parent directories are created.

    Returns:
        Native observation cube with source-projected covariance.
    """
    images, response, detector, geometry, parameters, valid, weights = example_inputs()
    options = dict(
        detector=detector,
        read_covariance=np.ones(8),
        parameters=parameters,
        parameter_covariance=np.array([0.01**2]),
        read_parameter_cross_covariance="zero",
        read_valid=valid,
        parameter_bias=np.array([-0.002]),
    )
    structured = calibrate_observations(images, response, **options)
    dense = calibrate_observations(images, response, representation="dense", **options)
    assembly = dict(shape=(1, 2, 2), band_names=("a", "b"), native_weights=weights)
    cube = form_observation_cube(structured, geometry, **assembly)
    reference = form_observation_cube(dense, geometry, **assembly)
    matrix = materialize_covariance(cube.covariance, max_bytes=3 * 3 * 8)
    np.testing.assert_allclose(matrix, reference.covariance, rtol=1e-14)
    np.testing.assert_allclose(cube.values, reference.values, equal_nan=True)
    np.testing.assert_allclose(projected_variances(cube.covariance), np.diag(matrix))
    np.testing.assert_array_equal(cube.observation_indices, [0, 1, 2])
    np.testing.assert_allclose(cube.values.ravel()[cube.observation_indices], [10.0, 15.0, 20.0])
    # Common fractional gain variance remains theta_variance * mean_radiance**2.
    common = cube.parameter_sensitivity[:, 0] ** 2 * 0.01**2
    np.testing.assert_allclose(common, [0.01, 0.0225, 0.04])
    print("Logical radiances (line, sample, band):", cube.values.tolist())
    print("Valid flat C-order indices:", cube.observation_indices.tolist())
    print("Marginal variances:", np.diag(matrix).tolist())
    print("Common gain variances (no independent-read reduction):", common.tolist())
    print("Dense and source-projected calibration/accumulation agree.")
    if figure is not None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        colors = {"a": "#3274a1", "b": "#d88732"}
        for i, (xmin, xmax, ymin, ymax) in enumerate(geometry.bounds_m):
            axes[0].add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor=colors[response.read_bands[i]],
                    linestyle="-" if valid[i] else ":",
                )
            )
            axes[0].text((xmin + xmax) / 2, (ymin + ymax) / 2, str(i), ha="center", fontsize=9)
        axes[0].autoscale_view()
        axes[0].set(
            xlabel="Local x (m)",
            ylabel="Local y (m)",
            title="Native supports: a blue, b orange; dotted invalid",
        )
        image = axes[1].imshow(matrix, cmap="viridis")
        axes[1].set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=["0,a", "0,b", "1,a"],
            yticklabels=["0,a", "0,b", "1,a"],
            title="Covariance in valid logical order",
        )
        fig.colorbar(image, ax=axes[1], label="Squared radiance units")
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=150)
        plt.close(fig)
    return cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path)
    run_example(parser.parse_args().figure)
