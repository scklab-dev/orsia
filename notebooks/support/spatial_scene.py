"""Compose a configurable bounded synthetic atmosphere/plume into native camera reads.

Run from the workspace with ``devbox run -- just python -m notebooks.support.spatial_scene``.
This application example owns all fixture choices. See docs/spatial-scenes.md for
coordinates, conservative support proof, approximation and derivative boundaries.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from orsia_dispersion import GaussianPuffField, GaussianPuffs
from orsia_geometry import affine_surface_points, intersect_affine_terrain
from orsia_geometry.frames import require_time_support
from orsia_instruments import (
    GaussianPSF,
    camera_acquisition_grid,
    filter_camera,
    filter_strip,
    frame_exposure_bounds,
    raster_read_schedule,
    regular_detector,
)
from orsia_instruments.photodetection.frames import frame_electron_tangents
from orsia_radiative import Absorber, LayerAtmosphere, SpectralDomain, SpectralGrid, wavenumber_grid
from orsia_scene import puff_molecular_columns, surface_radiance, surface_radiance_tangent
from scipy.special import ndtr


def synthetic_inputs():
    """Return fresh explicit fixture inputs, editable before binding the callbacks."""
    heights = np.array([0.0, 500.0, 1500.0])
    dry = np.diff(heights) * 2e25
    water = dry * 0.01
    background = LayerAtmosphere(
        heights,
        np.array([1e5, 9e4, 7e4]),
        np.array([9.5e4, 8e4]),
        np.array([290.0, 280.0]),
        dry + water,
        dry,
        water,
        (Absorber("CH4", dry * 2e-6, "dry_air"),),
    )
    plume = GaussianPuffs(
        0.0,
        np.array([0.0]),
        np.array([20.0]),
        np.array([[150.0, 0.0, 800.0]]),
        np.array([np.diag([180.0**2, 200.0**2, 120.0**2])]),
    )
    return dict(
        background=background,
        plume=plume,
        molar_mass_kg_mol=0.01604,
        height_m=250.0,
        slopes=np.zeros(2),
        sun_direction=np.array([0.6, 0.0, 0.8]),
        spectral_interval=(4000.0, 4100.0),
        time_interval=(0.0, 0.3),
        horizontal_interval=(-3000.0, 3000.0),
        reflectance_contrast=0.05,
        epoch=datetime(2023, 5, 16, tzinfo=UTC),
        angular_rate_rad_s=0.02,
    )


def chart_rotation(times_s, angular_rate_rad_s):
    """Return the exact synthetic chart-to-GCRS rotation at elapsed SI seconds."""
    angle = times_s * angular_rate_rad_s
    result = np.zeros((*times_s.shape, 3, 3))
    result[..., 0, 0] = result[..., 1, 1] = np.cos(angle)
    result[..., 1, 0] = np.sin(angle)
    result[..., 0, 1] = -np.sin(angle)
    result[..., 2, 2] = 1
    return result


@dataclass(frozen=True, eq=False)
class SyntheticRadianceModel:
    """Evaluate one bounded synthetic scene and its two physical sensitivities.

    Attributes:
        mode: Independent-column or heterogeneous direct-solar transfer policy.
        inputs: Shallow immutable mapping snapshot of scenario inputs. Arrays and
            physical objects are borrowed and must remain stable during use.
        plume_amplitude: Nonnegative dimensionless plume mass multiplier.
        reflectance_offset: Additive dimensionless Lambertian reflectance offset.

    The tangent's final axis is (plume_amplitude, reflectance_offset). Geometry,
    background, frame law and cross sections stay fixed in both operations.
    """

    mode: Literal["independent_column", "heterogeneous_direct_solar"]
    inputs: Mapping[str, Any]
    plume_amplitude: float
    reflectance_offset: float
    _background_columns: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate the supported scenario and prepare shared background columns."""
        mode, inputs = self.mode, self.inputs
        plume_amplitude, reflectance_offset = self.plume_amplitude, self.reflectance_offset
        if mode not in ("independent_column", "heterogeneous_direct_solar"):
            raise ValueError("unsupported scene transfer mode")
        if mode == "independent_column" and np.any(inputs["slopes"] != 0):
            raise ValueError("independent columns require horizontal terrain")
        if not np.isfinite(plume_amplitude) or plume_amplitude < 0:
            raise ValueError("plume amplitude must be finite and nonnegative")
        if not np.isfinite(inputs["molar_mass_kg_mol"]) or inputs["molar_mass_kg_mol"] <= 0:
            raise ValueError("positive molecular mass required")
        require_time_support(np.empty(0), inputs["time_interval"])
        low, high = inputs["spectral_interval"]
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("positive ordered spectral support required")
        contrast_bound = abs(inputs["reflectance_contrast"]) * max(
            abs(1 + (low - 4050) / 200), abs(1 + (high - 4050) / 200)
        )
        if (
            not np.isfinite([reflectance_offset, contrast_bound]).all()
            or 0.25 + reflectance_offset - contrast_bound < 0
            or 0.25 + reflectance_offset + contrast_bound > 1
        ):
            raise ValueError(
                "reflectance must stay within [0,1] throughout spectral and spatial support"
            )
        background = inputs["background"]
        if len(background.absorbers) != 1 or background.absorbers[0].species == "H2O":
            raise ValueError("example requires one non-water trace absorber")
        sun = inputs["sun_direction"]
        if (
            sun.shape != (3,)
            or not np.isfinite(sun).all()
            or not np.isclose(np.linalg.norm(sun), 1)
        ):
            raise ValueError("solar direction must be a finite unit vector")
        if sun[2] < 0.2:
            raise ValueError("solar path must leave the horizontal atmosphere")

        object.__setattr__(self, "inputs", MappingProxyType(dict(inputs)))
        object.__setattr__(
            self,
            "_background_columns",
            np.column_stack([gas.columns_molecules_m2 for gas in background.absorbers]),
        )

    def radiance(self, xy, view, times, spectral: SpectralGrid):
        """Return (N,K) radiance for flat metre, view, second and cm-1 queries."""
        if spectral.coordinate != "wavenumber_cm1":
            raise ValueError("synthetic scene requires wavenumber_cm1 queries")
        points = spectral.points
        return surface_radiance(
            xy,
            view,
            mode=self.mode,
            columns=self._columns,
            **self._physical_inputs(xy, view, times, points),
        )

    def radiance_tangent(self, xy, view, times, spectral: SpectralGrid):
        """Return radiance and (N,K,2) derivatives in the declared parameter order."""
        if spectral.coordinate != "wavenumber_cm1":
            raise ValueError("synthetic scene requires wavenumber_cm1 queries")
        points = spectral.points
        physical = self._physical_inputs(xy, view, times, points)
        dr = np.zeros((len(times), len(points), 2))
        dr[..., 1] = 1
        return surface_radiance_tangent(
            xy,
            view,
            mode=self.mode,
            columns=self._columns_tangent,
            reflectance_tangent=dr,
            **physical,
        )

    @overload
    def _columns(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        layers: NDArray[np.int64],
        *,
        scale_tangent: None = None,
    ) -> NDArray[np.float64]: ...

    @overload
    def _columns(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        layers: NDArray[np.int64],
        *,
        scale_tangent: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def _columns(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        layers: NDArray[np.int64],
        *,
        scale_tangent: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        return puff_molecular_columns(
            starts,
            ends,
            layers,
            height_edges_m=self.inputs["background"].height_boundaries_m,
            background_columns_molecules_m2=self._background_columns,
            plumes=(GaussianPuffField(self.inputs["plume"], "free"),),
            species_indices=np.array([0], dtype=np.int64),
            molar_masses_kg_mol=np.array([self.inputs["molar_mass_kg_mol"]]),
            scales=np.array([self.plume_amplitude]),
            scale_tangent=scale_tangent,
        )

    def _columns_tangent(self, starts, ends, layers):
        return self._columns(starts, ends, layers, scale_tangent=np.array([[1.0, 0.0]]))

    def _physical_inputs(self, xy, view, times, points):
        inputs = self.inputs
        low, high = inputs["spectral_interval"]
        low_xy, high_xy = inputs["horizontal_interval"]
        if times.ndim != 1 or xy.shape != (len(times), 2) or view.shape != (len(times), 3):
            raise ValueError("flat query axes must match (N,2), (N,3), (N,)")
        require_time_support(times, inputs["time_interval"])
        if (
            points.ndim != 1
            or len(points) == 0
            or not np.isfinite(points).all()
            or np.any(np.diff(points) <= 0)
            or points[0] < low
            or points[-1] > high
        ):
            raise ValueError("unsupported spectral nodes")
        # This fixture's support certificate assumes this narrower viewing cone.
        if np.any(view[:, 2] < 0.8):
            raise ValueError("unsupported viewing direction")
        reflectance = (
            0.25
            + self.reflectance_offset
            + inputs["reflectance_contrast"]
            * np.tanh(xy[:, 0, None] / 150)
            * (1 + (points - 4050) / 200)
        )
        return dict(
            height_m=inputs["height_m"],
            slopes=inputs["slopes"],
            height_edges_m=inputs["background"].height_boundaries_m,
            solar_direction=inputs["sun_direction"],
            cross_sections_m2=np.broadcast_to(
                synthetic_cross_section(points), (len(self._background_columns), 1, len(points))
            ),
            reflectance=reflectance,
            beam_normal_irradiance=np.full(reflectance.shape, 0.02),
            minimum_upward_cosine=0.2,
            horizontal_bounds_m=(low_xy, high_xy, low_xy, high_xy),
        )


def make_callbacks(*, mode, inputs, plume_amplitude, reflectance_offset):
    """Return bound radiance methods for callers using the original tuple interface.

    The shared SyntheticRadianceModel owns preparation and parameter consistency.
    Scenario arrays remain borrowed and stable, as with the model constructor.
    """
    model = SyntheticRadianceModel(mode, inputs, plume_amplitude, reflectance_offset)
    return model.radiance, model.radiance_tangent


def synthetic_cross_section(points):
    """Return fixed illustrative CH4-labelled cross sections, m2/molecule."""
    return 2e-25 * (0.2 + np.exp(-(((points - 4050) / 18) ** 2)))


def plume_excluded_mass_bound(inputs):
    """Bound prescribed Gaussian mass outside terrain/top/horizontal support, in kg.

    A union bound sums exact one-dimensional Gaussian half-space probabilities.
    Intersecting excluded regions can be counted twice, so this is a conservative
    upper bound. No mass renormalization or terrain-aware transport is applied.
    """
    plume = inputs["plume"]
    terrain = np.array([-inputs["slopes"][0], -inputs["slopes"][1], 1.0])
    sigma = np.sqrt(np.einsum("i,nij,j->n", terrain, plume.covariances_m2, terrain))
    probability = ndtr((inputs["height_m"] - plume.centres_m @ terrain) / sigma)
    top = inputs["background"].height_boundaries_m[-1]
    probability += ndtr((plume.centres_m[:, 2] - top) / np.sqrt(plume.covariances_m2[:, 2, 2]))
    low, high = inputs["horizontal_interval"]
    for coordinate in (0, 1):
        sigma = np.sqrt(plume.covariances_m2[:, coordinate, coordinate])
        probability += ndtr((low - plume.centres_m[:, coordinate]) / sigma)
        probability += ndtr((plume.centres_m[:, coordinate] - high) / sigma)
    return float(np.dot(plume.masses_kg, np.minimum(probability, 1)))


def make_acquisition(inputs, *, angular_order=5, time_order=2):
    """Build a rotating-chart synthetic camera and matching terrain projection.

    The local sensor translates at 800 m/s, at height 5000 m, with fixed oblique
    pointing. The analytic frame rotation is synthetic, without Earth/EOP claims.
    The coverage certificate requires atmospheric height edges within [0,1500] m
    and terrain inside that atmosphere, with slope norm at most 0.1.
    """
    detector = regular_detector(
        (2, 2),
        pitch_m=0.002,
        active_size_m=0.002,
        centre_xy_m=(0, 0),
        quantum_efficiency=0.7,
        dark_electrons_per_s=0,
        read_noise_electrons=0,
        full_well_electrons=1e12,
        electrons_per_dn=1,
        bias_dn=0,
        nonlinearity=0,
        adc_bits=24,
    )
    camera = filter_camera(
        detector,
        [
            filter_strip(
                i,
                i + 1,
                band=f"b{i}",
                transmission=lambda wavelength, i=i: (
                    0.5 + (1 if i else -1) * (0.01 / wavelength - 4050) / 250
                ),
            )
            for i in range(2)
        ],
        focal_length_m=0.3,
        f_number=4,
        spatial=GaussianPSF(0.0003),
        omitted_psf_mass=1e-6,
    )
    angle = -0.3
    cy, sy = np.cos(angle), np.sin(angle)
    pointing = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.diag([-1.0, 1.0, -1.0])
    origin_gcrs = np.array([6.4e6, 0.0, 0.0])

    def trajectory(times):
        rotation = chart_rotation(times, inputs["angular_rate_rad_s"])
        local = np.column_stack(
            (800 * times - np.tan(0.3) * (5000 - inputs["height_m"]), times * 0, times * 0 + 5000)
        )
        local_velocity = np.broadcast_to([800.0, 0.0, 0.0], local.shape).copy()
        local_velocity += np.cross(np.array([0.0, 0.0, inputs["angular_rate_rad_s"]]), local)
        return origin_gcrs + np.einsum("...ij,...j->...i", rotation, local), np.einsum(
            "...ij,...j->...i", rotation, local_velocity
        )

    def platform(times):
        return chart_rotation(times, inputs["angular_rate_rad_s"]) @ pointing

    def instrument(times):
        return np.broadcast_to(np.eye(3), (len(times), 3, 3))

    def projection(origins, directions, times):
        require_time_support(times.reshape(-1), inputs["time_interval"])
        rotation = chart_rotation(times, inputs["angular_rate_rad_s"])
        local = np.einsum("...ji,...j->...i", rotation, origins - origin_gcrs)
        rays = np.einsum("...ji,...j->...i", rotation, directions)
        hits, points = intersect_affine_terrain(
            local,
            rays,
            height_m=inputs["height_m"],
            slopes=inputs["slopes"],
            minimum_incidence_cosine=0.5,
        )
        return points[..., :2], -rays, hits

    exposures = frame_exposure_bounds(start_s=0, frame_count=2, frame_rate_hz=10, exposure_s=0.04)
    grid = camera_acquisition_grid(
        camera,
        raster_read_schedule(exposures, (2, 2), port=camera.optics.port),
        epoch=inputs["epoch"],
        validity_s=inputs["time_interval"],
        trajectory=trajectory,
        platform_attitude=platform,
        instrument_attitude=instrument,
        surface=projection,
        motion_bound_m_s=1600,
        angular_bound_m=20000,
        time_order=time_order,
        angular_order=angular_order,
    )

    # Certificate applies only to this stable grid from the known analytic laws.
    def coverage(current, spectral):
        result = np.zeros(len(current.read_bands), dtype=bool)
        if (
            current is not grid
            or current.rays.epoch != inputs["epoch"]
            or np.linalg.norm(inputs["slopes"]) > 0.1
        ):
            return result
        if spectral.domain is None or spectral.domain.coordinate != "wavenumber_cm1":
            return result
        lo, hi = inputs["spectral_interval"]
        if any(a < lo or b > hi for a, b in spectral.domain.intervals):
            return result
        # Continuous local view cone: fixed attitude, |u|,|v|<=0.02 gives view_z>0.8.
        fov = current.rays.acquisition.direction_bounds_uv
        if np.any(np.abs(fov) > 0.02):
            return result
        sun = inputs["sun_direction"]
        if sun[2] < 0.2 or not np.isclose(np.linalg.norm(sun), 1):
            return result
        bounds = current.bounds_m
        corners = np.stack(
            (bounds[:, [0, 2]], bounds[:, [0, 3]], bounds[:, [1, 2]], bounds[:, [1, 3]]), axis=1
        )
        surface, normal = affine_surface_points(
            corners, height_m=inputs["height_m"], slopes=inputs["slopes"]
        )
        minimum = surface[..., 2].min(axis=1)
        maximum = surface[..., 2].max(axis=1)
        bottom, top = inputs["background"].height_boundaries_m[[0, -1]]
        # The fixed angular bound assumes terrain between 0 and 1500 m.
        if bottom < 0 or top > 1500:
            return result
        # Straight rays remain above affine terrain if n dot d > 0, globally.
        if sun @ normal <= 0 or 0.8 * normal[2] - 0.6 * np.linalg.norm(normal[:2]) <= 0:
            return result
        displacement = (top - minimum) * max(0.75, np.max(np.abs(sun[:2])) / sun[2])
        xylo, xyhi = inputs["horizontal_interval"]
        times = current.schedule.reads.exposure_bounds_s
        tlo, thi = inputs["time_interval"]
        return (
            (minimum >= bottom)
            & (maximum < top)
            & (top < 5000)
            & (bounds[:, 0] - displacement >= xylo)
            & (bounds[:, 1] + displacement <= xyhi)
            & (bounds[:, 2] - displacement >= xylo)
            & (bounds[:, 3] + displacement <= xyhi)
            & (times[:, 0] >= tlo)
            & (times[:, 1] <= thi)
        )

    return camera, grid, coverage


def detector_prediction(
    *,
    mode,
    inputs,
    plume_amplitude=1.0,
    reflectance_offset=0.0,
    spectral_count=33,
    angular_order=21,
    time_order=2,
    batch_size=4,
    spectral_block_size=None,
):
    """Return native expected images and an electron Jacobian with final parameter axis."""
    camera, grid, coverage = make_acquisition(
        inputs, angular_order=angular_order, time_order=time_order
    )
    low, high = inputs["spectral_interval"]
    spectral = wavenumber_grid(
        np.linspace(low, high, spectral_count),
        domain=SpectralDomain("wavenumber_cm1", ((low, high),)),
    )
    model = SyntheticRadianceModel(
        mode=mode,
        inputs=inputs,
        plume_amplitude=plume_amplitude,
        reflectance_offset=reflectance_offset,
    )
    signal, jacobian = frame_electron_tangents(
        camera,
        grid,
        spectral,
        model.radiance_tangent,
        coverage,
        read_batch_size=batch_size,
        spectral_block_size=spectral_block_size,
    )
    return signal.total.reshape(grid.image_shape), jacobian.reshape((*grid.image_shape, 2))


if __name__ == "__main__":
    print(
        "Excluded prescribed plume mass upper bound [kg]",
        plume_excluded_mass_bound(synthetic_inputs()),
    )
    for selected_mode in ("independent_column", "heterogeneous_direct_solar"):
        images, jacobian = detector_prediction(mode=selected_mode, inputs=synthetic_inputs())
        print(selected_mode, "native image shape", images.shape)
        print("expected electrons", images)
        print("plume/reflectance Jacobian", jacobian)
