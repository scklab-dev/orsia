"""Project supplied trajectories and independent pointing laws onto surfaces."""

from orsia_geometry.attitude import (
    attitude_interval_speeds,
    axis_rotations,
    instrument_tracking,
    platform_tracking,
    radial_nadir,
)
from orsia_geometry.frames import (
    LocalCartesianFrame,
    frame_rotations,
    same_local_frame,
    transform_rays,
    transform_states,
)
from orsia_geometry.grids import LayeredCells, PointLattice, RectangularCells, resample_point_field
from orsia_geometry.los import AttitudeLaw, Trajectory, compute_lines_of_sight
from orsia_geometry.orbits import NumericalPropagator, SGP4MeanElements, SGP4Propagator
from orsia_geometry.rays import compose_rays, validate_rotations
from orsia_geometry.surfaces import (
    WGS84SurfaceProjection,
    ellipsoid_intersections,
    project_local_east_north,
    sphere_intersections,
    surface_visibility,
)
from orsia_geometry.terrain import (
    affine_surface_points,
    intersect_affine_terrain,
    intersect_terrain,
    segment_layer_paths,
    surface_points,
)

__all__ = [
    "LocalCartesianFrame",
    "same_local_frame",
    "PointLattice",
    "resample_point_field",
    "RectangularCells",
    "LayeredCells",
    "NumericalPropagator",
    "SGP4Propagator",
    "WGS84SurfaceProjection",
    "affine_surface_points",
    "intersect_affine_terrain",
    "intersect_terrain",
    "segment_layer_paths",
    "surface_points",
    "SGP4MeanElements",
    "AttitudeLaw",
    "Trajectory",
    "axis_rotations",
    "frame_rotations",
    "transform_states",
    "transform_rays",
    "compose_rays",
    "ellipsoid_intersections",
    "instrument_tracking",
    "compute_lines_of_sight",
    "platform_tracking",
    "attitude_interval_speeds",
    "project_local_east_north",
    "radial_nadir",
    "sphere_intersections",
    "surface_visibility",
    "validate_rotations",
]
