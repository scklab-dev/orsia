"""Define the WGS84 reference ellipsoid for terrestrial surface geometry.

The defining semimajor axis and inverse flattening follow NGA's WGS 84
Defining Parameters table: https://earth-info.nga.mil/index.php?action=wgs84&dir=wgs84
GeographicLib's Constants::WGS84_a and Constants::WGS84_f independently
implement these values: https://github.com/geographiclib/geographiclib/blob/main/include/GeographicLib/Constants.hpp
The remaining values derive from those definitions. Lengths are metres;
flattening and inverse flattening are dimensionless. The immutable semiaxes
tuple follows ITRF Cartesian x, y, z order, with z along the polar axis.
This reference surface has ellipsoidal height zero and excludes terrain.
"""

from typing import Final

WGS84_SEMIMAJOR_AXIS_M: Final = 6378137.0
WGS84_INVERSE_FLATTENING: Final = 298.257223563
WGS84_FLATTENING: Final = 1.0 / WGS84_INVERSE_FLATTENING
WGS84_SEMIMINOR_AXIS_M: Final = WGS84_SEMIMAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
WGS84_SEMIAXES_M: Final[tuple[float, float, float]] = (
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)
