"""Describe SGP4 mean elements and construct reusable GCRS orbit evaluators."""

import math
from dataclasses import dataclass
from datetime import datetime
from math import degrees, tau

import numpy as np
from numpy.typing import NDArray

from . import time_scales as _time_scales
from .frames import require_time_support, require_utc_epoch, transform_states
from .los import Trajectory


@dataclass(frozen=True, slots=True)
class SGP4MeanElements:
    """Retain one immutable scalar Earth-centred TEME/UTC SGP4 mean-element set.

    These are model-specific mean elements for classic SGP4, with WGS72/AFSPC
    propagation in Geometry. They are unsuitable as osculating Kepler elements.
    All numeric values must be finite. Epoch precision is one microsecond.

    Attributes:
        norad_id: Positive numeric NORAD catalogue identifier, at most 2147483647.
        epoch: Timezone-aware UTC element epoch.
        eccentricity: Dimensionless eccentricity in [0, 1).
        inclination_rad: Inclination in [0, pi] radians.
        raan_rad: Right ascension of the ascending node in [0, 2*pi) radians.
        argument_of_perigee_rad: Argument of perigee in [0, 2*pi) radians.
        mean_anomaly_rad: Mean anomaly in [0, 2*pi) radians.
        mean_motion_rad_s: Positive mean motion in radians per SI second.
        bstar_per_earth_radius: Signed B* drag parameter in inverse Earth radii.
        mean_motion_dot_rad_s2: Full first derivative of mean motion in rad/s².
            GP OMM/TLE stores half this derivative in rev/day²: multiply the
            source value by 2 * 2*pi / 86400².
        mean_motion_ddot_rad_s3: Full second derivative of mean motion in rad/s³.
            GP OMM/TLE stores one sixth in rev/day³: multiply the source value
            by 6 * 2*pi / 86400³. Classic SGP4 retains but does not use these
            two derivatives in its state calculation.
        object_name: Optional descriptive name; acquisition provenance is separate.

    Raises:
        ValueError: Identity, epoch or numerical values violate the supported domain.
    """

    norad_id: int
    epoch: datetime
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    argument_of_perigee_rad: float
    mean_anomaly_rad: float
    mean_motion_rad_s: float
    bstar_per_earth_radius: float
    mean_motion_dot_rad_s2: float = 0.0
    mean_motion_ddot_rad_s3: float = 0.0
    object_name: str | None = None

    def __post_init__(self) -> None:
        """Check physical support before values reach an optional backend.

        Raises:
            ValueError: Identity, epoch or numerical support is invalid.
        """
        if not 0 < self.norad_id <= 2147483647:
            raise ValueError("NORAD identity must be positive and fit a signed 32-bit integer")
        require_utc_epoch(self.epoch)
        values = (
            self.eccentricity,
            self.inclination_rad,
            self.raan_rad,
            self.argument_of_perigee_rad,
            self.mean_anomaly_rad,
            self.mean_motion_rad_s,
            self.bstar_per_earth_radius,
            self.mean_motion_dot_rad_s2,
            self.mean_motion_ddot_rad_s3,
        )
        if (
            not all(math.isfinite(v) for v in values)
            or not 0 <= self.eccentricity < 1
            or not 0 <= self.inclination_rad <= math.pi
            or not all(
                0 <= v < math.tau
                for v in (self.raan_rad, self.argument_of_perigee_rad, self.mean_anomaly_rad)
            )
            or self.mean_motion_rad_s <= 0
        ):
            raise ValueError("SGP4 mean elements outside supported finite domains")


def numerical_propagator(
    elements: NDArray[np.float64],
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    gravity_degree: int = 2,
    gravity_order: int = 0,
    sun_gravity: bool = False,
    moon_gravity: bool = False,
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-11,
) -> Trajectory:
    """Numerically propagate one classical osculating Earth orbit into a reusable dense evaluator.

    Args:
        elements: Borrowed (6,) [a,e,i,RAAN,arg_periapsis,true_anomaly] in GCRS,
            with a in metres, angles in radians, 0 <= e < 1 and 0 <= i <= pi.
        epoch: Timezone-aware UTC instant of the osculating elements.
        validity_s: Finite closed nonnegative interval with positive end, SI seconds.
            Integration starts at epoch even when the accepted interval starts later.
        gravity_degree: EGM96 spherical-harmonic degree in [0, 70]; zero is two-body gravity.
        gravity_order: EGM96 spherical-harmonic order in [0, gravity_degree].
        sun_gravity: Include Sun third-body gravity when True.
        moon_gravity: Include Moon third-body gravity when True.
        absolute_tolerance: Finite positive local integration error control in state units:
            metres for position components and metres per second for velocity components.
        relative_tolerance: Finite positive dimensionless local integration error control.
            Tolerances control adaptive integration and supply no global trajectory bound.

    Returns:
        Pure serial-use evaluator accepting finite (times,) elapsed SI seconds and
        returning independent writable float64 (times,3) GCRS positions m and velocities m/s.
        The closure owns the native dense result until released; the element array is
        not retained. Each construction uses fresh settings with EGM96 gravity, RKV98,
        dense interpolation and strict EOP coverage. Tides, relativity, space weather,
        drag, radiation pressure and thrust stay disabled for every scalar choice.
        SatKit may acquire normal public auxiliaries during construction; interpolation
        uses the retained result. Refinement and force-model suitability remain caller
        responsibilities. Concurrent native use is unqualified.

    Raises:
        ValueError: Elements, epoch, interval, gravity bounds or tolerances are invalid.
        RuntimeError: EOP coverage, auxiliary preparation or numerical propagation fails.
        ImportError: The required SatKit installation cannot load.
    """
    if (
        elements.shape != (6,)
        or not np.isfinite(elements).all()
        or elements[0] <= 0
        or not 0 <= elements[1] < 1
        or not 0 <= elements[2] <= np.pi
    ):
        raise ValueError(
            "classical elements require finite [a,e,i,RAAN,argp,nu] in supported domains"
        )
    require_utc_epoch(epoch)
    require_time_support(np.array(validity_s), validity_s)
    if validity_s[0] < 0 or validity_s[1] <= 0:
        raise ValueError("numerical trajectory requires a forward nonnegative interval")
    # SatKit 0.24.1 caps harmonic evaluation at degree 70; reject unsupported
    # truncations explicitly, including order greater than degree.
    if not 0 <= gravity_order <= gravity_degree <= 70:
        raise ValueError("gravity requires 0 <= order <= degree <= 70")
    if (
        not math.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0
    ):
        raise ValueError("integration tolerances must be finite and positive")
    import satkit

    settings = satkit.propsettings(
        gravity_degree=gravity_degree,
        gravity_order=gravity_order,
        gravity_model=satkit.gravmodel.egm96,
        integrator=satkit.integrator.rkv98,
        abs_error=absolute_tolerance,
        rel_error=relative_tolerance,
        use_sun_gravity=sun_gravity,
        use_moon_gravity=moon_gravity,
        use_spaceweather=False,
        tide_model=satkit.tidemodel.none,
        use_relativistic_correction=False,
        require_eop_coverage=True,
        enable_interp=True,
    )
    initial = satkit.kepler(*(float(v) for v in elements)).to_pv()
    begin, end = _time_scales.native_instants(np.array([0.0, validity_s[1]]), epoch=epoch)
    # The backend precomputes force/frame inputs on padded nodes beyond the arc;
    # require_eop_coverage also protects those backend nodes.
    result = satkit.propagate(
        pos=initial[0],
        vel=initial[1],
        begin=begin,
        end=end,
        propsettings=settings,
        satproperties=None,
        output_phi=False,
    )

    def evaluate(times_s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        require_time_support(times_s, validity_s)
        instants = _time_scales.native_instants(times_s, epoch=epoch)
        # SatKit's vector dense interpolator advances through nodes and assumes
        # chronological input. Sort privately, then restore order and repetitions.
        order = np.argsort(times_s, kind="stable")
        ordered_state = result.interp([instants[int(i)] for i in order])
        state = ordered_state[np.argsort(order)]
        if state.shape != (len(times_s), 6) or not np.isfinite(state).all():
            raise RuntimeError("numerical trajectory returned an invalid dense state")
        return state[:, :3].copy(), state[:, 3:].copy()

    return evaluate


def sgp4_propagator(
    elements: SGP4MeanElements,
    *,
    epoch: datetime,
    validity_s: tuple[float, float],
    derivative_step_s: float = 0.5,
) -> Trajectory:
    """Create a reusable GCRS evaluator from scalar SGP4 mean elements.

    Args:
        elements: Immutable Earth-centred TEME/UTC SGP4 mean elements. Geometry
            converts scalar values directly to a private native record without a
            textual TLE round trip. Use Data for local reading and object selection.
        epoch: Timezone-aware UTC reference for evaluator elapsed seconds.
        validity_s: Finite closed caller-accepted prediction interval in SI seconds;
            element availability itself supplies no accepted prediction accuracy.
        derivative_step_s: Central rotation-derivative half-step, at least 1e-4 seconds.

    Returns:
        Serial-use evaluator yielding owned writable float64 GCRS (times,3) positions
        m and velocities m/s. SatKit WGS72/AFSPC SGP4 outputs TEME; full IERS 2010
        rotation and its time derivative map that state into GCRS. Physical elapsed SI
        time follows SatKit across leap seconds, which differs from Vallado UTC-label
        subtraction over a crossing. Inputs and native metadata remain unchanged.
        Normal backend auxiliary search/download and global caches apply at evaluation.

    Raises:
        ValueError: Time support, derivative step, element conventions or SGP4 state is invalid.
        RuntimeError: Auxiliary coverage or backend evaluation fails.
        ImportError: The required SatKit installation cannot load.
    """
    require_utc_epoch(epoch)
    require_time_support(np.array(validity_s), validity_s)
    if not np.isfinite(derivative_step_s) or derivative_step_s < 1e-4:
        raise ValueError("derivative step must be at least 1e-4 seconds")
    import satkit

    native = satkit.TLE.from_omm(
        {
            "NORAD_CAT_ID": elements.norad_id,
            "EPOCH": elements.epoch.isoformat(),
            "ECCENTRICITY": elements.eccentricity,
            "INCLINATION": degrees(elements.inclination_rad),
            "RA_OF_ASC_NODE": degrees(elements.raan_rad),
            "ARG_OF_PERICENTER": degrees(elements.argument_of_perigee_rad),
            "MEAN_ANOMALY": degrees(elements.mean_anomaly_rad),
            "MEAN_MOTION": elements.mean_motion_rad_s / (tau / 86400),
            "BSTAR": elements.bstar_per_earth_radius,
            "MEAN_MOTION_DOT": elements.mean_motion_dot_rad_s2 / (2 * tau / 86400**2),
            "MEAN_MOTION_DDOT": elements.mean_motion_ddot_rad_s3 / (6 * tau / 86400**3),
        }
    )

    def evaluate(times_s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        require_time_support(times_s, validity_s)
        instants = _time_scales.native_instants(times_s, epoch=epoch)
        positions, velocities, errors = satkit.sgp4(
            native,
            instants,
            gravconst=satkit.sgp4_gravconst.wgs72,
            opsmode=satkit.sgp4_opsmode.afspc,
            errflag=True,
        )
        if (
            np.any(errors != 0)
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
        ):
            raise ValueError("SGP4 cannot propagate the supplied mean elements at requested times")
        return transform_states(
            times_s,
            positions,
            velocities,
            epoch=epoch,
            validity_s=validity_s,
            from_frame="TEME",
            to_frame="GCRS",
            derivative_step_s=derivative_step_s,
        )

    return evaluate
