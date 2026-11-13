"""Transport discrete mass with homogeneous bounded random flights in a local ENU chart.

Releases and turbulent renewal clocks define trajectories independently of output
requests. The module owns emission declarations, snapshots and transport policy.
"""

import heapq
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from orsia_dispersion.emissions import release_intervals
from orsia_dispersion.forcing import Boundary, FloatArray, Wind, validate_tensors


@dataclass(frozen=True)
class ParticleReleases:
    """Own the emission time, mass and position of each point particle.

    Attributes:
        release_times_s: Finite nondecreasing release times (particle,), seconds.
        masses_kg: Finite positive masses (particle,), kg, with finite total.
        positions_m: Finite east/north/altitude positions (particle,3), m.
            Construction copies arrays into independent read-only float64 storage.
            Empty histories are valid. Each entry represents one mass particle.
    """

    release_times_s: FloatArray
    masses_kg: FloatArray
    positions_m: FloatArray

    def __post_init__(self) -> None:
        """Snapshot and validate the declared release history.

        Raises:
            ValueError: Shapes, times, coordinates or masses are invalid.
        """
        _snapshot(self, ("release_times_s", "masses_kg", "positions_m"))
        _validate_population(self.release_times_s, self.masses_kg, self.positions_m)
        if np.any(np.diff(self.release_times_s) < 0):
            raise ValueError("particle release times must be nondecreasing")


@dataclass(frozen=True)
class MassParticles:
    """Own a discrete mass measure and turbulent velocities at one instant.

    Attributes:
        time_s: Finite observation time, seconds on the release clock.
        release_times_s: Original release times (particle,), at or before time_s.
        masses_kg: Positive masses (particle,), kg, with finite total.
        positions_m: East/north/altitude positions (particle,3), m.
        turbulent_velocities_m_s: Turbulent velocities (particle,3), m/s, excluding
            the mean wind. Reflection gives the physical, right-continuous velocity.
            All arrays are independent read-only float64 snapshots. Particle order
            follows the original release order; populations may be empty.
    """

    time_s: float
    release_times_s: FloatArray
    masses_kg: FloatArray
    positions_m: FloatArray
    turbulent_velocities_m_s: FloatArray

    def __post_init__(self) -> None:
        """Snapshot and validate the population and its common time.

        Raises:
            ValueError: Population, velocities or time are invalid.
        """
        _snapshot(self, ("release_times_s", "masses_kg", "positions_m", "turbulent_velocities_m_s"))
        _validate_population(self.release_times_s, self.masses_kg, self.positions_m)
        if not np.isfinite(self.time_s) or np.any(self.release_times_s > self.time_s):
            raise ValueError("particle time must be finite and at or after all releases")
        if (
            self.turbulent_velocities_m_s.shape != self.positions_m.shape
            or not np.isfinite(self.turbulent_velocities_m_s).all()
        ):
            raise ValueError("finite turbulent velocities with shape (particle,3) required")


@dataclass(frozen=True)
class ParticleDiagnostics:
    """Describe the cumulative mass balance and numerical effort at one output.

    Attributes:
        released_mass_kg: Total declared mass released by the output time, kg.
        represented_mass_kg: Total mass retained by the population, kg.
        mass_residual_kg: Represented minus released mass, kg; excludes no scene tails.
        transport_segments: Number of ballistic/Euler segments prepared so far.
        renewals: Number of turbulent velocity renewals after initial releases.
        turbulent_speed_bound_m_s: Homogeneous bound sqrt(3*lambda_max(R)), m/s.
    """

    released_mass_kg: float
    represented_mass_kg: float
    mass_residual_kg: float
    transport_segments: int
    renewals: int
    turbulent_speed_bound_m_s: float


@dataclass(frozen=True)
class ParticleMassField:
    """Bind a point-mass snapshot to its physical boundary and diagnostics.

    Attributes:
        state: Borrowed MassParticles record with owned read-only buffers.
        boundary: Free space or specular reflection at chart altitude zero.
        diagnostics: Optional mass balance at the same output time.
            Spatial integration deposits points directly, without smoothing.
    """

    state: MassParticles
    boundary: Boundary
    diagnostics: ParticleDiagnostics | None = None

    def __post_init__(self) -> None:
        """Validate the physical half-space.

        Raises:
            ValueError: Boundary is unknown or reflected points lie below zero.
        """
        if self.boundary not in ("free", "reflecting"):
            raise ValueError("boundary must be 'free' or 'reflecting'")
        if self.boundary == "reflecting" and np.any(self.state.positions_m[:, 2] < 0):
            raise ValueError("reflecting particle positions require z>=0")


@dataclass(frozen=True, kw_only=True)
class BoundedParticleDispersion:
    """Declare finite-speed random-flight transport and its numerical budgets.

    Attributes:
        releases: Point emissions with read-only owned arrays.
        wind: Pure existing Wind callback returning velocity and gradient. Only
            velocity is used. Arguments are independent buffers, in ENU metres and seconds.
        wind_speed_bound_m_s: Finite nonnegative global mean-wind speed bound, m/s,
            supplied by the caller. Sample checks cannot prove this global bound.
        turbulent_covariance_m2_s2: Homogeneous symmetric PSD velocity covariance
            R (3,3), m2/s2, copied to read-only float64 storage. Turbulent velocities
            are sqrt(3)*sqrt(R)*n for uniform unit directions n.
        correlation_time_s: Positive Poisson mean renewal time tau, seconds.
            Stationary homogeneous velocity covariance is R; its integral is tau*R.
        boundary: Free space or specular reflection at z=0. Reflection requires
            vertically separable R and identically zero vertical mean wind.
        max_step_s: Positive maximum Euler wind-hold duration, seconds. Uniform
            wind gives exact ballistic segments between renewals and reflections.
            Variable wind requires application-specific step convergence checks.
        max_steps: Positive total segment-preparation budget per sampling call.
        max_particles: Positive budget for the entire declared release population.
        discontinuities_s: Strictly increasing finite wind discontinuities, seconds.
            Turbulence is constant in space and time. Stratification, convection,
            chemistry and deposition are outside this closure's physical scope.
    """

    releases: ParticleReleases
    wind: Wind
    wind_speed_bound_m_s: float
    turbulent_covariance_m2_s2: FloatArray
    correlation_time_s: float
    boundary: Boundary
    max_step_s: float
    max_steps: int
    max_particles: int
    discontinuities_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot physical parameters and validate numerical budgets.

        Raises:
            ValueError: Bounds, covariance, reflection or budgets are invalid.
        """
        covariance = np.array(self.turbulent_covariance_m2_s2, dtype=np.float64, copy=True)
        validate_tensors(covariance[None], positive=False)
        covariance = covariance / 2 + covariance.T / 2
        covariance.setflags(write=False)
        object.__setattr__(self, "turbulent_covariance_m2_s2", covariance)
        r = self.releases
        object.__setattr__(
            self, "releases", ParticleReleases(r.release_times_s, r.masses_kg, r.positions_m)
        )
        scalars = (self.wind_speed_bound_m_s, self.correlation_time_s, self.max_step_s)
        if (
            not np.isfinite(scalars).all()
            or self.wind_speed_bound_m_s < 0
            or self.correlation_time_s <= 0
            or self.max_step_s <= 0
            or self.max_steps < 1
            or self.max_particles < 1
        ):
            raise ValueError("finite wind bound, positive time scales and budgets required")
        breaks = self.discontinuities_s
        if not np.isfinite(breaks).all() or np.any(np.diff(breaks) <= 0):
            raise ValueError("discontinuities_s must be finite and strictly increasing")
        if self.boundary not in ("free", "reflecting"):
            raise ValueError("boundary must be 'free' or 'reflecting'")
        if self.boundary == "reflecting" and (
            np.any(r.positions_m[:, 2] < 0) or np.any(covariance[:2, 2] != 0)
        ):
            raise ValueError(
                "reflection requires z>=0 releases and vertically separable turbulence"
            )
        bound = np.sqrt(3.0) * np.sqrt(max(0.0, float(np.linalg.eigvalsh(covariance)[-1])))
        if not np.isfinite(self.wind_speed_bound_m_s + bound):
            raise ValueError("combined wind and turbulence speed bound must be finite")


def discretize_particle_emissions(
    edges_s: FloatArray,
    rates_kg_s: FloatArray,
    positions_m: FloatArray,
    *,
    particles_per_release: int,
    partition_times_s: FloatArray | None = None,
    stop_time_s: float | None = None,
) -> ParticleReleases:
    """Represent integrated source intervals by equal-mass midpoint particles.

    Args:
        edges_s: Increasing finite source interval edges (n+1,), seconds.
        rates_kg_s: Nonnegative constant rates (n,), kg/s.
        positions_m: Source ENU positions (n,3), m, including zero-rate intervals.
        particles_per_release: Positive number of particles per retained subinterval.
        partition_times_s: Optional increasing times splitting source intervals.
        stop_time_s: Optional finite final inclusion time, seconds.

    Returns:
        Independent read-only particles. Integrated interval mass is preserved
        to floating-point summation precision; emission time uses midpoint quadrature.
        Select source partitioning independently of output requests for identical paths.

    Raises:
        ValueError: History, positions, population count or divided masses are invalid.
    """
    if particles_per_release < 1:
        raise ValueError("particles_per_release must be positive")
    times, masses, indices = release_intervals(edges_s, rates_kg_s, partition_times_s, stop_time_s)
    source = ParticleReleases(np.asarray(edges_s)[:-1], np.ones(len(edges_s) - 1), positions_m)
    return ParticleReleases(
        np.repeat(times, particles_per_release),
        np.repeat(masses / particles_per_release, particles_per_release),
        np.repeat(source.positions_m[indices], particles_per_release, axis=0),
    )


def simulate_particles(
    model: BoundedParticleDispersion, output_times_s: FloatArray, *, rng: np.random.Generator
) -> Iterator[ParticleMassField]:
    """Yield independent mass snapshots along output-independent random flights.

    Args:
        model: Emission history, homogeneous closure and explicit numerical policy.
        output_times_s: Finite strictly increasing vector, seconds on the release clock.
        rng: Caller-owned generator, consumed only by release and renewal events.
            Use a fresh identically seeded generator to reproduce a sampling call.
            Concurrent calls require separate generators and pure wind callbacks.

    Yields:
        Independent read-only states and cumulative diagnostics, including empty
        populations before emission. Inserting output times changes neither paths
        nor random draws at existing times. No particle is removed outside a scene.

    Raises:
        ValueError: Times or sampled velocities violate their declared contracts.
        RuntimeError: Population/segment budget is exhausted, an event cannot
            advance in float64, or a propagated position becomes nonfinite.
    """
    times = np.array(output_times_s, dtype=np.float64, copy=True)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("output_times_s must be finite and strictly increasing")
    releases = model.releases
    n = len(releases.masses_kg)
    if n > model.max_particles:
        raise RuntimeError("particle population budget exceeded")
    values, vectors = np.linalg.eigh(model.turbulent_covariance_m2_s2)
    root = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    speed_bound = float(np.sqrt(3.0) * np.sqrt(max(0.0, values[-1])))
    positions = releases.positions_m.copy()
    turbulent = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    last = releases.release_times_s.copy()
    renewal = np.full(n, np.inf)
    active = np.zeros(n, dtype=bool)
    events = [(float(t), i) for i, t in enumerate(releases.release_times_s)]
    heapq.heapify(events)
    steps = renewals = 0
    for output in times:
        while events and events[0][0] <= output:
            time, i = heapq.heappop(events)
            if steps >= model.max_steps:
                raise RuntimeError("particle transport segment budget exhausted")
            if active[i]:
                positions[i] += (time - last[i]) * velocity[i]
                if model.boundary == "reflecting":
                    _reflect(positions[i : i + 1], turbulent[i : i + 1], velocity[i : i + 1])
            else:
                active[i] = True
            if time >= renewal[i] or time == releases.release_times_s[i]:
                if time >= renewal[i]:
                    renewals += 1
                turbulent[i] = _draw_velocity(root, rng)
                renewal[i] = time + float(rng.exponential(model.correlation_time_s))
                if not np.isfinite(renewal[i]) or renewal[i] <= time:
                    raise RuntimeError("particle renewal clock cannot advance in float64")
            last[i] = time
            velocity[i] = _wind(model, positions[i : i + 1], time) + turbulent[i]
            stop = min(time + model.max_step_s, renewal[i])
            k = np.searchsorted(model.discontinuities_s, time, side="right")
            if k < len(model.discontinuities_s):
                stop = min(stop, model.discontinuities_s[k])
            if not np.isfinite(stop) or stop <= time:
                raise RuntimeError("particle transport clock cannot advance in float64")
            if not np.isfinite(positions[i]).all():
                raise RuntimeError("particle position became nonfinite")
            heapq.heappush(events, (float(stop), i))
            steps += 1
        observed = positions[active] + (output - last[active, None]) * velocity[active]
        observed_turbulent = turbulent[active].copy()
        if model.boundary == "reflecting":
            _reflect(observed, observed_turbulent, velocity[active])
        if not np.isfinite(observed).all():
            raise RuntimeError("particle position became nonfinite")
        state = MassParticles(
            float(output),
            releases.release_times_s[active],
            releases.masses_kg[active],
            observed,
            observed_turbulent,
        )
        released = float(releases.masses_kg[releases.release_times_s <= output].sum())
        represented = float(state.masses_kg.sum())
        yield ParticleMassField(
            state,
            model.boundary,
            ParticleDiagnostics(
                released,
                represented,
                represented - released,
                steps,
                renewals,
                speed_bound,
            ),
        )


def _snapshot(owner: ParticleReleases | MassParticles, names: tuple[str, ...]) -> None:
    for name in names:
        array = np.array(getattr(owner, name), dtype=np.float64, copy=True)
        array.setflags(write=False)
        object.__setattr__(owner, name, array)


def _validate_population(times: FloatArray, masses: FloatArray, positions: FloatArray) -> None:
    n = masses.size
    if (
        times.shape != (n,)
        or masses.shape != (n,)
        or positions.shape != (n, 3)
        or not all(np.isfinite(a).all() for a in (times, masses, positions))
    ):
        raise ValueError("finite particle arrays with shapes (n,), (n,), (n,3) required")
    if np.any(masses <= 0) or not np.isfinite(masses.sum()):
        raise ValueError("particle masses must be positive with finite total")


def _draw_velocity(root: FloatArray, rng: np.random.Generator) -> FloatArray:
    z = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2 * np.pi)
    radius = np.sqrt(max(0.0, 1 - z * z))
    direction = np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), z])
    return np.sqrt(3.0) * (root @ direction)


def _wind(model: BoundedParticleDispersion, positions: FloatArray, time: float) -> FloatArray:
    value, _ = model.wind(positions.copy(), time)
    wind = np.asarray(value, dtype=np.float64)
    if wind.shape != (1, 3) or not np.isfinite(wind).all():
        raise ValueError("particle wind must return finite velocities with shape (n,3)")
    speed = np.hypot.reduce(wind[0])
    if speed > model.wind_speed_bound_m_s * (1 + 16 * np.finfo(float).eps):
        raise ValueError("sampled wind exceeds wind_speed_bound_m_s")
    if model.boundary == "reflecting" and wind[0, 2] != 0:
        raise ValueError("reflecting particle wind requires zero vertical velocity")
    return wind[0]


def _reflect(positions: FloatArray, turbulent: FloatArray, velocity: FloatArray) -> None:
    reflected = (positions[:, 2] < 0) | ((positions[:, 2] == 0) & (velocity[:, 2] < 0))
    positions[:, 2] = np.abs(positions[:, 2])
    turbulent[reflected, 2] *= -1
