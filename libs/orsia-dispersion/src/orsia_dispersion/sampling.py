"""Sample stationary plumes, puff histories and particle flights at observation times."""

from collections.abc import Iterator

import numpy as np

from orsia_dispersion.particles import (
    BoundedParticleDispersion,
    ParticleMassField,
    simulate_particles,
)
from orsia_dispersion.puffs.simulation import (
    GaussianPuffDispersion,
    GaussianPuffField,
    simulate_puffs,
)
from orsia_dispersion.stationary import StationaryGaussianPlume
from orsia_dispersion.transport import FloatArray

type MassField = StationaryGaussianPlume | GaussianPuffField | ParticleMassField
type DispersionModel = StationaryGaussianPlume | GaussianPuffDispersion | BoundedParticleDispersion


def sample_fields(
    model: DispersionModel, times_s: FloatArray, *, rng: np.random.Generator | None = None
) -> Iterator[MassField]:
    """Yield mass fields in chronological order through one model-independent path.

    Args:
        model: Stationary plume, scheduled puffs or bounded particle dispersion.
        times_s: Finite strictly increasing observation times (time,), seconds
            in the same chart as the emission history. Empty requests are valid.
        rng: Explicit generator required for particles. Gaussian models consume
            no draws. Sampling calls require separate generators for concurrent use.

    Yields:
        One field per requested time. The stationary immutable declaration can
        be shared. Transient outputs own independent state and diagnostics;
        particle buffers are read-only. Advancing or mutating a yielded puff
        cannot change previous or future outputs.

    Raises:
        ValueError: Times, emissions, forcing or numerical parameters are invalid,
            or a particle model is sampled without an explicit generator.
        RuntimeError: Transient transport exhausts a numerical or population
            budget, or particle clocks/positions exceed float64 resolution/range.
    """
    times = np.array(times_s, dtype=np.float64, copy=True)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be a finite strictly increasing vector")
    if isinstance(model, StationaryGaussianPlume):
        for _ in times:
            yield model
    elif isinstance(model, BoundedParticleDispersion):
        if rng is None:
            raise ValueError("particle sampling requires an explicit rng")
        yield from simulate_particles(model, times, rng=rng)
    else:
        for state, diagnostics in simulate_puffs(
            model.releases,
            times,
            model.wind,
            model.diffusivity,
            boundary=model.boundary,
            rtol=model.rtol,
            atol_position_m=model.atol_position_m,
            atol_covariance_m2=model.atol_covariance_m2,
            min_step_s=model.min_step_s,
            max_step_s=model.max_step_s,
            max_steps=model.max_steps,
            envelope_sigma=model.envelope_sigma,
            wind_tolerance_m_s=model.wind_tolerance_m_s,
            diffusivity_tolerance_m2_s=model.diffusivity_tolerance_m2_s,
            split_fraction=model.split_fraction,
            max_puffs=model.max_puffs,
            max_split_rounds=model.max_split_rounds,
            discontinuities_s=model.discontinuities_s,
            kernel_cutoff_radius=model.kernel_cutoff_radius,
        ):
            yield GaussianPuffField(state, model.boundary, diagnostics, model.kernel_cutoff_radius)
