"""Source-resolved expected photoelectrons on physical reads, before detector noise."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .acquisition import Acquisition, ReadSchedule, read_schedule


@dataclass(frozen=True, eq=False)
class ExpectedElectrons:
    """Source-resolved mean photoelectrons for each physical read, before dark/read noise.

    Each named contribution is a finite nonnegative (reads,) writable copy.
    Sources are added only after their respective optical paths; names distinguish
    contributions, not a mandatory provenance framework. ``acquisition`` retains
    the exact supplied Acquisition or ReadSchedule for identity-preserving caller access.
    Already integrated photoelectrons receive no further QE or exposure factor.

    Attributes:
        acquisition: Physical read identities and exposure intervals for this charge.
        contributions: Named, owned mean-electron vectors of shape (reads,).
    """

    acquisition: Acquisition | ReadSchedule
    contributions: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        contributions = {}
        for name, values in self.contributions.items():
            array = np.copy(values)
            if not np.isfinite(array).all():
                raise ValueError("expected-electron contributions must be finite")
            if not name or array.shape != (len(self.acquisition.sample_ids),) or np.any(array < 0):
                raise ValueError(
                    "sources need names and nonnegative counts matching physical reads"
                )
            contributions[name] = array
        if not contributions:
            raise ValueError("at least one expected-electron contribution is required")
        if not np.all(np.isfinite(np.sum(list(contributions.values()), axis=0))):
            raise ValueError("total expected electrons overflow")
        object.__setattr__(self, "contributions", MappingProxyType(contributions))

    @property
    def reads(self) -> ReadSchedule:
        """Expose read identities and timing for charge alignment.

        Returns:
            Geometry-free physical reads, freshly snapshotted for an Acquisition.
        """
        return read_schedule(self.acquisition)

    @property
    def total(self) -> NDArray[np.float64]:
        """Combine named source charge before detector noise.

        Returns:
            Fresh writable (reads,) photoelectron means in electrons.
        """
        return np.sum(list(self.contributions.values()), axis=0)


def combine_source_paths(*paths: ExpectedElectrons) -> ExpectedElectrons:
    """Join independently transported nonnegative source means before Poisson noise.

    Each path owns its radiance support, optical response and throughput; only
    read identities and physical exposure intervals must agree. The first path
    supplies the retained acquisition geometry. Never pass internal emission
    through the external scene response merely to combine detector contributions.

    Args:
        *paths: Source-resolved charge paths with matching read identities and
            exposure intervals, each using distinct source names.

    Returns:
        Combined expected electrons with the first path's acquisition identity
        and copied named contributions.

    Raises:
        ValueError: Paths are empty, physical read identities or intervals
            differ, or source names are repeated.
    """
    if not paths:
        raise ValueError("at least one source path is required")
    first = paths[0].acquisition
    contributions: dict[str, NDArray[np.float64]] = {}
    for path in paths:
        for name in ("sample_ids", "pixel_ids", "setting_ids", "port_ids"):
            if getattr(path.acquisition, name) != getattr(first, name):
                raise ValueError("source paths must have matching physical-read identities")
        if not np.array_equal(path.acquisition.exposure_bounds_s, first.exposure_bounds_s):
            raise ValueError("source paths must have matching exposure intervals")
        if contributions.keys() & path.contributions.keys():
            raise ValueError("source names must be unique across paths")
        contributions.update(path.contributions)
    return ExpectedElectrons(first, contributions)
