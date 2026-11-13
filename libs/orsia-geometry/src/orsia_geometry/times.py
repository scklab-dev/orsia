"""Preserve physical SI durations at Geometry's implementation-only native time boundary.

This module is internal to Geometry and has no supported public exports. Frame and
orbit adapters validate epochs and time support before calling the shared conversion.
Native instants stay inside those adapters; public callers use UTC datetimes and arrays.
"""

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import satkit

__all__: list[str] = []


def _native_instants(times_s: NDArray[np.float64], *, epoch: datetime) -> list[satkit.time]:
    """Convert validated elapsed SI seconds for Geometry's native adapters.

    Args:
        times_s: Borrowed finite one-dimensional elapsed SI seconds within the
            caller-validated support. Order, repetitions and empty inputs are preserved.
        epoch: Caller-validated timezone-aware UTC datetime with zero offset.

    Returns:
        New list of native instants with microsecond resolution, formed by continuous
        duration addition across leap seconds. Import is deferred until use. Native
        objects are implementation details and remain inside Geometry's adapters.

    Raises:
        ImportError: The required SatKit installation cannot load.
    """
    import satkit

    origin = satkit.time.from_datetime(epoch)
    return [origin + satkit.duration(seconds=float(t)) for t in times_s]
