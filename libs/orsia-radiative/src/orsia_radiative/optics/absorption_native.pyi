"""Private fixed-cross-section absorption ABI with owned float64 outputs."""

from collections.abc import Buffer

import numpy as np
from numpy.typing import NDArray

def cross_section_optical_depth(
    cross_sections: Buffer, columns: Buffer, /
) -> NDArray[np.float64]: ...
def cross_section_optical_depth_tangent(
    cross_sections: Buffer,
    columns: Buffer,
    column_tangent: Buffer,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
