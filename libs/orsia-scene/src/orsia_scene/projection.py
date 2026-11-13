"""Project finite-cell layer columns using a declared centroid approximation."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


def project_layer_columns_by_centroid(
    amounts: FloatArray,
    heights_agl_m: FloatArray,
    slope_yx: tuple[float, float] | FloatArray,
    spacing_yx_m: tuple[float, float],
    *,
    continuation: Literal["finite", "infinite_uniform"],
) -> FloatArray:
    """Forward-deposit layer columns with positive bilinear weights.

    Each source cell contributes at its centre shifted by height times slope.
    Equal cell areas make this a conservative amount-per-area deposition inside
    the retained domain. Finite support loses contributions leaving the grid.

    Args:
        amounts: Finite nonnegative (y,x,layer) column amounts in a consistent
            area-density unit. Spatial axes must be nonempty; zero layers are valid.
        heights_agl_m: Matching finite nonnegative centroid heights above terrain, m.
        slope_yx: Finite dimensionless projection slopes in array (y,x) order.
        spacing_yx_m: Positive finite regular cell spacing (y,x), metres.
        continuation: Explicit finite-domain deposition or infinite uniform sheets.
            The latter requires spatially constant amounts and centroid height in
            each nonempty layer and returns those same columns. Uniform input never
            selects a continuation policy implicitly. Zero layers need no height match.

    Returns:
        Fresh float64 projected (y,x,layer) columns in the original units.

    Raises:
        ValueError: Shapes, physical domains, selected continuation or finite
            projected positions are invalid. Nonuniform infinite sheets fail.
    """
    slopes = np.asarray(slope_yx, dtype=np.float64)
    if (
        amounts.ndim != 3
        or not all(amounts.shape[:2])
        or heights_agl_m.shape != amounts.shape
        or not np.isfinite(amounts).all()
        or np.any(amounts < 0)
        or not np.isfinite(heights_agl_m).all()
        or np.any(heights_agl_m < 0)
        or slopes.shape != (2,)
        or not np.isfinite(slopes).all()
        or not np.isfinite(spacing_yx_m).all()
        or min(spacing_yx_m) <= 0
        or continuation not in ("finite", "infinite_uniform")
    ):
        raise ValueError("matching nonnegative layers, finite slopes and positive spacing required")
    if continuation == "infinite_uniform":
        active = amounts[0, 0] > 0
        if not np.all(amounts == amounts[0, 0]) or not np.all(
            heights_agl_m[..., active] == heights_agl_m[0, 0, active]
        ):
            raise ValueError("infinite_uniform requires uniform columns and active centroids")
        return amounts.copy()
    result = np.zeros_like(amounts)
    rows, cols, layers = amounts.shape
    yy, xx = np.indices((rows, cols))
    for layer in range(layers):
        with np.errstate(over="ignore", invalid="ignore"):
            y = yy + heights_agl_m[..., layer] * slopes[0] / spacing_yx_m[0]
            x = xx + heights_agl_m[..., layer] * slopes[1] / spacing_yx_m[1]
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("projected centroid positions exceed finite support")
        # Discard wholly outside deposits before conversion to integer indices.
        keep = (y > -1) & (y < rows) & (x > -1) & (x < cols)
        yf, xf = y[keep], x[keep]
        y0, x0 = np.floor(yf).astype(np.int64), np.floor(xf).astype(np.int64)
        field = amounts[..., layer][keep]
        for dy, wy in ((0, 1 - (yf - y0)), (1, yf - y0)):
            for dx, wx in ((0, 1 - (xf - x0)), (1, xf - x0)):
                valid = (y0 + dy >= 0) & (y0 + dy < rows) & (x0 + dx >= 0) & (x0 + dx < cols)
                np.add.at(
                    result[..., layer],
                    ((y0 + dy)[valid], (x0 + dx)[valid]),
                    (field * wy * wx)[valid],
                )
    if not np.isfinite(result).all():
        raise ValueError("projected column accumulation exceeds finite support")
    return result
