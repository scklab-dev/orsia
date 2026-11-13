"""Project signed registered detector errors into selected scalar spatial states."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from orsia_uncertainty.sparse import sparse_row_product as registration_row_product
from scipy.sparse import csr_matrix, vstack

from orsia_instruments.registration.resampling import STENCIL_BLOCK_CELLS, RegistrationMap

type FloatArray = NDArray[np.float64]


type ArrayInput = (
    NDArray[np.floating | np.integer | np.bool_] | Sequence[float] | Sequence[Sequence[float]]
)


def project_detector_covariance(
    registration: RegistrationMap | csr_matrix,
    source_variance: ArrayInput,
    local_gain: ArrayInput,
    state_support: ArrayInput,
) -> tuple[csr_matrix, NDArray[np.int64]]:
    """Project registered detector covariance into the scalar local posterior grid.

    Registration maps original to registered C-order cube cells. Source variance
    and local_gain have (row,column,band) axes; state_support selects valid scalar
    states. Returns sparse G R diag(V) R.T G.T on selected C-order states and their
    cell indices. Compact maps use bounded local row products, with no global R,
    GR or radiance covariance. Actual source-row overlap determines candidate
    states even for large or non-monotone shifts. Legacy CSR inputs remain readable.

    Args:
        registration: Compact signed map or archived CSR over C-order cube cells.
        source_variance: Borrowed (row,column,band) independent source variances.
        local_gain: Matching scalar-state derivatives per measured radiance unit.
        state_support: Boolean (row,column) mask selecting scalar states.

    Returns:
        Owned selected-state CSR covariance and int64 C-order spatial cell identities.
        Units are squared scalar-state units; compact scratch remains stencil-bounded.

    Raises:
        ValueError: Map axes, supported gains or used source variances are invalid.
    """
    variance = np.asarray(source_variance, float)
    gain = np.asarray(local_gain, float)
    mask = np.asarray(state_support, bool)
    if gain.shape != variance.shape or gain.ndim != 3 or mask.shape != gain.shape[:2]:
        raise ValueError("detector variance, local gain and support axes differ")
    indices = np.flatnonzero(mask.ravel())
    if isinstance(registration, RegistrationMap):
        if registration.cube_shape != variance.shape:
            raise ValueError("registration operator axes differ from cube")
        if not np.isfinite(gain[mask]).all():
            raise ValueError("supported local gain must be finite")
        return _compact_detector_covariance(registration, variance, gain, mask, indices), indices
    if registration.shape != (variance.size, variance.size):
        raise ValueError("registration operator axes differ from cube")
    if not np.isfinite(gain[mask]).all():
        raise ValueError("supported local gain must be finite")
    return _legacy_detector_covariance(registration, variance, gain, indices), indices


def _legacy_gain_projection(
    registration: csr_matrix, gain: FloatArray, pixels: NDArray[np.int64]
) -> csr_matrix:
    bands = gain.shape[-1]
    cells = (pixels[:, None] * bands + np.arange(bands)).ravel()
    local_gain = csr_matrix(
        (gain.flat[cells], (np.repeat(np.arange(len(pixels)), bands), np.arange(len(cells)))),
        shape=(len(pixels), len(cells)),
    )
    projection = csr_matrix(local_gain @ registration[cells])
    projection.eliminate_zeros()
    return projection


def _legacy_detector_covariance(
    registration: csr_matrix, variance: FloatArray, gain: FloatArray, indices: NDArray[np.int64]
) -> csr_matrix:
    # Arbitrary archived CSR maps can mix bands. Bounded pairwise products retain
    # that contract without a global GR; compact maps use geometric pruning below.
    count = max(1, STENCIL_BLOCK_CELLS // gain.shape[-1])
    blocks = []
    for start in range(0, len(indices), count):
        left = _legacy_gain_projection(registration, gain, indices[start : start + count])
        used = variance.flat[left.indices]
        if not np.isfinite(used).all() or np.any(used < 0):
            raise ValueError("used source variance must be finite and nonnegative")
        left.data *= used
        block = csr_matrix((left.shape[0], len(indices)))
        for offset in range(0, len(indices), count):
            right = _legacy_gain_projection(registration, gain, indices[offset : offset + count])
            local = registration_row_product(left, right)
            block += csr_matrix(
                (local.data, local.indices + offset, local.indptr), shape=block.shape
            )
        blocks.append(block)
    return csr_matrix(vstack(blocks, format="csr")) if blocks else csr_matrix((0, 0))


def _compact_detector_covariance(
    registration: RegistrationMap,
    variance: FloatArray,
    gain: FloatArray,
    mask: NDArray[np.bool_],
    indices: NDArray[np.int64],
) -> csr_matrix:
    """Assemble only scalar state covariance from bounded band-specific row products.

    Returns:
        Owned scalar-state CSR covariance in selected C-order.

    Raises:
        ValueError: A used source variance is nonfinite or negative.
    """
    rows, cols, bands = variance.shape
    selected = np.full(mask.size, -1, dtype=np.int64)
    selected[indices] = np.arange(len(indices))
    blocks = []
    for row in range(rows):
        pixels = row * cols + np.flatnonzero(mask[row])
        for start in range(0, len(pixels), STENCIL_BLOCK_CELLS):
            targets = pixels[start : start + STENCIL_BLOCK_CELLS]
            block = csr_matrix((len(targets), len(indices)))
            for band in range(bands):
                cells = targets * bands + band
                left = registration.rows(cells)
                left.data *= np.repeat(gain.flat[cells], np.diff(left.indptr))
                left.eliminate_zeros()
                if not left.nnz:
                    continue
                used_variance = variance.flat[left.indices]
                if not np.isfinite(used_variance).all() or np.any(used_variance < 0):
                    raise ValueError("used source variance must be finite and nonnegative")
                left.data *= used_variance
                candidates = registration.overlapping_rows(row, band)
                candidates = (candidates[:, None] * cols + np.arange(cols)).ravel()
                candidates = candidates[mask.flat[candidates]]
                for offset in range(0, len(candidates), STENCIL_BLOCK_CELLS):
                    other = candidates[offset : offset + STENCIL_BLOCK_CELLS]
                    other_cells = other * bands + band
                    right = registration.rows(other_cells)
                    right.data *= np.repeat(gain.flat[other_cells], np.diff(right.indptr))
                    right.eliminate_zeros()
                    local = registration_row_product(left, right)
                    contribution = csr_matrix(
                        (local.data, selected[other[local.indices]], local.indptr),
                        shape=block.shape,
                    )
                    block += contribution
            blocks.append(block)
    return csr_matrix(vstack(blocks, format="csr")) if blocks else csr_matrix((0, 0))
