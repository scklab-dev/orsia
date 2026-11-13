"""Retain signed registration dependencies with bounded local interpolation stencils."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

type ArrayInput = (
    float
    | np.floating
    | NDArray[np.floating | np.integer | np.bool_]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
)


type FloatArray = NDArray[np.float64]


type BoolArray = NDArray[np.bool_]


type IntArray = NDArray[np.int64]


STENCIL_BLOCK_CELLS = 2048


class RegistrationMap:
    """Signed cube resampling defined by translations, kernel and source support.

    Shape has (row,column,band) axes and C-order cell identities. Corrections are
    float64 pixel offsets with (band,2) or (row,band,2) axes. The reference band
    must have zero corrections. Source and output support include every declared
    stencil endpoint, even coefficients equal to zero. A row with both offsets
    exactly zero instead uses identity at every supported pixel, including edges.

    The map owns read-only copies of its defining arrays. Properties share those
    read-only buffers; callers must keep them unchanged. Operations allocate their
    results and bounded scratch, and can be called concurrently on stable inputs.
    No global coefficient matrix or per-cell coefficient cache is retained.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        corrections: ArrayInput,
        source_valid: ArrayInput,
        *,
        reference_band: int = 0,
        kernel: str = "lanczos5",
    ) -> None:
        """Copy and validate geometry; raise ValueError for incompatible inputs.

        Args:
            shape: Positive (row,column,band) integer dimensions.
            corrections: Finite row/column offsets in pixels, (band,2) or (row,band,2).
            source_valid: Boolean cube support copied into owned read-only storage.
            reference_band: Zero-based reference band, whose corrections must be zero.
            kernel: keys_cubic, lanczos3 or lanczos5 signed interpolation kernel.

        Raises:
            ValueError: Cube geometry, correction support, reference or kernel is invalid.
        """
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("positive integer cube shape required")
        support = np.asarray(source_valid)
        shifts = np.asarray(corrections, dtype=np.float64)
        rows, _, bands = shape
        if support.shape != shape or support.dtype != bool:
            raise ValueError("Boolean cube support required")
        if shifts.shape not in ((bands, 2), (rows, bands, 2)) or not np.isfinite(shifts).all():
            raise ValueError("correction axes differ or shifts are nonfinite")
        if type(reference_band) is not int or not 0 <= reference_band < bands:
            raise ValueError("invalid reference band")
        expanded = np.broadcast_to(shifts, (rows, bands, 2))
        if np.any(expanded[:, reference_band] != 0):
            raise ValueError("reference band must have exactly zero correction")
        self._shape = shape
        self._kernel = kernel
        self._reference_band = reference_band
        self._corrections = np.array(shifts, copy=True)
        self._source_valid = np.array(support, copy=True)
        self._shifts = np.broadcast_to(self._corrections, (rows, bands, 2))
        # Clipping changes only geometrically unsupported coordinates and avoids
        # integer overflow for arbitrary finite, out-of-scene supplied shifts.
        coordinates = np.clip(np.arange(rows)[:, None] + self._shifts[..., 0], -11, rows + 11)
        ri, rw, rv = interpolation_weights(coordinates, rows, kernel)
        self._identity = np.all(self._shifts == 0, axis=2)
        ri = np.clip(ri, 0, rows - 1)
        ri[self._identity] = np.broadcast_to(np.arange(rows)[:, None], (rows, bands))[
            self._identity, None
        ]
        rw[self._identity] = 0
        rw[..., 0][self._identity] = 1
        self._row_indices, self._row_weights = ri, rw
        self._row_possible = rv | self._identity
        self._valid = np.zeros(shape, bool)
        for cells in self.blocks():
            sources, _, possible = self._stencil(cells)
            self._valid.flat[cells] = possible & np.all(self._source_valid.flat[sources], axis=1)
        for array in (self._corrections, self._source_valid, self._valid):
            array.setflags(write=False)

    @property
    def cube_shape(self) -> tuple[int, int, int]:
        """Return the source/output (row,column,band) dimensions."""
        return self._shape

    @property
    def corrections_pixel(self) -> FloatArray:
        """Return a read-only view of the owned pixel shifts, retaining input axes."""
        return self._corrections.view()

    @property
    def source_valid(self) -> BoolArray:
        """Return a read-only view of the original Boolean cube support."""
        return self._source_valid.view()

    @property
    def valid(self) -> BoolArray:
        """Return a read-only view of complete-stencil output cube support."""
        return self._valid.view()

    @property
    def kernel(self) -> str:
        """Return the signed interpolation kernel name."""
        return self._kernel

    @property
    def reference_band(self) -> int:
        """Return the zero-shift reference band index."""
        return self._reference_band

    def blocks(self) -> Iterator[IntArray]:
        """Yield owned C-order cell-index blocks bounded by STENCIL_BLOCK_CELLS.

        Yields:
            Owned int64 C-order cell-index vectors of at most STENCIL_BLOCK_CELLS.
        """
        for start in range(0, int(np.prod(self._shape)), STENCIL_BLOCK_CELLS):
            yield np.arange(start, min(start + STENCIL_BLOCK_CELLS, int(np.prod(self._shape))))

    def apply(self, field: ArrayInput, *, squared: bool = False) -> FloatArray:
        """Transport a cube with signed or squared weights and zero invalid rows.

        field has cube axes; squared=True computes marginal variance transport
        from independent source variances. Returns an owned float64 cube. Invalid
        source cells are excluded before multiplication, including NaN gaps.
        Shape mismatch raises ValueError. Source inputs remain unchanged.

        Args:
            field: Borrowed source cube in any consistent physical unit.
            squared: Use squared coefficients for independent variance propagation.

        Returns:
            Owned float64 output cube, zero on invalid output rows.

        Raises:
            ValueError: Field axes differ from the declared cube.
        """
        array = np.asarray(field, dtype=np.float64)
        if array.shape != self._shape:
            raise ValueError("transport field axes differ from cube")
        result = np.empty(self._shape, dtype=np.float64)
        for cells in self.blocks():
            sources, weights, _ = self._stencil(cells)
            good = self._valid.flat[cells]
            values = np.where(good[:, None], array.flat[sources], 0)
            if squared:
                weights *= weights
            result.flat[cells] = np.sum(weights * values, axis=1)
        return result

    def propagate_any(self, field: ArrayInput) -> BoolArray:
        """Propagate Boolean dependency flags using all declared stencil endpoints.

        field has cube or spatial (row,column) axes. Returns an owned Boolean
        cube, false on invalid rows, with no coefficient-zero pruning. This also
        handles spatial masks without constructing a pixel-support matrix.
        Invalid shape or dtype raises ValueError. Inputs remain unchanged.

        Args:
            field: Borrowed Boolean source cube or (row,column) dependency mask.

        Returns:
            Owned Boolean cube indicating any declared dependency, false on invalid rows.

        Raises:
            ValueError: Boolean dtype or declared cube/spatial axes fail.
        """
        array = np.asarray(field)
        if array.dtype != bool or array.shape not in (self._shape, self._shape[:2]):
            raise ValueError("Boolean cube or spatial dependency mask required")
        result = np.empty(self._shape, dtype=bool)
        for cells in self.blocks():
            sources, _, _ = self._stencil(cells)
            if array.ndim == 2:
                sources = sources // self._shape[-1]
            result.flat[cells] = self._valid.flat[cells] & np.any(array.flat[sources], axis=1)
        return result

    def rows(self, output_cells: ArrayInput) -> csr_matrix:
        """Extract at most STENCIL_BLOCK_CELLS signed rows as an owned local CSR.

        output_cells is a one-dimensional integer array of C-order cube indices.
        Rows preserve supplied order and explicit zero coefficients; invalid rows
        are empty. Columns retain original cube identities. Invalid indices or an
        oversized extraction raise ValueError. Global map extraction is excluded.

        Args:
            output_cells: At most STENCIL_BLOCK_CELLS one-dimensional integer C-order cells.

        Returns:
            Owned CSR rows in supplied order with original source-cube column identities.
            Invalid rows are empty; supported zero coefficients remain explicit.

        Raises:
            ValueError: Cell indices or bounded extraction size are invalid.
        """
        cells = self._check_cells(output_cells)
        sources, weights, _ = self._stencil(cells)
        good = self._valid.flat[cells]
        width = weights.shape[1]
        indptr = np.r_[0, np.cumsum(good, dtype=np.int64) * width]
        return csr_matrix(
            (weights[good].ravel(), sources[good].ravel(), indptr),
            shape=(len(cells), int(np.prod(self._shape))),
        )

    def overlapping_rows(self, output_row: int, band: int) -> IntArray:
        """Return possible overlapping output rows from actual source-row stencils.

        Row and band are zero-based. The owned result includes every row that
        shares a declared source row, even for large or non-monotone shifts.
        Column and support filtering is left to local sparse multiplication.
        Invalid indices raise ValueError.

        Args:
            output_row: Zero-based output row.
            band: Zero-based output band.

        Returns:
            Owned int64 rows sharing a possible declared source row with the target.

        Raises:
            ValueError: Row or band is outside the cube.
        """
        rows, _, bands = self._shape
        if not 0 <= output_row < rows or not 0 <= band < bands:
            raise ValueError("row or band outside cube")
        if not self._row_possible[output_row, band]:
            return np.empty(0, dtype=np.int64)
        shared = np.isin(self._row_indices[:, band], self._row_indices[output_row, band]).any(
            axis=1
        )
        return np.flatnonzero(shared & self._row_possible[:, band])

    def _check_cells(self, output_cells: ArrayInput) -> IntArray:
        cells = np.asarray(output_cells)
        if (
            cells.ndim != 1
            or cells.dtype.kind not in "iu"
            or len(cells) > STENCIL_BLOCK_CELLS
            or np.any(cells < 0)
            or np.any(cells >= np.prod(self._shape))
        ):
            raise ValueError("bounded one-dimensional cube cell indices required")
        return cells.astype(np.int64, copy=False)

    def _stencil(self, cells: IntArray) -> tuple[IntArray, FloatArray, BoolArray]:
        rows, cols, bands = self._shape
        if len(cells) == 0:
            return np.empty((0, 1), dtype=np.int64), np.empty((0, 1)), np.empty(0, dtype=bool)
        band = cells % bands
        column = cells // bands % cols
        row = cells // (cols * bands)
        identity = self._identity[row, band]
        if identity.all():
            return cells[:, None], np.ones((len(cells), 1)), np.ones(len(cells), dtype=bool)
        ci, cw, cv = interpolation_weights(
            np.clip(column + self._shifts[row, band, 1], -11, cols + 11), cols, self._kernel
        )
        ci = np.clip(ci, 0, cols - 1)
        ci[identity] = column[identity, None]
        cw[identity] = 0
        cw[identity, 0] = 1
        ri, rw = self._row_indices[row, band], self._row_weights[row, band]
        sources = (ri[:, :, None] * cols + ci[:, None, :]) * bands + band[:, None, None]
        weights = rw[:, :, None] * cw[:, None, :]
        return (
            sources.reshape(len(cells), -1),
            weights.reshape(len(cells), -1),
            self._row_possible[row, band] & (cv | identity),
        )


def interpolation_stencil(
    coordinates: ArrayInput, sample_count: int, kernel: str = "lanczos5"
) -> tuple[IntArray, FloatArray, FloatArray, BoolArray]:
    """Return source indices, signed weights, coordinate derivatives and support.

    Coordinates are finite zero-based pixels. Outputs have (...,stencil) axes;
    valid has coordinate shape. Full stencils include zero-weight endpoints.
    Unsupported indices remain explicit and must be excluded before indexing.

    Args:
        coordinates: Finite zero-based pixel coordinates of arbitrary shape.
        sample_count: Positive source-axis length.
        kernel: keys_cubic, lanczos3 or lanczos5 signed interpolation kernel.

    Returns:
        Owned indices, signed weights and derivatives (...,stencil), plus Boolean support.
        Derivatives are per pixel; unsupported indices must be excluded before indexing.

    Raises:
        ValueError: Kernel, finite coordinates or source length is invalid.
    """
    indices, weights, derivative, valid = _interpolation_stencil(
        coordinates, sample_count, kernel, derivatives=True
    )
    assert derivative is not None
    return indices, weights, derivative, valid


def interpolation_weights(
    coordinates: ArrayInput, sample_count: int, kernel: str = "lanczos5"
) -> tuple[IntArray, FloatArray, BoolArray]:
    """Return signed interpolation coefficients and complete support without derivatives.

    Args:
        coordinates: Finite zero-based pixel coordinates of any shape.
        sample_count: Positive source-axis length.
        kernel: keys_cubic, lanczos3 or lanczos5 signed interpolation kernel.

    Returns:
        Owned indices and weights with (...,stencil) axes and a Boolean mask
        with coordinate shape. All endpoints remain explicit, including zero
        coefficients; exclude unsupported coordinates before source indexing.

    Raises:
        ValueError: Coordinates, source length or kernel are invalid.
    """
    indices, weights, _, valid = _interpolation_stencil(
        coordinates, sample_count, kernel, derivatives=False
    )
    return indices, weights, valid


def _interpolation_stencil(
    coordinates: ArrayInput, sample_count: int, kernel: str, *, derivatives: bool
) -> tuple[IntArray, FloatArray, FloatArray | None, BoolArray]:
    """Share coefficient normalization between value-only and tangent sampling.

    Returns:
        Indices, normalized weights, optional derivatives and complete endpoint support.

    Raises:
        ValueError: Kernel, coordinate finiteness or source length fails.
    """
    count = {"keys_cubic": 4, "lanczos3": 6, "lanczos5": 10}.get(kernel)
    x = np.asarray(coordinates, dtype=np.float64)
    if count is None or not np.isfinite(x).all() or sample_count < 1:
        raise ValueError("supported kernel and finite coordinates required")
    indices = np.floor(x).astype(int)[..., None] + np.arange(1 - count // 2, count // 2 + 1)
    distance = x[..., None] - indices
    a = np.abs(distance)
    derivative = None
    if kernel == "keys_cubic":
        weights = np.where(
            a <= 1,
            1.5 * a**3 - 2.5 * a**2 + 1,
            np.where(a < 2, -0.5 * a**3 + 2.5 * a**2 - 4 * a + 2, 0),
        )
        if derivatives:
            derivative = np.sign(distance) * np.where(
                a <= 1, 4.5 * a**2 - 5 * a, np.where(a < 2, -1.5 * a**2 + 5 * a - 4, 0)
            )
    else:
        radius = count / 2
        weights = np.where(a < radius, np.sinc(distance) * np.sinc(distance / radius), 0)
        if derivatives:
            derivative = np.where(
                a < radius,
                _sinc_derivative(distance) * np.sinc(distance / radius)
                + np.sinc(distance) * _sinc_derivative(distance / radius) / radius,
                0,
            )
        total = weights.sum(axis=-1, keepdims=True)
        weights /= total
        if derivative is not None:
            derivative = (derivative - weights * derivative.sum(axis=-1, keepdims=True)) / total
    valid = (indices[..., 0] >= 0) & (indices[..., -1] < sample_count)
    return indices, weights, derivative, valid


def _sinc_derivative(x: FloatArray) -> FloatArray:
    safe = np.where(abs(x) < 1e-5, 1, x)
    return np.where(
        abs(x) < 1e-5,
        -(np.pi**2) * x / 3 + np.pi**4 * x**3 / 30,
        (np.cos(np.pi * x) - np.sinc(x)) / safe,
    )
