"""Prepare species-generic spatial optical depths on exact uncompressed bases."""

from concurrent.futures import Executor, wait

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import LayeredCells
from orsia_radiative import (
    LayerCrossSections,
    LevelAtmosphere,
    SpectralGrid,
    WaterContinuumCoefficients,
    column_absorption_optical_depth,
    integrate_atmospheric_columns,
)

type FloatArray = NDArray[np.float64]


class SpatialOpticalDepth:
    """Own fixed atmospheric, spectral and spatial factors for repeated depth queries.

    Physical inputs are immutable records. Supplied extinction coefficient fields
    and spectral shapes are snapshotted. Active-layer material is represented by
    nonnegative amounts multiplying exact spectra; unchanged upper layers form a
    reference. This avoids subtracting nearly equal full-column optical depths.
    No spectral compression, detector response or mandatory gas identity is used.
    """

    def __init__(
        self,
        atmosphere: LevelAtmosphere,
        cross_sections: tuple[LayerCrossSections, ...],
        *,
        spectral: SpectralGrid,
        support: LayeredCells,
        continuum: WaterContinuumCoefficients | None = None,
        extinction_amplitudes: FloatArray | None = None,
        extinction_shapes: FloatArray | None = None,
        physical_mask: NDArray[np.bool_] | None = None,
        executor: Executor | None = None,
        batch_size: int = 256,
    ) -> None:
        """Prepare exact material/spectrum factors on the declared finite cells.

        Args:
            atmosphere: Immutable thermodynamics and species-labelled level ratios.
            cross_sections: Fixed coefficients at explicit evaluation conditions.
            spectral: Shared nm or cm-1 samples, matched exactly to all coefficients.
            support: Complete cells/frame/terrain with interfaces equal to atmospheric levels.
            continuum: Optional matching water density-pair coefficients.
            extinction_amplitudes: Optional nonnegative (*pixels,term) dimensionless amplitudes.
            extinction_shapes: Paired nonnegative (term,spectral) optical depths per unit amplitude.
            physical_mask: Optional Boolean (*pixels,) query support; default admits all cells.
                Every supplied terrain value must remain within atmospheric support.
            executor: Optional borrowed worker executor; default preparation is serial.
                Submitted work is settled on success or failure before returning.
            batch_size: Positive number of pixels per independently prepared block.

        Raises:
            ValueError: Support, fields, spectroscopy, terrain or numerical controls fail.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not np.array_equal(support.height_edges_m, atmosphere.level_altitude_m):
            raise ValueError("spatial and atmospheric layer interfaces must match exactly")
        shape = support.horizontal.shape
        if not support.surface_altitude_m.size:
            raise ValueError("spatial optical-depth support must be nonempty")
        if (extinction_amplitudes is None) != (extinction_shapes is None):
            raise ValueError("extinction amplitudes and shapes must be supplied together")
        k = spectral.points.size
        if extinction_amplitudes is None or extinction_shapes is None:
            amplitudes, shapes = np.empty((*shape, 0)), np.empty((0, k))
        else:
            amplitudes, shapes = extinction_amplitudes, extinction_shapes
            if (
                shapes.ndim != 2
                or shapes.shape[1] != k
                or amplitudes.shape != (*shape, shapes.shape[0])
                or not np.isfinite(amplitudes).all()
                or not np.isfinite(shapes).all()
                or np.any(amplitudes < 0)
                or np.any(shapes < 0)
            ):
                raise ValueError(
                    "matching finite nonnegative extinction fields and shapes required"
                )
        mask = np.ones(shape, dtype=bool) if physical_mask is None else physical_mask
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError("physical mask must be Boolean on the cell support")
        self._support, self._spectral, self._atmosphere = support, spectral, atmosphere
        self._tables, self._continuum = tuple(cross_sections), continuum
        self._mask = np.frombuffer(mask.tobytes(), dtype=np.bool_).reshape(shape)
        base = integrate_atmospheric_columns(
            atmosphere, surface_altitude_m=float(support.height_edges_m[0])
        )
        # Validate matching samples, geometry, identities and water pairing once.
        column_absorption_optical_depth(
            base, cross_sections, spectral=spectral, continuum=continuum
        )
        self._base_columns = base
        layers, species_count = base.layer_species_column_molecules_m2.shape
        active = min(
            layers,
            max(
                1,
                int(
                    np.searchsorted(
                        support.height_edges_m, support.surface_altitude_m.max(), side="right"
                    )
                ),
            ),
        )
        self._active = active
        sections = np.zeros((species_count, layers, k))
        for table in cross_sections:
            sections[atmosphere.species.index(table.species)] += (
                table.cross_sections_m2_per_molecule
            )
        own = (
            np.zeros((layers, k)) if continuum is None else continuum.self_coefficients_m5_molecule2
        )
        foreign = (
            np.zeros((layers, k))
            if continuum is None
            else continuum.foreign_coefficients_m5_molecule2
        )
        material_terms = species_count * active + 2 * active
        coefficients = np.empty((*shape, material_terms + shapes.shape[0]))
        coefficients[..., material_terms:] = amplitudes
        dry = np.empty(shape)
        flat = coefficients.reshape(-1, coefficients.shape[-1])
        heights = support.surface_altitude_m.ravel()
        dry_flat = dry.ravel()

        def prepare(start: int) -> None:
            for pixel in range(start, min(start + batch_size, heights.size)):
                columns = integrate_atmospheric_columns(
                    atmosphere, surface_altitude_m=float(heights[pixel])
                )
                flat[pixel, :material_terms] = np.concatenate(
                    (
                        columns.layer_species_column_molecules_m2[:active].T.ravel(),
                        columns.layer_water_self_density_product_molecules2_m5[:active],
                        columns.layer_water_foreign_density_product_molecules2_m5[:active],
                    )
                )
                dry_flat[pixel] = columns.layer_dry_air_column_molecules_m2.sum()

        starts = range(0, heights.size, batch_size)
        if executor is None:
            for start in starts:
                prepare(start)
        else:
            futures = []
            try:
                for start in starts:
                    futures.append(executor.submit(prepare, start))
                for future in futures:
                    future.result()
            finally:
                for future in futures:
                    future.cancel()
                wait(futures)
        basis = np.concatenate(
            (sections[:, :active].reshape(-1, k), own[:active], foreign[:active], shapes)
        )
        reference_layers = np.einsum("ls,slk->lk", base.layer_species_column_molecules_m2, sections)
        reference_layers += (
            base.layer_water_self_density_product_molecules2_m5[:, None] * own
            + base.layer_water_foreign_density_product_molecules2_m5[:, None] * foreign
        )
        reference_layers[:active] = 0
        self._reference = _snapshot(np.cumsum(reference_layers[::-1], axis=0)[-1])
        self._basis, self._coefficients, self._dry = map(_snapshot, (basis, coefficients, dry))
        self._species_reference = {
            name: _snapshot(
                base.layer_species_column_molecules_m2[active:, index] @ sections[index, active:]
            )
            for index, name in enumerate(atmosphere.species)
        }
        self._species_ranges = {
            name: ((index * active, (index + 1) * active),)
            for index, name in enumerate(atmosphere.species)
        }
        if continuum is not None:
            water = (
                base.layer_water_self_density_product_molecules2_m5[active:] @ own[active:]
                + base.layer_water_foreign_density_product_molecules2_m5[active:] @ foreign[active:]
            )
            self._species_reference["H2O"] = _snapshot(
                self._species_reference.get("H2O", np.zeros(k)) + water
            )
            self._species_ranges["H2O"] = self._species_ranges.get("H2O", ()) + (
                (species_count * active, material_terms),
            )

    @property
    def atmosphere(self) -> LevelAtmosphere:
        """Return the immutable level state used for physical preparation."""
        return self._atmosphere

    @property
    def continuum(self) -> WaterContinuumCoefficients | None:
        """Return the immutable supplied water continuum, if present."""
        return self._continuum

    def species_cross_sections(self, species: str) -> FloatArray:
        """Sum supplied isotopologue cross sections for one declared species.

        Args:
            species: Identity in the level state's species order.

        Returns:
            Fresh (layer,spectral) cross sections in m2/molecule. An explicitly
            unmodelled species has zero coefficients; water continuum is separate.

        Raises:
            ValueError: The species is absent from the atmospheric declaration.
        """
        if species not in self._atmosphere.species:
            raise ValueError("cross-section species is undeclared")
        result = np.zeros((self.support.height_edges_m.size - 1, self.spectral.points.size))
        for table in self._tables:
            if table.species == species:
                result += table.cross_sections_m2_per_molecule
        return result

    @property
    def support(self) -> LayeredCells:
        """Return the immutable complete geometric support."""
        return self._support

    @property
    def spectral(self) -> SpectralGrid:
        """Return the immutable spectral samples and declared coordinate."""
        return self._spectral

    @property
    def physical_mask(self) -> NDArray[np.bool_]:
        """Return the immutable mask of admitted physical pixel queries."""
        return self._mask

    @property
    def dry_air_columns_molecules_m2(self) -> FloatArray:
        """Return immutable retained dry-air columns (*pixels,), molecules/m2."""
        return self._dry

    def at_pixel(self, pixel: tuple[int, ...], *, species: str | None = None) -> FloatArray:
        """Evaluate a total or species depth spectrum on one supported cell.

        Args:
            pixel: Zero-based cell index tuple in the support's retained order.
            species: Optional declared species; H2O includes supplied water continuum.

        Returns:
            Fresh dimensionless optical depths (spectral,).

        Raises:
            ValueError: Pixel support or requested species is unavailable.
        """
        self._require_pixel(pixel)
        return self._evaluate(self._coefficients[pixel], slice(None), species)

    def spectral_block(self, start: int, stop: int, *, species: str | None = None) -> FloatArray:
        """Evaluate a bounded spectral slab, preserving all physical cell axes.

        Args:
            start: Inclusive zero-based spectral start.
            stop: Exclusive spectral stop, greater than start.
            species: Optional species contribution; default includes all extinction terms.

        Returns:
            Fresh dimensionless (*pixels,stop-start) depths, with NaN on masked cells.

        Raises:
            ValueError: Spectral bounds or requested species are invalid.
        """
        if not 0 <= start < stop <= self.spectral.points.size:
            raise ValueError("spectral block must lie inside the prepared samples")
        values = self._evaluate(self._coefficients, slice(start, stop), species)
        values[~self.physical_mask] = np.nan
        return values

    def spectral_indices(self, spectral: SpectralGrid) -> NDArray[np.int64]:
        """Locate exact prepared spectral samples without implicit interpolation.

        Args:
            spectral: Requested subset in the preparation's declared coordinate.

        Returns:
            Fresh indices (spectral,) into the prepared spectral axis. A requested
            physical domain does not enlarge the preparation's declared coverage.

        Raises:
            ValueError: Coordinates differ or any sample is absent.
        """
        if spectral.coordinate != self.spectral.coordinate:
            raise ValueError("query and prepared spectral coordinates must agree")
        indices = np.searchsorted(self.spectral.points, spectral.points)
        if np.any(indices >= self.spectral.points.size) or not np.array_equal(
            self.spectral.points[np.minimum(indices, self.spectral.points.size - 1)],
            spectral.points,
        ):
            raise ValueError("queries require exact prepared spectral samples")
        return indices.astype(np.int64)

    def at_pixels(
        self,
        pixels: NDArray[np.int64],
        spectral: SpectralGrid,
        *,
        species: str | None = None,
    ) -> FloatArray:
        """Evaluate selected cells and spectral samples without a full spatial slab.

        Args:
            pixels: Zero-based (query,spatial_axis) indices in the support's order.
                Empty queries and repetitions are permitted.
            spectral: Exact prepared subset with the same coordinate.
            species: Optional declared species contribution, including H2O continuum.

        Returns:
            Fresh dimensionless (query,spectral) optical depths.

        Raises:
            ValueError: Cell indices, physical mask, spectrum or species are unsupported.
        """
        shape = self.support.horizontal.shape
        if (
            pixels.ndim != 2
            or pixels.shape[1] != len(shape)
            or np.any(pixels < 0)
            or np.any(pixels >= np.array(shape))
        ):
            raise ValueError("pixel indices must lie inside spatial support")
        selected = tuple(pixels.T)
        if not self.physical_mask[selected].all():
            raise ValueError("pixel is outside admitted physical support")
        indices = self.spectral_indices(spectral)
        return self._evaluate(self._coefficients[selected], indices, species)

    def column_mixing_ratio(self, species: str) -> FloatArray:
        """Return a dry-air column ratio for one declared atmospheric species.

        Args:
            species: Species identity in the level state; conversion to ppm is caller-owned.

        Returns:
            Fresh dimensionless (*pixels,) column ratios, NaN on masked cells.

        Raises:
            ValueError: Species is absent or retained dry-air amount is nonpositive.
        """
        index = self._atmosphere.species.index(species)
        if np.any(self._dry <= 0):
            raise ValueError("column ratios require positive retained dry-air columns")
        lower = self._coefficients[..., index * self._active : (index + 1) * self._active].sum(
            axis=-1
        )
        upper = self._base_columns.layer_species_column_molecules_m2[self._active :, index].sum()
        ratio = (lower + upper) / self._dry
        ratio[~self.physical_mask] = np.nan
        return ratio

    def _evaluate(
        self, coefficients: FloatArray, selected: slice | NDArray[np.int64], species: str | None
    ) -> FloatArray:
        if species is None:
            result = self._reference[selected] + np.einsum(
                "...t,tk->...k", coefficients, self._basis[:, selected]
            )
            if not np.isfinite(result).all():
                raise ValueError("spatial optical depth overflow")
            return result
        if species not in self._species_reference:
            raise ValueError("requested optical-depth species is undeclared")
        result = np.broadcast_to(
            self._species_reference[species][selected],
            (*coefficients.shape[:-1], self.spectral.points[selected].size),
        ).copy()
        for first, last in self._species_ranges[species]:
            result += np.einsum(
                "...t,tk->...k", coefficients[..., first:last], self._basis[first:last, selected]
            )
        if not np.isfinite(result).all():
            raise ValueError("spatial optical depth overflow")
        return result

    def _require_pixel(self, pixel: tuple[int, ...]) -> None:
        if len(pixel) != len(self.support.horizontal.shape) or any(
            index < 0 or index >= size
            for index, size in zip(pixel, self.support.horizontal.shape, strict=True)
        ):
            raise ValueError("pixel lies outside spatial support")
        if not self.physical_mask[pixel]:
            raise ValueError("pixel is outside admitted physical support")


def _snapshot(values: FloatArray) -> FloatArray:
    if not np.isfinite(values).all():
        raise ValueError("spatial optical-depth preparation must remain finite")
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
