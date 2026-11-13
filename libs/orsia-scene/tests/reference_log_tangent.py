"""Retained numerical reference for the removed mixed log-tangent renderer.

This test-only method reproduces level_extinction_reference.npz. It retains the
historical exposure-mean log linearization, mean-view centroid projection and
filtered-surface diagnostic. Production Scene now exposes physical radiance only.
The fixture and this explicit comparison do not qualify the new exact-query law.
"""

from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from orsia_geometry import LayeredCells
from orsia_instruments.response.spatial import apply_spatial_response, spatial_response_support
from orsia_radiative.transfer.extinction import linearize_exposure_mean_radiance
from orsia_scene.gas import GasLayerColumns
from orsia_scene.optical_depth import SpatialOpticalDepth
from orsia_scene.projection import project_layer_columns_by_centroid

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]


class ReferenceBandCalculation(TypedDict):
    """Band charge, radiance, sensitivities and support on a supplied finite grid.

    Rendered arrays own writable storage. Spatial axes are row then column, and
    band order follows the supplied response. The per-nm units stay explicit.

    Attributes:
        signal_electrons: (row,column,band) spatially filtered nonnegative charge.
        local_column_signal_electrons: Matching charge for vertical-column plume optics
            with the spatially filtered surface, electrons.
        centre_wavelength_sensitivity_electrons_per_nm: Matching signed centre derivatives.
        filter_fwhm_sensitivity_electrons_per_nm: Matching signed FWHM derivatives.
        valid_mask: Boolean (row,column) filtered analysis support.
        electron_gradient_per_m: Optional caller-supplied spatial charge gradient,
            (row,column,band,coordinate=2) in row/column order, electrons/m.
        toa_radiance_w_m2_sr_nm: (row,column,band) response-normalized radiance per nm.
        radiometric_support_mask: Boolean (row,column) complete radiometric support.
        surface_coefficients: (row,column,band,anchor=2) charge per reflectance unit,
            retaining the supplied lower/upper spectral-anchor order.
        plume_layers_molecules_m2: (layer,y,x) excess molecular columns, molecules/m2.
        plume_heights_agl_m: Matching layer centroids above local terrain, m.
    """

    signal_electrons: FloatArray
    local_column_signal_electrons: FloatArray
    centre_wavelength_sensitivity_electrons_per_nm: FloatArray
    filter_fwhm_sensitivity_electrons_per_nm: FloatArray
    valid_mask: BoolArray
    electron_gradient_per_m: NotRequired[FloatArray]
    toa_radiance_w_m2_sr_nm: FloatArray
    radiometric_support_mask: BoolArray
    surface_coefficients: FloatArray
    plume_layers_molecules_m2: FloatArray
    plume_heights_agl_m: FloatArray


class ReferenceLogTangentRenderer:
    """Reuse level-column factors for reference-log-tangent band rendering.

    Inputs are borrowed unchanged for this object's lifetime, including worker
    shutdown after failure. Rebuild for changed atmosphere, spectrum, geometry,
    surface, masks or response. No global cache or mutable render scratch is shared.
    """

    def __init__(
        self,
        optical_depth: SpatialOpticalDepth,
        *,
        reference_optical_depth: FloatArray,
        support: BoolArray,
        valid_mask: BoolArray,
        solar: FloatArray,
        lower_nm: float,
        upper_nm: float,
        b11: FloatArray,
        b12: FloatArray,
        ranges: Sequence[IntArray],
        views: FloatArray,
        azimuth_deg: FloatArray,
        solar_angle: float,
        kernels: tuple[FloatArray, FloatArray, FloatArray],
        mtf: float,
        pixel_integration: bool,
    ) -> None:
        """Prepare unchanged atmospheric, spectral and surface factors once.

        Args:
            optical_depth: Physical atmospheric/extinction preparation on complete cells.
            reference_optical_depth: Explicit reference spectrum for the retained
                log-tangent coaddition approximation; dimensionless (wavelength,).
            support: Boolean radiometric support with the physical terrain shape.
            valid_mask: Boolean product-analysis support with that shape.
            solar: Beam-normal solar irradiance, W/m2/nm on absorption wavelengths.
            lower_nm: Wavelength of the first surface reflectance anchor, nm.
            upper_nm: Wavelength of the second surface reflectance anchor, nm.
            b11: First dimensionless surface reflectance (row,column).
            b12: Second dimensionless surface reflectance (row,column).
            ranges: Zero-based per-band exposure indices into views and azimuth_deg.
            views: Exposure zenith angles in degrees.
            azimuth_deg: Exposure row/column view azimuths, degrees.
            solar_angle: Common solar zenith, degrees on illuminated hemisphere.
            kernels: Nominal electron weights and centre/FWHM derivatives,
                each (band,wavelength), per W/m2/sr/nm and per nm for derivatives.
            mtf: Optical MTF at Nyquist, dimensionless in (0,1].
            pixel_integration: Include square-pixel integration in spatial response.

        Raises:
            ValueError: Atmospheric, aerosol, spectral or spatial support is invalid.
        """
        column = optical_depth.atmosphere
        species = column.species
        if optical_depth.spectral.coordinate != "wavelength_nm":
            raise ValueError("the retained band renderer requires per-nm optical sampling")
        wavelength = optical_depth.spectral.points
        cells = optical_depth.support
        sections = np.array([optical_depth.species_cross_sections(name) for name in species])
        terrain = cells.surface_altitude_m
        shape = terrain.shape
        lo, hi = cells.horizontal.lower_xy_m, cells.horizontal.upper_xy_m
        if terrain.ndim != 2 or not terrain.size:
            raise ValueError("scene cells must have nonempty (y,x) axes")
        widths = hi[0, 0] - lo[0, 0]
        yy, xx = np.indices(shape)
        centres = lo[0, 0] + widths / 2 + np.stack((xx * widths[0], yy * widths[1]), axis=-1)
        if (
            not np.array_equal(lo, centres - widths / 2)
            or not np.array_equal(hi, centres + widths / 2)
            or not np.array_equal(cells.height_edges_m, column.level_altitude_m)
        ):
            raise ValueError("regular tiled cells and matching atmospheric interfaces required")
        spacing = (float(widths[1]), float(widths[0]))
        self._cells = cells
        baseline_tau = reference_optical_depth
        if (
            baseline_tau.shape != wavelength.shape
            or not np.isfinite(baseline_tau).all()
            or np.any(baseline_tau < 0)
        ):
            raise ValueError(
                "reference optical depth must be finite nonnegative on spectral support"
            )
        bands = len(ranges)
        azimuth = np.deg2rad(azimuth_deg)
        slopes = np.tan(np.deg2rad(views))[:, None] * np.stack(
            (np.cos(azimuth), np.sin(azimuth)), axis=1
        )
        if (
            not 0 <= solar_angle < 90
            or views.ndim != 1
            or not np.isfinite(views).all()
            or np.any((views < 0) | (views >= 90))
        ):
            raise ValueError("illuminated forward-hemisphere angles required")
        mu = np.cos(np.deg2rad(solar_angle))
        masses = 1 / mu + 1 / np.cos(np.deg2rad(views))
        logsource, mass = [], []
        for indices in ranges:
            if (
                indices.ndim != 1
                or not indices.size
                or np.any((indices < 0) | (indices >= views.size))
            ):
                raise ValueError("nonempty in-range exposure indices required")
            _, log_value, effective = linearize_exposure_mean_radiance(
                mu / np.pi * solar, baseline_tau, masses[indices], np.ones(indices.size)
            )
            logsource.append(log_value)
            mass.append(effective)
        logsource, mass = np.array(logsource), np.array(mass)
        upper = (wavelength - lower_nm) / (upper_nm - lower_nm)
        filtered11 = apply_spatial_response(np.where(support, b11, 0), mtf, pixel_integration)
        filtered12 = apply_spatial_response(np.where(support, b12, 0), mtf, pixel_integration)
        self._prepared = (
            column,
            sections,
            shape,
            species,
            optical_depth,
            bands,
            ranges,
            slopes,
            wavelength,
            logsource,
            mass,
            solar_angle,
            terrain,
            spacing,
            upper,
            b11,
            b12,
            kernels,
            mtf,
            pixel_integration,
            filtered11,
            filtered12,
            support,
            valid_mask,
        )

    @property
    def cells(self) -> LayeredCells:
        """Return the immutable geometric support shared by all gas bindings."""
        return self._cells

    def render(
        self,
        *,
        continuation: Literal["finite", "infinite_uniform"],
        gas_layers: GasLayerColumns | None = None,
        solar_azimuth_deg: float | None = None,
        spectral_block_size: int = 32,
        executor: Executor | None = None,
    ) -> ReferenceBandCalculation:
        """Evaluate one scene with independent bounded scratch and owned outputs.

        Args:
            continuation: Explicit centroid boundary law. Finite support loses outgoing
                amounts; infinite_uniform requires spatially constant layer amounts/heights.
            gas_layers: Excess gas on exactly the prepared pixel/layer support.
                None renders background only. Each call can supply a new snapshot;
                one snapshot stays frozen over all exposures in this render.
                Nonzero excess H2O is rejected when continuum is prepared; its
                density products require an explicit physical-state update.
            solar_azimuth_deg: Optional Sun-to-ground azimuth for two-path projection.
            spectral_block_size: Positive maximum spectral samples in each temporary.
            executor: Optional borrowed CPU pool; None evaluates bands sequentially.
                Input arrays stay stable until tasks and owning pool cleanup finish.

        Returns:
            Fresh radiance, charge, derivatives, masks and plume-layer diagnostics.
            Spectral samples, band ordering and block summation order are retained.

        Raises:
            ValueError: Block size or resolved plume support is invalid.
        """
        if continuation not in ("finite", "infinite_uniform"):
            raise ValueError("continuation must be finite or infinite_uniform")
        if spectral_block_size < 1:
            raise ValueError("spectral_block_size must be a positive integer")
        (
            column,
            sections,
            shape,
            species,
            optical_depth,
            bands,
            ranges,
            slopes,
            wavelength,
            logsource,
            mass,
            solar_angle,
            terrain,
            spacing,
            upper,
            b11,
            b12,
            kernels,
            mtf,
            pixel_integration,
            filtered11,
            filtered12,
            support,
            valid_mask,
        ) = self._prepared
        layers = np.zeros((len(column.level_altitude_m) - 1, *shape))
        heights = np.zeros_like(layers)
        gas_index = 0
        if gas_layers is not None:
            if not gas_layers.support.matches(self.cells):
                raise ValueError(
                    "gas layers must match prepared horizontal cells, frame, terrain and interfaces"
                )
            if (
                gas_layers.species == "H2O"
                and optical_depth.continuum is not None
                and np.any(gas_layers.columns_molecules_m2 > 0)
            ):
                raise ValueError("excess water requires updated continuum density products")
            gas_index = species.index(gas_layers.species)
            layers = np.moveaxis(gas_layers.columns_molecules_m2, -1, 0).copy()
            heights = np.moveaxis(
                np.where(
                    gas_layers.columns_molecules_m2 > 0,
                    gas_layers.centroid_altitude_m - terrain[..., None],
                    0.0,
                ),
                -1,
                0,
            ).copy()
        outputs = [np.zeros((*shape, bands)) for _ in kernels]
        local = np.zeros_like(outputs[0])
        coefficients = np.zeros((*shape, bands, 2))
        nonzero_layers = np.any(layers != 0, axis=(1, 2))
        active_layers = layers[nonzero_layers]
        active_heights = heights[nonzero_layers]
        active_sections = (
            sections[gas_index, nonzero_layers]
            if np.any(nonzero_layers)
            else np.empty((0, len(wavelength)))
        )
        solar_layers = None
        if solar_azimuth_deg is not None and len(active_layers):
            slope = np.tan(np.deg2rad(solar_angle)) * np.array(
                [np.cos(np.deg2rad(solar_azimuth_deg)), np.sin(np.deg2rad(solar_azimuth_deg))]
            )
            solar_layers = np.moveaxis(
                project_layer_columns_by_centroid(
                    np.moveaxis(active_layers, 0, -1),
                    np.moveaxis(active_heights, 0, -1),
                    slope,
                    spacing,
                    continuation=continuation,
                ),
                -1,
                0,
            )

        def render_band(b: int) -> tuple[list[FloatArray], FloatArray, FloatArray]:
            indices = ranges[b]
            band_outputs = [np.zeros(shape) for _ in kernels]
            band_local = np.zeros(shape)
            band_coefficients = np.zeros((*shape, 2))
            projected = (
                np.moveaxis(
                    project_layer_columns_by_centroid(
                        np.moveaxis(active_layers, 0, -1),
                        np.moveaxis(active_heights, 0, -1),
                        slopes[indices].mean(axis=0),
                        spacing,
                        continuation=continuation,
                    ),
                    -1,
                    0,
                )
                if len(active_layers)
                else None
            )
            for start in range(0, len(wavelength), spectral_block_size):
                chunk = slice(start, min(start + spectral_block_size, len(wavelength)))
                tau = optical_depth.spectral_block(chunk.start, chunk.stop)
                plume_tau: FloatArray | float = 0.0
                if projected is not None:
                    plume_tau = np.einsum(
                        "krc,ks->rcs", projected, active_sections[:, chunk], optimize=True
                    )
                if solar_layers is not None:
                    solar_tau = np.einsum(
                        "krc,ks->rcs", solar_layers, active_sections[:, chunk], optimize=True
                    )
                    solar_weight = (1 / np.cos(np.deg2rad(solar_angle))) / mass[b, chunk]
                    plume_tau = solar_tau * solar_weight + plume_tau * (1 - solar_weight)
                response = np.exp(logsource[b, chunk] - mass[b, chunk] * (tau + plume_tau))
                surface = b11[..., None] * (1 - upper[chunk]) + b12[..., None] * upper[chunk]
                radiance = response * surface
                for out, kernel in zip(band_outputs, kernels, strict=True):
                    out += np.einsum("rcs,s->rc", radiance, kernel[b, chunk])
                band_coefficients[..., 0] += np.einsum(
                    "rcs,s->rc", response, kernels[0][b, chunk] * (1 - upper[chunk])
                )
                band_coefficients[..., 1] += np.einsum(
                    "rcs,s->rc", response, kernels[0][b, chunk] * upper[chunk]
                )
                vertical_response = response
                if len(active_layers):
                    vertical_tau = np.einsum(
                        "krc,ks->rcs", active_layers, active_sections[:, chunk], optimize=True
                    )
                    vertical_response = np.exp(
                        logsource[b, chunk] - mass[b, chunk] * (tau + vertical_tau)
                    )
                filtered_surface = (
                    filtered11[..., None] * (1 - upper[chunk])
                    + filtered12[..., None] * upper[chunk]
                )
                band_local += np.einsum(
                    "rcs,s->rc",
                    vertical_response * filtered_surface,
                    kernels[0][b, chunk],
                )
            return band_outputs, band_local, band_coefficients

        evaluated = (
            map(render_band, range(bands))
            if executor is None
            else executor.map(render_band, range(bands), buffersize=32)
        )
        for b, (band_outputs, band_local, band_coefficients) in enumerate(evaluated):
            for out, band in zip(outputs, band_outputs, strict=True):
                out[..., b] = band
            local[..., b] = band_local
            coefficients[..., b, :] = band_coefficients
        output_support = spatial_response_support(support, mtf, pixel_integration)
        for i, out in enumerate(outputs):
            out[~support] = 0
            outputs[i] = apply_spatial_response(out, mtf, pixel_integration)
            outputs[i][~output_support] = 0
        outputs[0] = np.maximum(outputs[0], 0)
        local[~output_support] = 0
        coefficients[~support] = 0
        return {
            "signal_electrons": outputs[0],
            "toa_radiance_w_m2_sr_nm": outputs[0] / kernels[0].sum(axis=1),
            "centre_wavelength_sensitivity_electrons_per_nm": outputs[1],
            "filter_fwhm_sensitivity_electrons_per_nm": outputs[2],
            "local_column_signal_electrons": local,
            "radiometric_support_mask": output_support,
            "valid_mask": spatial_response_support(valid_mask, mtf, pixel_integration),
            "surface_coefficients": coefficients,
            "plume_layers_molecules_m2": layers,
            "plume_heights_agl_m": heights,
        }
