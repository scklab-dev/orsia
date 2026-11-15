"""Prepare physical per-read filter response and its explicit area-preserving directions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from orsia_radiative import SpectralGrid
from orsia_radiative.spectral.grid import convert_spectral_grid, spectral_trapezoid_weights

from orsia_instruments.acquisition.schedule import ReadSchedule
from orsia_instruments.optics.filters import (
    FilterPassband,
    lorentzian_filter,
    super_gaussian_filter,
)
from orsia_instruments.photodetection.charge import ExpectedElectrons
from orsia_instruments.photodetection.coefficients import (
    Efficiency,
    QuantumEfficiency,
    efficiency_samples,
    photon_conversion_weights,
)
from orsia_instruments.photodetection.photoelectrons import integrate_photoelectrons

type FloatArray = NDArray[np.float64]
type Vector = Sequence[float] | NDArray[np.floating | np.integer]


@dataclass(frozen=True, eq=False)
class SpectralElectronResponse:
    """Own a geometry-free spectral radiance-to-electron map for actual native reads.

    Attributes:
        reads: Shared immutable physical read identities and exposure intervals.
        grid: Shared spectral samples and declared density coordinate, nm or cm-1.
        coefficients: Immutable float64 (read,spectral) coefficients, electrons per
            W/m2/sr per grid coordinate. They already include the supplied physical
            factors and quadrature. Single-sample evaluated maps are permitted;
            construction of a quadrature rule separately requires adequate sampling.
    """

    reads: ReadSchedule
    grid: SpectralGrid
    coefficients: FloatArray

    def __post_init__(self) -> None:
        """Validate and snapshot the supplied nonnegative electron map.

        Raises:
            ValueError: Physical values or read/spectral axes disagree.
        """
        values = self.coefficients
        if values.shape != (len(self.reads.sample_ids), self.grid.points.size) or np.any(
            values < 0
        ):
            raise ValueError("spectral coefficients require nonnegative (read,spectral) values")
        object.__setattr__(self, "coefficients", _immutable(values))


def spectral_electrons(
    response: SpectralElectronResponse,
    radiance: FloatArray,
    *,
    source_name: str = "scene",
) -> ExpectedElectrons:
    """Apply a spectral response to physical per-read or common radiance samples.

    Args:
        response: Fixed coefficients on the original native read schedule.
        radiance: Nonnegative finite (spectral,) or (read,spectral) W/m2/sr per grid
            coordinate. Each read's values represent its time/cone-averaged radiance
            under the selected geometry-free approximation; no averaging occurs here.
        source_name: Nonempty label for this optical source contribution.

    Returns:
        Owned native source-electron means before dark current, noise or coaddition.

    Raises:
        ValueError: Shape, physical radiance, source name or finite counts are invalid.
    """
    shape = response.coefficients.shape
    if (
        radiance.shape not in (shape, (shape[1],))
        or not np.isfinite(radiance).all()
        or np.any(radiance < 0)
    ):
        raise ValueError("radiance must be finite nonnegative on the response samples")
    return ExpectedElectrons(
        response.reads,
        {
            source_name: integrate_photoelectrons(
                response.coefficients, np.broadcast_to(radiance, shape)
            )
        },
    )


class FilterElectronResponse:
    """Own common spectral factors for repeated nominal and perturbed filter evaluation.

    FilterPassband owns shape/peak parameters. ReadSchedule owns chronology and
    durations. The caller supplies accepted etendue, other optical transmission
    (including blocking) and QE from its physical owners. Construction snapshots
    the resulting factors; callbacks are evaluated once and are not retained.
    No coadd count, inferred read chronology, detector noise or mutable cache exists.
    """

    def __init__(
        self,
        grid: SpectralGrid,
        reads: ReadSchedule,
        passbands: Mapping[str, FilterPassband],
        *,
        accepted_etendue_m2_sr: float | Vector,
        quantum_efficiency: Efficiency | QuantumEfficiency,
        transmission: Efficiency = 1.0,
    ) -> None:
        """Prepare photon factors and unblocked wavelength-area quadrature once.

        Args:
            grid: Complete endpoint-sampled spectral coverage in nm or cm-1.
            reads: Actual native identities and positive exposure intervals.
                Each setting_id selects a supplied passband; no reads are synthesized.
            passbands: Physical filter declarations keyed by read setting identity.
                Selected filters are retained in first-read-occurrence order.
            accepted_etendue_m2_sr: Nonnegative scalar or (read,) accepted etendue.
                Use square_pixel_etendue for the explicitly chosen square-cone model.
            quantum_efficiency: Fractions broadcastable to (read,spectral), or a pure
                callback (read_indices,wavelengths_m) returning that exact shape.
                Scalar, spectral vector, (read,1) and (read,spectral) arrays are supported.
            transmission: Other optical fractions with the same broadcasting contract,
                including the caller's blocking law. Filter peak transmission is separate.

        Raises:
            ValueError: Required filters, support, represented durations, etendue or
                fractions are invalid. Supplied QE callback errors propagate.
        """
        if not reads.sample_ids:
            raise ValueError("filter preparation requires at least one native read")
        ids = tuple(dict.fromkeys(reads.setting_ids))
        if any(name not in passbands for name in ids):
            raise ValueError("every read setting must select a supplied filter passband")
        self._grid, self._reads = grid, reads
        self._filter_ids = ids
        self._passbands = tuple(passbands[name] for name in ids)
        positions = {name: i for i, name in enumerate(ids)}
        self._indices = np.array([positions[name] for name in reads.setting_ids], dtype=np.int64)
        self._wave_nm = _immutable(
            grid.points if grid.coordinate == "wavelength_nm" else 1e7 / grid.points
        )
        wavelength_grid = (
            grid if grid.coordinate == "wavelength_nm" else convert_spectral_grid(grid)
        )
        areas = spectral_trapezoid_weights(wavelength_grid)
        self._area_weights_nm = _immutable(
            areas if grid.coordinate == "wavelength_nm" else areas[::-1]
        )
        photon = photon_conversion_weights(grid)
        n, k = len(reads.sample_ids), grid.points.size
        etendue = np.broadcast_to(np.asarray(accepted_etendue_m2_sr, dtype=np.float64), (n,))
        if not np.isfinite(etendue).all() or np.any(etendue < 0):
            raise ValueError("accepted etendue must be finite nonnegative (read,) or scalar")
        if callable(quantum_efficiency):
            qe = quantum_efficiency(np.arange(n, dtype=np.int64), self._wave_nm * 1e-9)
            if qe.shape != (n, k):
                raise ValueError("QE callback must return (read,spectral)")
        else:
            qe = quantum_efficiency
        qe = efficiency_samples(qe, (n, k), "QE")
        optical = efficiency_samples(transmission, (n, k), "transmission")
        with np.errstate(over="ignore", invalid="ignore"):
            durations = np.diff(np.asarray(reads.exposure_bounds_s, dtype=np.float64), axis=1)[:, 0]
            if not np.isfinite(durations).all() or np.any(durations <= 0):
                raise ValueError("represented read durations must be finite and positive")
            self._base = _immutable(qe * optical * photon * (etendue * durations)[:, None])

    @property
    def grid(self) -> SpectralGrid:
        """Return the immutable prepared samples and density coordinate."""
        return self._grid

    @property
    def reads(self) -> ReadSchedule:
        """Return the actual immutable native read schedule."""
        return self._reads

    @property
    def filter_ids(self) -> tuple[str, ...]:
        """Return filter identities defining all trial-vector and area axes."""
        return self._filter_ids

    @property
    def passbands(self) -> tuple[FilterPassband, ...]:
        """Return immutable passband declarations in filter_ids order."""
        return self._passbands

    def coefficients(
        self,
        *,
        centres_nm: Vector | None = None,
        fwhm_nm: Vector | None = None,
        fixed_area_nm: Vector | None = None,
    ) -> SpectralElectronResponse:
        """Evaluate nominal or trial filters as a per-read electron map.

        Args:
            centres_nm: Optional positive centre vector in filter_ids order, nm.
            fwhm_nm: Optional positive FWHM vector in that order, nm.
            fixed_area_nm: Optional positive reference areas per filter, nm. Scale
                trial profiles by reference/trial sampled area before peak, blocking
                and photon weighting. Finite trial feasibility remains caller-owned.

        Returns:
            Immutable (read,spectral) coefficients on the actual read schedule.
            Durations enter once; coaddition is a separate explicit linear map.

        Raises:
            ValueError: Trials, sampled-area normalization or finite coefficients fail.
        """
        weights, _ = self._evaluate(centres_nm, fwhm_nm, fixed_area_nm)
        return SpectralElectronResponse(self.reads, self.grid, weights)

    def profile_areas_nm(
        self,
        *,
        centres_nm: Vector | None = None,
        fwhm_nm: Vector | None = None,
    ) -> FloatArray:
        """Integrate unit-peak profiles over the declared sampled wavelength domain.

        Args:
            centres_nm: Optional positive trial centres in filter_ids order, nm.
            fwhm_nm: Optional positive trial FWHM values in that order, nm.

        Returns:
            Fresh (filter,) sampled areas in nm, before peak scaling, blocking, QE
            or photon conversion. These are truncated sampled areas, not infinite-tail areas.

        Raises:
            ValueError: Trial shapes or physical/numerical domains are invalid.
        """
        return self._profiles(centres_nm, fwhm_nm) @ self._area_weights_nm

    def fixed_area_filter_sensitivities(
        self,
        step_nm: float = 0.01,
    ) -> tuple[SpectralElectronResponse, FloatArray, FloatArray]:
        """Evaluate nominal coefficients and area-preserving centre/FWHM directions.

        Args:
            step_nm: Positive finite symmetric difference step below every nominal
                FWHM and centre wavelength, nm. It must move every scalar in float64.
                Sampling and physical factors stay fixed.

        Returns:
            Nominal response and fresh signed (read,spectral) centre and FWHM
            derivatives per nm. Each row differentiates its selected filter; reads
            sharing a setting therefore share the same physical filter perturbation.
            All five evaluations reuse the prepared QE/transmission/photon factors.

        Raises:
            ValueError: Step, shifted filters or positive sampled-area requirements fail.
        """
        centres = np.array([p.centre_nm for p in self.passbands])
        widths = np.array([p.fwhm_nm for p in self.passbands])
        if not np.isfinite(step_nm) or not 0 < step_nm < min(widths.min(), centres.min()):
            raise ValueError("positive finite step below every centre and width required")
        if any(
            np.any(values + step_nm == values) or np.any(values - step_nm == values)
            for values in (centres, widths)
        ):
            raise ValueError("step must representably change every centre and width")
        nominal, areas = self._evaluate(None, None, None)
        plus, _ = self._evaluate(centres + step_nm, None, areas)
        minus, _ = self._evaluate(centres - step_nm, None, areas)
        centre = (plus - minus) / (2 * step_nm)
        plus, _ = self._evaluate(None, widths + step_nm, areas)
        minus, _ = self._evaluate(None, widths - step_nm, areas)
        width = (plus - minus) / (2 * step_nm)
        if not np.isfinite(centre).all() or not np.isfinite(width).all():
            raise ValueError("filter sensitivity arithmetic overflow")
        return SpectralElectronResponse(self.reads, self.grid, nominal), centre, width

    def _evaluate(self, centres, widths, fixed):
        profiles = self._profiles(centres, widths)
        areas = profiles @ self._area_weights_nm
        if fixed is not None:
            fixed = np.asarray(fixed, dtype=np.float64)
            if (
                fixed.shape != areas.shape
                or not np.isfinite(fixed).all()
                or np.any(fixed <= 0)
                or np.any(areas <= 0)
            ):
                raise ValueError("positive finite fixed and sampled trial areas required")
            profiles *= (fixed / areas)[:, None]
        peak = np.array([p.peak_transmission for p in self.passbands])
        with np.errstate(over="ignore", invalid="ignore"):
            weights = self._base * profiles[self._indices] * peak[self._indices, None]
        if not np.isfinite(weights).all():
            raise ValueError("filter coefficient arithmetic overflow")
        return weights, areas

    def _profiles(self, centres, widths):
        centres = (
            np.array([p.centre_nm for p in self.passbands])
            if centres is None
            else np.asarray(centres, dtype=np.float64)
        )
        widths = (
            np.array([p.fwhm_nm for p in self.passbands])
            if widths is None
            else np.asarray(widths, dtype=np.float64)
        )
        shape = (len(self.passbands),)
        if (
            centres.shape != shape
            or widths.shape != shape
            or not np.isfinite(centres).all()
            or not np.isfinite(widths).all()
            or np.any(centres <= 0)
            or np.any(widths <= 0)
        ):
            raise ValueError("matching positive finite filter centres and widths required")
        profiles = []
        for p, centre, width in zip(self.passbands, centres, widths, strict=True):
            if p.profile == "lorentzian":
                values = lorentzian_filter(self._wave_nm * 1e-9, centre * 1e-9, width * 1e-9)
            else:
                assert p.exponent is not None
                if p.exponent >= 2:
                    values = super_gaussian_filter(
                        self._wave_nm * 1e-9, centre * 1e-9, width * 1e-9, order=p.exponent / 2
                    )
                else:
                    values = np.exp(
                        -np.log(2) * np.abs(2 * (self._wave_nm - centre) / width) ** p.exponent
                    )
            profiles.append(values)
        return np.stack(profiles)


def _immutable(values: Vector) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("retained response values must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
