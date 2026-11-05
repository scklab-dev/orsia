"""Load the five-band scene and verified local inputs as named Pydantic fields.

The acquisition manifest keeps its existing schema and provenance. This module
only loads settings and resolves paths; scientific calculations stay in cells.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from orsia_data.acquisition import file_sha256
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from notebooks.support.notebook_five_band_support import verify_inputs


class LineAbsorber(BaseModel):
    """A species' local HITRAN files, support and optional dry-air layer fractions.

    Mass is in atomic mass units; pressure support is in Pa. Fractions are
    bottom-to-top mol/mol dry air and are absent for water or MIPAS abundances.
    Other acquisition metadata remains in the unchanged source manifest.
    """

    species: str
    lines: Path
    partition: Path
    molecule_id: int
    isotopologue_id: int
    mass_u: float
    pressure_range_pa: tuple[float, float]
    included_cases: list[str]
    dry_air_mole_fraction: list[float] | None = None


class TraceGasProfiles(BaseModel):
    """A local MIPAS file and the species selected for pressure-layer projection."""

    file: Path
    species: list[str]


class MeasuredCrossSections(BaseModel):
    """Native UV files with laboratory temperatures [K] and interpolation policy.

    Extra acquisition metadata remains available during explicit offline collection
    preparation. The public Data reader owns native units and interpretation.
    """

    model_config = ConfigDict(extra="allow")

    species: str
    format: Literal["serdyuchenko", "bogumil"]
    files: list[Path]
    temperatures_k: list[float]
    included_cases: list[str]
    temperature_policy: Literal["clip", "error"]
    pressure_policy: str


class FiveBandInputs(BaseModel):
    """Local atmospheric/spectral files and the manifest's sampling coordinates.

    Pressure interfaces are bottom-to-top Pa. Coordinates are native names in
    (time, pressure, latitude, longitude) order. Angles are degrees north/east.
    The loader resolves file paths relative to the acquisition manifest.
    """

    era5: Path
    solar_csv: Path
    coordinates: tuple[str, str, str, str]
    backend: Literal["classic", "hdf5"]
    valid_time: datetime
    latitude_deg: float
    longitude_deg: float
    pressure_interfaces_pa: list[float]
    spectroscopy: list[LineAbsorber]
    mipas: TraceGasProfiles
    cross_sections: list[MeasuredCrossSections]


class SolarDiscretization(BaseModel):
    """Successive-scattering, angular-quadrature and physical-layer subdivisions."""

    model_config = ConfigDict(extra="forbid")

    max_order: int = Field(ge=1)
    angular_order: int = Field(ge=1)
    layer_subdivisions: int = Field(ge=1)


class SpectralBand(BaseModel):
    """A vacuum-wavelength interval [nm] and its solar or dark-thermal controls."""

    model_config = ConfigDict(extra="forbid")

    low_nm: float = Field(gt=0)
    high_nm: float = Field(gt=0)
    thermal: bool
    solar: SolarDiscretization | None = None

    @model_validator(mode="after")
    def check_window_and_transport(self) -> Self:
        """Require an ordered window and solar orders exactly for solar bands."""
        if self.high_nm <= self.low_nm:
            raise ValueError("high_nm must exceed low_nm")
        if self.thermal != (self.solar is None):
            raise ValueError("solar orders are required exactly for nonthermal bands")
        return self


class ExponentialAerosol(BaseModel):
    """Solar aerosol column with an exponential profile and an Angstrom spectrum.

    Attributes:
        optical_depth: Dimensionless column extinction at reference_wavelength_nm.
        reference_wavelength_nm: Vacuum reference wavelength in nm.
        angstrom_exponent: Dimensionless alpha in extinction proportional to wavelength**(-alpha).
        scale_height_m: Exponential e-folding height in m, above the bottom interface.
        single_scattering_albedo: Scattering/extinction ratio, uniform across layers/bands.
        asymmetry: Dimensionless HG asymmetry, uniform across layers/bands.

    The column is normalized over the actual bottom and top interfaces. These
    illustrative parameters apply to UV/VNIR/SWIR only.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    optical_depth: float = Field(ge=0)
    reference_wavelength_nm: float = Field(gt=0)
    angstrom_exponent: float = Field(ge=0)
    scale_height_m: float = Field(gt=0)
    single_scattering_albedo: float = Field(ge=0, le=1)
    asymmetry: float = Field(ge=-0.8, le=0.8)


class ScenePhysics(BaseModel):
    """Surface, viewing/solar angles, dry-air Rayleigh law and solar aerosol column.

    Reflectance/emissivity are dimensionless, temperature is K, angles are
    degrees, and Rayleigh reference area/wavelength are m2/molecule and microns.
    The reference worker fixes incoming-to-view relative azimuth at zero.
    """

    model_config = ConfigDict(extra="forbid")

    albedo: float
    emissivity: float
    surface_temperature_k: float
    solar_zenith_deg: float
    relative_azimuth_deg: Literal[0.0]
    viewing_zenith_deg: float
    rayleigh_cross_section_m2: float
    rayleigh_wavelength_um: float
    solar_aerosol: ExponentialAerosol


class Sasktran2Settings(BaseModel):
    """SASKTRAN2 discretization and persistent-cache access settings.

    Cells count interior altitude nodes per physical layer; streams count
    angular directions. The loader resolves the cache path beside the scenario.
    """

    model_config = ConfigDict(extra="forbid")

    cells: int = Field(ge=1)
    streams: int = Field(ge=1)
    cache_directory: Path
    cache_mode: Literal["reuse", "refresh", "off"]


class FiveBandScenario(BaseModel):
    """Five-band physical assumptions, solver settings and verified local inputs.

    The manifest and scenario remain separate files. Spectral points are uniform
    in increasing cm-1; timing_runs counts warm complete-call repetitions.
    The notebook keeps gravity, broadening conventions and thermal method explicit.
    """

    model_config = ConfigDict(extra="forbid")

    manifest: Path
    spectroscopy_collection: Path
    inputs: FiveBandInputs
    bands: dict[str, SpectralBand]
    physics: ScenePhysics
    points: int = Field(ge=2)
    line_margin_cm1: float = Field(gt=0)
    wing_cutoff_cm1: float = Field(gt=0)
    timing_runs: int = Field(ge=1)
    reference: Sasktran2Settings


def load_scenario(path: Path) -> tuple[FiveBandScenario, list[Path], dict[str, str]]:
    """Load settings, verify manifest checksums and resolve named input paths.

    Args:
        path: Scenario JSON. Its manifest/cache paths are relative to this file;
            atmospheric/spectral paths are relative to the acquisition manifest.

    Returns:
        Validated scenario, original input-path list and SHA-256 provenance.
        Provenance retains the original manifest and inputs and adds the scenario.

    Raises:
        OSError: A local scenario, manifest or input file cannot be read.
        ValueError: JSON or a declared Pydantic field is invalid.
        AssertionError: The manifest lacks enrichment fields or a checksum differs.
    """
    path = path.resolve()
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("scenario JSON must be an object")
    manifest_name = TypeAdapter(Path).validate_python(settings.get("manifest"))
    manifest_path = (path.parent / manifest_name).resolve()
    manifest, base, input_paths, provenance = verify_inputs(manifest_path)
    scenario = FiveBandScenario.model_validate(
        {**settings, "manifest": manifest_path, "inputs": manifest}
    )
    scenario.spectroscopy_collection = (path.parent / scenario.spectroscopy_collection).resolve()
    inputs = scenario.inputs
    inputs.era5 = (base / inputs.era5).resolve()
    inputs.solar_csv = (base / inputs.solar_csv).resolve()
    inputs.mipas.file = (base / inputs.mipas.file).resolve()
    for gas in inputs.spectroscopy:
        gas.lines = (base / gas.lines).resolve()
        gas.partition = (base / gas.partition).resolve()
    for entry in inputs.cross_sections:
        entry.files = [(base / filename).resolve() for filename in entry.files]
    scenario.reference.cache_directory = (
        path.parent / scenario.reference.cache_directory
    ).resolve()
    provenance[str(path)] = file_sha256(path)
    return scenario, input_paths, provenance
