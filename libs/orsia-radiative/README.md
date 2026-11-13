# Orsia Radiative

`orsia-radiative` composes materialised atmospheric state, spectroscopy and optical properties into reflected-solar or non-scattering LTE thermal radiance. It provides isolated Lorentz/Voigt lines, fixed cross sections, water continuum, explicit Rayleigh/Henyey-Greenstein aerosol roles and experimental fixed-order solar scattering. Explicit analytical tangents cover fixed-cross-section column absorption, solar transfer and LTE thermal transfer. Numerical calls use in-memory inputs without files or providers.

## Architecture

![Caller-supplied state and tangent seeds feed optical and solar or thermal calculations; native tangent kernels use shared Numerics headers.](docs/diagrams/architecture.svg)

Solid arrows show selected value flow; dashed arrows show native tangent kernels using Numerics headers. The caller supplies scene, spectral support and illumination to the selected transfer family. Tangents cover fixed-cross-section column absorption, solar constituents and non-scattering LTE; physical derivative formulas remain in Radiative. [Full-size SVG](docs/diagrams/architecture.svg) · [Editable Draw.io source](docs/diagrams/architecture.drawio).

## Minimal solar example

Run the example with this distribution and its dependencies installed, or in the [Orsia development environment](../../README.md#development-environment). A synthetic transparent layer and normal illumination recover Lambertian reflection on a two-point grid.

```python
import numpy as np
from orsia_radiative import LayerAtmosphere, ObservationGeometry, ColumnScene, SpectralIrradiance, SurfaceState, atmospheric_optics, solar_source, wavenumber_grid
from orsia_radiative.transfer import solar

grid = wavenumber_grid(np.array([800.0, 1000.0]))  # cm-1 point samples
atmosphere = LayerAtmosphere(
    np.array([0.0, 1000.0]),
    np.array([100000.0, 90000.0]),
    np.array([95000.0]),
    np.array([285.0]),
    np.array([2.0e29]),
    np.array([1.9e29]),
    np.array([1.0e28]),
)
surface = SurfaceState(grid, reflectance=np.array([0.1, 0.2]))
scene = ColumnScene(atmosphere, surface, ObservationGeometry(0.0))
source = solar_source(SpectralIrradiance(grid, np.array([1.0, 2.0])), direction=(0.0, 0.0))
optics = atmospheric_optics(atmosphere, grid)
result = solar.reflected_radiance(
    scene,
    source=source,
    optics=optics,
    max_order=0,
    delta_m_moments=None,
)
np.testing.assert_allclose(result.values, surface.reflectance * source.irradiance.values / np.pi)
print(np.round(result.values, 6).tolist())
```

Expected output: `[0.031831, 0.127324]`, in W m-2 sr-1 per (cm-1). The two samples illustrate the analytic limit; they do not resolve a real spectrum.

## Physical conventions and limits

- Spectral points are positive, increasing float64 samples. `Radiance`, `SpectralIrradiance` and fixed layer cross-section tables accept nm or cm-1; column transfer retains its cm-1 contract. Coverage domains, coordinate conversion and spectral-density Jacobians are explicit. Grid spacing and instrumental resolution are distinct.
- Atmospheric layers run bottom to top: height boundaries [m], pressure [Pa], temperature [K] and molecular columns [molecules/m2]. Optical depths have axes `(layer, point)`.
- Evaluated physical fields own writable copies; reusable line/laboratory references and spectral coordinates own read-only snapshots. Keep shared inputs stable during calls and recompute affected spectroscopy after state changes. Freshness checks require separately retained evaluation atmospheres.
- Solar geometry is local plane-parallel. Relative azimuth uses incoming photon propagation to outgoing view; use the [angular contract](docs/transfer.md#angular-contract-and-outward-ray-conversion) when adapting outward rays.
- [Cross-section optical-depth arrays](docs/spectroscopy.md#cross-section-optical-depth-arrays) provide `cross_section_optical_depth` and `cross_section_optical_depth_tangent` from the package root or `optics.cross_sections`. They compute direct depth or fused value/column tangents at fixed cross sections, independently of spectroscopy objects.
- [Analytical solar tangents](docs/transfer.md#solar-sos-constituent-tangents) cover absorption, Rayleigh and Henyey-Greenstein (HG) aerosol extinction/scattering depth, asymmetry and Lambertian reflectance, including fixed-order successive orders of scattering (SOS) and Delta-M. `solar.reflected_radiance_tangent` defaults to `max_order=0`; set it explicitly for scattering. The value-only `solar.reflected_radiance` and order-resolved `orders.scattering_orders_tangent` default to two orders. The required `optical_depth_tangent` seeds absorption only; optional constituent seeds supply Rayleigh and aerosol directions separately.
- [LTE thermal tangents](docs/transfer.md#lte-thermal-tangents) cover total absorption depth, layer and surface temperatures, and spectral emissivity. All transfer tangent direction axes are last, float64, with arbitrary runtime direction count, including zero; illumination, geometry and numerical controls stay fixed.
- Solar scattering requires separate order, angular and vertical convergence checks. LTE thermal transfer rejects nonzero scattering. Clouds, ice and a combined solar/thermal call are outside scope.

## Read next

- [Physical spectral bands](docs/bands-and-cubes.md): normalized spectral responses, ground `(y,x,band)` products, coverage and NaN validity.
- [Physical state and spectra](docs/state-and-spectra.md): source responsibilities, density conversion, support, ownership and freshness.
- [Atmospheric columns](docs/columns.md): hydrostatic construction, [log-linear level paths](docs/columns.md#level-atmospheres-and-column-integrals) with endpoint-product water continuum, and native-profile composition.
- [Spatial scenes and terrain/path contracts](../../docs/spatial-scenes.md): partial height-layer retention, separate path optics, direct Lambertian slant-depth transfer and detector Jacobians.
- [Spectroscopy and optics](docs/spectroscopy.md): reusable line/laboratory references, explicit evaluation, detailed tables and continuum.
- [Transfer reference](docs/transfer.md): solar/LTE APIs, executable examples, Delta-M and evidence limits.
- [Native execution](docs/native-execution.md): implementation details and precision contracts.

Run `devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests` or `devbox run -- just --justfile tests/justfile pytest libs/orsia-radiative/tests/test_thermal.py libs/orsia-radiative/tests/test_radiometry.py` from the workspace root. Synthetic numerical agreement leaves broad atmospheric accuracy, external-solver parity, instrument response and retrieval performance unqualified.

## Sampled absorption responses and aerosol mixtures

[`transfer.extinction`](src/orsia_radiative/transfer/extinction.py) supplies spectral exposure means, source/depth tangents and an explicit reference log-tangent approximation. Air masses and averaging weights are physical inputs. Band integration and fitted-coordinate binding belong to callers. The [extinction guide](docs/extinction.md) defines the numerical laws, the distinct mean-air-mass approximation and migration from the removed mixed callbacks.

[`lognormal_aerosol_optics`](src/orsia_radiative/optics/aerosol_mixture.py) accepts volume fractions, number-median radii, geometric widths, refractive-index tables and an explicit spherical-Mie callback. It integrates six log-radius standard deviations with Gauss-Legendre quadrature and retains extinction/scattering/asymmetry weighting. Population definitions, reference normalization wavelength and provider selection belong to the caller. The function imports no Mie backend and performs no data acquisition.

[`LevelAtmosphere` and `integrate_log_linear_columns`](src/orsia_radiative/atmosphere/levels.py) provide explicit species-labelled level state and log-linear material reconstruction. [`AtmosphericColumnIntegrals` and `dry_air_molar_density`](src/orsia_radiative/atmosphere/columns.py) retain material and explicit dry-air diagnostics. Spectroscopy stays separate in `LayerCrossSections` and `WaterContinuumCoefficients`; [column optical-depth contraction](src/orsia_radiative/optics/absorption.py) checks their support and pairing. The [column guide](docs/columns.md#level-atmospheres-and-column-integrals) defines axes, optional zero-based indices and numerical laws.
