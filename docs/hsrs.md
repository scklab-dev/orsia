# TSIS-1 HSRS local integration

This document records the implemented offline boundary for a TSIS-1-style solar irradiance CSV and the experimental LISIRD acquisition adapter; it does not authenticate a provider edition or qualify the spectrum scientifically.

## Implemented path

`orsia_data.solar.read_solar_spectrum_csv(path)` reads the supported native wavelength-point columns: wavelength [nm], irradiance [W m-2 nm-1], and optional supplied uncertainty and bandwidth [nm]. It rejects malformed, empty, nonfinite, negative, duplicate, decreasing and unsupported data. It returns radiative `SolarIrradianceSpectrum`, with independent writable float64 arrays and no provenance fields.

`orsia_data.solar.lisird.fetch_solar_spectrum(product=..., edition="current", directory=...)` explicitly selects either `tsis1_hsrs` or `tsis1_hsrs_p025nm`, downloads the native CSV into a new directory and returns `SolarIrradianceSpectrum` after strict decoding. Users need not construct URLs or supply column mappings. The adapter remains experimental: the endpoint convention is based on inspected SCKLab code, not verified live LASP responses. Only the mutable current service is supported, not a pinned `v2` or archival edition. Reuse rights, reference distance, uncertainty meaning and source records stay with the application. The former solar artifact/manifests and acquire-snapshot APIs remain removed without aliases.

`orsia_radiative.solar_irradiance` converts wavelength point density to the canonical increasing cm-1 point density with the explicit Jacobian and performs piecewise-linear interpolation onto a caller-selected `SpectralGrid`. It rejects extrapolation and declared gaps and does not perform bin averaging, convolution or uncertainty propagation.

`scale_spectral_irradiance` applies only the caller-supplied inverse-square ratio between two positive distances. Date, ephemeris, geometry and any interpretation of native uncertainty remain consumer policy.

## Evidence boundary

LASP public access timed out during both the offline implementation and the acquisition extension. The product edition, source bandwidth/sample convention, reference distance, uncertainty meaning and reuse terms therefore remain caller-supplied declarations. The repository does not mark the inspected SCKLab labels as verified LASP evidence and does not include provider data. The implemented adapter is synthetically tested, not live-qualified.

| Subject | Evidence and implemented contract | Remaining limit |
| --- | --- | --- |
| Intended products | The experimental adapter resolves explicitly selected `tsis1_hsrs` and `tsis1_hsrs_p025nm` to the CSV endpoint convention inspected in SCKLab, with no product default. | The [LASP landing page](https://lasp.colorado.edu/lisird/data/tsis1_hsrs/) and attempted LaTiS accesses timed out; neither product has a verified live schema or edition here. |
| CSV schema | The inspected SCKLab reader supports the two, three and four ordered column schemas below. Synthetic Orsia tests use those exact headers. | No real-provider file was decoded in this pass. Other schemas are rejected rather than guessed. |
| Irradiance and uncertainty | CSV headers declare W m-2 nm-1; native uncertainty is retained independently of nominal target values. | Distribution, confidence level, correlations and absolute-density interpretation are not independently verified. Do not treat the field as one-sigma or interpolate it as covariance. |
| Bandwidth and sampling | Native bandwidth is retained in nm; irregular wavelength sampling is allowed. | Source bandwidth/resolution, spacing and integration-bin width are distinct. SCKLab's 0.025 nm resolution / 0.005 nm spacing and v2 label are unverified descriptors, not provider guarantees. |
| Distance and rights | These declarations are application-owned, not mandatory fields in a file or physical-value record. | Distance scaling requires an explicitly established reference distance; no default 1 AU or permission to redistribute provider bytes is inferred. |

`SolarIrradianceSpectrum` is an aligned native tabulation with ancillary fields and independent array ownership, not a file or transfer-method result. It is now owned/exported by `orsia-radiative`. This is a breaking migration; see the [data migration map](../libs/orsia-data/README.md#breaking-migration).

The schemas are exactly `wavelength (nm),irradiance (W/m^2/nm)`, the same pair followed by `uncertainty (W/m^2/nm)`, or that triple followed by `bandwidth (nm)`. Optional uncertainty and bandwidth are unavailable when absent, not zero. Headers, comments and empty rows are handled explicitly; partially missing numeric fields are errors.

## Offline public-API example

This example uses synthetic bytes, not HSRS data. Replace the local file only after checking its scientific contract and any applicable rights. Keep native ancillary fields separately from the adapted nominal irradiance.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from orsia_data.solar import read_solar_spectrum_csv
from orsia_radiative import (
    Atmosphere, ObservationGeometry, RadiativeScene, SpectralGrid, SurfaceState,
    compose_atmospheric_optics, solar_irradiance,
    scale_spectral_irradiance, solar_source, transfer,
)

with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "synthetic.csv").write_text(
        "wavelength (nm),irradiance (W/m^2/nm),uncertainty (W/m^2/nm),bandwidth (nm)\n"
        "500,1,0.1,0.025\n750,2,0.2,0.025\n1000,3,0.3,0.025\n",
        encoding="utf-8",
    )
    native = read_solar_spectrum_csv(root / "synthetic.csv")
    # Explicit regular cm-1 point grid, not wavelength-bin averages.
    target = SpectralGrid("wavenumber_cm1", np.linspace(10000.0, 20000.0, 11))
    nominal = solar_irradiance(native, target)
    # The 1 AU reference is a declaration of this synthetic example only.
    scaled = scale_spectral_irradiance(nominal, distance_au=2.0, reference_distance_au=1.0)
    source = solar_source(scaled, direction=(30.0, 0.0))
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]), np.array([100000.0, 90000.0]),
        np.array([95000.0]), np.array([296.0]), np.array([2e29]),
        np.array([1.9e29]), np.array([1e28]),
    )
    scene = RadiativeScene(
        atmosphere, SurfaceState(target, reflectance=np.full(11, 0.2)),
        ObservationGeometry(10.0),
    )
    optics = compose_atmospheric_optics(atmosphere, source)
    radiance = transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
    np.testing.assert_allclose(
        radiance.values, scaled.values * 0.2 * np.cos(np.deg2rad(30.0)) / np.pi,
    )
    assert radiance.grid is target
    assert native.uncertainty_per_nm is not None and native.bandwidth_nm is not None
```

## Acquisition and replay limits

`download_file` remains a low-level exact-URL primitive with a positive network timeout and a byte limit (128 MiB by default), Content-Length checks and optional SHA-256 verification before atomic no-clobber publication. `fetch_solar_spectrum` uses it and then the offline reader; existing directories are rejected before acquisition and new directories are removed on failure. Successful downloads remain as `<product>.csv` for explicit offline replay. No archival edition resolver, total elapsed-time deadline, resumable transfer or authenticated access is supplied. Without a trusted digest or declared Content-Length, transport cannot establish provider completeness. Raw files must not be concurrently mutated while reading or acquiring.

## Validation scope and plan status

The local CSV contract supports native field retention, explicit gaps supplied to materialisation, Jacobian/order, interpolation support, analytic coordinate-change checks, distance factors and transparent Lambertian source scaling. Controlled transport is independent of CSV/provider identity. Tests additionally exercise product resolution, transport-option forwarding, failure cleanup and native auxiliary-field retention through the experimental acquisition function. Authentic files for both products and live LASP access remain unqualified. The parent plan is retained unchanged as a historical proposal, not the current API authority.

The former local CSV probe is retired. [Synthetic interoperability tests](../tests/test_solar_interoperability.py) retain raw-row agreement, an independently evaluated density Jacobian/interpolation, nonnegative finite adapted values and deterministic replay, using the public reader and materialisation functions shown above. `devbox run -- just test` includes these checks. They do not authenticate an HSRS edition or establish upstream, atmospheric, retrieval or instrument accuracy; authentic-file verification still requires separately identified local bytes and evidence.

The implementation deliberately does not add conservative bin integration, instrument response, automatic Sun-Earth distance, solar variability, target-grid covariance or study-specific spectral defaults.
