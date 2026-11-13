# Physical spectral bands

`orsia_radiative.spectral.bands` owns `SpectralBand` and `spectral_band`. These physical spectral means are independent of instrument descriptions, acquisition schedules and file providers. Import them from `orsia_radiative` or their defining module.

## Sampled physical band means

`SpectralBand(name, grid, weights)` describes a positive discrete response on at least two increasing positive cm^-1 nodes. Weights are finite, nonnegative and sum to one within relative tolerance `1e-12`. The band mean is `sum(radiance_per_cm1 * weights)`, with units W m^-2 sr^-1 per (cm^-1). Weights include spectral quadrature. The constructor snapshots weights into writable float64 storage and shares the canonical `SpectralGrid` with immutable coordinates.

`spectral_band(name, grid, response_density)` multiplies a supplied positive physical density by nonuniform trapezoidal weights, then normalizes the integral. The density's meaning determines the band's meaning. An energy transmission and a photon sensitivity generally give different normalized responses. For example, a photon detector has an additional factor proportional to `1 / wavenumber_cm1`. Instruments retains that achieved response after scalar calibration.

When `grid.domain` is declared, `spectral_band` requires one connected interval and raises `ValueError` for disconnected domains. This restriction keeps trapezoids inside declared coverage. With `grid.domain=None`, the helper constructs discrete weights from the supplied nodes and density; physical coverage remains unknown.

The product stores the response actually supplied. It performs no spectral extrapolation, inversion or conversion between energy and photon means. Callers keep arrays stable while passing bands to other computations. A frozen record fixes attributes while its array buffers remain writable.

## Instrument-owned registered products

`BandRadianceCube` is owned by Instruments and imported from `orsia_instruments` or `orsia_instruments.observation.measurements`. Its registered `(y,x,band)` values, coverage and validity retain the existing product contract. See the [registered-product reference](../../orsia-instruments/docs/observation-cubes.md#ground-registered-band-products). Radiative retains `SpectralBand` because a physical spectral mean is independent of a camera.

## Validation

`libs/orsia-radiative/tests/test_bands.py` checks connected-domain and unknown-domain trapezoidal means, disconnected-domain rejection, energy/photon distinctions, independent array storage, invalid band inputs. Instruments tests own cube axis/validity checks. Camera integration tests additionally exercise the producer's `(y,x,band)` convention and missing-band behavior. General map interpolation and reusable synthetic-scene builders remain outside these products.
