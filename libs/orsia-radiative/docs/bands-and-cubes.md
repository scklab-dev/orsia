# Physical spectral bands and radiance cubes

`orsia_radiative.bands` owns `SpectralBand`, `spectral_band` and `RadianceCube`. These numerical products are independent of instrument descriptions, acquisition schedules and file providers. Import the products from `orsia_radiative` or their owning module.

## Sampled physical band means

`SpectralBand(name, grid, weights)` describes a positive discrete response on at least two increasing positive cm^-1 nodes. Weights are finite, nonnegative and sum to one within relative tolerance `1e-12`. The band mean is `sum(radiance_per_cm1 * weights)`, with units W m^-2 sr^-1 per (cm^-1). Weights include spectral quadrature. The constructor snapshots weights into writable float64 storage and shares the canonical `SpectralGrid` with immutable coordinates.

`spectral_band(name, grid, response_density)` multiplies a supplied positive physical density by nonuniform trapezoidal weights, then normalizes the integral. The density's meaning determines the band's meaning. An energy transmission and a photon sensitivity generally give different normalized responses. For example, a photon detector has an additional factor proportional to `1 / wavenumber_cm1`. Instruments retains that achieved response after scalar calibration.

When `grid.domain` is declared, `spectral_band` requires one connected interval and raises `ValueError` for disconnected domains. This restriction keeps trapezoids inside declared coverage. With `grid.domain=None`, the helper constructs discrete weights from the supplied nodes and density; physical coverage remains unknown.

The product stores the response actually supplied. It performs no spectral extrapolation, inversion or conversion between energy and photon means. Callers keep arrays stable while passing bands to other computations. A frozen record fixes attributes while its array buffers remain writable.

## Ground-registered cube

`RadianceCube(x_m, y_m, bands, values, coverage, valid)` stores band means with axes `(y,x,band)`. The x and y axes are finite increasing cell centres in metres. A single centre is supported. The caller declares the surface chart, terrestrial frame and datum externally; numerical coordinates alone do not identify a map projection.

`values` has units W m^-2 sr^-1 per (cm^-1). Every valid value is finite and has positive coverage. Invalid elements contain NaN. `coverage` is finite and nonnegative, with meaning established by the producer. The fixed-filter reconstruction uses dimensionless accumulated normalized spatial response weight. Coverage is distinct from physical covered area.

The constructor validates axes, shapes, coverage and NaN/validity consistency. It snapshots coordinate, value, coverage and validity arrays into writable storage. The band tuple retains its supplied objects; keep their buffers stable during use. Cube construction performs no resampling or scene construction.

The [fixed-filter camera guide](../../orsia-instruments/docs/filter-camera.md) describes an implemented producer. Its reconstruction retains compatible achieved spectral responses and uses separately supplied estimated geometry. Missing bands remain invalid with NaN values. Calibration and reconstruction belong to Instruments; this module contains physical products that other algorithms can also consume.

## Validation

`libs/orsia-radiative/tests/test_bands.py` checks connected-domain and unknown-domain trapezoidal means, disconnected-domain rejection, energy/photon distinctions, independent array storage, invalid inputs and cube axis/validity rules. Camera integration tests additionally exercise the producer's `(y,x,band)` convention and missing-band behavior. General map interpolation and reusable synthetic-scene builders remain outside these products.
