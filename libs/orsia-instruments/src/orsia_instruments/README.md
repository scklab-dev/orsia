# Inside `orsia_instruments`

`orsia_instruments` maps explicitly supported radiance fields through acquisition geometry and optical response to photoelectrons and optional detector readout. It neither computes atmospheric radiance nor reconstructs spectra or retrieves physical parameters. See the [API reference](../../README.md) for the detailed image and advanced-kernel contracts.

For a timed platform state and Earth-frame sight lines, use the [line-of-sight read-to-DN example](../../line-of-sight.md); it uses the optical charge kernel and detector sampler with an explicit radiance coverage assertion.

## Architecture at a glance

The [editable architecture diagram](../../docs/diagrams/architecture.drawio) and [SVG rendering](../../docs/diagrams/architecture.svg) show local angular imaging and Earth-frame line-of-sight sampling as separate scene inputs. Both paths use the photon-response and detector-sampling kernels. The module table maps each responsibility to its source.

### Where to look

| Source | Responsibility | Boundary |
| --- | --- | --- |
| `scene.py` | Radiance fields, sampling and the explicit uniform-field adapter | Spatial/time coverage is supplied, never inferred from a directional radiance. |
| `acquisition.py`, `imaging/acquisition.py` | Native reads, pixels, exposures, pointing and quadrature | Geometry is distinct from optical throughput and detector electronics. |
| `line_of_sight.py` | Earth-frame sight lines from read schedules and platform state | Caller supplies time, attitude and mount conventions. |
| `line_of_sight_radiance.py` | Radiance along sight lines with full-support coverage checks | Provider certifies the continuous time/FOV and spectral domain. |
| `quadrature.py` | Gauss-Legendre nodes and interval measure weights for acquisition and pixels | Owns the implementation; weights are not physical response weights or probabilities. |
| `optics/` | PSFs, finite pixels and filter/dispersive/two-beam/cavity responses | Named factories are conveniences within the supported physical contract, not a closed instrument registry. |
| `integration.py`, `charge.py` | Radiance integration, expected electrons and source-path combination | Optical fractions, area, exposure and QE each enter once. |
| `detector.py` | Detector maps, conditional moments, random readout and digitisation | Expected photoelectrons are not mean DN or a random detector realization. |
| `imaging/charge.py`, `imaging/readout.py` | Complete image charge and readout with per-exposure common noise | External photoelectrons join optical charge before sampling. |
| `imaging/measurements.py`, `imaging/simulation.py` | Dense measurements and composed image calls | Arrays have explicit exposure, row and column axes. |
| `tolerances.py` | Native sensitivities, uncertainty budgets and Monte Carlo transport | Perturbations and source laws remain explicit. |

The library uses public radiative contracts and uncertainty algebra. Array copying and physical validation use NumPy directly; `orsia_numerics.arrays` has been removed. The library does not import data services. Interval quadrature belongs to `orsia_instruments.quadrature`; the former `orsia_numerics.quadrature` module is removed without a compatibility facade.

## Progressive examples

The [dense response regression](../../tests/test_dense_imaging.py) exercises `image_acquisition`, `photoelectron_response`, `detector_response` and `response_electrons`. The [module map](../../README.md#module-organization) distinguishes acquisition geometry, positive scene response and detector readout.

Run the blocks **in order in one Python session** with `orsia-instruments` installed. This small uniform scene and idealized imager are teaching inputs, not a calibrated instrument. The low angular order keeps the example cheap and is not a convergence recommendation.

### 1. Give a directional spectrum spatial and temporal support

**Goal:** state the uniform-scene approximation instead of silently treating one radiance as an image.

```python
from functools import partial
import numpy as np
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid
from orsia_instruments import (
    DetectorMaps,
    exposure_sequence,
    filter_imager,
    gaussian_filter,
    regular_pixel_bounds,
    simulate_images,
    uniform_radiance_field,
)

radiance = Radiance(
    wavenumber_grid(np.linspace(4900.0, 5100.0, 65)),
    np.full(65, 1e-3),
    ObservationGeometry(0.0),
)
field = uniform_radiance_field(
    radiance,
    time_bounds_s=(0.0, 0.1),
    u_bounds=(-0.004, 0.004),
    v_bounds=(-0.004, 0.004),
    spectral_bounds_cm1=(4900.0, 5100.0),
)
assert field.values.shape == (2, 2, 2, 65)  # (time, u, v, spectrum)
```

Input radiance is W m⁻² sr⁻¹ per (cm⁻¹). The angular coordinates are the field's `u,v` coordinates; explicit bounds cover the accepted sky patch and all exposure times. The adapter does not establish scene homogeneity physically, extrapolate beyond coverage or simulate an atmosphere.

### 2. Integrate expected photoelectrons before asking for noise

**Goal:** compose geometry, a spectral filter, finite-pixel response and QE once.

```python
pixels = regular_pixel_bounds((2, 2), pitch_m=30e-6, active_size_m=28e-6)
optics = filter_imager(
    focal_length_m=0.02,
    collecting_area_m2=1e-8,
    psf_sigma_m=30e-6,
    filter=partial(gaussian_filter, centre_m=2e-6, width_m=20e-9),
    throughput=0.8,
)
exposures = exposure_sequence(
    start_s=[0.0, 0.05],
    duration_s=0.01,
    direction_bounds_uv=(-0.004, 0.004, -0.004, 0.004),
)
detector = DetectorMaps(
    quantum_efficiency=0.7,
    dark_electrons_per_s=2.0,
    read_noise_electrons=3.0,
    electrons_per_dn=2.0,
    bias_dn=100.0,
)
configuration = dict(
    pixel_bounds_m=pixels,
    optics=optics,
    exposures=exposures,
    detector=detector,
    angular_order=3,
)
expected = simulate_images(field, **configuration)
assert expected.photoelectrons.shape == (2, 2, 2)  # (exposure, row, column)
assert np.all(expected.photoelectrons > 0)
assert expected.dn is None
```

The filter callback receives wavelengths in metres, whereas the radiance grid is in cm⁻¹; the integrator owns the conversion. `width_m` is the Gaussian standard deviation, not FWHM or grid spacing. Photoelectrons precede dark current and electronics; omitting the RNG does not request mean DN. Spectral spacing, angular/time quadrature and PSF support need separate convergence checks in a real application.

### 3. Request readout with an explicit random generator

**Goal:** keep RNG ownership and output axes visible.

```python
noisy = simulate_images(
    field,
    **configuration,
    rng=np.random.default_rng(71),
    realisations=2,
)
assert noisy.dn.shape == (2, 2, 2, 2)  # (realisation, exposure, row, column)
np.testing.assert_allclose(noisy.photoelectrons, expected.photoelectrons)
assert noisy.full_well_saturated.shape == noisy.dn.shape
assert noisy.adc_clipped.shape == noisy.dn.shape

repeat = simulate_images(field, **configuration, rng=np.random.default_rng(71), realisations=2)
np.testing.assert_array_equal(repeat.dn, noisy.dn)
```

The same seed and complete input reproduce the detector samples. Inspect saturation and clipping flags alongside DN. Storage scales with the full acquisition, scene interpolation, physical response and requested realizations.

## Composition and limits

Use `image_photoelectrons`, `image_charge`, `combine_image_source_paths` and `image_readout` when charge sources or optical paths need separate control. Read the [composition guide](../../composition.md), [optical factory and PSF contracts](../../README.md#physical-description-image-api) and [explicit optical reuse contract](../../README.md#explicit-optical-reuse). Native measurements remain detector-space data; full-image resource costs, instrument/retrieval performance and concurrent-native safety require separate qualification.
