# Detector-owned LVF, two-port and cavity imaging

These chains share finite-pixel acquisition, physical photon integration, detector readout and supplied calibration. `focal_plane_imager` attaches an explicit detector and finite point-spread-function (PSF) support to `lvf_imager`, `interferometric_imager` or `cavity_imager`. `camera_acquisition_grid` selects the incident rays and exposure-motion quadrature. `simulate_frames` evaluates caller-supplied radiance at those coordinates and returns expected electrons, with optional seeded digital numbers (DN).

The examples below are local analytic fixtures. Run the Python blocks in order. They require Instruments and Inverse, including their declared dependencies. All callbacks are pure and stable; returned numerical arrays own writable storage unless the API declares immutable schedules or spectral coordinates.

## Shared detector, scene and acquisition

The detector has two 20-micrometre pixels, a 4-micrometre Gaussian PSF and a caller-certified 25-micrometre support radius. The response clips incident centroids outside each expanded pixel rectangle without renormalizing the omitted tails. The imager boundary applies pupil projection once. The optical kernels integrate the finite pixel, including joint spatial transmission.

```python
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
from orsia_instruments import ExpectedElectrons, detector_operating_points, electron_moments, GaussianPSF, calibrate_frame_electrons, calibrate_frames, camera_acquisition_grid, cavity_imager, focal_plane_imager, frame_response, group_native_bands, interferometric_imager, lvf_imager, raster_read_schedule, reconstruct_radiance_cube, regular_detector, simulate_frames, spatial_spectral_response, stack_frame_electrons
from orsia_inverse.linear import solve_linear
from orsia_radiative import SpectralDomain, wavenumber_grid

spectral = wavenumber_grid(
    np.linspace(4500.0, 5500.0, 161),
    domain=SpectralDomain("wavenumber_cm1", ((4500.0, 5500.0),)),
)
detector = regular_detector(
    (1, 2),
    pitch_m=20e-6,
    active_size_m=20e-6,
    centre_xy_m=(0, 0),
    quantum_efficiency=0.8,
    dark_electrons_per_s=2.0,
    read_noise_electrons=1.0,
    full_well_electrons=1e8,
    electrons_per_dn=1.0,
    bias_dn=100.0,
    nonlinearity=0,
    adc_bits=24,
)
optical = dict(focal_length_m=0.3, collecting_area_m2=0.004, spatial=GaussianPSF(4e-6))
starts = np.arange(8) * 0.01
intervals = np.column_stack((starts, starts + 0.005))


def trajectory(times):
    positions = np.column_stack((times * 0.2, times * 0, times * 0 + 300))
    return positions, np.broadcast_to([0.2, 0, 0], positions.shape)


def platform(times):
    return np.broadcast_to(np.diag([-1.0, 1.0, -1.0]), (len(times), 3, 3))


def attitude(times):
    return np.broadcast_to(np.eye(3), (len(times), 3, 3))


def surface(origins, directions, times):
    distance = -origins[..., 2] / directions[..., 2]
    return (origins + distance[..., None] * directions)[..., :2], -directions, distance > 0


def acquisition(camera, trajectory_law=trajectory):
    return camera_acquisition_grid(
        camera,
        raster_read_schedule(intervals, detector.shape, port=camera.optics.port),
        epoch=datetime(2023, 5, 16, tzinfo=UTC),
        validity_s=(0, 1),
        trajectory=trajectory_law,
        platform_attitude=platform,
        instrument_attitude=attitude,
        surface=surface,
        motion_bound_m_s=0.2,
        angular_bound_m=301.0,
        time_order=2,
        angular_order=14,
    )


def scene(xy, directions, times, spectral):
    # Static, direction-independent, two-cell/two-bin spectral radiance density.
    points = spectral.points
    return np.where(
        xy[..., 0, None] < 0,
        np.where(points < 5000, 1e-5, 3e-5),
        np.where(points < 5000, 4e-5, 2e-5),
    )


def coverage(grid, spectral_grid):
    # This analytic fixture is defined over all selected rays and the declared band.
    return np.ones(len(grid.read_bands), dtype=bool)
```

The surface fixture explicitly shares GCRS Cartesian axes with its chart and returns reversed outward views. `scene` receives flat `(N,2)` coordinates, `(N,3)` chart-local unit views, `(N,)` elapsed seconds and `(K,)` wavenumbers, then returns `(N,K)` radiance. Instruments flattens reads then nodes in C order. Scene values are W m^-2 sr^-1 per (cm^-1). The source's coverage callback certifies the complete continuous acquisition support and declared spectral interval. Grid endpoints alone cannot establish physical rejection outside that interval. Supply order blocking or certify that excluded illumination is negligible for the intended model.

## LVF: fixed detector-plane law and achieved bands

The LVF centre is a fixed function of detector x. Scanning changes the surface viewed by each pixel while preserving that law. Finite-pixel PSF/filter integration determines the achieved spectral coefficients. `group_native_bands` merges only normalized photon responses that satisfy the supplied pairwise L1 tolerance. It keeps complete spectral shapes, including multiple peaks. Physical read, exposure and port identities remain unchanged.

```python
lvf = focal_plane_imager(
    detector,
    lvf_imager(**optical, reference_m=2e-6, gradient=0.01, sigma_m=50e-9, pixel_order=14),
    support_radius_m=25e-6,
)
lvf_truth_grid = acquisition(lvf)
lvf_frames = simulate_frames(
    lvf,
    lvf_truth_grid,
    spectral,
    scene,
    coverage,
    rng=np.random.default_rng(42),
)
# Independently supplied characterization; this closure case uses identical estimates.
estimated_detector = replace(detector)
estimated_lvf = replace(lvf, detector=estimated_detector)
lvf_response = frame_response(estimated_lvf, lvf_truth_grid, spectral)
grouped_lvf, lvf_bands = group_native_bands(lvf_response)
lvf_calibrated = calibrate_frames(
    lvf_frames.images,
    grouped_lvf,
    detector=estimated_detector,
)
lvf_estimated_grid = acquisition(estimated_lvf)
lvf_cube = reconstruct_radiance_cube(
    lvf_calibrated,
    lvf_estimated_grid,
    x_edges_m=np.linspace(-0.1, 0.1, 21),
    y_edges_m=np.array([-0.1, 0.1]),
    bands=lvf_bands,
)
assert len(lvf_bands) == 2
assert np.any(lvf_cube.valid)
```

This cube registers native achieved-band means using the flat-spectrum spatial marginal. Heterogeneous spectra retain spectral/spatial mixing and smoothing. `lvf_cube.coverage` records accumulated spatial weight; missing bands and cells remain invalid NaN. Use the joint operator below when a piecewise-constant spatial/spectral inverse model is appropriate.

## Two-beam: separate detectors and a joint inverse

Both physical ports remain nonnegative photon-counting paths. OPD is optical path difference in metres, constant during each exposure; the optional detector-plane `opd_offset_m` is integrated jointly with the PSF. Correct each port's gain, pedestal and dark before any signed combination. The joint operator uses both corrected ports directly, preserving their common intensity information and full supplied covariance.

```python
ports = [
    focal_plane_imager(
        replace(detector, electrons_per_dn=gain),
        interferometric_imager(
            **optical,
            opd_m=np.linspace(0, 30e-6, 8),
            port=sign,
            opd_offset_m=lambda x, y: 0.05 * x,
        ),
        support_radius_m=25e-6,
    )
    for sign, gain in (("+", 1.0), ("-", 2.0))
]
port_grids = [acquisition(camera) for camera in ports]
port_frames = [
    simulate_frames(camera, grid, spectral, scene, coverage, rng=np.random.default_rng(seed))
    for camera, grid, seed in zip(ports, port_grids, (21, 22), strict=True)
]
estimated_ports = [replace(camera, detector=replace(camera.detector)) for camera in ports]
estimated_grids = [acquisition(camera) for camera in estimated_ports]
response = spatial_spectral_response(
    estimated_ports,
    estimated_grids,
    spectral,
    x_edges_m=np.array([-1.0, 0, 1]),
    y_edges_m=np.array([-1.0, 1]),
    spectral_edges_cm1=np.array([4500.0, 5000.0, 5500.0]),
    read_batch_size=4,
    max_bytes=100_000_000,
)
# Known synthetic photon/electronic model, independent of the sampled DN.
# No common-mode noise was simulated; these per-read variances are independent.
charges = []
dn_variances = []
for frames, physical, estimated in zip(port_frames, ports, estimated_ports, strict=True):
    schedule = frames.images.schedule
    _, rows, columns = schedule.native_indices.T
    operating = detector_operating_points(physical.detector, rows=rows, columns=columns)
    mean_signal = ExpectedElectrons(schedule.reads, {"scene": frames.images.photoelectrons.ravel()})
    _, analogue_variance = electron_moments(mean_signal, operating)
    # Independent uniform rounding error is an approximation, in DN squared.
    variance_dn = analogue_variance / operating.electrons_per_dn**2 + 1 / 12
    dn_variances.append(variance_dn)
    charges.append(calibrate_frame_electrons(
        frames.images, detector=estimated.detector, covariance=variance_dn, mask_flagged=True,
    ))
observed, valid = stack_frame_electrons(response, charges)
covariance = np.concatenate([charge.covariance for charge in charges])
estimate = solve_linear(
    response.matrix, observed, covariance=covariance, valid=valid,
    rcond=1e-10, max_bytes=10_000_000,
)
assert estimate.rank == 4
recovered = estimate.state.reshape(response.state_shape)

# Separate noiseless expectation diagnostic; the primary estimate above uses DN.
expectation_charges = [
    calibrate_frame_electrons(frames.images, detector=camera.detector, use_expectation=True)
    for frames, camera in zip(port_frames, estimated_ports, strict=True)
]
expected_observed, expected_valid = stack_frame_electrons(response, expectation_charges)
expectation_estimate = solve_linear(
    response.matrix, expected_observed, covariance=covariance, valid=expected_valid,
    rcond=1e-10, max_bytes=10_000_000,
)
np.testing.assert_allclose(expectation_estimate.state, [1e-5, 3e-5, 4e-5, 2e-5], rtol=1e-10)
assert not np.array_equal(estimate.state, expectation_estimate.state)
```

The primary inverse above uses actual sampled DN after estimated electronic correction. `electron_moments` supplies analogue shot, dark and independent read-noise variance from the known synthetic model. The added 1/12 DN-squared term approximates independent uniform quantization error; it is not an exact quantized-noise covariance. The separately labelled expectation diagnostic verifies exact closure of the piecewise-constant source.

The matrix stacks separate `RasterReadSchedule` objects in the supplied order. `stack_frame_electrons` verifies every corrected observation schedule against the response, including port identity, native read order and exposure timing. A schedule index plus its native read identity identifies an observation; sample IDs can coincide across ports. Selected or permuted reads preserve their original correspondence. `calibrate_frame_electrons` optionally propagates supplied DN covariance through the estimated gain. A combined covariance can include cross-port correlations. Independent simulation calls generate independent stochastic detector noise; they do not synthesize cross-port common fluctuations.

`SpatialSpectralResponse` uses the same batched joint coefficient generator as frame simulation. It retains actual motion, PSF, spectral transmission, QE and finite-pixel response. Spatial cells must enclose every conservative continuous footprint, including exterior illumination. Spectral bins must enclose every declared spectral node. The state is static and direction-independent, with constant radiance density in each `(y,x,spectral_bin)` cell. Refine quadrature near cell/bin discontinuities. Unmodelled temporal, angular or within-bin spectral variation can bias the inverse.

`solve_linear` uses covariance whitening and a rank-truncated singular-value decomposition. Its signed `resolution` describes the expected estimated-state response and remains separate from positive `SpectralBand` objects. `coverage` measures physical sensitivity; `identifiable` tests whether individual state coordinates are resolved. Missing reads, repeated OPDs or zero visibility can reduce rank despite positive coverage. Unresolved state components follow a minimum-norm convention. Their true uncertainty is unconstrained; the returned covariance measures noise in that estimator and does not assign certainty to its null space. The memory budget is a conservative dense-array estimate, excluding backend allocator overhead. Oversized construction or solving raises `MemoryError` before the large dense allocation.

### Bounded response products

`imaging.response.joint.spatial_spectral_operator` applies the same cell basis and photon coefficients as `spatial_spectral_response`. Each product recomputes bounded read batches without allocating the full response matrix. State coordinates use C-order `(y, x, spectral_bin)` radiance density in W m-2 sr-1 per (cm-1). Output coordinates stack ports in supplied order and preserve each native read schedule.

An optional fixed `valid` mask selects rows in that order. The mask and cell edges are snapshotted; instruments, acquisition grids and spectral support are borrowed and must remain stable for the operator lifetime. The transpose accumulates the physical electron response coefficients without normalization. Normalized radiance reconstruction has its own contract.

The per-call memory estimate covers coefficient workspace, gathered direction blocks and results. Provider-internal allocations and numerical-adapter copies are outside its scope. A budget failure raises `MemoryError` before applying the requested block. The [Numerics adapter](../../orsia-numerics/docs/tangents.md#general-linear-operators) defines block shapes and ownership.

## Fabry-Perot: retain the complete transmitted response

The cavity uses an equal-mirror local Airy law. Spacing is fixed or supplied per exposure; a spatial offset can vary local thickness. The declared internal incidence, refractive index, intensity reflectivity, coating phase and order-selection transmission determine the response. Every transmitted peak inside the spectral support remains in its achieved band. Local-cavity physics excludes coherent lateral coupling and walk-off.

```python
cavity = focal_plane_imager(
    detector,
    cavity_imager(
        **optical,
        spacing_m=np.linspace(4e-6, 4.4e-6, 8),
        refractive_index=1.0,
        internal_angle_rad=0.0,
        reflectivity=0.7,
        coating_phase_rad=0.0,
        order_transmission=1.0,
    ),
    support_radius_m=25e-6,
)
cavity_grid = acquisition(cavity)
cavity_frames = simulate_frames(cavity, cavity_grid, spectral, scene, coverage, rng=np.random.default_rng(43))
estimated_cavity = replace(cavity, detector=replace(detector))
cavity_response = frame_response(estimated_cavity, cavity_grid, spectral)
grouped_cavity, cavity_bands = group_native_bands(cavity_response)
cavity_calibrated = calibrate_frames(
    cavity_frames.images, grouped_cavity, detector=estimated_cavity.detector
)
cavity_cube = reconstruct_radiance_cube(
    cavity_calibrated,
    acquisition(estimated_cavity),
    x_edges_m=np.array([-1.0, 1]),
    y_edges_m=np.array([-1.0, 1]),
    bands=cavity_bands,
)
assert len(cavity_bands) == 8
assert cavity_cube.valid.all()
```

Electronic calibration supports linear operating points only: estimated gain, pedestal and mean dark correction. Nonzero detector nonlinearity is rejected; nonlinear inversion is outside the implemented correction. All characterization in this guide is supplied. Calibration estimation, optical design feasibility and retrieval-level performance require separate evidence. The software tests cover analytic limiting cases, native cubes, joint heterogeneous closure, covariance transport, rank loss, flags, support rejection and bounded allocation.

## Breaking consolidation

`frame_response` replaces the identical `camera_response` and `spectrometer_response` operations. Direct consumers import the shared operation. Geometric `read_bands` are descriptive labels; reconstruction checks physical read identities, timing and quadrature independently of spectral grouping labels. Existing fixed-filter and slit acquisition semantics are preserved.
