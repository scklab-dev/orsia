#!/usr/bin/env bash
set -euo pipefail

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for package in orsia-numerics orsia-uncertainty orsia-radiative orsia-data orsia-instruments orsia-inverse; do
    uv build --package "$package" --out-dir "$tmp/dist"
done
shopt -s nullglob
wheels=("$tmp"/dist/*.whl)
if (( ${#wheels[@]} != 6 )); then
    echo "expected exactly six built wheels" >&2
    exit 1
fi
# Each distribution must provision its own native requirements: installing both
# together could otherwise hide an undeclared SciPy dependency in uncertainty.
for wheel in "${wheels[@]}"; do
    uv venv --python "$(cat .python-version)" "$tmp/standalone"
    # Resolve only declared dependencies from independently built local wheels;
    # do not install unrelated workspace packages to mask missing declarations.
    uv pip install --python "$tmp/standalone/bin/python" --find-links "$tmp/dist" "$wheel"
    "$tmp/standalone/bin/python" -I - "$wheel" <<'PY'
import sys
from pathlib import Path

import numpy as np

if "orsia_numerics" in Path(sys.argv[1]).name:
    from orsia_numerics.execution import CpuThreadPoolExecutor
    from importlib.util import find_spec
    assert find_spec("orsia_numerics.arrays") is None
    assert find_spec("orsia_numerics.interpolation") is None
    assert find_spec("orsia_numerics.quadrature") is None
    import os
    with CpuThreadPoolExecutor(1) as executor:
        assert executor.submit(pow, 3, 2).result(timeout=10) == 9
        assert len(executor.submit(os.sched_getaffinity, 0).result(timeout=10)) == 1
        assert list(executor.map(abs, [-2, -1], buffersize=1)) == [2, 1]
    values = np.arange(3.)
    retained = np.copy(values)
    values.fill(0)
    np.testing.assert_array_equal(retained, [0., 1., 2.])
    assert retained.flags.writeable
    retained[0] = 10
    assert values[0] == 0
    assert "site-packages" in str(Path(sys.modules["orsia_numerics"].__file__).resolve())
    assert not ({"orsia_radiative", "orsia_uncertainty", "orsia_instruments", "orsia_data"} & sys.modules.keys())
elif "orsia_inverse" in Path(sys.argv[1]).name:
    from importlib.util import find_spec
    from orsia_inverse import solve_map, retrieval_diagnostics
    assert find_spec("orsia_numerics") is None
    assert find_spec("orsia_uncertainty") is None
    estimate = solve_map(lambda q: q**2, lambda q: np.diag(2*q),
        np.array([4.0]), np.array([1.0]), observation_covariance=np.ones(1))
    assert estimate.success
    assert estimate.state.flags.writeable
    np.testing.assert_allclose(estimate.state, [2.0])
    diagnostics = retrieval_diagnostics(estimate, include_gain=True, include_averaging_kernel=True)
    np.testing.assert_allclose(diagnostics.posterior_covariance, [[1/16]])
    np.testing.assert_allclose(diagnostics.gain, [[1/4]])
    np.testing.assert_allclose(diagnostics.averaging_kernel, [[1.]])
    assert "site-packages" in str(Path(sys.modules["orsia_inverse"].__file__).resolve())
    assert not ({"orsia_radiative", "orsia_instruments", "orsia_data"} & sys.modules.keys())
elif "orsia_uncertainty" in Path(sys.argv[1]).name:
    from importlib.util import find_spec
    from orsia_uncertainty import covariance_solver, whitener, validate_covariance
    import orsia_uncertainty as uncertainty
    assert find_spec("orsia_numerics") is None
    assert find_spec("orsia_uncertainty.layout") is None
    for obsolete in ("ArrayLayout", "Covariance", "DiagonalCovariance", "SignedBias", "Contribution", "UncertaintySource"):
        assert not hasattr(uncertainty, obsolete)
    covariance = validate_covariance(np.array([[4.0, 1.0], [1.0, 3.0]]))
    assert covariance.flags.writeable
    np.testing.assert_allclose(covariance_solver(covariance, method="cholesky")(
        np.array([7.0, -1.0])), [2.0, -1.0], atol=1e-14)
    white = whitener(covariance)(np.eye(2))
    np.testing.assert_allclose(white @ covariance @ white.T, np.eye(2), atol=1e-14)
    assert "site-packages" in str(Path(sys.modules["orsia_uncertainty"].__file__).resolve())
elif "orsia_radiative" in Path(sys.argv[1]).name:
    from orsia_radiative import Atmosphere, Absorber, Line, LineSet, line_absorption
    from orsia_radiative.spectroscopy.components import values_equal
    from orsia_radiative.transfer.formal import integrated_exponential, propagation_coefficients, apply_propagation
    from orsia_radiative.transfer import formal
    from orsia_radiative.transfer.native import propagation_native
    assert formal._native_apply is not None and formal._native_prepare is not None
    assert "site-packages" in str(Path(propagation_native.__file__).resolve())
    from importlib.util import find_spec
    from array import array
    import sysconfig
    from zipfile import ZipFile
    assert propagation_native.__name__ == "orsia_radiative.transfer.native.propagation_native"
    assert Path(propagation_native.__file__).parent.name == "native"
    assert find_spec("orsia_radiative.transfer.propagation_native") is None
    with ZipFile(sys.argv[1]) as archive:
        names = set(archive.namelist())
        native_prefix = "orsia_radiative/transfer/native/"
        for name in ("__init__.py", "propagation_native.cpp", "propagation_native.pyi",
                     "orders_native.h", "thermal_native.h", Path(propagation_native.__file__).name):
            assert native_prefix + name in names
            assert "orsia_radiative/transfer/" + name not in names or name == "__init__.py"
    assert find_spec("pybind11") is None, "pybind11 must remain build-only"
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    for format in ("f", "d"):
        depth, mu = array(format, [0.0, 0.4]), array(format, [-0.8, 0.8])
        matrices = [memoryview(array(format, [0] * 4)).cast("B").cast(format, shape=[2, 2])
                    for _ in range(4)]
        propagation_native.prepare(memoryview(depth).toreadonly(), mu, 0,
                                   *matrices, None, None)
        exits = array(format, [0, 0])
        source = memoryview(array(format, [1] * 4)).cast("B").cast(format, shape=[2, 2])
        propagation_native.apply(mu, source, matrices[0], matrices[2], None, None,
                                 2, None, exits)
        dtype = np.float32 if format == "f" else np.float64
        coefficients = propagation_coefficients(np.array(depth, dtype=dtype), np.array(mu, dtype=dtype))
        _, reference = apply_propagation(np.ones((2, 2), dtype=dtype), coefficients,
                                        bottom_radiance=2, return_centres=False)
        np.testing.assert_array_equal(exits, reference)
        depth.append(1)
        mu.append(1)
        exits.append(1)
    for dtype in (np.float32, np.float64):
        for solar_mu in (None, 0.8):
            coefficients = propagation_coefficients(np.array([0.0, 0.4], dtype=dtype),
                np.array([0.8, -0.8], dtype=dtype), solar_mu=solar_mu)
            source = np.ones((2, 2), dtype=dtype)
            field, exits = apply_propagation(source, coefficients,
                bottom_radiance=2.0, solar_source=solar_mu is not None)
            weight = coefficients.source_weight if solar_mu is None else coefficients.solar_source_weight
            half_weight = coefficients.centre_source_weight if solar_mu is None else coefficients.solar_centre_source_weight
            reference = formal._apply_numpy(source, coefficients, weight, half_weight, 2.0, True)
            assert field.dtype == exits.dtype == coefficients.transmission.dtype == dtype
            np.testing.assert_array_equal(field, reference[0])
            np.testing.assert_array_equal(exits, reference[1])
            centreless, exits_only = apply_propagation(source, coefficients,
                bottom_radiance=2.0, solar_source=solar_mu is not None, return_centres=False)
            assert centreless is None
            np.testing.assert_array_equal(exits_only, exits)
    assert not sys._is_gil_enabled()
    from scipy.constants import atomic_mass
    assert values_equal.__module__ == "orsia_radiative.spectroscopy.components"
    assert values_equal((np.arange(2), (3,)), (np.arange(2), (3,)))
    np.testing.assert_array_equal(integrated_exponential(np.zeros(2), np.zeros(2)), 1)
    atmosphere = Atmosphere(
        np.array([0.0, 1000.0]), np.array([100000.0, 90000.0]),
        np.array([95000.0]), np.array([296.0]), np.array([2e29]),
        np.array([1.9e29]), np.array([1e28]),
        (Absorber("X", np.array([1e25]), "dry_air"),),
    )
    from orsia_radiative import (
        Absorption, AtmosphericOptics, ObservationGeometry, RadiativeScene, SurfaceState,
        wavenumber_grid,
    )
    from orsia_radiative.radiometry import planck_wavenumber
    from orsia_radiative.transfer import thermal
    from scipy.special import expn
    thermal_grid = wavenumber_grid(np.array([800.0, 1000.0]))
    thermal_depth = np.array([[0.2, 0.7]])
    thermal_scene = RadiativeScene(atmosphere, SurfaceState(
        thermal_grid, emissivity=np.full(2, 0.4), temperature_k=300.0), ObservationGeometry(30.0))
    thermal_optics = AtmosphericOptics(
        thermal_grid, Absorption(thermal_grid, thermal_depth), None, None)
    transmission = np.exp(-thermal_depth[0] / np.cos(np.deg2rad(30.0)))
    layer_planck = planck_wavenumber(thermal_grid.points, 296.0)
    expected = 0.4 * planck_wavenumber(thermal_grid.points, 300.0) * transmission
    expected += layer_planck * (1 - transmission)
    expected += 0.6 * layer_planck * (1 - 2 * expn(3, thermal_depth[0])) * transmission
    for method in ("quadrature", "expn"):
        thermal_result = thermal.schwarzschild_radiance(
            thermal_scene, optics=thermal_optics, angular_order=128, hemispheric_method=method)
        np.testing.assert_allclose(thermal_result.values, expected, rtol=3e-13)
    assert not sys._is_gil_enabled()
    print("isolated radiative wheel: native thermal quadrature and expn passed")
    inputs = LineSet("X", "synthetic", (Line(1000.0, 1e-25, 0.0, 0.1, 0.2, 0.7),),
                     np.ones(1), (270.0, 320.0), (80000.0, 100000.0),
                     molecular_mass_kg=44 * atomic_mass)
    result = line_absorption(atmosphere, np.array([999.0, 1000.0, 1001.0]), inputs)
    assert result.kind == "voigt_lines" and np.all(result.optical_depth > 0)
    from orsia_radiative import LineDatabase, PartitionFunctionTable, line_set
    database = LineDatabase(inputs.lines, 1, 1, (900.0, 1100.0))
    partitions = PartitionFunctionTable(1, 1, np.array([270.0, 320.0]), np.ones(2))
    materialised = line_set(database, partitions, atmosphere,
        np.array([999.0, 1000.0, 1001.0]), species="X", molecular_mass_kg=44 * atomic_mass,
        pressure_range_pa=(80000.0, 100000.0), wing_cutoff_cm1=25.0)
    assert not hasattr(materialised, "source_id")
    np.testing.assert_array_equal(line_absorption(atmosphere, result.grid, materialised).optical_depth,
                                  result.optical_depth)
    assert "orsia_data" not in sys.modules
    from orsia_radiative import SpectralGrid, SolarIrradianceSpectrum, solar_irradiance, scale_spectral_irradiance
    wavelength = SpectralGrid("wavelength_nm", np.array([500.0, 1000.0]))
    target = SpectralGrid("wavenumber_cm1", np.array([10000.0, 15000.0, 20000.0]))
    solar = solar_irradiance(SolarIrradianceSpectrum(wavelength.points, np.array([1.0, 2.0])), target)
    scaled = scale_spectral_irradiance(solar, distance_au=2.0, reference_distance_au=1.0)
    assert solar.grid is scaled.grid is target
    np.testing.assert_array_equal(scaled.values, solar.values / 4)
    assert "site-packages" in str(Path(sys.modules["orsia_radiative"].__file__).resolve())
    from importlib import import_module
    from orsia_radiative import atmosphere as atmosphere_package
    from orsia_radiative.atmosphere import models, layers, hydrostatic
    assert atmosphere_package.Atmosphere is Atmosphere is models.Atmosphere
    assert atmosphere_package.Absorber is Absorber is models.Absorber
    assert atmosphere_package.AirBroadener is models.AirBroadener
    from orsia_radiative import gas_columns_from_fraction, remap_layer_columns
    assert gas_columns_from_fraction is hydrostatic.gas_columns_from_fraction
    assert remap_layer_columns is layers.remap_layer_columns
    pressure = np.array([100000., 60000., 20000.])
    gas = gas_columns_from_fraction(pressure, np.zeros(2), np.full(2, 1e-6),
                                   convention="dry_air_mole", gravity_m_s2=10.)
    column_state = hydrostatic.hydrostatic_atmosphere(
        pressure, np.array([280., 240.]), np.zeros(2),
        target_pressure_boundaries_pa=pressure[[0, 2]], gravity_m_s2=10.,
        bottom_height_m=0., gas_columns_molecules_m2={"CH4": gas}, air_broadener="dry_air")
    assert isinstance(column_state, Atmosphere)
    np.testing.assert_allclose(column_state.absorbers[1].columns_molecules_m2, [gas.sum()])
    for module in ("atmosphere", "atmosphere.models", "atmosphere.layers", "atmosphere.hydrostatic",
                   "surface", "geometry", "scene", "profiles",
                   "spectral.grid", "spectral.conversions", "spectral.interpolation",
                   "solar.source", "solar.spectrum", "spectroscopy.lines",
                   "spectroscopy.line_tables", "spectroscopy.cross_sections",
                   "spectroscopy.continuum", "spectroscopy.components", "optics.absorption",
                   "optics.phases", "optics.scattering", "optics.atmosphere"):
        installed = import_module(f"orsia_radiative.{module}")
        assert "site-packages" in str(Path(installed.__file__).resolve())
    for module in ("sources", "arrays", "state", "optical", "absorption", "continuum", "line_tables",
                   "spectral_grid", "solar_spectrum", "atmospheric_columns", "atmosphere.column"):
        try:
            import_module(f"orsia_radiative.{module}")
        except ModuleNotFoundError as error:
            assert error.name == f"orsia_radiative.{module}"
        else:
            raise AssertionError(f"Removed radiative module installed: {module}")
elif "orsia_instruments" in Path(sys.argv[1]).name:
    import orsia_instruments as instruments
    from importlib import import_module
    from importlib.util import find_spec
    from orsia_instruments.quadrature import interval_quadrature, validate_quadrature_order
    assert find_spec("orsia_numerics.quadrature") is None
    assert interval_quadrature.__module__ == "orsia_instruments.quadrature"
    assert validate_quadrature_order.__module__ == "orsia_instruments.quadrature"
    nodes, weights = interval_quadrature(-1., 1., 3)
    np.testing.assert_allclose(weights @ nodes**4, 2/5)
    validate_quadrature_order(3)
    assert not hasattr(instruments, "interval_quadrature")
    assert "interval_quadrature" not in instruments.__all__
    assert not hasattr(import_module("orsia_instruments.optics.psfs"), "bilinear_zero")
    np.testing.assert_allclose(instruments.tabulated_psf(
        np.array([[[0.5, 0.5], [1., 1.], [2., 0.5]]]), np.ones(3), np.zeros((1, 2)), 0,
        x_offsets_m=[0., 1.], y_offsets_m=[0., 1.], density_per_m2=np.ones((2, 2)),
    ), [[1., 1., 0.]])
    # Check canonical subpackages and absence of removed facades in the built wheel.
    for module in ("charge", "quadrature", "optics.response", "optics.psfs", "optics.pixels",
                   "optics.filters", "optics.dispersion", "optics.two_beam", "optics.cavities",
                   "imaging.acquisition", "imaging.charge", "imaging.readout",
                   "imaging.measurements", "imaging.simulation", "imaging.response"):
        installed = import_module(f"orsia_instruments.{module}")
        assert "site-packages" in str(Path(installed.__file__).resolve())
    from orsia_instruments.optics.cavities import cavity_imager
    from orsia_instruments.imaging.readout import image_readout
    from orsia_instruments.charge import ExpectedElectrons
    from orsia_instruments.detector import DetectorMaps
    assert cavity_imager is instruments.cavity_imager
    assert image_readout is instruments.image_readout
    assert ExpectedElectrons is instruments.ExpectedElectrons
    assert DetectorMaps is instruments.DetectorMaps
    for removed in ("AlongTrackReadSpecification", "GroundSpatialSupport", "AlongTrackIntegrationOrders"):
        assert not hasattr(instruments, removed)
    from orsia_radiative import SpectralGrid, SpectralDomain
    trajectory = instruments.LinearTrajectory(height_m=100., speed_m_s=1.)
    target = np.zeros((1, 2))
    centres = np.zeros(1)
    times = np.array([-0.5, 0., 0.5])
    np.testing.assert_allclose(instruments.ground_fixed_pointing(trajectory, target, centres)(0, times)[:, 0],
                               [0.005, 0., -0.005])
    np.testing.assert_allclose(instruments.ground_scroll_pointing(
        trajectory, target, centres, ground_rate_m_s=0.25)(0, times)[:, 0],
        [0.00375, 0., -0.00375])
    reads = instruments.along_track_read_schedule(
        sample_ids=("scan",), pixel_ids=("pixel",), setting_ids=("setting",), port_ids=("+",),
        target_along_m=np.zeros(1), ground_along_offsets_m=np.zeros(1),
        exposure_s=1., trajectory=trajectory,
    )
    scan = instruments.linear_scan_acquisition(
        reads, trajectory, target_xy_m=np.zeros((1, 2)),
        pixel_bounds_m=np.array([[-1., 1., -1., 1.]]),
        along_edges_m=np.array([-1., 1.]), cross_edges_m=np.array([-1., 1.]),
        angular_order=2, time_order=2, focal_length_m=1.,
        pointing_law=lambda read, times: np.zeros((len(times), 2)),
    )
    spectral_grid = SpectralGrid("wavenumber_cm1", np.array([4000., 6000.]),
                                 SpectralDomain("wavenumber_cm1", ((4000., 6000.),)))
    coordinates = scan.directions_uv * 100
    coordinates[..., 0] += scan.times_s
    response = instruments.detector_response(
        spectral_grid, scan, lambda r, w, a: np.ones((a.times_s.shape[1], len(w))),
        ground_xy_m=coordinates, along_m=np.array([-1., 1.]), cross_m=np.array([-1., 1.]),
        support_bounds_m=np.array([[-1., 1., -1., 1.]]), collecting_area_m2=1.,
    )
    assert response.__class__ is instruments.DetectorResponse
    np.testing.assert_allclose(instruments.response_electrons(response, np.ones((2, 2, 2))).total,
                               response.coefficients.sum(axis=(1, 2, 3)))
    for module in ("arrays", "responses", "psfs", "tabulations", "image_charge", "image_sequence", "imaging_optics"):
        try:
            import_module(f"orsia_instruments.{module}")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"Removed facade installed: {module}")
    assert "interferometric_imager" in instruments.__all__
    optics = instruments.interferometric_imager(
        focal_length_m=0.02, collecting_area_m2=1e-8,
        psf_sigma_m=3e-5, opd_m=[0., 1e-5], port="+",
    )
    assert optics.port == "+"
    np.testing.assert_array_equal(optics.settings["opd_m"], [0., 1e-5])
    from orsia_instruments import rectangular_acquisition, ExpectedElectrons, DetectorElectronics, sample_readout
    acquisition = rectangular_acquisition(
        sample_ids=("r",), pixel_ids=("p",), setting_ids=("s",), port_ids=("+",),
        exposure_bounds_s=np.array([[0., 1.]]), pixel_bounds_m=np.array([[-1., 1., -1., 1.]]),
        direction_bounds_uv=(-0.1, 0.1, -0.1, 0.1),
        pointing=lambda t: np.zeros((len(t), 2)), focal_length_m=1.,
    )
    readings = sample_readout(ExpectedElectrons(acquisition, {"scene": np.array([100.])}),
                              DetectorElectronics(), np.random.default_rng(71))
    assert readings.dn.shape == (1, 1)
    # Composition entry points must also work without workspace or optical geometry.
    from dataclasses import replace
    external = instruments.image_charge(
        {"external": np.full((1, 1, 2), 12.)}, exposure_bounds_s=np.array([[0., 0.1]]))
    assert isinstance(external.signal.acquisition, instruments.ReadSchedule)
    assert not hasattr(external.signal.reads, "weights_sr_s")
    background = instruments.image_charge(
        {"warm": np.full((1, 1, 2), 3.)}, exposure_bounds_s=np.array([[0., 0.1]]))
    composed = instruments.image_readout(
        instruments.combine_image_source_paths(external, background),
        detector=instruments.DetectorMaps(quantum_efficiency=0, bias_dn=100.),
        rng=np.random.default_rng(72), realisations=2, row_noise_electrons=1.)
    np.testing.assert_array_equal(composed.photoelectrons, 15.)
    assert set(composed.contributions) == {"external", "warm"}
    assert composed.dn.shape == (2, 1, 1, 2)
    try:
        replace(optics, settings={"opd_m": [1., 2.]})
    except ValueError:
        pass
    else:
        raise AssertionError("optical description must not be independently replaceable")
    # Exercise new physical-law exports against the installed wheel, not sources.
    from functools import partial
    wave = np.array([1.98e-6, 2e-6, 2.01e-6])
    for name in ("lorentzian_filter", "super_gaussian_filter", "rectangular_filter",
                 "longpass_filter", "shortpass_filter", "notch_filter"):
        assert name in instruments.__all__
    np.testing.assert_allclose(instruments.lorentzian_filter(wave[1:], 2e-6, 2e-8), [1., 0.5])
    band = partial(instruments.super_gaussian_filter, centre_m=2e-6, fwhm_m=3e-8)
    common = dict(focal_length_m=0.02, collecting_area_m2=1e-8, psf_sigma_m=3e-5)
    small_acquisition = rectangular_acquisition(
        sample_ids=("r",), pixel_ids=("p",), setting_ids=("s",), port_ids=("direct",),
        exposure_bounds_s=np.array([[0., 0.01]]),
        pixel_bounds_m=np.array([[-2e-5, 2e-5, -2e-5, 2e-5]]),
        direction_bounds_uv=(-0.001, 0.001, -0.001, 0.001),
        pointing=lambda t: np.zeros((len(t), 2)), focal_length_m=0.02,
        time_order=1, angular_order=2,
    )
    optical_models = (
        instruments.lvf_imager(**common, reference_m=2e-6, gradient=1e-4,
            profile=partial(instruments.lorentzian_filter, fwhm_m=2e-8)),
        instruments.spatial_filter_imager(**common, filter=lambda x, y, w: band(w)),
        instruments.dispersive_imager(**common, transmission=band,
            image_map=partial(instruments.polynomial_image_map, reference_m=2e-6, dispersion=4)),
        instruments.interferometric_imager(**common, opd_m=1e-5, port="+", transmission=band,
            visibility=lambda w: np.full_like(w, 0.8), opd_offset_m=lambda x, y: 0.01*x),
        instruments.cavity_imager(**common, spacing_m=1e-5, order_transmission=band,
            refractive_index=lambda w: np.full_like(w, 1.4), spacing_offset_m=lambda x, y: 1e-4*x),
    )
    for model in optical_models:
        accepted = model.response(0, 0, wave, small_acquisition)
        assert accepted.shape == (len(small_acquisition.times_s[0]), len(wave))
        assert np.all((accepted > 0) & (accepted <= 1))
    for name in ("PointSpreadFunction", "gaussian_psf", "tabulated_psf"):
        assert name in instruments.__all__
    gaussian = partial(instruments.gaussian_psf, sigma_m=(3e-5, 2e-5))
    table = partial(instruments.tabulated_psf,
        x_offsets_m=[-1e-4, 1e-4], y_offsets_m=[-1e-4, 1e-4],
        density_per_m2=np.full((2, 2), 1/(2e-4)**2))
    for psf in (gaussian, table):
        model = instruments.filter_imager(focal_length_m=0.02, collecting_area_m2=1e-8,
            filter=band, psf=psf)
        accepted = model.response(0, 0, wave, small_acquisition)
        assert accepted.shape == (len(small_acquisition.times_s[0]), len(wave))
        assert np.all((accepted > 0) & (accepted < 1))
    np.testing.assert_array_equal(instruments.tabulated_transmission(
        [1e-6, 2e-6], [1e-6, 2e-6], [0.2, 0.8]), [0.2, 0.8])
    points = instruments.DetectorOperatingPoints(
        np.zeros(1), np.ones(1), np.full(1, 1000.), np.ones(1),
        np.full(1, 10.), np.zeros(1))
    def evaluate(parameters):
        return ExpectedElectrons(acquisition, {"scene": np.array([100. + 2*parameters["offset"]])})
    nominal, jacobian = instruments.native_sensitivities(
        evaluate, {"offset": 0.}, {"offset": 0.01})
    budget = instruments.native_tolerance_budget(
        nominal, ("offset",), jacobian, np.array([0.1]),
        np.array([[0.25]]),
        instruments.electron_covariance(nominal, points))
    np.testing.assert_allclose(budget.signed_bias, [0.2])
    np.testing.assert_allclose(budget.instrument_covariance, [[1.]])
    analogue, _ = instruments.analogue_dn_budget(budget, nominal, points)
    np.testing.assert_array_equal(analogue, [110.])
    assert "site-packages" in str(Path(instruments.__file__).resolve())
    assert "orsia_data" not in sys.modules
else:
    import tempfile
    from orsia_data import spectroscopy
    from orsia_data.spectroscopy import continuum, cross_sections, lines, partition_sums
    from orsia_data import solar
    from orsia_data.solar import irradiance as solar_irradiance
    for module in (solar, solar_irradiance):
        assert "site-packages" in str(Path(module.__file__).resolve())
    for module in (spectroscopy, continuum, cross_sections, lines, partition_sums):
        assert "site-packages" in str(Path(module.__file__).resolve())
    assert spectroscopy.read_aer_lines is lines.read_aer_lines
    assert spectroscopy.read_partition_table is partition_sums.read_partition_table
    assert spectroscopy.read_mt_ckd_water is continuum.read_mt_ckd_water
    from orsia_data import download_mt_ckd_water, fetch_mt_ckd_water
    assert spectroscopy.download_mt_ckd_water is continuum.download_mt_ckd_water is download_mt_ckd_water
    assert spectroscopy.fetch_mt_ckd_water is continuum.fetch_mt_ckd_water is fetch_mt_ckd_water
    assert spectroscopy.read_cross_section_table is cross_sections.read_cross_section_table
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "partition.csv").write_text(
            "molecule_id,isotopologue_id,temperature_k,partition_sum\n1,1,200,100\n1,1,350,175\n"
        )
        table = spectroscopy.read_partition_table(root / "partition.csv", molecule_id=1, isotopologue_id=1)
        np.testing.assert_array_equal(table.temperatures_k, [200.0, 350.0])
        np.testing.assert_array_equal(table.partition_sums, [100.0, 175.0])
        (root / "solar.csv").write_text(
            "wavelength (nm),irradiance (W/m^2/nm),uncertainty (W/m^2/nm),bandwidth (nm)\n"
            "500,1,0.1,0.025\n1000,2,0.2,0.025\n"
        )
        native_solar = solar.read_solar_spectrum_csv(root / "solar.csv")
        np.testing.assert_array_equal(native_solar.wavelengths_nm, [500.0, 1000.0])
        np.testing.assert_array_equal(native_solar.bandwidth_nm, [0.025, 0.025])
        assert native_solar.irradiance_per_nm.flags.writeable
        from datetime import UTC, datetime
        from scipy.io import netcdf_file
        from orsia_data import read_pressure_level_atmosphere
        with netcdf_file(root / "meteorology.nc", "w") as dataset:
            for name, values, units in (
                ("time", [0], "hours since 2025-01-01"), ("level", [1000, 500], "hPa"),
                ("latitude", [0], "degrees_north"), ("longitude", [0], "degrees_east"),
            ):
                dataset.createDimension(name, len(values))
                variable = dataset.createVariable(name, "f8", (name,))
                variable.units = units
                variable[:] = values
            for name, units, values in (("t", "K", [280, 240]), ("q", "1", [0.01, 0.01])):
                variable = dataset.createVariable(name, "f8", ("time", "level", "latitude", "longitude"))
                variable.units = units
                variable[:] = np.array(values).reshape(1, 2, 1, 1)
        atmosphere = read_pressure_level_atmosphere(
            root / "meteorology.nc", temperature_variable="t", specific_humidity_variable="q",
            specific_humidity_units="1", coordinates=("time", "level", "latitude", "longitude"),
            pressure_units="hPa", valid_time=datetime(2025, 1, 1, tzinfo=UTC),
            latitude_deg=0, longitude_deg=0, surface_pressure_pa=100000,
            pressure_boundaries_pa=np.array([100000.0, 75000, 50000]), gravity_m_s2=10,
            bottom_height_m=0, gas_columns_molecules_m2={}, air_broadener="dry_air",
        )
        np.testing.assert_allclose(atmosphere.temperature_k, [270, 250])
        assert atmosphere.temperature_k.flags.writeable
        assert "site-packages" in str(Path(sys.modules["xarray"].__file__).resolve())
        assert not ({"cfgrib", "eccodes", "h5netcdf", "h5py", "netCDF4", "cdsapi"} & sys.modules.keys())
    assert "site-packages" in str(Path(sys.modules["orsia_data"].__file__).resolve())
assert not sys._is_gil_enabled(), "standalone native calls enabled the GIL"
PY
    rm -rf "$tmp/standalone"
done
uv venv --python "$(cat .python-version)" "$tmp/venv"
uv pip install --python "$tmp/venv/bin/python" "${wheels[@]}"
readme="$PWD/libs/orsia-radiative/README.md"
uncertainty_readme="$PWD/libs/orsia-uncertainty/README.md"
hsrs_doc="$PWD/docs/hsrs.md"
atmosphere_doc="$PWD/docs/atmospheric-data.md"
atmosphere_tests="$PWD/libs/orsia-data/tests/test_atmosphere_native.py"
instruments_readme="$PWD/libs/orsia-instruments/README.md"
instruments_source_readme="$PWD/libs/orsia-instruments/src/orsia_instruments/README.md"
instrument_cases="$PWD/tests/instrument_cases.py"
lvf_tests="$PWD/tests/test_lvf_scan.py"
composition_tests="$PWD/tests/test_instrument_composition.py"
cd "$tmp"
"$tmp/venv/bin/python" -I - "$instruments_readme" "$instruments_source_readme" "$instrument_cases" "$lvf_tests" "$composition_tests" <<'PY'
import re
import runpy
import sys
from pathlib import Path
import orsia_instruments
import numpy as np
from orsia_instruments import DetectorElectronics, expected_electrons, sample_readout, uniform_radiance_field

assert "site-packages" in str(Path(orsia_instruments.__file__).resolve())
document = Path(sys.argv[1])
examples = re.findall(r"```python\n(.*?)```", document.read_text(), flags=re.DOTALL)
assert len(examples) == 5
for example in examples:
    exec(compile(example, str(document), "exec"), {"__name__": "__instrument_docs__"})
namespace = runpy.run_path(sys.argv[3])
detector = DetectorElectronics(bias_dn=100)
for family in namespace["FAMILIES"]:
    signal = namespace["simulate"](family)
    reads = sample_readout(signal, detector, np.random.default_rng(718))
    assert reads.acquisition is signal.acquisition and np.all(signal.total > 0)
    assert not reads.full_well_saturated.any() and not reads.adc_clipped.any()
_, acquisition, response = namespace["make_case"]("filter")
for radiance in namespace["radiative_inputs"]():
    field = uniform_radiance_field(
        radiance, time_bounds_s=(-0.1, 0.2), u_bounds=(-0.004, 0.004),
        v_bounds=(-0.004, 0.004), spectral_bounds_cm1=(4900.0, 5100.0),
    )
    signal = expected_electrons(field, acquisition, response, collecting_area_m2=1e-8)
    sample_readout(signal, detector, np.random.default_rng(720))
    assert np.all(signal.total > 0)
runpy.run_path(sys.argv[4])["test_lvf_scan_dense_image_preserves_read_geometry_and_electrons"]()
runpy.run_path(sys.argv[5])["test_source_paths_and_detector_only"]()
for guide in (document, Path(sys.argv[2]), document.with_name("composition.md")):
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", guide.read_text()):
        path = target.split("#", 1)[0]
        if path and "://" not in path and not path.startswith("mailto:"):
            assert (guide.parent / path).exists(), (guide, target)
assert not sys._is_gil_enabled()
assert "orsia_data" not in sys.modules
print("isolated instrument wheel: README examples, composition, all response families, solar/thermal inputs and local links passed")
PY
"$tmp/venv/bin/python" -I - "$readme" "$uncertainty_readme" "$hsrs_doc" "$atmosphere_doc" <<'PY'
import re
import sys
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import numpy
import orsia_uncertainty as u
import orsia_radiative as r

assert not sys._is_gil_enabled(), "native imports enabled the GIL"
assert "site-packages" in str(Path(u.__file__).resolve())
assert "site-packages" in str(Path(r.__file__).resolve())
assert "AtmosphericOptics" in r.__all__
assert "compose_atmospheric_optics" in r.__all__
assert "validate_atmospheric_optics" in r.__all__
for obsolete in ("PreparedOpticalInputs", "prepare_optical_inputs", "validate_prepared_optical_inputs"):
    assert not hasattr(r, obsolete) and obsolete not in r.__all__
# Shared phase exports and broadcasting work independently of workspace imports.
cosine = numpy.array([-1.0, 0.0, 1.0])
numpy.testing.assert_array_equal(r.rayleigh_phase(cosine), [1.5, 0.75, 1.5])
numpy.testing.assert_array_equal(r.henyey_greenstein_phase(0.0, cosine), numpy.ones(3))
assert r.henyey_greenstein_phase(numpy.array([[0.0], [0.2]]), cosine).shape == (2, 3)
assert type(r.henyey_greenstein_phase(0.2, 0.0)) is float
assert type(r.rayleigh_phase(0.0)) is float
covariance = numpy.eye(2, dtype=numpy.float64)
numpy.testing.assert_array_equal(u.solve_covariance(covariance, numpy.ones(2)), numpy.ones(2))
numpy.testing.assert_array_equal(u.covariance_solver(covariance)(numpy.ones(2)), numpy.ones(2))
numpy.testing.assert_array_equal(
    u.gaussian_sampler(covariance)(numpy.random.default_rng(71), 8),
    u.sample_gaussian(covariance, numpy.random.default_rng(71), 8),
)
atmosphere = r.Atmosphere(
    numpy.array([0.0, 1000.0]), numpy.array([100000.0, 90000.0]),
    numpy.array([95000.0]), numpy.array([285.0]),
    numpy.array([2.0e29]), numpy.array([1.9e29]), numpy.array([1.0e28]),
    (r.Absorber("X", numpy.array([1.0e25]), "dry_air"),),
)
wavenumbers_cm1 = r.wavelength_to_wavenumber(numpy.array([760.0, 761.0]))
grid = r.wavenumber_grid(wavenumbers_cm1)
assert type(grid) is r.SpectralGrid and grid.domain is None
surface = r.SurfaceState(grid, reflectance=numpy.array([0.2, 0.3]))
scene = r.RadiativeScene(atmosphere, surface, r.ObservationGeometry(10.0))
numpy.testing.assert_array_equal(scene.atmosphere.absorbers[0].columns_molecules_m2, [1.0e25])
input_values = numpy.array([10.0, 20.0])
solar = r.SpectralIrradiance(grid, input_values)
input_values.fill(0)
numpy.testing.assert_array_equal(solar.values, [10.0, 20.0])
assert not numpy.shares_memory(solar.values, input_values)
assert solar.values.flags.writeable
absorption = r.AbsorptionComponent(
    grid, "X", "synthetic", "lorentz_lines", numpy.zeros((1, 2))
)
source = r.solar_source(solar, direction=(30.0, 20.0))
total = r.compose_absorption(atmosphere, source, (absorption,))
optics = r.compose_atmospheric_optics(scene.atmosphere, source, absorption=total)
assert type(optics) is r.AtmosphericOptics
radiance = r.transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
assert radiance.grid is optics.grid is source.irradiance.grid is grid
assert not hasattr(r, "RadianceComponent") and not hasattr(r, "RadianceDiagnostic")
from dataclasses import fields
assert [field.name for field in fields(radiance)] == ["grid", "values", "geometry"]
assert optics.absorption is total and not hasattr(total, "components")
assert numpy.array_equal(radiance.wavenumbers_cm1, wavenumbers_cm1)
assert radiance.values[0] > 0.0
constituent = r.AerosolComponent(
    grid, numpy.array([[0.02, 0.03]]), numpy.array([[0.01, 0.02]]),
    numpy.array([[0.2, 0.3]]),
)
scattered = r.transfer.solar.reflected_radiance(
    scene, source=source,
    optics=r.compose_atmospheric_optics(scene.atmosphere, source, absorption=total, aerosol=constituent),
    max_order=1, delta_m_moments=None,
)
from orsia_radiative.transfer import orders

scattered_terms = orders.scattering_orders(
    scene, source=source,
    optics=r.compose_atmospheric_optics(scene.atmosphere, source, absorption=total, aerosol=constituent),
    max_order=1, delta_m_moments=None,
)
assert len(scattered_terms) == 2
assert numpy.array_equal(scattered_terms.sum(axis=0), scattered.values)
assert numpy.array_equal(r.wavenumber_to_wavelength(wavenumbers_cm1), [760.0, 761.0])
assert not hasattr(r, "ScatteringComponent")
assert not hasattr(constituent, "as_scattering_component")
rayleigh = r.RayleighComponent(grid, numpy.array([[0.01, 0.02]]))
aerosol = r.AerosolComponent(
    grid, constituent.optical_depth,
    constituent.scattering_depth, constituent.asymmetry,
)
optics = r.compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol)
typed_beam = r.transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
sos = r.transfer.solar.reflected_radiance(
    scene, source=source, optics=optics,
    max_order=5, angular_order=4, layer_subdivisions=4,
)
order_values = orders.scattering_orders(
    scene, source=source, optics=optics, max_order=5, angular_order=4, layer_subdivisions=4,
)
numpy.testing.assert_array_equal(order_values.sum(axis=0), sos.values)
numpy.testing.assert_array_equal(orders.scattering_orders(
    scene, source=source, optics=optics, max_order=0,
).sum(axis=0), typed_beam.values)
assert all(a < b for a, b in zip(typed_beam.values, sos.values, strict=True))
from orsia_radiative.radiance import SpectralIrradiance

assert SpectralIrradiance is r.SpectralIrradiance
assert SpectralIrradiance.__module__ == "orsia_radiative.radiance"
from orsia_radiative.transfer import solar as solar_transfer
from orsia_radiative.transfer.orders import scattering_orders
from orsia_radiative.transfer.solar import reflected_radiance

assert solar_transfer is r.transfer.solar
assert scattering_orders is orders.scattering_orders
assert not hasattr(solar_transfer, "scattering_orders")
assert reflected_radiance is solar_transfer.reflected_radiance
assert reflected_radiance.__name__ == "reflected_radiance"
assert "site-packages" in str(Path(solar_transfer.__file__).resolve())
assert r.transfer.__all__ == ["solar", "thermal"]
assert solar_transfer.__all__ == ["reflected_radiance"]
assert r.solar.__name__ == "orsia_radiative.solar"
assert not hasattr(r.solar, "direct_beam")
for removed in ("direct_beam", "single_scattering", "single_scattering_terms", "unscattered", "single_event", "solar_sos", "sos"):
    assert not hasattr(r, removed)
    assert not hasattr(r.transfer, removed)
    assert not hasattr(solar_transfer, removed)
for method in solar_transfer.__all__:
    assert not hasattr(r, method)
    assert not hasattr(r.transfer, method)
    assert getattr(solar_transfer, method).__module__ == "orsia_radiative.transfer.solar"
assert importlib.util.find_spec("orsia_radiative.sos") is None
from orsia_radiative.transfer import thermal
from orsia_radiative.radiometry import planck_wavenumber
assert thermal is r.transfer.thermal
assert "site-packages" in str(Path(thermal.__file__).resolve())
thermal_grid = numpy.array([800.0, 1000.0])
thermal_sampling = r.wavenumber_grid(thermal_grid)
thermal_scene = replace(scene, surface=r.SurfaceState(
    thermal_sampling, emissivity=numpy.ones(2), temperature_k=300.0,
))
thermal_absorption = r.compose_absorption_on_grid(atmosphere, thermal_sampling, ())
thermal_optics = r.atmospheric_optics(
    atmosphere, thermal_sampling, absorption=thermal_absorption,
)
thermal_result = thermal.schwarzschild_radiance(thermal_scene, optics=thermal_optics)
numpy.testing.assert_array_equal(thermal_result.values, planck_wavenumber(thermal_grid, 300.0))
thermal_terms = thermal.schwarzschild_terms(thermal_scene, optics=thermal_optics)
assert numpy.array_equal(sum(thermal_terms.values()), thermal_result.values)
reflecting_thermal_scene = replace(thermal_scene, surface=replace(
    thermal_scene.surface, emissivity=numpy.full(2, 0.4),
))
thin_absorption = r.Absorption(thermal_sampling, numpy.full((1, 2), 1e-20))
thin_optics = r.atmospheric_optics(atmosphere, thermal_sampling, absorption=thin_absorption)
thin_result = thermal.schwarzschild_radiance(
    reflecting_thermal_scene, optics=thin_optics, hemispheric_method="expn",
)
thin_terms = thermal.schwarzschild_terms(
    reflecting_thermal_scene, optics=thin_optics, hemispheric_method="expn",
)
assert numpy.all(thin_terms["reflected_atmospheric_emission"] > 0)
numpy.testing.assert_array_equal(sum(thin_terms.values()), thin_result.values)
assert not sys._is_gil_enabled(), "SciPy-backed thermal calls enabled the GIL"
for module in ("_kernels", "_sos"):
    assert importlib.util.find_spec(f"orsia_radiative.{module}") is None
for removed in ("CollimatedSource", "collimated_source", "DirectionalSource", "directional_source"):
    assert not hasattr(r, removed)
constructed_source = r.SolarSource(solar, 30.0, 20.0)
assert constructed_source != source
assert constructed_source.irradiance is source.irradiance is solar
assert constructed_source.zenith_deg == source.zenith_deg == 30.0
assert constructed_source.relative_azimuth_deg == source.relative_azimuth_deg == 20.0
for record in (scene, source, solar, atmosphere, total, optics, radiance):
    assert record != replace(record)
    try:
        hash(record)
    except TypeError:
        pass
    else:
        raise AssertionError("radiative record unexpectedly hashable")
# Installed-wheel geometry: outward Sun and view coincide at 45 degrees.
# Incoming propagation is opposite, so the scattering cosine is -1, not 0.
geometry_scene = replace(scene, geometry=r.ObservationGeometry(45.0),
                          surface=replace(surface, reflectance=numpy.zeros(2)))
geometry_source = r.solar_source(solar, direction=(45.0, 180.0))
geometry_aerosol = r.AerosolComponent(
    grid, numpy.full((1, 2), 0.2), numpy.full((1, 2), 0.1), numpy.full((1, 2), 0.6),
)
mu = numpy.sqrt(0.5)
phase_back = (1 - 0.6**2) / (1 + 0.6)**3
expected = numpy.asarray(solar.values) / (4*numpy.pi*mu) * 0.5 * phase_back * (
    -numpy.expm1(-0.4/mu) / (2/mu)
)
geometry_optics = r.compose_atmospheric_optics(
    geometry_scene.atmosphere, geometry_source, aerosol=geometry_aerosol,
)
for max_order in (1, 2):
    rows = orders.scattering_orders(
        geometry_scene, source=geometry_source, optics=geometry_optics, max_order=max_order,
    )
    numpy.testing.assert_allclose(rows[1], expected, rtol=3e-13, atol=1e-15)
from orsia_radiative.transfer import formal, orders

for module in (formal, solar_transfer, orders):
    assert "site-packages" in str(Path(module.__file__).resolve())
assert not hasattr(scene.geometry, "solar_zenith_deg")
# Atmospheric optics are independent of the selected transfer calculation.
optics = r.compose_atmospheric_optics(scene.atmosphere, source, rayleigh=rayleigh, aerosol=aerosol)
repeated_beam = r.transfer.solar.reflected_radiance(scene, source=source, optics=optics, max_order=0, delta_m_moments=None)
assert repeated_beam.grid == typed_beam.grid
assert repeated_beam.geometry == typed_beam.geometry
numpy.testing.assert_array_equal(repeated_beam.values, typed_beam.values)
repeated_sos = r.transfer.solar.reflected_radiance(
    scene, source=source, optics=optics, max_order=5, angular_order=4, layer_subdivisions=4,
)
assert repeated_sos.grid == sos.grid
assert repeated_sos.geometry == sos.geometry
numpy.testing.assert_array_equal(repeated_sos.values, sos.values)
table = r.CrossSectionField(
    "X", "synthetic", "cross_section", grid,
    atmosphere.temperature_k, atmosphere.pressure_pa, numpy.array([[1e-27, 2e-27]]),
)
prepared_table = r.cross_sections(wavenumbers_cm1, table)
original_x = r.evaluate_table_absorption(atmosphere, prepared_table)
optics = r.compose_atmospheric_optics(
    scene.atmosphere, source, absorption=r.compose_absorption(atmosphere, source, (original_x,)),
    rayleigh=rayleigh, aerosol=aerosol,
)
changed = replace(scene, atmosphere=replace(
    atmosphere, absorbers=(replace(atmosphere.absorbers[0], columns_molecules_m2=numpy.array([2e25])),),
))
updated_x = r.evaluate_table_absorption(changed.atmosphere, prepared_table)
numpy.testing.assert_equal(
    asdict(updated_x), asdict(r.table_absorption(changed.atmosphere, wavenumbers_cm1, table))
)
updated = r.compose_atmospheric_optics(
    changed.atmosphere, source,
    absorption=r.compose_absorption(changed.atmosphere, source, (updated_x,)),
    rayleigh=rayleigh, aerosol=aerosol,
)
assert updated.rayleigh is optics.rayleigh
assert updated.aerosol is optics.aerosol
assert not hasattr(updated.absorption, "input_source_ids")
assert r.transfer.solar.reflected_radiance(changed, source=source, optics=updated, max_order=0, delta_m_moments=None).values.shape == (2,)
assert not hasattr(r, "update_optical_inputs")
try:
    r.validate_atmospheric_optics(changed, source, optics, reference_atmosphere=atmosphere)
except ValueError as error:
    assert "stale" in str(error)
else:
    raise AssertionError("stale installed-wheel composition was accepted")
# Execute the literal documentation examples against installed wheels, in their
# documented order. The README is read as text, never used for import resolution.
readme = Path(sys.argv[1])
text = readme.read_text()
examples = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
assert len(examples) == 7
namespace = {"__name__": "__documentation_examples__"}
for example in examples:
    exec(compile(example, str(readme), "exec"), namespace)
uncertainty_readme = Path(sys.argv[2])
uncertainty_examples = re.findall(r"```python\n(.*?)```", uncertainty_readme.read_text(), flags=re.DOTALL)
assert len(uncertainty_examples) == 2
uncertainty_namespace = {"__name__": "__uncertainty_documentation_examples__"}
for example in uncertainty_examples:
    exec(compile(example, str(uncertainty_readme), "exec"), uncertainty_namespace)
hsrs_doc = Path(sys.argv[3])
hsrs_examples = re.findall(r"```python\n(.*?)```", hsrs_doc.read_text(), flags=re.DOTALL)
assert len(hsrs_examples) == 1
hsrs_namespace = {"__name__": "__hsrs_documentation_examples__"}
for example in hsrs_examples:
    exec(compile(example, str(hsrs_doc), "exec"), hsrs_namespace)
atmosphere_doc = Path(sys.argv[4])
atmosphere_examples = re.findall(r"```python\n(.*?)```", atmosphere_doc.read_text(), flags=re.DOTALL)
assert len(atmosphere_examples) == 1
for example in atmosphere_examples:
    exec(compile(example, str(atmosphere_doc), "exec"), {"__name__": "__atmosphere_examples__"})
links_checked = 0
for document in (readme, uncertainty_readme, hsrs_doc, atmosphere_doc,
                 readme.parents[2] / "README.md", readme.parent.parent / "orsia-data" / "README.md"):
    content = document.read_text()
    assert sum(line.startswith("```") for line in content.splitlines()) % 2 == 0
    assert all(line == line.rstrip() for line in content.splitlines())
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
        path = target.split("#", 1)[0]
        if path and "://" not in path:
            assert (document.parent / path).exists(), (document, path)
            links_checked += 1
assert not sys._is_gil_enabled(), "radiative execution enabled the GIL"
print(f"isolated wheel composable transfer, solar SOS, LTE thermal, optical updates, native runtime "
      f"and {len(examples) + len(uncertainty_examples) + len(hsrs_examples) + len(atmosphere_examples)} examples passed; "
      f"{links_checked} local Markdown links exist")
PY
if [[ "${ORSIA_TEST_ATMOSPHERE_BACKENDS:-0}" == 1 ]]; then
    data_wheels=("$tmp"/dist/orsia_data-*.whl)
    # Native ecCodes is supplied by Devbox. The PyPI eccodeslib binary bundle
    # has no cp314t wheel; retain all other declared/transitive dependencies.
    printf 'eccodeslib\n' > "$tmp/native-excludes.txt"
    uv pip install --python "$tmp/venv/bin/python" --excludes "$tmp/native-excludes.txt" "${data_wheels[0]}[atmosphere]" pytest
    "$tmp/venv/bin/python" -I - "$atmosphere_tests" <<'PY'
import sys
from pathlib import Path
import orsia_data
import pytest

assert "site-packages" in str(Path(orsia_data.__file__).resolve())
raise SystemExit(pytest.main([sys.argv[1], "-c", "/dev/null", "--import-mode=importlib",
                             "-p", "no:cacheprovider", "-q"]))
PY
fi
