"""Opt-in synthetic refinement/statistics/resource regressions, not qualification."""

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from instrument_cases import FAMILIES, make_case
from orsia_instruments import (
    DetectorElectronics,
    ExpectedElectrons,
    central_difference,
    digitize,
    electron_moments,
    expected_electrons,
    sample_readout,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORSIA_TEST_INSTRUMENT_CHARACTERIZATION") != "1",
    reason="opt-in refinement/resource checks: just test-instrument-characterization",
)

# Predeclared synthetic acceptance, never a mission requirement.
TOLERANCES = {
    "spectral": 2e-5,
    "time": 2e-7,
    "angular": 2e-5,
    "pixel": 2e-5,
    "derivative": 2e-3,
}
SECONDS_BUDGET = 30.0
RSS_MIB_BUDGET = 512.0


def expectation(family: str, **kwargs) -> ExpectedElectrons:
    """Complete expectation call, including construction of the explicit sampled case."""
    field, acquisition, response = make_case(family, **kwargs)
    return expected_electrons(
        field,
        acquisition,
        response,
        collecting_area_m2=1e-8,
        throughput=0.8,
        quantum_efficiency=0.7,
    )


def relative_error(actual, reference) -> float:
    """Maximum absolute difference scaled by maximum reference signal."""
    return float(np.max(abs(actual - reference)) / max(float(np.max(abs(reference))), 1e-30))


def test_numerical_characterization() -> None:
    """Refine axes independently, test families, perturbations and detector statistics."""
    results: dict[str, dict] = {}
    for family in FAMILIES:
        # Baseline deliberately sampled finely enough for these smooth synthetic cases.
        baseline_kwargs = dict(spectral_points=1025, time_order=5, angular_order=9, pixel_order=12)
        baseline = expectation(family, **baseline_kwargs)
        errors = {}
        for name, change in (
            ("spectral", {"spectral_points": 2049}),
            ("time", {"time_order": 9}),
            ("angular", {"angular_order": 15}),
            ("pixel", {"pixel_order": 20}),
        ):
            refined = expectation(family, **(baseline_kwargs | change))
            errors[name] = relative_error(baseline.total, refined.total)
        step = 1e-7 if family == "dispersive" else 1e-10

        def evaluate(x, family=family, baseline_kwargs=baseline_kwargs):
            return expectation(family, shift=x, **baseline_kwargs)

        coarse = central_difference(evaluate, 0.0, step)
        fine = central_difference(evaluate, 0.0, step / 2)
        errors["derivative"] = relative_error(coarse, fine)
        # Independent physical throughput law gives exactly linear native counts.
        field, acq, response = make_case(family, **baseline_kwargs)

        def throughput_evaluate(x, field=field, acq=acq, response=response):
            return expected_electrons(
                field, acq, response, collecting_area_m2=1e-8, throughput=x, quantum_efficiency=0.7
            )

        throughput_derivative = central_difference(throughput_evaluate, 0.8, 0.001)
        errors["throughput_derivative"] = relative_error(
            throughput_derivative, baseline.total / 0.8
        )
        if family == "two-beam":
            # Each consecutive pair shares geometry and differs only by port.
            def sum_ports(i, wavelength, acquisition, response=response):
                pair = i - i % 2
                return response(pair, wavelength, acquisition) + response(
                    pair + 1, wavelength, acquisition
                )

            combined = expected_electrons(
                field,
                acq,
                sum_ports,
                collecting_area_m2=1e-8,
                throughput=0.8,
                quantum_efficiency=0.7,
            )
            errors["port_balance"] = relative_error(
                baseline.total[::2] + baseline.total[1::2], combined.total[::2]
            )
        results[family] = {
            "relative_errors": errors,
            "passed": all(value <= TOLERANCES.get(name, 5e-10) for name, value in errors.items()),
        }
    nominal = expectation("filter")
    geometric = {}
    for parameter, step in (("pointing_shift", 1e-5), ("time_shift", 1e-4)):

        def evaluate_geometry(x, parameter=parameter):
            return expectation("filter", **{parameter: x})

        coarse = central_difference(evaluate_geometry, 0, step)
        fine = central_difference(evaluate_geometry, 0, step / 2)
        geometric[parameter] = relative_error(coarse, fine)
    results["pointing_timing"] = {
        "relative_errors": geometric,
        "passed": all(v < 0.002 for v in geometric.values()),
    }
    detector = DetectorElectronics(
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        electrons_per_dn=0.01,
        full_well_electrons=1e6,
        bias_dn=10000,
        adc_bits=32,
    )
    mean, variance = electron_moments(nominal, detector)
    draws = sample_readout(nominal, detector, np.random.default_rng(7319), realisations=100000)
    electrons = (draws.dn - detector.bias_dn) * detector.electrons_per_dn
    z = float(np.max(abs(electrons.mean(axis=0) - mean) / np.sqrt(variance / len(electrons))))
    variance_error = float(np.max(abs(electrons.var(axis=0) / variance - 1)))
    # Noiseless unsaturated detector gain law, independent of photon integration.
    noise = np.zeros((1, len(mean)))
    dn = digitize(nominal.acquisition, mean[None, :], noise, detector).dn
    dn_reference = np.rint(mean / 0.01 + 10000)
    results["detector"] = {
        "mean_z": z,
        "variance_relative_error": variance_error,
        "passed": bool(
            z < 6
            and variance_error < 0.03
            and np.array_equal(dn[0], dn_reference)
            and not draws.full_well_saturated.any()
            and not draws.adc_clipped.any()
        ),
    }
    assert all(item["passed"] for item in results.values()), results


@pytest.mark.parametrize("family", FAMILIES)
def test_resource_budget(family):
    # Fresh interpreter; inherit the launcher's CPU affinity and numerical thread limits.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from test_instrument_characterization import _resource_case; "
            "print(json.dumps(_resource_case(sys.argv[2])))",
            str(Path(__file__).resolve().parent),
            family,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = json.loads(child.stdout)
    assert result["passed"], result


def _resource_case(family: str) -> dict:
    """Fresh process: one warmup and three complete construction/integration/readout timings."""
    repeats = 2 if family == "two-beam" else 4
    detector = DetectorElectronics(bias_dn=100)
    timings = []
    for index in range(4):
        start = perf_counter()
        signal = expectation(family, spectral_points=2049, repeats=repeats)
        readings = sample_readout(signal, detector, np.random.default_rng(918))
        elapsed = perf_counter() - start
        if index > 0:
            timings.append(elapsed)
        assert readings.dn.shape == (1, 16)
    assert not sys._is_gil_enabled()
    # Linux /proc VmHWM belongs to this executed process. ru_maxrss can retain a
    # pre-exec/fork high-water mark from the parent numerical campaign.
    status = Path("/proc/self/status").read_text()
    rss_kib = next(
        float(line.split()[1]) for line in status.splitlines() if line.startswith("VmHWM:")
    )
    rss_mib = rss_kib / 1024
    median = float(np.median(timings))
    return {
        "family": family,
        "reads": 16,
        "nodes_per_read": 75,
        "spectral_points": 2049,
        "dtype": "float64",
        "timings_s": timings,
        "median_s": median,
        "peak_process_rss_mib": rss_mib,
        "peak_rss_source": "Linux /proc/self/status VmHWM after exec",
        "ru_maxrss_mib_including_possible_pre_exec_highwater": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        / 1024,
        "gil_enabled": sys._is_gil_enabled(),
        "passed": median <= SECONDS_BUDGET and rss_mib <= RSS_MIB_BUDGET,
    }
