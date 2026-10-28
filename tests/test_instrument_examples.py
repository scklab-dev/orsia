"""Public multi-family examples, deterministic perturbations and runtime contracts."""

import re
import sys
from pathlib import Path

import instrument_cases as example
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_complete_public_examples():
    from orsia_instruments import (
        DetectorElectronics,
        expected_electrons,
        sample_readout,
        uniform_radiance_field,
    )

    detector = DetectorElectronics(
        dark_electrons_per_s=2,
        read_noise_electrons=3,
        full_well_electrons=1e5,
        electrons_per_dn=2,
        bias_dn=100,
    )
    for family in example.FAMILIES:
        signal = example.simulate(family)
        reads = sample_readout(signal, detector, np.random.default_rng(718))
        assert reads.acquisition is signal.acquisition
        assert np.all(signal.total > 0)
        assert not reads.full_well_saturated.any() and not reads.adc_clipped.any()
    _, acquisition, response = example.make_case("filter")
    for radiance in example.radiative_inputs():
        field = uniform_radiance_field(
            radiance,
            time_bounds_s=(-0.1, 0.2),
            u_bounds=(-0.004, 0.004),
            v_bounds=(-0.004, 0.004),
            spectral_bounds_cm1=(4900.0, 5100.0),
        )
        signal = expected_electrons(field, acquisition, response, collecting_area_m2=1e-8)
        sample_readout(signal, detector, np.random.default_rng(720))
        assert np.all(signal.total > 0)
    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("family", example.FAMILIES)
def test_family_full_grid_and_parameter_derivatives(family):
    from orsia_instruments import central_difference, native_difference

    baseline = example.simulate(family)
    np.testing.assert_allclose(example.simulate(family).total, baseline.total, rtol=5e-13)
    step = 1e-7 if family == "dispersive" else 1e-10

    def evaluate(x):
        return example.simulate(family, shift=x)

    derivative = central_difference(evaluate, 0, step)
    refined = central_difference(evaluate, 0, step / 2)
    np.testing.assert_allclose(derivative, refined, rtol=0.002, atol=1e-4)
    assert np.max(np.abs(derivative)) > 0
    np.testing.assert_array_equal(native_difference(baseline, baseline), 0)


def test_supplied_pointing_timing_changes_propagate_scene_support():
    from orsia_instruments import expected_electrons, native_difference

    nominal = example.simulate("filter")
    for pointing_shift, time_shift in ((1e-4, 0.0), (0.0, 0.001)):
        field, acq, response = example.make_case(
            "filter", pointing_shift=pointing_shift, time_shift=time_shift
        )
        changed = expected_electrons(
            field, acq, response, collecting_area_m2=1e-8, throughput=0.8, quantum_efficiency=0.7
        )
        assert np.max(abs(native_difference(nominal, changed))) > 0


def test_instrument_document_links_fences_and_whitespace():
    for relative in ("README.md", "libs/orsia-instruments/README.md"):
        document = ROOT / relative
        text = document.read_text()
        assert sum(line.startswith("```") for line in text.splitlines()) % 2 == 0
        assert all(line == line.rstrip() for line in text.splitlines())
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            path = target.split("#", 1)[0]
            if path and "://" not in path:
                assert (document.parent / path).exists(), (relative, target)
