"""Exercise the six explicit notebook chains with a bounded synthetic spectrum."""

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from orsia_instruments import (
    calibrate_frame_electrons,
    camera_acquisition_grid,
    reconstruct_radiance_cube,
    stack_frame_electrons,
)
from orsia_inverse import solve_linear
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid

from notebooks.support import replay_instrument_response
from notebooks.support.instrument_notebook_checks import check_strip_boundaries
from notebooks.support.notebook_frame_display import source_text
from notebooks.support.notebook_ground_pixel import synthetic_scene
from notebooks.support.replay_instrument_response import (
    compare_arrays,
    execute_instrument_cells,
    instrument_cells,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    radiance = Radiance(
        wavenumber_grid(np.linspace(1e7 / 1655, 1e7 / 1645, 33)),
        np.linspace(0.0008, 0.0012, 33),
        ObservationGeometry(0.0),
    )
    # Property tests use lower integration orders to keep coverage tracing bounded.
    # The fresh 801-node notebook execution separately checks the physical default
    # and its quadrature refinements; these tests check composition and identities.
    cells = instrument_cells(ROOT)
    for cell in cells:
        source = "".join(cell["source"])
        if cell["id"] == "geometry":
            source += "\npixel_order, angular_order = 2, 8\nshared_geometry['time_order'] = 1\n"
        if cell["id"] == "slit-instrument":
            source = source.replace("slit_order = 48", "slit_order = 16")
        cell["source"] = source.splitlines(keepends=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(replay_instrument_response, "instrument_cells", lambda root: cells)
        return execute_instrument_cells(
            ROOT, tmp_path_factory.mktemp("chains"), radiance, checks=False, archive=False
        )


def test_six_literal_end_to_end_sections_and_species():
    notebook = json.loads((ROOT / "notebooks/era5_hitran_instrument_responses.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    order = list(cells)
    for family in ("uniform", "mosaic", "lvf", "slit", "two-beam", "cavity"):
        positions = [
            order.index(family + suffix)
            for suffix in ("-instrument", "-measure", "-reconstruct", "-product")
        ]
        assert positions == sorted(positions)
        assert "simulate_frames(" in cells[family + "-measure"]
        assert "use_expectation=True" not in cells[family + "-reconstruct"]
        assert positions[-1] < order.index("summary")
    for family in ("uniform", "mosaic", "lvf", "slit", "cavity"):
        text = cells[family + "-reconstruct"]
        for call in (
            "frame_response(",
            "group_native_bands(",
            "calibrate_frames(",
            "reconstruct_radiance_cube(",
        ):
            assert call in text
        assert f"{family}_estimated_grid" in text
    assert "read_spectroscopy_collection(" in cells["atmosphere"]
    for species in ("h2o", "co2", "ch4"):
        assert "evaluate_line_absorption(" in cells["absorption-" + species]


def test_same_scene_native_order_and_selected_target_coverage(chains):
    expected_maps, expected_spectra = synthetic_scene(chains["radiance"])
    np.testing.assert_array_equal(chains["maps"], expected_maps)
    np.testing.assert_array_equal(chains["source_spectra"], expected_spectra)
    for name, count in (("uniform", 1), ("mosaic", 8), ("lvf", 16), ("slit", 16), ("cavity", 8)):
        frames = chains[name + "_frames"]
        schedule = frames.images.schedule
        assert len(schedule.native_indices) == 4 * count
        flat = np.ravel_multi_index(schedule.native_indices.T, schedule.native_shape)
        assert np.all(np.diff(flat) > 0)
        assert not schedule.complete
        assert frames.images.dn is not None
        assert not frames.images.full_well_saturated.any()
        assert not frames.images.adc_clipped.any()
        assert chains[name + "_estimated_grid"] is not chains[name + "_truth_grid"]
        assert chains[name + "_estimated"].detector is not chains[name + "_detector"]
        np.testing.assert_allclose(
            abs(chains[name + "_along_m"][chains[name + "_selected"]]), 25, atol=1e-11
        )
        cube, expected = chains[name + "_cube"], chains[name + "_expectation_cube"]
        assert len(cube.bands) == count
        assert cube.valid[3:5, 3:5].all()
        assert np.isnan(cube.values[~cube.valid]).all()
        assert not np.array_equal(cube.values[cube.valid], expected.values[cube.valid])
        for band in cube.bands:
            assert np.all(band.weights >= 0)
            np.testing.assert_allclose(band.weights.sum(), 1)
    assert chains["two_beam_plus_detector"].shape == (100, 32)
    assert chains["slit_detector"].shape == (100, 16)
    for sign, schedule in zip(("+", "-"), chains["two_beam_schedules"], strict=True):
        assert set(schedule.reads.port_ids) == {sign}
        assert len(schedule.native_indices) == 128
    assert chains["two_beam_response"].matrix.shape == (256, 128)
    assert chains["two_beam_estimate"].rank <= 128
    assert not chains["two_beam_estimate"].identifiable.all()
    assert np.any(chains["two_beam_cell_bin_residual"] != 0)


def test_fixed_strip_boundary_quadrature(chains):
    check_strip_boundaries(
        chains["two_beam_plus_detector"],
        chains["two_beam_fixed_opd"],
        chains["two_beam_opd_m"],
        order=12,
    )
    check_strip_boundaries(
        chains["cavity_detector"],
        chains["cavity_fixed_offset"],
        chains["cavity_gaps_m"],
        base=chains["cavity_gaps_m"][0],
        order=12,
    )
    assert np.ndim(chains["two_beam_true"][0].optics.settings["opd_m"]) == 0
    assert np.ndim(chains["cavity_true"].optics.settings["spacing_m"]) == 0


def test_noisy_two_port_correction_covariance_and_identity(chains):
    frames = chains["two_beam_frames"][0]
    detector = chains["two_beam_estimated"][0].detector
    response = chains["two_beam_response"]
    charges = list(chains["two_beam_charges"])
    changed_dn = frames.images.dn.copy()
    changed_dn[0, 0] += 100
    charges[0] = calibrate_frame_electrons(
        replace(frames.images, dn=changed_dn),
        detector=detector,
        covariance=chains["two_beam_dn_variances"][0],
    )
    observed, valid = stack_frame_electrons(response, charges)
    assert observed[0] != chains["two_beam_observed"][0]
    changed = solve_linear(
        response.matrix,
        observed,
        covariance=chains["two_beam_covariance"],
        valid=valid,
        max_bytes=chains["two_beam_solve_budget"],
    )
    assert not np.allclose(changed.state, chains["two_beam_estimate"].state, rtol=1e-8, atol=0)
    assert charges[0].covariance is not None
    np.testing.assert_allclose(charges[0].covariance, chains["two_beam_dn_variances"][0] * 4)
    with pytest.raises(ValueError, match="schedule|port|identit|alignment"):
        stack_frame_electrons(response, charges[::-1])
    with pytest.raises(MemoryError):
        solve_linear(
            response.matrix,
            observed,
            covariance=chains["two_beam_covariance"],
            valid=valid,
            max_bytes=1,
        )


def test_estimated_geometry_controls_registration(chains):
    def shifted(times):
        positions, velocities = chains["local_trajectory"](times)
        positions[:, 0] += 50
        return positions, velocities

    geometry: dict = dict(chains["shared_geometry"], trajectory=shifted)
    estimated = camera_acquisition_grid(
        chains["uniform_estimated"],
        chains["uniform_schedule"],
        **geometry,
        instrument_attitude=chains["row_strip_instrument"],
        angular_order=chains["angular_order"],
    )
    cube = reconstruct_radiance_cube(
        chains["uniform_calibrated"],
        estimated,
        x_edges_m=chains["product_edges_m"],
        y_edges_m=chains["product_edges_m"],
        bands=chains["uniform_bands"],
    )
    assert not np.array_equal(cube.coverage, chains["uniform_cube"].coverage)
    np.testing.assert_allclose(
        cube.coverage[:, 1:], chains["uniform_cube"].coverage[:, :-1], atol=1e-12
    )


def test_presentation_renders_actual_noisy_data(chains, monkeypatch):
    from notebooks.support.notebook_frame_display import show_native, show_product

    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    show_native("Uniform", chains["uniform_frames"])
    show_product(
        "Uniform",
        chains["uniform_cube"],
        chains["uniform_expectation_cube"],
        chains["uniform_reference"],
        chains["points"],
    )
    assert len(figures) == 2
    native_images = [artist for ax in figures[0].axes for artist in ax.images]
    np.testing.assert_array_equal(
        np.sort(np.ma.asarray(native_images[1].get_array()).compressed()),
        np.sort(chains["uniform_frames"].images.dn.ravel()),
    )
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)


@pytest.mark.parametrize(
    "changed", ([np.inf, 1.0], [-np.inf, -np.inf], [np.inf, np.nan], [np.inf, np.inf])
)
def test_replay_rejects_changed_nonfinite_values(changed):
    original = {"values": np.array([np.inf, -np.inf])}
    assert compare_arrays(original, original)["values"]["max_abs"] == 0
    with pytest.raises(AssertionError):
        compare_arrays(original, {"values": np.array(changed)})


def test_replay_comparison_and_extraction():
    cells = instrument_cells(ROOT)
    assert cells[0]["id"] == "imports"
    assert not any(cell["id"] == "atmosphere" for cell in cells)
    original = {"dn": np.array([1, 2]), "cube": np.array([1.0, np.nan])}
    assert len(compare_arrays(original, original)) == 2
    with pytest.raises(AssertionError):
        compare_arrays(original, dict(original, dn=np.array([1, 3])))
    extracted = source_text(
        {
            "cells": [
                {
                    "id": "test",
                    "cell_type": "code",
                    "source": ["x = 1"],
                    "outputs": [{"data": {"image/png": "ignored"}}],
                }
            ]
        }
    )
    assert "x = 1" in extracted and "ignored" not in extracted
