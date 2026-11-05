"""Bounded acquisition-to-reconstructed-scene-pixel notebook contracts."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
from orsia_instruments import (
    Acquisition,
    DetectorElectronics,
    ExpectedElectrons,
    ReadSchedule,
    electron_covariance,
)
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid
from scipy.constants import c, h

from notebooks.support import notebook_ground_pixel as reference_model
from notebooks.support.notebook_ground_pixel import (
    FAMILIES,
    acquisition_window,
    architecture,
    comparison_settings,
    detector_channels,
    direct_scene_read,
    image_channels,
    physical_kernel,
    reconstruction,
    synthetic_scene,
)
from notebooks.support.notebook_instrument_responses import extract_figures

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/era5_hitran_instrument_responses.ipynb"
SNAPSHOT = ROOT / "notebooks/data/gesat-toulouse-2025.json"


def read_notebook(path: Path) -> nbformat.NotebookNode:
    return cast(nbformat.NotebookNode, nbformat.read(path, as_version=4))


@pytest.fixture(scope="module")
def settings():
    return comparison_settings(json.loads(SNAPSHOT.read_text()))


@pytest.fixture(scope="module")
def radiance():
    return Radiance(
        wavenumber_grid(np.linspace(1e7 / 1655, 1e7 / 1645, 101)),
        np.linspace(0.0008, 0.0012, 101),
        ObservationGeometry(0.0),
    )


def test_kernel_requires_declared_entrance_band(settings, radiance):
    broader = Radiance(
        wavenumber_grid(np.linspace(1e7 / 1660, 1e7 / 1640, 5)),
        np.full(5, 1e-3),
        radiance.geometry,
    )
    with pytest.raises(ValueError, match="entrance-band endpoints"):
        physical_kernel(architecture("Uniform filter", settings), broader, settings)


@pytest.mark.parametrize("family", FAMILIES)
def test_complete_edge_exposures_and_mapping(family, settings):
    layout = architecture(family, settings)
    window = acquisition_window(layout, settings)
    bounds = window.exposure_bounds_s
    assert bounds.shape == (100, layout.read_count, 2)
    assert window.complete_edge_channels == layout.read_count
    centres = bounds.mean(axis=-1)
    achieved = centres * settings.speed_m_s + layout.target_offsets_at_exposure_centre_m
    np.testing.assert_allclose(
        achieved, np.broadcast_to((np.arange(100)[:, None] - 49.5) * 50, achieved.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.diff(bounds, axis=-1), settings.exposure_s)
    assert bounds.min() == window.first_exposure_s
    assert bounds.max() == window.last_exposure_s
    np.testing.assert_allclose(
        window.window_s,
        ((99 + np.ptp(layout.target_offsets_at_exposure_centre_m) / 50) * 50 + 25)
        / settings.speed_m_s,
    )
    assert layout.active_pixels <= 6400
    assert settings.exposure_s * settings.speed_m_s == 25


@pytest.mark.parametrize("family", FAMILIES)
def test_physical_stencil_against_direct_public_acquisition(family, settings, radiance):
    layout = architecture(family, settings)
    kernel = physical_kernel(layout, radiance, settings)
    assert kernel.shape == (layout.read_count, 5, 5, 101)
    assert np.all(kernel >= 0)
    # Asymmetric spatial and spectral structure exercises signed channel mixing,
    # detector position, both spectral extremes, finite exposure and scene halo.
    x, y = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    scene = (0.5 + 0.04 * x + 0.02 * y + 0.03 * (x > 2))[..., None] * radiance.values
    for read in sorted({0, layout.read_count // 2, layout.read_count - 1}):
        direct = direct_scene_read(layout, read, radiance, settings, scene)[0]
        composed = np.sum(kernel[read] * scene)
        np.testing.assert_allclose(composed, direct, rtol=3e-8)
    weights, effective, diagnostic = reconstruction(layout, kernel, radiance)
    np.testing.assert_allclose(effective, weights @ kernel.sum(axis=(1, 2)), atol=1e-15)
    if family == "Two-beam interferometer":
        assert diagnostic["basis_rank"] == 8
        assert diagnostic["condition_number"] < 100
        assert diagnostic["basis_closure_max"] < 1e-12
        assert weights.min() < 0
        assert effective.min() < 0
    else:
        assert weights.min() >= 0
    np.testing.assert_allclose(effective.sum(axis=1), 1, atol=1e-12)


def test_ports_sum_to_relay_and_exposure_scaling(settings, radiance):
    layout = architecture("Two-beam interferometer", settings)
    kernel = physical_kernel(layout, radiance, settings)
    relay_layout = architecture("Uniform filter", settings)
    # Identical Gaussian geometry makes complementary port sum independent of OPD.
    total = kernel[:32] + kernel[32:]
    np.testing.assert_allclose(
        total, np.broadcast_to(total[:1], total.shape), rtol=2e-12, atol=1e-14
    )
    changed = replace(settings, throughput=settings.throughput / 2)
    half = physical_kernel(relay_layout, radiance, changed)
    np.testing.assert_allclose(
        2 * half, physical_kernel(relay_layout, radiance, settings), rtol=1e-13
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_separable_quadrature_and_photon_conversion(family, settings, radiance):
    """Check spatial quadrature and photon conversion against separable integration."""
    layout = architecture(family, settings)
    order = 4
    kernel = physical_kernel(layout, radiance, settings, angular_order=order)
    nodes, weights = np.polynomial.legendre.leggauss(order)

    def interval(edges, current_nodes=nodes, current_weights=weights):
        low, high = np.asarray(edges[:-1]), np.asarray(edges[1:])
        return (
            (low[:, None] + high[:, None]) / 2 + (high - low)[:, None] * current_nodes / 2
        ).ravel(), ((high - low)[:, None] * current_weights / 2).ravel()

    offsets = np.arange(-2, 3) * 50
    y, wy = interval(offsets)
    y_hat = np.maximum(0, 1 - np.abs(y[:, None] / 50 - np.arange(-2, 3)))
    y_fraction = reference_model._pixel_fraction(y * settings.pitch_m / 50, settings)
    y_kernel = y_hat.T @ (wy * y_fraction)
    centre_fraction = reference_model._pixel_fraction(np.array([0.0]), settings)[0]
    points = radiance.grid.points
    trapezoid = np.r_[np.diff(points)[0], points[2:] - points[:-2], np.diff(points)[-1]] / 2
    spectral = (
        trapezoid
        / (h * c * 100 * points)
        * np.pi
        * (settings.diameter_m / 2) ** 2
        * settings.throughput
        * settings.qe
    )
    for read in (0, layout.read_count - 1):
        centre = -layout.target_offsets_at_exposure_centre_m[read] / settings.speed_m_s
        half = settings.exposure_s / 2
        t_nodes, t_weights = np.polynomial.legendre.leggauss(settings.time_order)
        times = centre + half * t_nodes
        wt = half * t_weights
        if family == "Dispersive imager":
            ray_x, wx = interval([-25.0, 0.0, 25.0])
            ray, tt = np.meshgrid(ray_x, times)
            ground = ray + settings.speed_m_s * tt
        else:
            x, wx = interval(offsets)
            ground, tt = np.meshgrid(x, times)
            ray = ground - settings.speed_m_s * tt
        uv = np.column_stack((ray.ravel() / settings.height_m, np.zeros(ray.size)))
        pitch = settings.pitch_m
        detector = layout.detector_x_m[read]
        acquisition = Acquisition(
            ReadSchedule(
                ("legacy",),
                ("pixel",),
                ("setting",),
                (layout.ports[read],),
                np.array([[centre - half, centre + half]]),
            ),
            tt.reshape(1, -1),
            uv[None],
            (uv * settings.focal_length_m)[None],
            (wt[:, None] * wx[None, :] / settings.height_m**2).reshape(1, -1),
            np.array([[detector - pitch / 2, detector + pitch / 2, -pitch / 2, pitch / 2]]),
            np.array([[uv[:, 0].min() - 1e-9, uv[:, 0].max() + 1e-9, -1e-3, 1e-3]]),
        )
        optics = reference_model._optics(layout, read, settings, settings.pixel_order)
        accepted = optics.response(0, 0, 0.01 / points, acquisition)
        x_hat = np.maximum(0, 1 - np.abs(ground.ravel()[:, None] / 50 - np.arange(-2, 3)))
        x_kernel = x_hat.T @ (accepted * acquisition.weights_sr_s[0, :, None] / centre_fraction)
        legacy = np.einsum("ik,j,k->ijk", x_kernel, y_kernel, spectral)
        np.testing.assert_allclose(kernel[read], legacy, rtol=2e-12, atol=1e-14)


def test_native_detector_identities_groups_draws_and_covariance(settings):
    layout = architecture("Two-beam interferometer", settings)
    shape = (64, 3, 4)
    reads, groups = reference_model.native_reads(layout, settings, shape)
    along = (np.arange(shape[1]) - (shape[1] - 1) / 2) * settings.sampling_m
    passage = along[None, :] - layout.target_offsets_at_exposure_centre_m[:, None]
    original_centres = np.broadcast_to(passage[:, :, None] / settings.speed_m_s, shape).ravel()
    original_half = settings.exposure_s / 2
    np.testing.assert_array_equal(
        reads.exposure_bounds_s,
        np.column_stack((original_centres - original_half, original_centres + original_half)),
    )
    indices = list(np.ndindex(shape))
    port_rows = tuple(f"{layout.ports[r]}-strip-{layout.stage[r]}" for r, i, j in indices)
    assert reads.sample_ids == tuple(f"read-{r}-ground-{i}-{j}" for r, i, j in indices)
    assert reads.pixel_ids == tuple(
        f"{row}-column-{j}" for row, (_, _, j) in zip(port_rows, indices, strict=True)
    )
    assert reads.setting_ids == tuple(str(layout.stage[r]) for r, i, j in indices)
    assert reads.port_ids == tuple(layout.ports[r] for r, i, j in indices)
    assert groups["row_ids"] == port_rows
    assert groups["column_ids"] == tuple(f"{layout.ports[r]}-column-{j}" for r, i, j in indices)
    assert groups["exposure_ids"] == tuple(
        f"passage-{float(passage[r, i]).hex()}" for r, i, j in indices
    )
    assert len(reads.sample_ids) == np.prod(shape)
    assert len(set(reads.sample_ids)) == np.prod(shape)
    assert len(set(reads.pixel_ids)) == 64 * 4
    times = reads.exposure_bounds_s.mean(axis=1).reshape(shape)
    np.testing.assert_allclose(
        times[:, :, 0] * settings.speed_m_s + layout.target_offsets_at_exposure_centre_m[:, None],
        np.broadcast_to(np.array([-50.0, 0.0, 50.0]), (64, 3)),
        atol=2e-13,
    )
    exposure = np.array(groups["exposure_ids"]).reshape(shape)
    np.testing.assert_array_equal(exposure[:32], exposure[32:])
    assert len(set(exposure[0, 0])) == 1
    assert exposure[0, 0, 0] != exposure[0, 1, 0]
    native = np.full(shape, 20.0)
    W = np.zeros((1, 64))
    W[0, [0, 32]] = [1, -1]
    custom = DetectorElectronics(
        dark_electrons_per_s=1,
        read_noise_electrons=0,
        full_well_electrons=1e6,
        electrons_per_dn=4,
        bias_dn=7,
        adc_bits=20,
    )
    custom_dn, custom_channels, custom_report = detector_channels(
        layout, settings, native, W, electronics=custom
    )
    exposure = np.diff(reads.exposure_bounds_s, axis=1).ravel().reshape(shape)
    np.testing.assert_array_equal(custom_dn, np.rint((native + exposure) / 4 + 7))
    np.testing.assert_allclose(
        custom_channels, np.einsum("cr,rij->cij", W, 4 * (custom_dn - 7) - exposure)
    )
    assert custom_report["zero_light_analogue_dn"] == 7 + settings.exposure_s / 4
    assert custom_report["quantization_bound"] == [4.0]
    reading, channels, actual_groups = reference_model.sample_detector_channels(
        layout,
        settings,
        native,
        W,
        np.random.default_rng(42),
        realisations=3,
        exposure_noise_electrons=2,
    )
    assert reading.dn.shape == (3, np.prod(shape))
    assert channels.shape == (3, 1, 3, 4)
    assert reading.reads.sample_ids == reads.sample_ids and actual_groups == groups
    dark = 2 * np.diff(reads.exposure_bounds_s, axis=1).ravel()
    np.testing.assert_allclose(
        channels, np.einsum("cr,brij->bcij", W, (2 * (reading.dn - 100) - dark).reshape(3, *shape))
    )
    electronics = DetectorElectronics(dark_electrons_per_s=2, read_noise_electrons=3)
    covariance = electron_covariance(
        ExpectedElectrons(reads, {"scene": native.ravel()}),
        electronics,
        **groups,
        exposure_noise_electrons=2,
    )
    projector = np.zeros((1, np.prod(shape)))
    projector[0, 0] = 1
    projector[0, 32 * 3 * 4] = -1
    # Balanced, simultaneous ports cancel common exposure noise exactly.
    np.testing.assert_allclose(
        projector @ covariance @ projector.T, [[2 * (20 + 2 * settings.exposure_s + 9)]]
    )


def test_scene_edges_and_detector_combination(settings, radiance):
    layout = architecture("Two-beam interferometer", settings)
    kernel = physical_kernel(layout, radiance, settings)
    weights, _, _ = reconstruction(layout, kernel, radiance)
    maps, spectra = synthetic_scene(radiance)
    native, channels = image_channels(
        kernel, weights, maps, spectra, layout=layout, radiance=radiance, settings=settings
    )
    assert native.shape == (64, 100, 100)
    assert channels.shape == (8, 100, 100)
    for row, column in [(0, 0), (0, 99), (99, 0), (99, 99), (49, 49)]:
        tile = np.einsum("cij,ck->ijk", maps[:, row : row + 5, column : column + 5], spectra)
        expected = np.einsum("rijk,ijk->r", kernel, tile)
        np.testing.assert_allclose(native[:, row, column], expected, rtol=2e-14)
    dn, digitized, detector = detector_channels(layout, settings, native, weights)
    np.testing.assert_array_equal(dn, np.rint((native + 2 * settings.exposure_s) / 2 + 100))
    np.testing.assert_allclose(
        digitized,
        np.einsum("cr,rij->cij", weights, 2 * (dn - 100) - 2 * settings.exposure_s),
        atol=1e-16,
    )
    bound = np.asarray(detector["quantization_bound"])
    assert np.all(np.max(np.abs(digitized - channels), axis=(1, 2)) <= bound * 1.000001)
    assert detector["unsaturated"]


def test_signed_reconstructed_response_against_all_direct_ports(settings, radiance):
    layout = architecture("Two-beam interferometer", settings)
    kernel = physical_kernel(layout, radiance, settings)
    weights, effective, _ = reconstruction(layout, kernel, radiance)
    spectrum = radiance.values * (1 + 0.25 * np.sin(np.arange(101) * 0.37))
    # Deliberately outside the eight-bin basis: test the high-resolution response.
    scene = np.broadcast_to(spectrum, (5, 5, 101))
    direct = np.array(
        [
            direct_scene_read(layout, read, radiance, settings, scene)[0]
            for read in range(layout.read_count)
        ]
    )
    np.testing.assert_allclose(weights @ direct, effective @ spectrum, rtol=3e-8)


@pytest.mark.parametrize("family", FAMILIES)
def test_effective_response_gallery_plots_every_channel(family, settings, radiance, monkeypatch):
    from notebooks.support.notebook_instrument_support import show_effective_responses

    layout = architecture(family, settings)
    kernel = physical_kernel(layout, radiance, settings)
    _, effective, _ = reconstruction(layout, kernel, radiance)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    namespace = dict(
        np=np,
        plt=plt,
        points=radiance.grid.points,
        settings=settings,
        layouts={family: layout},
        results={family: SimpleNamespace(effective=effective)},
        show_effective_responses=show_effective_responses,
    )
    namespace["cm_weights"] = (
        np.r_[
            np.diff(radiance.grid.points)[0],
            radiance.grid.points[2:] - radiance.grid.points[:-2],
            np.diff(radiance.grid.points)[-1],
        ]
        / 2
    )
    show_effective_responses(
        radiance.grid.points,
        namespace["cm_weights"],
        settings,
        {family: layout},
        {family: SimpleNamespace(effective=effective)},
    )
    try:
        wave_nm = 1e7 / radiance.grid.points[::-1]
        plotted = []
        for figure in figures:
            for ax in figure.axes:
                for line in ax.lines:
                    x = np.asarray(line.get_xdata())
                    y = np.asarray(line.get_ydata())
                    in_band = (x >= wave_nm.min()) & (x <= wave_nm.max())
                    if in_band.sum() == wave_nm.size and np.allclose(x[in_band], wave_nm):
                        plotted.append(y[in_band])
                        np.testing.assert_array_equal(y[~in_band], 0)
        for response in effective:
            assert any(
                np.allclose(curve, (response / namespace["cm_weights"])[::-1]) for curve in plotted
            )
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)


def test_notebook_code_is_valid():
    notebook = read_notebook(NOTEBOOK)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")


def test_figure_extraction(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / NOTEBOOK.name).write_text(
        json.dumps({"cells": [{"id": "demo", "outputs": [{"data": {"image/png": "cG5n"}}]}]})
    )
    paths = extract_figures(tmp_path)
    assert paths[0].read_bytes() == b"png"
