"""Bounded acquisition-to-reconstructed-scene-pixel notebook contracts."""

import ast
import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from orsia_instruments import (
    Acquisition,
    DetectorElectronics,
    ExpectedElectrons,
    electron_covariance,
)
from orsia_radiative import ObservationGeometry, Radiance, wavenumber_grid
from scipy.constants import c, h

from scripts.replay_instrument_response import _retained_notebook_outputs

ROOT = Path(__file__).resolve().parents[2]
HELPERS = runpy.run_path(str(ROOT / "notebooks/support/notebook_ground_pixel.py"))
FAMILIES = HELPERS["FAMILIES"]
acquisition_window = HELPERS["acquisition_window"]
architecture = HELPERS["architecture"]
comparison_settings = HELPERS["comparison_settings"]
detector_channels = HELPERS["detector_channels"]
direct_scene_read = HELPERS["direct_scene_read"]
image_channels = HELPERS["image_channels"]
physical_kernel = HELPERS["physical_kernel"]
reconstruction = HELPERS["reconstruction"]
synthetic_scene = HELPERS["synthetic_scene"]
IO = runpy.run_path(str(ROOT / "notebooks/support/notebook_instrument_responses.py"))
extract_figures = IO["extract_figures"]
preparation_sources = IO["preparation_sources"]
NOTEBOOK = ROOT / "notebooks/era5_hitran_instrument_responses.ipynb"
SNAPSHOT = ROOT / "notebooks/data/gesat-toulouse-2025.json"
SOURCE = ROOT / "notebooks/era5_hitran_sasktran2_five_bands.ipynb"


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


def test_preparation_preserves_source_statements():
    original = {cell.id: cell.source for cell in nbformat.read(SOURCE, as_version=4).cells}
    selected = preparation_sources(SOURCE)
    assert set(selected) == {"imports", "config", "atmosphere", "materialize"}
    for name, text in selected.items():
        before = {ast.dump(node) for node in ast.parse(original[name]).body}
        assert all(ast.dump(node) in before for node in ast.parse(text).body)
    assert "prepare_case" in selected["materialize"]
    assert "cached_reference" not in "\n".join(selected.values())


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
def test_notebook_public_scan_preserves_native_order_ports_and_timing(family, settings):
    sources = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    namespace = {}
    exec(compile(sources["imports"], f"{NOTEBOOK}:imports", "exec"), namespace)
    exec(compile(sources["scan-example"], f"{NOTEBOOK}:scan-example", "exec"), namespace)
    layout = architecture(family, settings)
    shape = (layout.read_count, 3, 4)
    actual = namespace["native_reads"](layout, settings, shape)
    reference, _ = HELPERS["native_reads"](layout, settings, shape)
    assert actual.sample_ids == reference.sample_ids
    assert actual.pixel_ids == reference.pixel_ids
    assert actual.setting_ids == reference.setting_ids
    assert actual.port_ids == reference.port_ids
    np.testing.assert_allclose(
        actual.exposure_bounds_s, reference.exposure_bounds_s, rtol=0, atol=3e-17
    )
    if family == "Two-beam interferometer":
        assert actual.port_ids[: 32 * 3 * 4] == ("+",) * (32 * 3 * 4)
        assert actual.port_ids[32 * 3 * 4 :] == ("-",) * (32 * 3 * 4)
    notebook_layout = namespace["DetectorLayout"](
        family=family,
        stages=layout.stages,
        stage=layout.stage,
        ports=layout.ports,
        detector_x_m=layout.detector_x_m,
        target_offsets_at_exposure_centre_m=layout.target_offsets_at_exposure_centre_m,
        channel_nm=layout.channel_nm,
        image_rows=100,
    )
    window = namespace["acquisition_window"](notebook_layout, settings)
    reference_window = acquisition_window(layout, settings)
    np.testing.assert_allclose(
        window.exposure_bounds_s, reference_window.exposure_bounds_s, rtol=0, atol=3e-16
    )
    assert abs(window.first_exposure_s - reference_window.first_exposure_s) < 3e-16
    assert abs(window.last_exposure_s - reference_window.last_exposure_s) < 3e-16


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
def test_original_separable_arithmetic(family, settings, radiance):
    """Preserve the original helper's dt dx dy/h² and independent spectral factor."""
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
    y_fraction = HELPERS["_pixel_fraction"](y * settings.pitch_m / 50, settings)
    y_kernel = y_hat.T @ (wy * y_fraction)
    centre_fraction = HELPERS["_pixel_fraction"](np.array([0.0]), settings)[0]
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
            ("legacy",),
            ("pixel",),
            ("setting",),
            (layout.ports[read],),
            np.array([[centre - half, centre + half]]),
            tt.reshape(1, -1),
            uv[None],
            (uv * settings.focal_length_m)[None],
            (wt[:, None] * wx[None, :] / settings.height_m**2).reshape(1, -1),
            np.array([[detector - pitch / 2, detector + pitch / 2, -pitch / 2, pitch / 2]]),
            np.array([[uv[:, 0].min() - 1e-9, uv[:, 0].max() + 1e-9, -1e-3, 1e-3]]),
        )
        optics = HELPERS["_optics"](layout, read, settings, settings.pixel_order)
        accepted = optics.response(0, 0, 0.01 / points, acquisition)
        x_hat = np.maximum(0, 1 - np.abs(ground.ravel()[:, None] / 50 - np.arange(-2, 3)))
        x_kernel = x_hat.T @ (accepted * acquisition.weights_sr_s[0, :, None] / centre_fraction)
        legacy = np.einsum("ik,j,k->ijk", x_kernel, y_kernel, spectral)
        np.testing.assert_allclose(kernel[read], legacy, rtol=2e-12, atol=1e-14)


def test_native_detector_identities_groups_draws_and_covariance(settings):
    layout = architecture("Two-beam interferometer", settings)
    shape = (64, 3, 4)
    reads, groups = HELPERS["native_reads"](layout, settings, shape)
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
    reading, channels, actual_groups = HELPERS["sample_detector_channels"](
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


def test_response_sections_follow_operator_dependencies():
    cells = nbformat.read(NOTEBOOK, as_version=4).cells
    order = [cell.id for cell in cells]
    expected = [
        "inputs",
        "settings-doc",
        "method",
        "physical-response",
        "plots-doc",
        "tables",
        "scan-preview-doc",
        "scan-preview",
        "scan-timing-doc",
        "scan-timing",
        "native-image-doc",
        "native-image",
        "local-cube-doc",
        "local-cube",
        "acquisition-archive-doc",
        "operators-images",
        "flat-doc",
        "flat-gallery",
        "atmosphere-doc",
        "swir-gallery",
        "evidence",
    ]
    assert [name for name in order if name in expected] == expected
    assert order.index("imports") < order.index("scan-example") < order.index("inputs")
    assert order.index("method") < order.index("scene-example") < order.index("physical-response")
    scan = next(cell.source for cell in cells if cell.id == "scan-example")
    model = "\n".join(cell.source for cell in cells if cell.id in ("scan-example", "scene-example"))
    assert "notebook_ground_pixel" not in "\n".join(cell.source for cell in cells)
    assert "notebook_scan_support" not in scan
    assert "IMAGE_ROWS = 100" in scan
    assert "entrance_nm=(1645.0, 1655.0)" in scan
    assert "return along_track_raster_reads(" in scan
    assert "sample_id=lambda r, i, j:" in scan
    assert "pixel_id=lambda r, i, j:" in scan
    assert all(
        name in model
        for name in (
            "def native_reads(",
            "def scan_templates(",
            "def acquisition_window(",
            "def reconstruction(",
            "def synthetic_scene(",
        )
    )
    assert "def native_read_inputs(" not in scan
    assert "row_ids=" not in scan and "exposure_ids=" not in scan
    sources = {cell.id: cell.source for cell in cells}
    physical = sources["physical-response"]
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(physical))
        if isinstance(node, ast.Call)
    ]
    for name in (
        "detector_layout",
        "acquisition_window",
        "synthetic_scene",
        "scan_templates",
        "gaussian_along_track_response",
        "evaluate_instrument",
        "response_electrons",
        "reconstruction",
        "electron_moments",
        "digitize",
        "correct_digital_readings",
        "electron_covariance",
        "propagate_covariance",
    ):
        assert name in calls
    assert "groups" not in physical
    assert "exposure_groups" not in physical
    assert "for family, layout in layouts.items():" not in physical
    assert "FAMILIES" not in scan + physical
    labels = [
        node.value.value
        for node in ast.walk(ast.parse(physical))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "label" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert labels == list(FAMILIES)
    assert "maps, spectra=source_spectra" in physical
    assert "scene_pixel" not in sources["inputs"]
    assert "layouts =" not in sources["inputs"]
    assert "windows =" not in sources["inputs"]
    assert "return SceneResponseEvaluation(" in physical
    assert [
        cell.id
        for cell in cells
        if cell.cell_type == "code"
        and cell.id
        not in ("imports", "scan-example", "scene-example", "inputs", "physical-response")
    ][0] == "tables"
    assert sources["operators-images"].strip().startswith("save_acquisition(")
    assert "response.coefficients" in physical
    assert "native" in sources["native-image"]
    assert "SceneRadiance" in sources["local-cube"]
    assert "scene_pixel" in physical
    assert "all channels of **W @ H**" in sources["flat-doc"]
    assert "effective kernels above" in sources["atmosphere-doc"]


def test_scan_preview_separates_sight_line_and_six_family_timing(settings, monkeypatch, tmp_path):
    sources = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    namespace = {}
    exec(compile(sources["imports"], f"{NOTEBOOK}:imports", "exec"), namespace)
    exec(compile(sources["scan-example"], f"{NOTEBOOK}:scan-example", "exec"), namespace)
    layouts = {family: architecture(family, settings) for family in FAMILIES}
    windows = {family: acquisition_window(layout, settings) for family, layout in layouts.items()}
    namespace.update(layouts=layouts, windows=windows, settings=settings)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    exec(compile(sources["scan-preview"], f"{NOTEBOOK}:scan-preview", "exec"), namespace)
    exec(compile(sources["scan-timing"], f"{NOTEBOOK}:scan-timing", "exec"), namespace)
    assert len(shown) == 2
    assert "AlongTrackSpatialSupport" not in sources["scan-preview"]
    assert "preview_rows" not in sources["scan-preview"]
    assert "### 2a. The sight line" in sources["scan-preview-doc"]
    assert "### 2b. When that pixel is read" in sources["scan-timing-doc"]
    sight = shown[0].axes[0]
    line = sight.lines[2]
    np.testing.assert_allclose(line.get_ydata(), [1, 0])
    assert line.get_xdata()[1] == -25
    assert sight.get_yticklabels()[1].get_text() == "Platform (schematic height)"
    assert sight.get_xlabel() == "Along-track ground position [m]"
    np.testing.assert_allclose(
        line.get_xdata()[1] - line.get_xdata()[0],
        layouts["Filter mosaic"].target_offsets_at_exposure_centre_m[0],
    )
    shown[0].savefig(tmp_path / "scan-preview.png")
    timing = shown[1].axes[0]
    assert [label.get_text() for label in timing.get_yticklabels()] == list(FAMILIES)
    assert len(timing.lines) == len(FAMILIES) + 1
    assert len(timing.collections) == len(FAMILIES)
    for row, family in enumerate(FAMILIES):
        bounds = windows[family].exposure_bounds_s[49]
        centres = np.unique(1000 * (bounds.mean(axis=1) + 25 / settings.speed_m_s))
        np.testing.assert_allclose(timing.lines[row].get_xdata(), centres, atol=1e-12)
        np.testing.assert_array_equal(timing.lines[row].get_ydata(), np.full(centres.shape, row))
        np.testing.assert_allclose(
            timing.collections[row].get_segments()[0][[0, 1], 0],
            1000 * (np.array([bounds.min(), bounds.max()]) + 25 / settings.speed_m_s),
            atol=1e-12,
        )
    assert len(timing.lines[3].get_xdata()) == 1  # Simultaneous dispersive columns.
    assert len(timing.lines[4].get_xdata()) == 32  # Two ports per strip time.
    shown[1].savefig(tmp_path / "scan-timing.png")


def test_native_photoelectron_gallery_has_separate_scene_and_six_reads(
    settings, monkeypatch, tmp_path
):
    sources = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    maps = np.ones((2, 104, 104))
    spectra = np.full((2, 9), 1e-3)
    images = {
        family: SimpleNamespace(native=np.full((1, 100, 100), index + 1.0))
        for index, family in enumerate(FAMILIES)
    }
    namespace = dict(
        np=np,
        maps=maps,
        source_spectra=spectra,
        points=np.linspace(1e7 / 1655, 1e7 / 1645, 9),
        layouts={family: architecture(family, settings) for family in FAMILIES},
        results=images,
    )
    exec(compile(sources["native-image"], f"{NOTEBOOK}:native-image", "exec"), namespace)
    assert len(shown) == 1
    shown[0].savefig(tmp_path / "native-image.png")
    axes = namespace["axes"].flat
    assert sum(len(ax.images) for ax in axes) == 7
    np.testing.assert_array_equal(axes[0].images[0].get_array(), np.full((100, 100), 0.002))
    for index, ax in enumerate(list(axes)[1:7], start=1):
        np.testing.assert_array_equal(ax.images[0].get_array(), np.full((100, 100), float(index)))
        assert ax.get_title() == FAMILIES[index - 1]
    assert "expected electrons per exposure" in shown[0]._suptitle.get_text()


def test_notebook_executes_public_instrument_response_chain(settings, radiance, monkeypatch):
    """Run the actual canonical code cells on a small offline spectrum."""
    sources = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    stages = ("physical-response", "operators-images")
    calls = {
        ast.unparse(node.func)
        for name in ("scan-example", *stages)
        for node in ast.walk(ast.parse(sources[name]))
        if isinstance(node, ast.Call)
    }
    assert {
        "scan_templates",
        "gaussian_along_track_response",
        "response_electrons",
    } <= calls
    assert {"physical_kernel", "image_channels"}.isdisjoint(calls)
    namespace = {}
    exec(compile(sources["imports"], f"{NOTEBOOK}:imports", "exec"), namespace)
    exec(compile(sources["scan-example"], f"{NOTEBOOK}:scan-example", "exec"), namespace)
    exec(compile(sources["scene-example"], f"{NOTEBOOK}:scene-example", "exec"), namespace)
    namespace.update(
        radiance=radiance,
        points=radiance.grid.points,
        settings=settings,
        # Compare the computation directly, leaving archival writes to their own workflow.
        save_acquisition=lambda *args: None,
    )
    trace = []

    def observed(function, name):
        def call(*args, **kwargs):
            value = function(*args, **kwargs)
            trace.append(name)
            if name == "response_electrons":
                index = trace.count("response_electrons") - 1
                layout = architecture(FAMILIES[index], settings)
                expected_reads, _ = HELPERS["native_reads"](
                    layout, settings, (layout.read_count, 100, 100)
                )
                assert value.reads.sample_ids == expected_reads.sample_ids
                assert value.reads.pixel_ids == expected_reads.pixel_ids
                assert value.reads.setting_ids == expected_reads.setting_ids
                assert value.reads.port_ids == expected_reads.port_ids
                np.testing.assert_allclose(
                    value.reads.exposure_bounds_s,
                    expected_reads.exposure_bounds_s,
                    rtol=0,
                    atol=3e-17,
                )
            return value

        return call

    # Trace the public boundaries in the actual notebook computation cell.
    for name in ("gaussian_along_track_response", "along_track_raster_reads"):
        namespace[name] = observed(namespace[name], name)
    for name in ("response_electrons", "correct_digital_readings"):
        if name == "response_electrons":
            original_array_apply = namespace[name]
        namespace[name] = observed(namespace[name], name)
    for name in stages:
        exec(compile(sources[name], f"{NOTEBOOK}:{name}", "exec"), namespace)
    expected_trace = [
        "gaussian_along_track_response",
        "along_track_raster_reads",  # Native scene image.
        "response_electrons",
        "correct_digital_readings",
    ] * len(FAMILIES)
    assert trace == expected_trace
    namespace["response_electrons"] = original_array_apply
    exec(compile(sources["local-cube"], f"{NOTEBOOK}:local-cube", "exec"), namespace)
    np.testing.assert_allclose(
        namespace["cube_electrons"],
        namespace["results"]["Uniform filter"].native[:, 49, 49],
        rtol=1e-12,
    )
    for family in FAMILIES:
        layout = architecture(family, settings)
        actual = namespace["results"][family]
        kernel = physical_kernel(layout, radiance, settings)
        weights, effective, diagnostics = reconstruction(layout, kernel, radiance)
        maps, spectra = synthetic_scene(radiance)
        native, channels = image_channels(
            kernel, weights, maps, spectra, layout=layout, radiance=radiance, settings=settings
        )
        dn, digitized, detector = detector_channels(layout, settings, native, weights)
        for name, expected in dict(
            kernel=kernel,
            weights=weights,
            effective=effective,
            native=native,
            channels=channels,
            dn=dn,
            digitized=digitized,
        ).items():
            np.testing.assert_array_equal(getattr(actual, name), expected)
        assert actual.diagnostics == diagnostics
        # The reference retains nonzero common-noise grouping experiments; the
        # notebook's zero-common-noise example omits that inactive diagnostic.
        assert "exposure_groups" not in actual.detector
        assert actual.detector == {k: v for k, v in detector.items() if k != "exposure_groups"}


@pytest.mark.parametrize("family", FAMILIES)
def test_effective_response_gallery_plots_every_channel(family, settings, radiance, monkeypatch):
    from notebooks.support.notebook_instrument_support import show_effective_responses

    cells = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
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
    exec(compile(cells["flat-gallery"], f"{NOTEBOOK}:flat-gallery", "exec"), namespace)
    (figure,) = figures
    _, right = figure.axes
    lines = [line for line in right.lines if line.get_label().startswith("Ch ")]
    labels = [f"Ch {index + 1}: {centre:.3f} nm" for index, centre in enumerate(layout.channel_nm)]
    assert len(lines) == len(effective) == len(labels)
    assert len({to_rgba(line.get_color()) for line in lines}) == len(labels)
    assert [line.get_label() for line in lines] == labels
    assert [text.get_text() for text in right.get_legend().get_texts()] == labels
    for index, line in enumerate(lines):
        np.testing.assert_array_equal(line.get_xdata()[2:-2], 1e7 / radiance.grid.points[::-1])
        np.testing.assert_allclose(
            line.get_ydata()[2:-2], (effective[index] / namespace["cm_weights"])[::-1]
        )
        np.testing.assert_array_equal(line.get_ydata()[[0, 1, -2, -1]], 0)
    interpretation = "Signed" if family == "Two-beam interferometer" else "Positive"
    assert right.get_title().startswith(interpretation)
    if interpretation == "Signed":
        assert min(line.get_ydata().min() for line in lines) < 0
        assert right.get_ylim()[0] < 0
    else:
        assert all(np.all(line.get_ydata() >= 0) for line in lines)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    legend_box = right.get_legend().get_window_extent(renderer)
    assert legend_box.x0 > right.get_window_extent(renderer).x1
    assert figure.bbox.contains(legend_box.x0, legend_box.y0)
    assert figure.bbox.contains(legend_box.x1, legend_box.y1)


def test_retained_notebook_outputs_allow_removed_and_reordered_cells():
    def cell(name, *, output=None, metadata=None):
        return dict(
            id=name,
            cell_type="code",
            metadata={} if metadata is None else metadata,
            outputs=[] if output is None else [output],
            execution_count=1,
        )

    before = dict(
        cells=[
            cell("imports"),
            cell("inputs"),
            cell("removed", output={"text": "old"}),
            cell("operators-images", output={"text": "saved"}, metadata={"execution": 1}),
        ]
    )
    current = dict(
        cells=[
            cell("imports"),
            cell("operators-images", output={"text": "saved"}, metadata={"execution": 1}),
            cell("physical-response", metadata={"new": True}),
            cell("inputs"),
        ]
    )
    outputs, missing = _retained_notebook_outputs(before, current)
    assert set(outputs) == {"imports", "inputs", "operators-images"}
    assert (
        outputs["operators-images"]
        == hashlib.sha256(json.dumps([{"text": "saved"}], sort_keys=True).encode()).hexdigest()
    )
    assert missing == ["removed"]
    current["cells"][1]["outputs"][0]["text"] = "changed"
    with pytest.raises(AssertionError):
        _retained_notebook_outputs(before, current)
    current["cells"][1]["outputs"][0]["text"] = "saved"
    current["cells"][1]["metadata"] = {"execution": 2}
    with pytest.raises(AssertionError):
        _retained_notebook_outputs(before, current)


def test_saved_notebook_and_receipts():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK}:{cell.id}", "exec")
    text = "\n".join(cell.source for cell in notebook.cells)
    for phrase in (
        "reconstructed",
        "continuous geocentric nadir",
        "W @ H",
        "Toulouse",
        "0°",
        "2024-01-15",
    ):
        assert phrase in text
    receipt = ROOT / "outputs/instrument-responses/evidence.json"
    # Scientific execution is a separate explicit recipe. Once saved, validate it.
    if (
        receipt.exists()
        and json.loads(receipt.read_text()).get("comparison") == "reconstructed ground-image pixel"
    ):
        evidence = json.loads(receipt.read_text())
        assert evidence["scenario_sha256"] == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
        # A fresh local execution starts without a replay. If a local replay
        # exists, validate its source and input identities independently.
        local_replay = ROOT / "outputs/ground-response-integration/comparison.json"
        if local_replay.exists():
            replay = json.loads(local_replay.read_text())
            for name, sha in replay["reference_sha256"].items():
                assert hashlib.sha256((receipt.parent / name).read_bytes()).hexdigest() == sha
            assert all(
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
                for path, sha in replay["source_sha256"].items()
            )
            assert all(
                item.get("exact", False) or item["max_abs"] <= item["tolerance_abs"]
                for item in replay["comparisons"].values()
            )
        assert evidence["runtime"]["free_threaded"]
        assert len(evidence["families"]) == 6
        assert not evidence["runtime"]["gil_enabled"]
        assert set(evidence["runtime"]["affinity"]) <= set(range(8))
        assert set(evidence["runtime"]["thread_pools"].values()) == {"1"}
        assert all(
            max(
                check["direct_relative_error"],
                check["angular_refinement"],
                check["pixel_refinement"],
            )
            < 0.003
            for check in evidence["checks"].values()
        )


def test_figure_extraction(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / NOTEBOOK.name).write_text(
        json.dumps({"cells": [{"id": "demo", "outputs": [{"data": {"image/png": "cG5n"}}]}]})
    )
    paths = extract_figures(tmp_path)
    assert paths[0].read_bytes() == b"png"
