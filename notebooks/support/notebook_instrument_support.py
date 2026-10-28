"""Display, local archives and bounded checks for the ground-pixel notebook.

The notebook keeps the scientific acquisition/reconstruction steps explicit.
Arrays passed to display and archive functions are borrowed read-only; the
bounded checks add the out-of-basis diagnostic to the interferometer record.
"""

import hashlib
import json
import os
import sys
import sysconfig
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import Markdown, display
from orsia_instruments import (
    LinearTrajectory,
    RadianceField,
    along_track_acquisition,
    expected_electrons,
)
from orsia_radiative import Radiance, SpectralDomain, SpectralGrid, wavenumber_grid


def configure_display():
    """Configure inline retina figures and the retained notebook typography."""
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("retina", bbox_inches=None)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 15,
        }
    )


def runtime_settings():
    """Record and check imported free-threaded runtime and bounded CPU settings."""
    runtime = dict(
        python=sys.version,
        free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        gil_enabled=sys._is_gil_enabled(),
        affinity=sorted(os.sched_getaffinity(0)),
        thread_pools={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    )
    assert runtime["free_threaded"] and not runtime["gil_enabled"]
    assert set(runtime["affinity"]) <= set(range(8))
    assert set(runtime["thread_pools"].values()) == {"1"}
    return runtime


def show_layouts(layouts, windows, settings):
    """Display detector allocations and complete-image acquisition windows."""
    rows = [
        "| Family | Active detector pixels | Physical reads / scene pixel | "
        "Reconstructed channels | First start [s] | Last end [s] | "
        "Complete image [s] | One-pixel span [ms] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, layout in layouts.items():
        window = windows[family]
        rows.append(
            f"| {family} | {layout.active_pixels} | {layout.read_count} | "
            f"{len(layout.channel_nm)} | {window.first_exposure_s:.6f} | "
            f"{window.last_exposure_s:.6f} | {window.window_s:.6f} | "
            f"{1000 * window.per_pixel_span_s:.3f} |"
        )
    display(Markdown("\n".join(rows)))
    display(
        Markdown(
            f"The 5 km central ground traversal is **{5000 / settings.speed_m_s:.6f} s**. "
            "Table bounds use the 100 pixel centres (−2475…2475 m), finite exposure bounds "
            "and every required spectral read; the image boundary lies half a sampling "
            "interval beyond each end centre. Spatial optical support uses the declared scene halo."
        )
    )


def save_acquisition(
    OUT, layouts, windows, results, points, maps, source_spectra, scene_pixel, settings
):
    """Save the original operator, image and channel archives with unchanged field names."""
    operator_arrays, image_arrays, channel_receipts = {}, {}, {}
    for number, (family, layout) in enumerate(layouts.items()):
        value = results[family]
        kernel, weights, effective = value.kernel, value.weights, value.effective
        native, channels = value.native, value.channels
        dn, digitized = value.dn, value.digitized
        flat_channels = settings.flat_radiance * effective.sum(axis=1)
        key = f"family_{number}"
        operator_arrays.update(
            {
                key + "_K": kernel,
                key + "_H": kernel.sum(axis=(1, 2)),
                key + "_W": weights,
                key + "_effective": effective,
                key + "_exposure_bounds_s": windows[family].exposure_bounds_s,
            }
        )
        image_arrays.update(
            {
                key + "_native_electrons": native,
                key + "_native_dn": dn,
                key + "_channels": channels,
                key + "_digitized_channels": digitized,
            }
        )
        channel_receipts[family] = dict(
            channel_nm=layout.channel_nm.tolist(),
            ideal_centre=channels[:, 49, 49].tolist(),
            digitized_centre=digitized[:, 49, 49].tolist(),
            uniform_scene_at_centre=(effective @ scene_pixel).tolist(),
            full_flat_source_channels=flat_channels.tolist(),
            native_centre_electrons=native[:, 49, 49].tolist(),
            native_centre_dn=dn[:, 49, 49].tolist(),
        )
    operator_arrays["wavenumber_cm1"] = points
    operator_arrays["family_names"] = np.array(tuple(layouts))
    image_arrays.update(maps=maps, source_spectra=source_spectra)
    np.savez_compressed(OUT / "operators.npz", **operator_arrays)
    np.savez_compressed(OUT / "scene-images.npz", **image_arrays)
    (OUT / "channels.json").write_text(json.dumps(channel_receipts, indent=2) + "\n")


def show_scene(points, maps, source_spectra, layouts, results):
    """Display the heterogeneous scene and each reconstructed channel image."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    wave_index = int(np.argmin(abs(1e7 / points - 1650)))
    truth_image = np.einsum("cij,c->ij", maps[:, 2:-2, 2:-2], source_spectra[:, wave_index])
    panels = [
        (truth_image, "Input scene at 1650 nm"),
        (maps[1, 2:-2, 2:-2], "Synthetic absorption-template amplitude"),
    ]
    for family, value in results.items():
        index = (len(layouts[family].channel_nm) - 1) // 2
        panels.append(
            (
                value.channels[index],
                f"{family}\nChannel {index + 1}: {layouts[family].channel_nm[index]:.3f} nm",
            )
        )
    for ax, (image, title) in zip(axes.flat, panels, strict=True):
        color = ax.imshow(
            image, origin="lower", extent=(-2500, 2500, -2500, 2500), cmap="viridis", aspect="equal"
        )
        ax.plot(-25, -25, marker="+", color="white", ms=10, mew=1.5)
        ax.set(title=title, xlabel="Cross-track [m]", ylabel="Along-track [m]")
        fig.colorbar(color, ax=ax, shrink=0.75, format="%.2g")
    fig.suptitle(
        "100 × 100 scene and reconstructed channel images | white cross: shared scene pixel"
    )
    plt.show()
    plt.close(fig)


def show_effective_responses(points, cm_weights, settings, layouts, results):
    """Display every effective channel response including signed sidelobes."""
    for family, value in results.items():
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(14, 5), layout="constrained", width_ratios=(1, 1.3)
        )
        left.plot([1635, 1665], [settings.flat_radiance] * 2, color="#a35a24")
        left.axvspan(1645, 1655, color="#d9ad62", alpha=0.25, label="Ideal entrance acceptance")
        left.set(
            title="Spatially uniform constant source",
            xlabel="Source wavelength [nm]",
            ylabel="Radiance [W m⁻² sr⁻¹ per cm⁻¹]",
            xlim=(1635, 1665),
            ylim=(0, 0.0012),
        )
        left.legend(fontsize=10)
        channels = len(value.effective)
        colours = plt.get_cmap("tab20")(np.linspace(0, 1, channels))
        for channel in range(channels):
            response = value.effective[channel] / cm_weights
            wave = np.r_[1635, 1645 - 1e-6, 1e7 / points[::-1], 1655 + 1e-6, 1665]
            density = np.r_[0, 0, response[::-1], 0, 0]
            right.plot(
                wave,
                density,
                color=colours[channel],
                label=f"Ch {channel + 1}: {layouts[family].channel_nm[channel]:.3f} nm",
            )
        right.axhline(0, color="#555555", lw=0.7)
        interpretation = "Signed" if np.any(value.effective < 0) else "Positive"
        right.set(
            title=f"{interpretation} effective channel responses",
            xlabel="Diagnostic source wavelength [nm]",
            ylabel="Response density [(cm⁻¹)⁻¹]",
            xlim=(1635, 1665),
        )
        right.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        for ax in (left, right):
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        fig.suptitle(f"SWIR | {family} | flat source and effective spectral response")
        plt.show()
        plt.close(fig)


def show_atmospheric_channels(points, scene_pixel, layouts, results):
    """Display the common input spectrum and reconstructed scene-pixel channels."""
    for family, value in results.items():
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        left.plot(1e7 / points[::-1], scene_pixel[::-1], color="#a35a24", lw=1.4)
        left.set(
            title="Full-band incident scene-pixel radiance",
            xlabel="Source wavelength [nm]",
            ylabel="Radiance [W m⁻² sr⁻¹ per cm⁻¹]",
            xlim=(1645, 1655),
            ylim=(0, None),
        )
        labels = layouts[family].channel_nm
        right.plot(
            labels,
            value.channels[:, 49, 49],
            "o-",
            label="Reconstructed mean charge",
            color="#245a81",
        )
        right.errorbar(
            labels,
            value.digitized[:, 49, 49],
            yerr=value.detector["centre_noise_sigma"],
            fmt="x",
            capsize=3,
            label="After native DN transfer ±1σ",
            color="#9c482c",
        )
        right.set(
            title="Same reconstructed ground pixel (49, 49)",
            xlabel="Nominal channel centre [nm]",
            ylabel="Channel radiance [W m⁻² sr⁻¹ per cm⁻¹]",
            xlim=(1645, 1655),
        )
        right.legend(fontsize=10)
        for ax in (left, right):
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        fig.suptitle(f"SWIR | {family} | full spectral acquisition")
        plt.show()
        plt.close(fig)


def check_acquisition(
    radiance,
    points,
    maps,
    source_spectra,
    scene_pixel,
    cm_weights,
    settings,
    layouts,
    results,
    *,
    kernel_probe,
    optics_for_read,
    scan_geometry,
):
    """Run bounded refinements and an independent public field/acquisition check."""
    checks = {}
    for family, layout in layouts.items():
        value = results[family]
        indices = np.array(sorted({0, layout.read_count // 2, layout.read_count - 1}))
        probe = replace(
            layout,
            stage=layout.stage[indices],
            detector_x_m=layout.detector_x_m[indices],
            target_offsets_at_exposure_centre_m=layout.target_offsets_at_exposure_centre_m[indices],
            ports=tuple(layout.ports[index] for index in indices),
        )
        base = value.kernel[indices]
        angular_fine = kernel_probe(probe, radiance, angular_order=14, time_order=5)
        pixel_fine = kernel_probe(probe, radiance, pixel_order=16)
        # Bounded independent direct-acquisition checks use 41 source nodes across the full band.
        sparse_points = points[::20]
        sparse = Radiance(
            wavenumber_grid(sparse_points),
            np.interp(sparse_points, points, radiance.values),
            radiance.geometry,
        )
        sparse_kernel = kernel_probe(probe, sparse)
        centre_tile = np.einsum("cij,ck->ijk", maps[:, 49:54, 49:54], source_spectra[:, ::20])
        errors = []
        for read in range(len(indices)):
            direct = _direct_scene_read(
                probe, read, sparse, settings, centre_tile, optics_for_read, scan_geometry
            )[0]
            errors.append(float(abs(np.sum(sparse_kernel[read] * centre_tile) - direct) / direct))
        fine_points = np.linspace(points[0], points[-1], 1601)
        fine_radiance = Radiance(
            wavenumber_grid(fine_points),
            np.interp(fine_points, points, radiance.values),
            radiance.geometry,
        )
        fine_kernel = kernel_probe(probe, fine_radiance)
        angular_change = relative(base, angular_fine)
        pixel_change = relative(base, pixel_fine)
        spectral_change = relative(
            np.einsum("rijk,k->r", base, radiance.values),
            np.einsum("rijk,k->r", fine_kernel, fine_radiance.values),
        )
        edge_error = 0.0
        for row, column in ((0, 0), (0, 99), (99, 0), (99, 99)):
            tile = np.einsum(
                "cij,ck->ijk", maps[:, row : row + 5, column : column + 5], source_spectra
            )
            expected = np.einsum("rijk,ijk->r", value.kernel, tile)
            edge_error = max(edge_error, relative(value.native[:, row, column], expected))
        checks[family] = dict(
            direct_relative_error=max(errors),
            angular_refinement=angular_change,
            pixel_refinement=pixel_change,
            spectral_refinement=spectral_change,
            edge_stencil_error=edge_error,
            representative_native_reads=indices.tolist(),
        )
        assert max(max(errors), angular_change, pixel_change, spectral_change) < 0.003
        assert edge_error < 1e-12
        np.testing.assert_allclose(value.effective.sum(axis=1), 1, atol=1e-12)

    interferometer = results["Two-beam interferometer"]
    wave = 1e7 / points
    bin_number = np.minimum(((wave - 1645) / 1.25).astype(int), 7)
    ideal_bins = np.array(
        [
            np.average(scene_pixel[bin_number == index], weights=cm_weights[bin_number == index])
            for index in range(8)
        ]
    )
    projected_bins = interferometer.effective @ scene_pixel
    interferometer.diagnostics["out_of_basis_relative_max"] = relative(projected_bins, ideal_bins)

    return checks


def _direct_scene_read(
    layout, read, radiance, settings, scene_values, optics_for_read, scan_geometry
):
    """Check the composed stencil against public time-dependent field integration.

    A bilinear 5×5 radiance field supplies exactly the same scene samples to
    expected_electrons; this path does not apply the response kernel under test.
    """
    offsets = np.arange(-2, 3) * settings.sampling_m
    trajectory = LinearTrajectory(height_m=settings.height_m, speed_m_s=settings.speed_m_s)
    reads, pixels = scan_geometry(layout, read, settings)
    dispersive = layout.family == "Dispersive imager"
    acquisition = along_track_acquisition(
        reads,
        trajectory,
        target_xy_m=np.zeros((1, 2)),
        pixel_bounds_m=pixels,
        along_edges_m=np.array([-25.0, 0.0, 25.0]) if dispersive else offsets,
        cross_edges_m=offsets,
        support_frame="platform" if dispersive else "ground",
        time_order=settings.time_order,
        angular_order=settings.angular_order,
        cross_order=settings.angular_order,
        focal_length_m=settings.focal_length_m,
        pointing_law=lambda read_index, times_s: np.zeros((len(times_s), 2)),
    )
    height, speed = settings.height_m, settings.speed_m_s
    time_grid = np.unique(np.r_[acquisition.exposure_bounds_s[0], acquisition.times_s[0]])
    u = np.unique(((offsets[:, None] - speed * time_grid[None, :]) / height).ravel())
    u = np.unique(np.r_[u, acquisition.direction_bounds_uv[0, :2]])
    v = offsets / height
    values = np.empty((len(time_grid), len(u), len(v), len(radiance.grid.points)))
    for index, time in enumerate(time_grid):
        x = u * height + speed * time
        for cross in range(5):
            for spectral in range(len(radiance.grid.points)):
                values[index, :, cross, spectral] = np.interp(
                    x, offsets, scene_values[:, cross, spectral]
                )
    points = radiance.grid.points
    grid = SpectralGrid(
        "wavenumber_cm1",
        points,
        SpectralDomain("wavenumber_cm1", ((points[0], points[-1]),)),
    )
    optics = optics_for_read(layout.family, layout, read)
    return expected_electrons(
        RadianceField(grid, time_grid, u, v, values),
        acquisition,
        lambda current_read, wave, current: optics.response(0, current_read, wave, current),
        collecting_area_m2=optics.collecting_area_m2,
        throughput=optics.throughput,
        quantum_efficiency=settings.qe,
    ).total


def relative(a, b):
    """Return the original maximum norm-relative discrepancy."""
    return float(np.max(abs(a - b)) / max(float(np.max(abs(b))), 1e-30))


def save_evidence(
    ROOT,
    OUT,
    *,
    scenario_path,
    runtime,
    settings,
    layouts,
    windows,
    results,
    checks,
    input_hashes,
    preparation_log,
    compute_wall_seconds,
):
    """Save source/input lineage, original model scope and numerical checks."""
    source_paths = [
        "notebooks/support/notebook_ground_pixel.py",
        "notebooks/support/notebook_instrument_responses.py",
        "notebooks/support/notebook_five_band_support.py",
        "notebooks/support/notebook_instrument_support.py",
        "notebooks/support/five_band_gases.py",
        "notebooks/support/notebook_sasktran2_cache.py",
        "notebooks/support/notebook_sasktran2_layers.py",
        "notebooks/era5_hitran_sasktran2_five_bands.ipynb",
    ]
    family_evidence = {}
    for family, layout in layouts.items():
        timing = windows[family]
        window = dict(
            first_exposure_s=timing.first_exposure_s,
            last_exposure_s=timing.last_exposure_s,
            window_s=timing.window_s,
            per_pixel_span_s=timing.per_pixel_span_s,
            target_rows=timing.target_rows,
            complete_edge_channels=timing.complete_edge_channels,
        )
        family_evidence[family] = dict(
            active_pixels=layout.active_pixels,
            read_count=layout.read_count,
            channel_nm=layout.channel_nm.tolist(),
            timing=window,
            detector=results[family].detector,
            reconstruction=results[family].diagnostics,
        )
    evidence = dict(
        comparison="reconstructed ground-image pixel",
        scenario_sha256=hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        source_hashes={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths
        },
        runtime=runtime,
        settings=asdict(settings),
        families=family_evidence,
        checks=checks,
        atmospheric_points=201,
        instrument_spectral_nodes=801,
        input_hashes=input_hashes,
        preparation_log=preparation_log,
        atmosphere_provenance=(
            "ERA5 0N 0E 2024-01-15T12:00Z, illustrative inherited gas profiles; "
            "nominal nadir and Toulouse solar elevation"
        ),
        spatial_model=(
            "bilinear 50m scene, Gaussian sigma10m, 25m exposure smear, "
            "100m explicit halo, paraxial radial local plane"
        ),
        compute_wall_seconds=compute_wall_seconds,
    )
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def show_checks(checks, diagnostics):
    """Display refinement checks and finite-basis reconstruction diagnostics."""
    rows = [
        "| Family | Direct/public relative error | Angular/time refinement | "
        "Pixel refinement | Spectral refinement | Edge stencil error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, check in checks.items():
        rows.append(
            "| "
            + family
            + " | "
            + " | ".join(
                f"{check[key]:.3g}"
                for key in (
                    "direct_relative_error",
                    "angular_refinement",
                    "pixel_refinement",
                    "spectral_refinement",
                    "edge_stencil_error",
                )
            )
            + " |"
        )
    display(Markdown("\n".join(rows)))
    diag = diagnostics
    display(
        Markdown(
            f"Interferometric eight-bin rank **{diag['basis_rank']}**, "
            f"condition **{diag['condition_number']:.3g}**, "
            f"basis closure **{diag['basis_closure_max']:.3g}**. "
            "The supplied high-resolution scene spectrum differs from its ideal bin averages by "
            f"**{100 * diag['out_of_basis_relative_max']:.3f}%** after W @ H. "
            "The effective kernels describe this finite-basis sensitivity. "
            "Local numerical checks use the declared **0.3%** integration-sensitivity threshold "
            "and representative samples; the atmospheric 201-point interpolation remains "
            "the spectral source model."
        )
    )


def show_acquisition_timing(windows):
    """Display complete-image bounds and every read of the shared scene pixel."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for index, (family, window) in enumerate(windows.items()):
        left.barh(index, window.window_s, left=window.first_exposure_s, height=0.65)
        starts = windows[family].exposure_bounds_s[49, :, 0]
        right.scatter(starts, np.full(len(starts), index), s=15)
    left.set(
        yticks=np.arange(len(windows)),
        yticklabels=tuple(windows),
        xlabel="Time relative to centre passage [s]",
        title="Complete-image acquisition bounds",
    )
    right.set(
        yticks=np.arange(len(windows)),
        yticklabels=[],
        xlabel="Exposure start relative to passage [s]",
        title="Same scene pixel: every physical read",
    )
    for ax in (left, right):
        ax.axvline(0, color="#444444", lw=0.7)
    fig.suptitle(
        "Spectral edge coverage determines acquisition time | 0.702014 s central ground traversal"
    )
    plt.show()
    plt.close(fig)
