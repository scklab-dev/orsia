"""Present detector-owned notebook measurements and products without running physics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display


def show_scene(points, maps, spectra, scene_axis):
    """Show the complete source image, executed patch and representative spectra.

    Args:
        points: Increasing wavenumbers in cm^-1.
        maps: Dimensionless (factor,along,cross) scene factors.
        spectra: (factor,wavenumber) radiance in W m^-2 sr^-1 per cm^-1.
        scene_axis: Shared along/cross factor-map coordinates in metres.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    image = np.einsum("fij,f->ij", maps, spectra[:, np.argmin(abs(1e7 / points - 1650))])
    extent = (-2600, 2600, -2600, 2600)
    for ax, limits in zip(axes[:2], ((-2500, 2500), (-200, 200)), strict=True):
        artist = ax.imshow(image, origin="lower", extent=extent, cmap="viridis")
        ax.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50], color="white")
        ax.plot(-25, -25, "+", color="red")
        ax.set(xlim=limits, ylim=limits, xlabel="Cross-track [m]", ylabel="Along-track [m]")
        fig.colorbar(artist, ax=ax, label="Radiance at 1650 nm", shrink=0.8)
    axes[0].set_title("Shared 5 km scene; white executed ROI")
    axes[1].set_title("ROI detail; red shared target")
    for along, cross in ((-25, -25), (425, -675), (-1025, 975)):
        i, j = np.argmin(abs(scene_axis - along)), np.argmin(abs(scene_axis - cross))
        axes[2].plot(1e7 / points, maps[:, i, j] @ spectra, label=f"({along}, {cross}) m")
    axes[2].set(
        xlabel="Wavelength [nm]", ylabel="Radiance per cm^-1", title="Same scene, different spectra"
    )
    axes[2].legend(fontsize=8)
    plt.show()
    plt.close(fig)


def show_native(label, frames):
    """Display selected native reads as frame by physical-pixel panels.

    Args:
        label: Family or physical-port display name.
        frames: FrameMeasurements with expected electrons and at least one DN realization.
    """
    schedule = frames.images.schedule
    exposure, row, column = schedule.native_indices.T
    pixels = row * schedule.native_shape[2] + column
    unique = np.unique(pixels)
    frame_ids = np.unique(exposure)
    image = np.full((len(unique), len(frame_ids)), np.nan)
    ix, iy = np.searchsorted(frame_ids, exposure), np.searchsorted(unique, pixels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, values, title in zip(
        axes,
        (frames.images.photoelectrons.ravel(), frames.images.dn[0].ravel()),
        ("Expected photoelectrons / exposure", "Sampled noisy DN"),
        strict=True,
    ):
        shown = image.copy()
        shown[iy, ix] = values
        artist = ax.imshow(shown, origin="lower", aspect="auto", interpolation="nearest")
        ax.set(
            title=title,
            xlabel="Selected exposure index (chronological)",
            ylabel="Selected physical pixel index",
        )
        fig.colorbar(artist, ax=ax, shrink=0.85)
    fig.suptitle(f"{label}: native reads; white cells were not selected")
    plt.show()
    plt.close(fig)


def show_product(label, cube, expectation_cube, reference, points, target=(-25.0, -25.0)):
    """Show native-band radiance, coverage, target references and defining responses.

    Args:
        label: Instrument family name.
        cube: Noisy registered BandRadianceCube on the notebook's 50 m grid.
        expectation_cube: Separately labelled noiseless cube on the same grid.
        reference: Achieved-band point references with the cube's (y,x,band) shape.
        points: Band integration wavenumbers in cm^-1.
        target: Along/cross coordinates in metres of the shared target.
    """
    iy = int(np.argmin(abs(cube.y_m - target[1])))
    ix = int(np.argmin(abs(cube.x_m - target[0])))
    bands = cube.bands
    middle = len(bands) // 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    extent = (cube.x_m[0] - 25, cube.x_m[-1] + 25, cube.y_m[0] - 25, cube.y_m[-1] + 25)
    for ax, data, title in (
        (axes[0, 0], cube.values[..., middle], f"Noisy radiance: achieved band {middle}"),
        (axes[0, 1], cube.coverage[..., middle], "Accumulated spatial weight"),
    ):
        artist = ax.imshow(data, origin="lower", extent=extent, aspect="equal")
        ax.plot(*target, "+", color="red")
        ax.set(title=title, xlabel="Along-track [m]", ylabel="Cross-track [m]")
        fig.colorbar(artist, ax=ax, shrink=0.85)
    index = np.arange(len(bands))
    axes[1, 0].plot(index, reference[iy, ix], "o-", label="Point source under achieved band")
    axes[1, 0].plot(
        index, expectation_cube.values[iy, ix], "s--", label="Noiseless registered diagnostic"
    )
    axes[1, 0].plot(index, cube.values[iy, ix], ".-", label="Noisy DN reconstruction")
    axes[1, 0].set(
        xlabel="Achieved band index",
        ylabel="Radiance per cm^-1",
        title="Shared target (-25, -25) m",
    )
    axes[1, 0].legend(fontsize=8)
    if len(bands) == 1:
        axes[1, 0].set_xticks([0])
        axes[1, 0].set_xlim(-0.5, 0.5)
    weights = np.r_[np.diff(points)[0], points[2:] - points[:-2], np.diff(points)[-1]] / 2
    for i, band in enumerate(bands):
        axes[1, 1].plot(1e7 / points, band.weights / weights, label=str(i))
    axes[1, 1].set(
        xlabel="Source wavelength [nm]",
        ylabel="Response density [(cm^-1)^-1]",
        title="All achieved photon-weighted bands",
    )
    axes[1, 1].legend(ncol=4, fontsize=7, title="Band")
    fig.suptitle(f"{label}: registered (y, x, band) product; radiance in W m^-2 sr^-1 per cm^-1")
    plt.show()
    plt.close(fig)


def show_inverse(response, estimate, expectation_estimate, reference, physical_response):
    """Show the bounded inverse, physical coverage and signed state resolution.

    Args:
        response: Joint operator with actual spatial/spectral cell edges.
        estimate: Noisy LinearEstimate including estimator covariance and resolution.
        expectation_estimate: Noiseless estimate using the same model.
        reference: Flattened (y,x,spectral_bin) cell/bin mean source radiance.
        physical_response: Ordered (port label, NativeBandResponse) pairs.
    """
    shape = response.state_shape
    state = estimate.state.reshape(shape)
    iy = np.searchsorted(response.y_edges_m, -25.0, side="right") - 1
    ix = np.searchsorted(response.x_edges_m, -25.0, side="right") - 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for ax, data, title in (
        (axes[0, 0], state[..., 4], "Minimum-norm noisy estimate: bin 4"),
        (
            axes[0, 1],
            response.coverage.reshape(shape).sum(axis=-1),
            "Physical sensitivity, summed bins",
        ),
        (
            axes[0, 2],
            estimate.identifiable.reshape(shape).sum(axis=-1),
            "Identifiable coordinates / 8",
        ),
    ):
        limits = {"vmin": 0, "vmax": shape[-1]} if ax is axes[0, 2] else {}
        art = ax.pcolormesh(response.x_edges_m, response.y_edges_m, data, shading="flat", **limits)
        ax.set_aspect("equal")
        ax.plot(-25, -25, "+", color="red")
        ax.set(title=title, xlabel="Along-track [m]", ylabel="Cross-track [m]")
        fig.colorbar(art, ax=ax, shrink=0.75)
    bins = np.arange(shape[-1])
    axes[1, 0].plot(bins, reference.reshape(shape)[iy, ix], "o-", label="Cell/bin mean reference")
    axes[1, 0].plot(
        bins,
        expectation_estimate.state.reshape(shape)[iy, ix],
        "s--",
        label="Noiseless model diagnostic",
    )
    axes[1, 0].errorbar(
        bins,
        state[iy, ix],
        yerr=np.sqrt(np.maximum(0, np.diag(estimate.covariance).reshape(shape)[iy, ix])),
        fmt=".-",
        label="Noisy estimate +/- noise sigma",
    )
    axes[1, 0].set(
        xlabel="Spectral bin (increasing cm^-1)",
        ylabel="Radiance per cm^-1",
        title="Target cell; null-space uncertainty unbounded",
    )
    axes[1, 0].legend(fontsize=7)
    limit = max(abs(estimate.resolution.min()), abs(estimate.resolution.max()))
    art = axes[1, 1].imshow(
        estimate.resolution, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )
    axes[1, 1].set(
        title=f"Signed resolution, rank {estimate.rank}/{len(estimate.state)}",
        xlabel="True state coordinate",
        ylabel="Estimated coordinate",
    )
    fig.colorbar(art, ax=axes[1, 1], shrink=0.75)
    for _port, native in physical_response:
        # Eight native reads illustrate each physical port's positive spectral weights.
        coefficients = native.spectral_coefficients
        for r in np.linspace(0, len(coefficients) - 1, 8, dtype=int):
            axes[1, 2].plot(
                1e7 / native.grid.points, coefficients[r] / coefficients[r].sum(), alpha=0.5
            )
    axes[1, 2].set(
        title="Physical port weights (8 reads per port)",
        xlabel="Source wavelength [nm]",
        ylabel="Normalized sampled photon weights",
    )
    status = "rank-deficient" if estimate.rank < len(estimate.state) else "full-rank"
    fig.suptitle(f"Two-beam: {status} discretized estimate; inverse quadrature unresolved")
    plt.show()
    plt.close(fig)


def show_summary(rows):
    """Display completed-chain counts, valid fractions and wall seconds.

    Args:
        rows: Family-name mapping with reads, components, valid_fraction and wall_s fields.
    """
    headings = ("Family", "Reads", "Bands / states", "Valid fraction", "Wall seconds")
    lines = ["| " + " | ".join(headings) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['reads']} | {row['components']} | "
            f"{row['valid_fraction']:.3f} | {row['wall_s']:.2f} |"
        )
    display(Markdown("\n".join(lines)))


def show_target_timing(schedules, nadir_time_s):
    """Plot supplied shared-target read times, measured relative to nadir passage.

    Args:
        schedules: Mapping from family name to selected (read,2) exposure bounds in seconds.
        nadir_time_s: Shared target's nominal nadir-passage time, seconds from epoch.
    """
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for row, bounds in enumerate(schedules.values()):
        relative_ms = 1000 * (bounds - nadir_time_s)
        centres = np.unique(relative_ms.mean(axis=1))
        ax.hlines(row, relative_ms.min(), relative_ms.max(), color="0.6")
        ax.plot(centres, np.full_like(centres, row), "o", ms=4)
    ax.set_yticks(range(len(schedules)), list(schedules))
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", linestyle=":")
    ax.set(
        xlabel="Time from shared target at nadir [ms]",
        title="Actual selected reads at (-25, -25) m; two ports simultaneous",
    )
    plt.show()
    plt.close(fig)


def source_text(notebook: dict) -> str:
    """Extract complete cell IDs, types and sources without binary outputs.

    Args:
        notebook: Parsed notebook JSON containing ordered cells.

    Returns:
        Source-only UTF-8-compatible text with a heading for every cell.
    """
    return (
        "\n\n".join(
            f"### {i} {cell['id']} {cell['cell_type']}\n{''.join(cell['source'])}"
            for i, cell in enumerate(notebook["cells"])
        )
        + "\n"
    )


def save_source_text(path: Path, notebook: dict) -> None:
    """Write an inspection copy of all editable notebook sources.

    Args:
        path: Destination UTF-8 file, replaced on success.
        notebook: Parsed notebook JSON containing ordered cells.

    Raises:
        OSError: The destination cannot be written.
    """
    path.write_text(source_text(notebook))
