"""
Plotting script for RAPSD and timestep weighting metrics.
Generates publication-quality figures for:
1. Retained Signal (RSignal)
2. Delta Retained Signal (ΔRSignal)
3. CSF-weighted Delta Retained Signal (CSF * ΔRSignal)
4. RAPSD
5. CSFlow importance weights

"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.frequencies_utils import FrequencyConverter, csf_barten
from src.metrics_calculator import MetricsCalculator

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "mathtext.fontset": "cm",
    }
)


def _to_numpy(array):
    """Convert array-like or tensor-like inputs to a numpy array."""
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.array(array)


def load_rapsd_results(rapsd_cache_path):
    with open(rapsd_cache_path, "rb") as f:
        mean_rapsd_data, frequencies, mean_rapsd_noise = pickle.load(f)
    return mean_rapsd_data, mean_rapsd_noise, frequencies


def load_training_weights(training_path):
    with open(training_path, "rb") as f:
        timestep_scalar_weights, timesteps = pickle.load(f)
    return timestep_scalar_weights, timesteps


def compute_metrics_from_rapsd(
    mean_rapsd_data, mean_rapsd_noise, frequencies, timesteps
):
    """Recompute the metric surfaces from RAPSD."""
    metrics_calc = MetricsCalculator(
        mean_rapsd_data=mean_rapsd_data,
        mean_rapsd_noise=mean_rapsd_noise,
        timesteps=timesteps,
        frequencies=frequencies,
    )

    freq_converter = FrequencyConverter(mode=0, frequencies=frequencies)
    frequencies_converted = freq_converter.frequencies_converted
    csf_vals = csf_barten(frequencies_converted)

    rsignal = _to_numpy(metrics_calc.retained_signal)
    delta_rsignal = _to_numpy(metrics_calc.delta_retained_signal)
    csf_delta_rsignal = delta_rsignal * csf_vals[None, :]

    return rsignal, delta_rsignal, csf_delta_rsignal, frequencies_converted, timesteps


def plot_timestep_weights(timestep_scalar_weights, timesteps, output_path):
    """Plot training weights as interval-steps and interpolated weight function."""
    timestep_scalar_weights = _to_numpy(timestep_scalar_weights)
    timesteps = _to_numpy(timesteps)

    # Ensure monotonic timesteps for stable plotting/interpolation.
    sort_idx = np.argsort(timesteps)
    timesteps = timesteps[sort_idx]
    timestep_scalar_weights = timestep_scalar_weights[sort_idx]

    # Interval plot: use true timestep boundaries so the domain starts at 0 and ends at 1.
    # If timesteps has N boundary points, interval weights are defined on N-1 intervals.
    if len(timesteps) < 2:
        raise ValueError("Need at least two timesteps to construct interval weights")

    interval_left = timesteps[:-1]
    interval_right = timesteps[1:]
    interval_weights = timestep_scalar_weights[:-1]

    # Normalize by interval width so the plot shows w / Δt
    delta_t = interval_right - interval_left
    normalized_weights = interval_weights / delta_t

    base_path = Path(output_path)
    interval_path = base_path

    step_x = np.append(interval_left, interval_right[-1])
    step_y = np.append(normalized_weights, normalized_weights[-1])

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.step(step_x, step_y, where="post", color="midnightblue", linewidth=2.8)
        ax.fill_between(step_x, step_y, step="post", color="cornflowerblue", alpha=0.2)
        ax.set_xlabel("Timestep (t)", fontsize=18)
        ax.set_ylabel("Weight", fontsize=18)
        ax.set_title(r"$w_{\mathrm{CSFlow}}$", fontsize=22)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_xlim(timesteps[0], timesteps[-1])
        fig.tight_layout()
        fig.savefig(interval_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {interval_path}")
        plt.close(fig)


def plot_heatmap(
    x,
    y,
    z,
    title,
    xlabel,
    ylabel,
    cbar_label,
    output_path,
    cmap="viridis",
    smooth_sigma=1.5,
    force_minmax_ticks=True,
):
    """Generic heatmap plotter."""
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Force the axes box (plot area) to be square, independent of data scales
    ax.set_box_aspect(1)

    # Smooth z before contouring for a cleaner appearance
    z_smooth = gaussian_filter(z, sigma=smooth_sigma)

    # Preserve original value range while smoothing spatial variation.
    vmin, vmax = z_smooth.min(), z_smooth.max()

    # contourf for smooth visualization
    # x: horizontal axis, y: vertical axis
    ax.contourf(x, y, z_smooth, levels=np.linspace(vmin, vmax, 125), cmap=cmap)

    # Force the axes box (plot area) to be square, independent of data scales
    ax.set_box_aspect(1)

    # Smooth z before contouring for a cleaner appearance
    z_smooth = gaussian_filter(z, sigma=smooth_sigma)

    # Preserve original value range while smoothing spatial variation.
    vmin, vmax = z_smooth.min(), z_smooth.max()

    # contourf for smooth visualization
    # x: horizontal axis, y: vertical axis
    ax.contourf(x, y, z_smooth, levels=np.linspace(vmin, vmax, 125), cmap=cmap)

    # Use a ScalarMappable so the colorbar is a continuous gradient, not banded
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78)
    cbar.set_label(cbar_label, rotation=270, labelpad=28, fontsize=21)
    cbar.ax.tick_params(labelsize=17)
    # Keep a compact, readable colorbar scale and round to 2 decimals.
    if force_minmax_ticks:
        ticks = np.linspace(vmin, vmax, 6)
    else:
        ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    ax.set_xlabel(xlabel, fontsize=22, labelpad=14)
    ax.set_ylabel(ylabel, fontsize=22, labelpad=14)
    ax.set_title(title, fontsize=30, pad=16)
    ax.tick_params(axis="both", labelsize=17)

    # Optional: Log scale for frequency if needed, but usually linear is fine for these plots
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _rapsd_axes_style(ax):
    """Apply shared clean paper style to a RAPSD axes."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mean_rapsd_loglog(mean_rapsd_data, mean_rapsd_noise, frequencies, output_path):
    """Plot mean RAPSD in loglog scale for both axes."""

    # Avoid log(0) if frequency starts at 0
    start_idx = 1 if frequencies[0] == 0 else 0
    f = frequencies[start_idx:]
    y_data = mean_rapsd_data[start_idx:]
    y_noise = mean_rapsd_noise[start_idx:]

    # Drop non-positive values so log scaling is always valid.
    valid = (f > 0) & (y_data > 0) & (y_noise > 0)
    f = f[valid]
    y_data = y_data[valid]
    y_noise = y_noise[valid]

    base = Path(output_path)
    variants = [
        # (path suffix,  xscale,  yscale)
        ("", "log", "log"),
    ]

    for suffix, xscale, yscale in variants:
        out = base.with_name(f"{base.stem}{suffix}{base.suffix}")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        fig.patch.set_facecolor("white")

        ax.plot(f, y_data, color="#1f4e79", linewidth=3.5, label="ImageNet")
        ax.plot(f, y_noise, color="#b03a2e", linewidth=3.5, label="Gaussian noise")

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel("Frequency (low to high)", fontsize=14)
        ax.set_ylabel("Power", fontsize=14)
        ax.set_title("Average frequency power", fontsize=16, pad=8)
        ax.legend(loc="best", frameon=False, fontsize=11)

        _rapsd_axes_style(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
        plt.close(fig)


def main(config):
    figures_dir = os.path.join(
        config.figures.directory,
        f"{config.experiment.name}_{config.image.resolution}_weights_steps{config.weights_calculation.num_steps}_plots_steps{config.figures.plot_num_steps}",
    )
    os.makedirs(figures_dir, exist_ok=True)

    results_dir = os.path.join(
        config.output.directory,
        f"{config.experiment.name}_{config.image.resolution}_steps{config.weights_calculation.num_steps}",
    )

    print(f"Loading RAPSD results from {config.rapsd.cache}...")

    # Load data
    mean_rapsd_data, mean_rapsd_noise, frequencies = load_rapsd_results(
        config.rapsd.cache
    )

    print(f"Loading results from {results_dir}...")

    timestep_scalar_weights, timesteps = load_training_weights(
        os.path.join(results_dir, "training_weights.pkl")
    )

    valid_mask = frequencies <= 0.5
    frequencies = frequencies[valid_mask]
    mean_rapsd_data = mean_rapsd_data[valid_mask]
    mean_rapsd_noise = mean_rapsd_noise[valid_mask]

    timesteps = _to_numpy(timesteps)
    timesteps_plots = np.linspace(
        0.0, 1 - (1 / config.figures.plot_num_steps), config.figures.plot_num_steps
    )
    timesteps_plots = np.concatenate(
        [
            timesteps_plots,
            np.array([1.0]),
        ],
        axis=0,
    )

    rsignal, delta_rsignal, csf_delta_rsignal, _, _ = compute_metrics_from_rapsd(
        mean_rapsd_data=mean_rapsd_data,
        mean_rapsd_noise=mean_rapsd_noise,
        frequencies=frequencies,
        timesteps=timesteps_plots,
    )

    # Plot Mean RAPSD (Log-Log)
    plot_mean_rapsd_loglog(
        mean_rapsd_data,
        mean_rapsd_noise,
        frequencies,
        os.path.join(figures_dir, "mean_rapsd_loglog.png"),
    )

    rsignal_numpy = _to_numpy(rsignal)
    delta_rsignal_numpy = _to_numpy(delta_rsignal)
    csf_delta_rsignal_numpy = _to_numpy(csf_delta_rsignal)
    timesteps_plots_delta = timesteps_plots[1:]  # Adjust timesteps for delta metrics

    # normalized to [0, 1] for plotting
    csf_delta_rsignal_plot = (
        csf_delta_rsignal_numpy / np.abs(csf_delta_rsignal_numpy).max()
    )

    frequencies = _to_numpy(frequencies)
    timesteps_plots = _to_numpy(timesteps_plots)
    timesteps_plots_delta = _to_numpy(timesteps_plots_delta)

    print("Generating plots...")

    plot_timestep_weights(
        timestep_scalar_weights,
        timesteps,
        os.path.join(figures_dir, "timestep_weights.png"),
    )

    # RSignal Heatmap (Timesteps on X-axis)
    plot_heatmap(
        x=timesteps_plots,
        y=frequencies,
        z=rsignal_numpy.T,
        title=r"$r_{\mathrm{signal}}$",
        xlabel="Timestep (t)",
        ylabel="Frequency",
        cbar_label=r"$r_{\mathrm{signal}}$",
        output_path=os.path.join(figures_dir, "RS_heatmap.png"),
    )

    # delta RSignal Heatmap (Timesteps on X-axis)
    plot_heatmap(
        x=timesteps_plots_delta,
        y=frequencies,
        z=delta_rsignal_numpy.T,
        title=r"$\partial_t\, r_{\mathrm{signal}}$",
        xlabel="Timestep (t)",
        ylabel="Frequency",
        cbar_label=r"$\partial_t\, r_{\mathrm{signal}}$",
        output_path=os.path.join(figures_dir, "DeltaRS_heatmap.png"),
        cmap="RdYlBu_r",
        force_minmax_ticks=False,
    )

    # CSF * delta RSignal Heatmap (Timesteps on X-axis)
    plot_heatmap(
        x=timesteps_plots_delta,
        y=frequencies,
        z=csf_delta_rsignal_plot.T,
        title=r"CSF $\cdot\, \partial_t\, r_{\mathrm{signal}}$",
        xlabel="Timestep (t)",
        ylabel="Frequency",
        cbar_label=r"CSF $\cdot\, \partial_t\, r_{\mathrm{signal}}$",
        output_path=os.path.join(figures_dir, "CSF_DeltaRS_heatmap.png"),
        cmap="RdYlBu_r",
        force_minmax_ticks=False,
    )

    print(f"Done. Plots saved to {figures_dir}")
