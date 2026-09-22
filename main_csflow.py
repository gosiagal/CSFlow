import argparse
import os
import pickle

import numpy as np
import torch

from src.config import load_config
from src.data_utils_blip3o import DataLoaderBLIP3o
from src.data_utils_imagenet import DataLoaderImagenet
from src.frequencies_utils import FrequencyConverter, csf_barten
from src.metrics_calculator import MetricsCalculator
from src.rapsd_calculator_gpu import RapsdCalculatorGPU


def run_rapsd_gpu(config, rapsd_cache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load RAPSD for the configured dataset."""

    if config.rapsd.use_precomputed and os.path.exists(rapsd_cache):
        print(f"Loading cached RAPSD from {rapsd_cache}")

        with open(rapsd_cache, "rb") as f:
            mean_rapsd_data, frequencies, mean_rapsd_noise = pickle.load(f)

        return (
            mean_rapsd_data,
            mean_rapsd_noise,
            frequencies,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Computing RAPSD on device: {device}")

    calculator = RapsdCalculatorGPU(
        device=device,
        image_size=config.image.resolution,
    )

    if config.dataset.type.lower() == "blip3o":

        print("Using BLIP3o dataloader for tar files...")

        dataloader = DataLoaderBLIP3o(
            data_dir=config.dataset.path,
            resolution=config.image.resolution,
            batch_size=config.weights_calculation.batch_size,
            max_images=config.dataset.max_images,
        )

    elif config.dataset.type.lower() == "imagenet":

        print("Using ImageNet dataloader...")

        dataloader = DataLoaderImagenet(
            data_dir=config.dataset.path,
            resolution=config.image.resolution,
            batch_size=config.weights_calculation.batch_size,
            max_images=config.dataset.max_images,
        )

    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset.type}")

    print("Computing RAPSD for real data...")

    mean_rapsd_data, frequencies = calculator.compute_dataset_mean_rapsd(dataloader)

    print("Computing RAPSD for noise...")

    mean_rapsd_noise = calculator.compute_noise_rapsd(num_samples=dataloader.num_images)

    cache_dir = os.path.dirname(rapsd_cache)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(rapsd_cache, "wb") as f:
        pickle.dump(
            (
                mean_rapsd_data,
                frequencies,
                mean_rapsd_noise,
            ),
            f,
        )

    print(f"Saved RAPSD cache to {rapsd_cache}")

    return (
        mean_rapsd_data,
        mean_rapsd_noise,
        frequencies,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated weighted timesteps computation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def main(config):

    os.makedirs(
        config.output.directory,
        exist_ok=True,
    )

    # --------------------------------------------------
    # RAPSD cache
    # --------------------------------------------------

    if config.rapsd.cache is None:

        rapsd_cache_path = os.path.join(
            "rapsd_cache",
            f"rapsd_results_{config.experiment.name}_{config.image.resolution}.pkl",
        )

    else:
        rapsd_cache_path = config.rapsd.cache

    # --------------------------------------------------
    # Validate dataset
    # --------------------------------------------------

    if not os.path.exists(rapsd_cache_path) and config.dataset.path is None:
        raise ValueError(
            "RAPSD cache does not exist and " "dataset.path was not provided."
        )

    # --------------------------------------------------
    # Compute/load RAPSD
    # --------------------------------------------------

    mean_rapsd_data, mean_rapsd_noise, frequencies = run_rapsd_gpu(
        config=config,
        rapsd_cache=rapsd_cache_path,
    )

    # --------------------------------------------------
    # Diffusion timesteps
    # --------------------------------------------------

    timesteps = np.linspace(
        0.0,
        1 - (1 / config.weights_calculation.num_steps),
        config.weights_calculation.num_steps,
    )

    timesteps = np.concatenate(
        [
            timesteps,
            np.array([1.0]),
        ],
        axis=0,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metrics = MetricsCalculator(
        mean_rapsd_data=mean_rapsd_data,
        mean_rapsd_noise=mean_rapsd_noise,
        timesteps=timesteps,
        frequencies=frequencies,
    )

    # --------------------------------------------------
    # Frequency conversion and CSF calculation
    # --------------------------------------------------

    frequency_converter = FrequencyConverter(
        mode=config.frequencies.mode,
        pixel_size=config.frequencies.pixel_size,
        viewing_distance=config.frequencies.viewing_distance,
        frequencies=frequencies,
    )

    frequencies_converted = frequency_converter.frequencies_converted

    csf_vals = csf_barten(frequencies_converted)

    # --------------------------------------------------
    # Calculate timestep weights
    # --------------------------------------------------

    timestep_weighted = metrics.delta_retained_signal * csf_vals[None, :]

    timestep_scalar_weights = np.trapezoid(
        timestep_weighted,
        x=frequencies_converted,
        axis=1,
    )

    timestep_scalar_weights /= np.trapezoid(
        metrics.delta_retained_signal,
        x=frequencies_converted,
        axis=1,
    )

    timestep_scalar_weights /= np.sum(timestep_scalar_weights)

    if (
        np.any(~np.isfinite(timestep_scalar_weights))
        or timestep_scalar_weights.sum() <= 0
    ):
        raise ValueError(
            "Could not construct finite, positive " "timestep importance weights."
        )

    if config.weights_calculation.padded:
        timestep_scalar_weights_padded = np.append(
            timestep_scalar_weights,
            0.0,
        )
    else:
        timestep_scalar_weights_padded = timestep_scalar_weights

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    results_subdir = os.path.join(
        config.output.directory,
        f"{config.experiment.name}_{config.image.resolution}_steps{config.weights_calculation.num_steps}",
    )

    os.makedirs(
        results_subdir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save training weights
    # --------------------------------------------------

    training_weights_path = os.path.join(
        results_subdir,
        "training_weights.pkl",
    )

    with open(training_weights_path, "wb") as f:
        pickle.dump(
            (
                timestep_scalar_weights_padded,
                timesteps,
            ),
            f,
        )

    print(f"Saved training weights to " f"{training_weights_path}")

    print("\nOutput file:")
    print(f"  - {training_weights_path}")


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)

    print(f"Running experiment: " f"{config.experiment.name}")

    main(config)

    if config.figures.reproduce_figures:
        import reproduce_figures

        print(f"\nReproducing figures in " f"{config.figures.directory}")

        reproduce_figures.main(config=config)
