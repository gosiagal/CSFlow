from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str
    seed: int


@dataclass
class DatasetConfig:
    type: str
    path: str | None
    max_images: int


@dataclass
class ImageConfig:
    resolution: int


@dataclass
class WeightsConfig:
    batch_size: int
    num_steps: int
    padded: bool


@dataclass
class RapsdConfig:
    use_precomputed: bool
    cache: str | None


@dataclass
class FrequenciesConfig:
    mode: int
    pixel_size: float
    viewing_distance: float


@dataclass
class FiguresConfig:
    reproduce_figures: bool
    directory: str
    plot_num_steps: int


@dataclass
class OutputConfig:
    directory: str


@dataclass
class Config:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    image: ImageConfig
    weights_calculation: WeightsConfig
    rapsd: RapsdConfig
    frequencies: FrequenciesConfig
    figures: FiguresConfig
    output: OutputConfig


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Configuration file is empty: {path}")

    config = Config(
        experiment=ExperimentConfig(
            **data["experiment"]
        ),
        dataset=DatasetConfig(
            **data["dataset"]
        ),
        image=ImageConfig(
            **data["image"]
        ),
        weights_calculation=WeightsConfig(
            **data["weights_calculation"]
        ),
        rapsd=RapsdConfig(
            **data["rapsd"]
        ),
        frequencies=FrequenciesConfig(
            **data["frequencies"]
        ),
        figures=FiguresConfig(
            **data["figures"]
        ),
        output=OutputConfig(
            **data["output"]
        ),
    )

    validate_config(config)

    return config


def validate_config(config: Config) -> None:
    """Validate configuration values."""

    if config.dataset.type not in {"imagenet", "blip3o"}:
        raise ValueError(
            f"Unsupported dataset type: "
            f"{config.dataset.type}. "
            f"Expected 'imagenet' or 'blip3o'."
        )

    if config.image.resolution <= 0:
        raise ValueError(
            "image.resolution must be positive."
        )

    if config.weights_calculation.batch_size <= 0:
        raise ValueError(
            "weights_calculation.batch_size must be positive."
        )

    if config.weights_calculation.num_steps <= 0:
        raise ValueError(
            "weights_calculation.num_steps must be positive."
        )
    
    if config.weights_calculation.padded not in {True, False}:
        raise ValueError(
            "weights_calculation.padded must be a boolean."
        )
    
    if config.dataset.max_images < 0:
        raise ValueError(
            "dataset.max_images must be >= 0."
        )

    if config.frequencies.mode not in {0, 1}:
        raise ValueError(
            f"Unsupported frequencies mode: "
            f"{config.frequencies.mode}. "
            f"Expected 0 or 1."
        )

    if config.frequencies.pixel_size <= 0:
        raise ValueError(
            "frequencies.pixel_size must be positive."
        )
    
    if config.frequencies.viewing_distance <= 0:
        raise ValueError(
            "frequencies.viewing_distance must be positive."
        )

    if not config.experiment.name:
        raise ValueError(
            "experiment.name cannot be empty."
        )
    
    if config.figures.plot_num_steps <= 0:
        raise ValueError(
            "figures.plot_num_steps must be positive."
        )
