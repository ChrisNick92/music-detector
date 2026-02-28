"""Train a music detection model using DeepAudioX."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from deepaudiox import AudioClassifier, Trainer

from dataset import CLASS_MAPPING, build_music_detection_dataset


@dataclass
class DatasetConfig:
    """Configuration for the training and validation datasets.

    Attributes:
        train_data_dir: Directory containing the training ``.wav`` files.
        train_mapping: Path to the JSON mapping YouTube IDs to class labels
            for the training set.
        valid_data_dir: Directory containing the validation ``.wav`` files.
        valid_mapping: Path to the JSON mapping YouTube IDs to class labels
            for the validation set.
        sample_rate: Target sampling rate in Hz for audio loading.
        segment_duration: Duration in seconds to segment each audio file.
            Set to ``null`` in the YAML to load full files.
    """

    train_data_dir: str
    train_mapping: str
    valid_data_dir: str
    valid_mapping: str
    sample_rate: int = 16_000
    segment_duration: float | None = 10.0


@dataclass
class ModelConfig:
    """Configuration for the AudioClassifier architecture.

    Attributes:
        backbone: Pretrained backbone to use for feature extraction.
        pooling: Pooling layer to aggregate features. ``null`` defaults to GAP.
        freeze_backbone: Whether to freeze the backbone weights during training.
        classifier_hidden_layers: Hidden layer sizes for the MLP classifier head.
        activation: Activation function for the classifier head.
        apply_batch_norm: Whether to apply batch normalization in the classifier.
        pretrained: Whether to load pretrained weights for the backbone.
    """

    backbone: Literal["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"] = "mobilenet_05_as"
    pooling: Literal["gap", "simpool", "ep"] | None = "ep"
    freeze_backbone: bool = True
    classifier_hidden_layers: list[int] | None = field(default_factory=list)
    activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu"
    apply_batch_norm: bool = True
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes:
        learning_rate: Initial learning rate for the optimizer.
        epochs: Maximum number of training epochs.
        patience: Number of epochs without improvement before early stopping.
        batch_size: Number of samples per training batch.
        num_workers: Number of data-loading worker processes.
        checkpoint_path: File path to save the best model checkpoint.
        device_index: GPU device index to use for training. If not specified,
            falls back to CPU.
    """

    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 15
    batch_size: int = 16
    num_workers: int = 4
    checkpoint_path: str = "pretrained_models/checkpoint.pt"
    device_index: int | None = None


@dataclass
class Config:
    """Top-level configuration for the training pipeline.

    Expected YAML structure::

        dataset:
          train_data_dir: data/audio_set_train/train_wav
          train_mapping: data/audio_set_train/music_non_music_map.json
          valid_data_dir: data/audio_set_valid/valid_wav
          valid_mapping: data/audio_set_valid/valid_music_non_music_map.json
          sample_rate: 16000
          segment_duration: 10.0

        model:
          backbone: mobilenet_05_as
          pooling: ep
          freeze_backbone: true
          classifier_hidden_layers: []
          activation: relu
          apply_batch_norm: true
          pretrained: true

        training:
          learning_rate: 0.001
          epochs: 100
          patience: 15
          batch_size: 16
          num_workers: 4
          checkpoint_path: pretrained_models/checkpoint.pt
          device_index: 0
    """

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(config_path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated Config instance.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )


def train(config: Config) -> None:
    """Run the training pipeline from a parsed configuration.

    Args:
        config: A validated Config instance.
    """
    train_dset = build_music_detection_dataset(
        path_to_data=config.dataset.train_data_dir,
        path_to_json_mapping=config.dataset.train_mapping,
        sample_rate=config.dataset.sample_rate,
        segment_duration=config.dataset.segment_duration,
    )

    valid_dset = build_music_detection_dataset(
        path_to_data=config.dataset.valid_data_dir,
        path_to_json_mapping=config.dataset.valid_mapping,
        sample_rate=config.dataset.sample_rate,
        segment_duration=config.dataset.segment_duration,
    )

    model = AudioClassifier(
        num_classes=len(CLASS_MAPPING),
        backbone=config.model.backbone,
        pooling=config.model.pooling,
        freeze_backbone=config.model.freeze_backbone,
        sample_rate=config.dataset.sample_rate,
        classifier_hidden_layers=config.model.classifier_hidden_layers,
        activation=config.model.activation,
        apply_batch_norm=config.model.apply_batch_norm,
        pretrained=config.model.pretrained,
    )

    trainer = Trainer(
        train_dset=train_dset,
        model=model,
        validation_dset=valid_dset,
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        patience=config.training.patience,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        path_to_checkpoint=config.training.checkpoint_path,
        device_index=config.training.device_index,
    )

    trainer.train()


def main() -> None:
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train a music detection model using DeepAudioX.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to the YAML configuration file (default: configs/training_config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()