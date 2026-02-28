"""Run real-time music detection from microphone input using a trained DeepAudioX model."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import sounddevice as sd
import torch
import yaml
from deepaudiox import AudioClassifier

from dataset import CLASS_MAPPING



@dataclass
class ModelConfig:
    """Configuration for reconstructing the trained AudioClassifier.

    Attributes:
        backbone: Pretrained backbone used during training.
        pooling: Pooling layer used during training. ``null`` defaults to GAP.
        freeze_backbone: Whether the backbone was frozen during training.
        classifier_hidden_layers: Hidden layer sizes for the MLP classifier head.
        activation: Activation function used in the classifier head.
        apply_batch_norm: Whether batch normalization was applied in the classifier.
        pretrained: Whether pretrained backbone weights were loaded.
        checkpoint_path: Path to the saved model checkpoint (``.pt`` file).
    """

    checkpoint_path: str = "pretrained_models/checkpoint.pt"
    backbone: Literal["beats", "passt", "mobilenet_05_as", "mobilenet_10_as", "mobilenet_40_as"] = "mobilenet_05_as"
    pooling: Literal["gap", "simpool", "ep"] | None = "ep"
    freeze_backbone: bool = True
    classifier_hidden_layers: list[int] | None = field(default_factory=list)
    activation: Literal["relu", "gelu", "tanh", "leakyrelu"] = "relu"
    apply_batch_norm: bool = True
    pretrained: bool = True


@dataclass
class InferenceConfig:
    """Configuration for real-time audio capture and inference.

    Attributes:
        sample_rate: Sampling rate in Hz for microphone capture. Must match the
            rate used during training.
        segment_duration: Duration in seconds of each audio segment to classify.
    """

    sample_rate: int = 16_000
    segment_duration: float = 10.0


@dataclass
class OnlineInferenceConfig:
    """Top-level configuration for real-time music detection.

    Expected YAML structure::

        model:
          backbone: mobilenet_05_as
          pooling: ep
          freeze_backbone: true
          classifier_hidden_layers: []
          activation: relu
          apply_batch_norm: true
          pretrained: true
          checkpoint_path: pretrained_models/checkpoint.pt

        inference:
          sample_rate: 16000
          segment_duration: 10.0
    """

    model: ModelConfig
    inference: InferenceConfig


def load_config(config_path: str | Path) -> OnlineInferenceConfig:
    """Load and validate a YAML configuration file for online inference.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated OnlineInferenceConfig instance.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return OnlineInferenceConfig(
        model=ModelConfig(**raw.get("model", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
    )


def load_model(config: ModelConfig, device: torch.device) -> AudioClassifier:
    """Reconstruct the AudioClassifier and load trained weights.

    Args:
        config: Model configuration matching the architecture used during training.
        device: Device to load the model onto.

    Returns:
        The trained AudioClassifier in evaluation mode.
    """
    model = AudioClassifier(
        num_classes=len(CLASS_MAPPING),
        backbone=config.backbone,
        pooling=config.pooling,
        freeze_backbone=config.freeze_backbone,
        sample_rate=16_000,
        classifier_hidden_layers=config.classifier_hidden_layers,
        activation=config.activation,
        apply_batch_norm=config.apply_batch_norm,
        pretrained=config.pretrained,
    )

    state_dict = torch.load(config.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run(config: OnlineInferenceConfig) -> None:
    """Continuously capture audio from the microphone and classify each segment.

    Each segment is printed with a UTC timestamp and the predicted class
    (``Music`` or ``Non-Music``).

    Args:
        config: Validated online inference configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.model, device)

    sample_rate = config.inference.sample_rate
    segment_duration = config.inference.segment_duration
    segment_samples = int(segment_duration * sample_rate)

    print(
        f"Listening... (segment duration: {segment_duration}s, "
        f"sample rate: {sample_rate} Hz). Press Ctrl+C to stop.\n"
    )

    try:
        while True:
            audio = sd.rec(
                frames=segment_samples,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            waveform = audio.squeeze()

            result = model.inference_on_waveform(
                waveform,
                sample_rate=sample_rate,
                class_mapping=CLASS_MAPPING,
                segment_duration=segment_duration,
            )

            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            label = result["final_label"]
            posterior = result["final_posterior"]
            print(f"[{timestamp}] {label} ({posterior:.2%})")

    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    """Entry point for the online inference script."""
    parser = argparse.ArgumentParser(
        description="Real-time music detection from microphone input using a trained DeepAudioX model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/online_inference_config.yaml",
        help="Path to the YAML configuration file (default: configs/online_inference_config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
