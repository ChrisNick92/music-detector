"""Run real-time music detection from microphone input using a trained DeepAudioX model."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import torch
import yaml
from deepaudiox import AudioClassifier
from typing import Literal

from dataset import CLASS_MAPPING



@dataclass
class ModelConfig:
    """Configuration for loading the trained AudioClassifier checkpoint.

    Attributes:
        checkpoint_path: Path to the saved model checkpoint (``.pt`` file).
            Architecture and weights are restored automatically via
            ``AudioClassifier.from_checkpoint``.
    """

    checkpoint_path: str = "pretrained_models/checkpoint.pt"


@dataclass
class InferenceConfig:
    """Configuration for real-time audio capture and inference.

    Attributes:
        sample_rate: Sampling rate in Hz for microphone capture. Must match the
            rate used during training.
        segment_duration: Duration in seconds of each audio segment to classify.
        device: Device to use for inference. One of ``"cuda"``, ``"mps"``, or
            ``"cpu"``.
        device_index: GPU device index. Only used when ``device="cuda"``.
    """

    sample_rate: int = 16_000
    segment_duration: float = 10.0
    device: Literal["cuda", "mps", "cpu"] = "cpu"
    device_index: int | None = None


@dataclass
class OnlineInferenceConfig:
    """Top-level configuration for real-time music detection.

    Expected YAML structure::

        model:
          checkpoint_path: pretrained_models/checkpoint.pt

        inference:
          sample_rate: 16000
          segment_duration: 10.0
          device: cpu
          device_index: null
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


def load_model(config: ModelConfig, inference: "InferenceConfig") -> AudioClassifier:
    """Load the AudioClassifier from a self-describing checkpoint.

    Args:
        config: Model configuration holding the checkpoint path.
        inference: Inference configuration with device settings.

    Returns:
        The trained AudioClassifier in evaluation mode.
    """
    device = torch.device(
        f"cuda:{inference.device_index}" if inference.device == "cuda" and inference.device_index is not None
        else inference.device
    )
    model = AudioClassifier.from_checkpoint(config.checkpoint_path)
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
    model = load_model(config.model, config.inference)

    sample_rate = config.inference.sample_rate
    segment_duration = config.inference.segment_duration
    segment_samples = int(segment_duration * sample_rate)

    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    MUSIC_ICON = "\u266b"
    SILENCE_ICON = "\u2205"

    print(f"\n{BOLD}{'=' * 58}{RESET}")
    print(f"{BOLD}  Music Detector — Real-Time Inference{RESET}")
    print(f"{DIM}  Segment: {segment_duration}s | Sample rate: {sample_rate} Hz{RESET}")
    print(f"{BOLD}{'=' * 58}{RESET}")
    print(f"{DIM}  Press Ctrl+C to stop.{RESET}\n")

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

            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            label = result["final_label"]
            posterior = result["final_posterior"]

            if label == "Music":
                icon = MUSIC_ICON
                color = GREEN
            else:
                icon = SILENCE_ICON
                color = YELLOW

            bar_len = int(posterior * 20)
            bar = f"{'|' * bar_len}{'.' * (20 - bar_len)}"

            print(
                f"  {DIM}{timestamp}{RESET}  "
                f"{color}{BOLD}{icon} {label:<10}{RESET} "
                f"{DIM}[{bar}]{RESET} {posterior:.1%}"
            )

    except KeyboardInterrupt:
        print(f"\n{DIM}  Stopped.{RESET}\n")


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
