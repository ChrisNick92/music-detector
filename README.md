# Music Detector

>## **Create an efficient and lightweight Music Detector with a few commands 🚀**

### See it in action 🎬


https://github.com/user-attachments/assets/c34ac982-a87e-46cc-83f0-5895898154f8


<sub>Real-time music detection running from the microphone — predictions update every second with confidence scores.</sub>

This is a repository to showcase the capabilities of [DeepAudioX](https://github.com/deepaudiox/deepaudiox). A lightweight Pytorch-based library that me and my team in [MagCIL](https://labs-repos.iit.demokritos.gr/MagCIL/) have developed. DeepAudioX provides simple APIs for training / evaluating / and inference audio models using pretrained models as feature extractors!

In this project we train a binary classifier that detects **Music** vs **Non-Music** excerpts. We use a small sample of the [AudioSet](https://research.google.com/audioset/) dataset that you can download from Kaggle (see the instructions below).

Let's get started 🚀

## Project Structure

```
music-detector/
├── configs/
│   ├── training_config.yaml          # Training hyperparameters and dataset paths
│   ├── online_inference_config.yaml  # Real-time inference settings
│   ├── music_non_music_map.json      # Train set: YouTube ID -> class label mapping
│   └── valid_music_non_music_map.json # Validation set: YouTube ID -> class label mapping
├── scripts/
│   ├── dataset.py                    # Dataset construction utilities
│   ├── train.py                      # Training and evaluation script
│   └── online-inference.py           # Real-time microphone inference
├── pretrained_models/                # Trained model checkpoints (.pt files)
├── misc/                             # Demo videos and media assets
├── data/                             # AudioSet audio files (not tracked in git)
├── pyproject.toml
└── README.md
```

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- PortAudio (required by `sounddevice` for microphone access)

### Installation

```bash
git clone https://github.com/ChrisNick92/music-detector.git
cd music-detector
uv sync
```

### Download the Data

Download the AudioSet train and validation sets from Kaggle:

- **Train set:** https://www.kaggle.com/datasets/zfturbo/audioset
- **Validation set:** https://www.kaggle.com/datasets/zfturbo/audioset-valid

Place the downloaded data so the directory structure looks like:

```
data/
├── audio_set_train/
│   └── train_wav/           # .wav files
└── audio_set_valid/
    └── valid_wav/            # .wav files
```

The class label mappings (`music_non_music_map.json` and `valid_music_non_music_map.json`) are already included in the `configs/` directory. By default, `training_config.yaml` expects them under `data/` — either copy them there or update the `train_mapping` / `valid_mapping` paths in the config to point to `configs/` instead.

## Training

Train a music detection model using a pretrained backbone:

```bash
uv run python scripts/train.py
```

Or specify a custom config:

```bash
uv run python scripts/train.py --config configs/training_config.yaml
```

### Training Configuration

Edit `configs/training_config.yaml` to customize the training pipeline:

| Section   | Key                        | Description                                                                 |
|-----------|----------------------------|-----------------------------------------------------------------------------|
| `dataset` | `train_data_dir`           | Path to the directory with training `.wav` files                           |
| `dataset` | `train_mapping`            | Path to JSON mapping YouTube IDs to `"Music"` / `"Non-Music"`             |
| `dataset` | `valid_data_dir`           | Path to the directory with validation `.wav` files                         |
| `dataset` | `valid_mapping`            | Path to JSON mapping for validation set                                    |
| `dataset` | `sample_rate`              | Audio sampling rate in Hz (default: `16000`)                               |
| `dataset` | `segment_duration`         | Segment length in seconds; `null` for full files (default: `10.0`)         |
| `model`   | `backbone`                 | Pretrained backbone: `beats`, `passt`, `mobilenet_05_as`, `mobilenet_10_as`, `mobilenet_40_as` |
| `model`   | `pooling`                  | Pooling method: `gap`, `simpool`, `ep`, or `null` (default: `ep`)          |
| `model`   | `freeze_backbone`          | Freeze backbone weights during training (default: `true`)                  |
| `model`   | `classifier_hidden_layers` | List of hidden layer sizes for the MLP head, e.g. `[256]` or `[]`          |
| `model`   | `activation`               | Activation function: `relu`, `gelu`, `tanh`, `leakyrelu`                   |
| `model`   | `pretrained`               | Load pretrained backbone weights (default: `true`)                         |
| `training`| `learning_rate`            | Initial learning rate (default: `0.001`)                                   |
| `training`| `epochs`                   | Maximum training epochs (default: `100`)                                   |
| `training`| `patience`                 | Early stopping patience (default: `15`)                                    |
| `training`| `batch_size`               | Batch size (default: `16`)                                                 |
| `training`| `num_workers`              | DataLoader workers (default: `4`)                                          |
| `training`| `checkpoint_path`          | Where to save the best model checkpoint                                    |
| `training`| `device_index`             | GPU index to use; omit or `null` for CPU                                   |

After training completes, the script automatically evaluates the best checkpoint on the validation set and prints a classification report.

## Real-Time Inference

Run music detection from your microphone in real time:

```bash
uv run python scripts/online-inference.py
```

Or with a custom config:

```bash
uv run python scripts/online-inference.py --config configs/online_inference_config.yaml
```

The script captures audio segments from your microphone and prints predictions continuously:

```
==========================================================
  Music Detector — Real-Time Inference
  Segment: 1s | Sample rate: 16000 Hz
==========================================================
  Press Ctrl+C to stop.

  2025-02-28 14:34:56  ♫ Music      [||||||||||||||||....] 82.3%
  2025-02-28 14:34:57  ∅ Non-Music  [||||||||||||||......] 71.5%
  2025-02-28 14:34:58  ♫ Music      [||||||||||||||||||||] 97.1%
```

Press `Ctrl+C` to stop.

### Inference Configuration

Edit `configs/online_inference_config.yaml`:

| Section     | Key                        | Description                                                    |
|-------------|----------------------------|----------------------------------------------------------------|
| `model`     | `backbone`                 | Must match the backbone used during training                   |
| `model`     | `pooling`                  | Must match the pooling used during training                    |
| `model`     | `classifier_hidden_layers` | Must match the architecture used during training               |
| `model`     | `checkpoint_path`          | Path to the trained `.pt` checkpoint                           |
| `inference` | `sample_rate`              | Must match the sample rate used during training                |
| `inference` | `segment_duration`         | Duration of each audio segment to classify (in seconds)        |
| `inference` | `device_index`             | GPU index to use; omit or `null` for CPU                       |

## Scripts

### `scripts/dataset.py`

Provides `build_music_detection_dataset()` — constructs a DeepAudioX `AudioClassificationDataset` from a directory of `.wav` files and a JSON class mapping. Used by both training and can be imported for custom workflows.

### `scripts/train.py`

End-to-end training pipeline. Loads config, builds datasets, constructs an `AudioClassifier` with a pretrained backbone, trains with early stopping, and evaluates the best checkpoint. All configuration is driven by YAML — no code changes needed to experiment.

### `scripts/online-inference.py`

Real-time inference from microphone input. Loads a trained checkpoint, captures audio in fixed-duration segments, and prints timestamped predictions with confidence scores.

## Quick Start — Try It Now! 🎧

A pretrained checkpoint (`pretrained_models/mobilenet_05.pt`) is included in the repo so you can test the music detector right away — no training required!

Just run:

```bash
uv run python scripts/online-inference.py
```

Then play some music from your phone, speakers, or any source near your microphone and watch the predictions roll in. 🎶
