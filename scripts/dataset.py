"""Utilities for constructing an AudioClassificationDataset for music detection."""

import json
import os
from pathlib import Path

from deepaudiox import AudioClassificationDataset

CLASS_MAPPING: dict[str, int] = {"Music": 0, "Non-Music": 1}
"""Mapping from class labels to integer IDs used by the classifier."""


def build_music_detection_dataset(
    path_to_data: str | Path,
    path_to_json_mapping: str | Path,
    sample_rate: int = 16_000,
    segment_duration: float | None = 10.0,
) -> AudioClassificationDataset:
    """Build an AudioClassificationDataset for music / non-music classification.

    Reads a JSON mapping of YouTube IDs to class labels and pairs each entry
    with the corresponding ``.wav`` file found in ``path_to_data``. Files that
    are listed in the mapping but missing from disk are silently skipped.

    Args:
        path_to_data: Directory containing the ``.wav`` audio files.
        path_to_json_mapping: Path to a JSON file mapping YouTube IDs to class
            labels (``"Music"`` or ``"Non-Music"``).
        sample_rate: Target sampling rate in Hz for audio loading.
        segment_duration: Duration in seconds to segment each audio file into
            fixed-length chunks. Each chunk becomes an individual sample. Set to
            ``None`` to load full audio files without segmentation.

    Returns:
        An AudioClassificationDataset ready for use with a Trainer or Evaluator.
    """
    path_to_data = Path(path_to_data)
    path_to_json_mapping = Path(path_to_json_mapping)

    with open(path_to_json_mapping) as f:
        id_to_class: dict[str, str] = json.load(f)

    file_to_class: dict[str | os.PathLike, str] = {
        str(path_to_data / f"{youtube_id}.wav"): label
        for youtube_id, label in id_to_class.items()
        if (path_to_data / f"{youtube_id}.wav").exists()
    }

    return AudioClassificationDataset(file_to_class, sample_rate, CLASS_MAPPING, segment_duration)
