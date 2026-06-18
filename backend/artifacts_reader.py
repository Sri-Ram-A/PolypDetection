# 1. Imports
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import MODELS_DIR, ModelSpec


# 2. Known Static Artifact Names
# Every PNG below is checked for existence per model - segmentation folders
# only ship a subset of these, classification folders ship a larger set.
IMAGE_ARTIFACT_NAMES = [
    "confusion_matrix.png",
    "roc_per_class.png",
    "pvsr_per_class.png",
    "training_stats.png",
    "train_img_samaples.png",
    "training_history_detailed.png",
]


# 3. Path Helpers
def get_model_folder(spec: ModelSpec) -> Path:
    """Return the on-disk folder for a given model spec."""
    return MODELS_DIR / spec.folder


def list_available_images(spec: ModelSpec) -> dict[str, Path]:
    """Return {artifact_name: path} for every known PNG that exists."""
    folder = get_model_folder(spec)
    return {
        name: folder / name
        for name in IMAGE_ARTIFACT_NAMES
        if (folder / name).exists()
    }


# 4. Training History Loader
def load_history(spec: ModelSpec) -> Optional[dict[str, list]]:
    """Load history.json if present; classification models ship this file."""
    history_path = get_model_folder(spec) / "history.json"
    if not history_path.exists():
        return None
    try:
        with open(history_path, "r") as history_file:
            return json.load(history_file)
    except (json.JSONDecodeError, OSError):
        return None


# 5. Checkpoint Footprint
def get_checkpoint_size_mb(spec: ModelSpec) -> float:
    """Return the on-disk size in MB of the primary weights file."""
    folder = get_model_folder(spec)
    candidate = folder / (spec.complete_file or spec.weights_file)
    if candidate.exists():
        return round(candidate.stat().st_size / (1024 * 1024), 2)
    return 0.0


# 6. Raw File Listing
def list_all_files(spec: ModelSpec) -> list[str]:
    """List every file present in a model's folder, for transparency."""
    folder = get_model_folder(spec)
    if not folder.exists():
        return []
    return sorted(item.name for item in folder.iterdir() if item.is_file())
