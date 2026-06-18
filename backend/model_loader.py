# 1. Imports
from __future__ import annotations
import torch
import torch.nn as nn
import streamlit as st
from config import CLASSIFICATION, MODELS_DIR, ModelSpec
import architectures


# 2. Device Resolution
def get_device() -> torch.device:
    """Return the CUDA device if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. Classification Backbone Builder
def _build_classification_backbone(spec: ModelSpec) -> nn.Module:
    """Reconstruct a timm architecture so a saved state_dict can be loaded.

    timm model names map 1:1 with spec.architecture for every classification
    entry in the registry (mobilenetv3_small_100, efficientnet_b0, etc).
    """
    import timm  # local import keeps timm optional for segmentation-only runs

    model = timm.create_model(
        spec.architecture,
        pretrained=False,
        num_classes=spec.num_classes,
    )
    return model


# 4. State Dict Unwrapping Helper
def _unwrap_state_dict(checkpoint):
    """Handle checkpoints saved as {'model_state_dict': ...} or a raw dict."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
    return checkpoint

# 5. Unified Model Loader
@st.cache_resource
def load_model(spec: ModelSpec) -> nn.Module:
    """Load a single model checkpoint and return it ready for inference.

    # ! Important: segmentation and detection checkpoints are restored from
    # ! a fully pickled nn.Module ("..._complete.pth"). This avoids needing
    # ! to re-implement custom architectures (PraNet, Residual U-Net, etc)
    # ! here, but it does mean the original training module must be on the
    # ! PYTHONPATH if torch.load needs to resolve custom classes by name.
    """
    device = get_device()
    folder = MODELS_DIR / spec.folder

    if spec.architecture == "complete_pickle":
        checkpoint_path = folder / spec.complete_file
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(model, dict):
            raise ValueError(
                f"{checkpoint_path.name} contains a state_dict, not a full "
                "model object. Add an architecture builder for this model "
                "in model_loader.py and load it via its weights_file instead."
            )
    elif spec.category == CLASSIFICATION:
        model = _build_classification_backbone(spec)
        state_path = folder / spec.weights_file
        checkpoint = torch.load(state_path, map_location=device, weights_only=False)
        state_dict = _unwrap_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"No loading strategy is defined for model '{spec.key}'.")

    model.to(device)
    model.eval()
    return model
