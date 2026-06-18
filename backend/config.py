# #* config.py
# 1. Imports
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# 2. Base Paths
# backend/config.py -> backend/ -> project root, so this works regardless
# of whether Streamlit is launched from the project root or from backend/.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# 3. Task Categories
CLASSIFICATION = "classification"
SEGMENTATION = "segmentation"
DETECTION = "detection"

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyp",
    "ulcerative-colitis",
]
POLYP_CLASS_INDEX = 6
POLYP_DECISION_THRESHOLD = 0.5


# 4. Model Metadata Schema
@dataclass(frozen=True)
class ModelSpec:
    """Static description of a single model checkpoint stored on disk."""

    key: str  # unique short id used across the app
    display_name: str  # human readable name shown in the UI
    folder: str  # folder name under MODELS_DIR
    category: str  # classification | segmentation | detection
    architecture: str  # timm name, or "complete_pickle"
    weights_file: str  # state_dict / checkpoint file name
    complete_file: Optional[str] = None  # fully pickled nn.Module file, if any
    num_classes: int = len(CLASS_NAMES)
    input_size: int = 224
    notes: str = ""


# 5. Classification Model Registry
# Loaded by reconstructing the timm backbone and restoring its state_dict,
# since these folders only contain "best_*.pt" weight files.
CLASSIFICATION_MODELS = [
    ModelSpec(
        key="mobilenetv3_small_100",
        display_name="MobileNetV3 - Small 100",
        folder="10-mobilenet-v3-small",
        category=CLASSIFICATION,
        architecture="mobilenetv3_small_100",
        weights_file="best_mobilenetv3_small_100.pt",
        notes="Lightweight CNN, well suited to edge / mobile deployment.",
    ),
    ModelSpec(
        key="efficientnet_b0",
        display_name="EfficientNet - B0",
        folder="11-efficientnet-b0",
        category=CLASSIFICATION,
        architecture="efficientnet_b0",
        weights_file="best_efficientnet_b0.pt",
        notes="Compound-scaled CNN balancing accuracy and efficiency.",
    ),
    ModelSpec(
        key="convnext_tiny",
        display_name="ConvNeXt - Tiny",
        folder="12-convnext-tiny",
        category=CLASSIFICATION,
        architecture="convnext_tiny",
        weights_file="best_convnext-tiny.pt",
        notes="Modernised ConvNet with transformer-inspired design choices.",
    ),
    ModelSpec(
        key="densenet121",
        display_name="DenseNet - 121",
        folder="13-densenet-121",
        category=CLASSIFICATION,
        architecture="densenet121",
        weights_file="best_densenet121.pt",
        notes="Densely connected CNN with strong gradient flow.",
    ),
    ModelSpec(
        key="ghostnet_100",
        display_name="GhostNet - 100",
        folder="14-ghostnet-100",
        category=CLASSIFICATION,
        architecture="ghostnet_100",
        weights_file="best_ghostnet-100.pt",
        notes="Cheap feature-map generation for low-resource inference.",
    ),
]


# 6. Segmentation Model Registry
# ! Important: every folder also ships a fully pickled nn.Module
# ! ("..._complete.pth"). The loader uses that file directly, so no
# ! architecture reconstruction code is required for these custom networks.
# ! Input resolutions below are reasonable defaults - adjust per model if
# ! your training pipeline used a different size.
SEGMENTATION_MODELS = [
    ModelSpec(
        key="residual_unet",
        display_name="Residual U-Net",
        folder="2-residual-unet",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_unet_model.pth",
        complete_file="final_unet_model_complete.pth",
        input_size=256,
        notes="U-Net encoder-decoder with residual blocks.",
    ),
    ModelSpec(
        key="squeeze_excite_unet",
        display_name="Squeeze-and-Excite U-Net",
        folder="3-squeeze-excite",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_squeeze-excite_model.pth",
        complete_file="final_squeeze-excite_model_complete.pth",
        input_size=256,
        notes="U-Net with channel attention (squeeze-and-excite blocks).",
    ),
    ModelSpec(
        key="unet_plus_plus",
        display_name="UNet++",
        folder="5-unet-plus-plus",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_unet-plus-plus_model.pth",
        complete_file="final_unet-plus-plus_model_complete.pth",
        input_size=256,
        notes="Nested, densely skip-connected U-Net variant.",
    ),
    ModelSpec(
        key="deeplab_v3_plus",
        display_name="DeepLabV3+",
        folder="6-deeplab-v3-plus",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_deeplab-v3-plus_model.pth",
        complete_file="final_deeplab-v3-plus_model_complete.pth",
        input_size=256,
        notes="Atrous spatial pyramid pooling with a lightweight decoder.",
    ),
    ModelSpec(
        key="fpn",
        display_name="Feature Pyramid Network (FPN)",
        folder="7-fpn",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_fpn_model.pth",
        complete_file="final_fpn_model_complete.pth",
        input_size=256,
        notes="Multi-scale pyramid features fused for dense prediction.",
    ),
    ModelSpec(
        key="pranet",
        display_name="PraNet",
        folder="8-pranet",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_pranet_model.pth",
        complete_file="final_pranet_model_complete.pth",
        input_size=352,
        notes="Reverse-attention parallel decoder for polyp segmentation.",
    ),
    ModelSpec(
        key="pranet_full",
        display_name="PraNet (Full)",
        folder="9-pranet-full",
        category=SEGMENTATION,
        architecture="complete_pickle",
        weights_file="best_pranet-full_model.pth",
        complete_file="final_pranet-full_model_complete.pth",
        input_size=352,
        notes="Full-capacity PraNet variant trained for longer.",
    ),
]


# 7. Detection / Localisation Model Registry
DETECTION_MODELS = [
    ModelSpec(
        key="retinanet",
        display_name="RetinaNet (ResNet-50 FPN)",
        folder="4-retinanet",
        category=DETECTION,
        architecture="complete_pickle",
        weights_file="retinanet_state_dict.pth",
        complete_file="retinanet_complete.pth",
        num_classes=2,
        notes="One-stage detector with focal loss for polyp localisation.",
    ),
]


# 8. Convenience Lookups
ALL_MODELS = CLASSIFICATION_MODELS + SEGMENTATION_MODELS + DETECTION_MODELS
MODELS_BY_KEY = {model.key: model for model in ALL_MODELS}
MODELS_BY_CATEGORY = {
    CLASSIFICATION: CLASSIFICATION_MODELS,
    SEGMENTATION: SEGMENTATION_MODELS,
    DETECTION: DETECTION_MODELS,
}
