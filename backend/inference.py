# 1. Imports
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES, POLYP_CLASS_INDEX, POLYP_DECISION_THRESHOLD, ModelSpec
from model_loader import get_device, load_model
from profiler import ProfileResource


# 2. Image Preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess_normalised(image: Image.Image, size: int) -> torch.Tensor:
    """Resize, tensorise, and ImageNet-normalise an RGB image."""
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


# 3. Classification Inference
def run_classification(
    spec: ModelSpec, image: Image.Image, project_name: str = "polyp_app"
) -> dict[str, Any]:
    """Run a single classification model and capture compute metrics."""
    device = get_device()
    result: dict[str, Any] = {
        "key": spec.key,
        "display_name": spec.display_name,
        "category": spec.category,
    }

    with ProfileResource(project_name=project_name, method_name=spec.key) as profile:
        model = load_model(spec)
        tensor = _preprocess_normalised(image, spec.input_size).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = (
        CLASS_NAMES[predicted_index]
        if predicted_index < len(CLASS_NAMES)
        else str(predicted_index)
    )

    result.update(
        {
            "probabilities": probabilities.tolist(),
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": float(probabilities[predicted_index]),
            "metrics": profile.metrics,
        }
    )
    return result


# 4. Triage Decision
def aggregate_polyp_decision(
    classification_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine multiple classification model outputs into one polyp call."""
    polyp_probs = [
        r["probabilities"][POLYP_CLASS_INDEX]
        for r in classification_results
        if len(r["probabilities"]) > POLYP_CLASS_INDEX
    ]
    mean_polyp_probability = float(np.mean(polyp_probs)) if polyp_probs else 0.0
    votes_for_polyp = sum(1 for p in polyp_probs if p >= POLYP_DECISION_THRESHOLD)

    return {
        "mean_polyp_probability": mean_polyp_probability,
        "votes_for_polyp": votes_for_polyp,
        "total_models": len(polyp_probs),
        "is_polyp": mean_polyp_probability >= POLYP_DECISION_THRESHOLD,
    }


# 5. Segmentation Output Extraction
def _extract_mask_tensor(output: Any) -> torch.Tensor:
    """Some segmentation models (e.g. PraNet) return multi-scale tuples."""
    if isinstance(output, (tuple, list)):
        return output[-1]
    if isinstance(output, dict):
        return output.get("out", next(iter(output.values())))
    return output


# 6. Segmentation Inference
def run_segmentation(
    spec: ModelSpec, image: Image.Image, project_name: str = "polyp_app"
) -> dict[str, Any]:
    """Run a single segmentation model and return a predicted polyp mask."""
    device = get_device()
    result: dict[str, Any] = {
        "key": spec.key,
        "display_name": spec.display_name,
        "category": spec.category,
    }

    with ProfileResource(project_name=project_name, method_name=spec.key) as profile:
        model = load_model(spec)
        tensor = _preprocess_normalised(image, spec.input_size).to(device)
        with torch.no_grad():
            raw_output = model(tensor)
            mask_logits = _extract_mask_tensor(raw_output)
            mask_probability = torch.sigmoid(mask_logits).cpu().numpy()

    mask_probability = np.squeeze(mask_probability)
    if mask_probability.ndim == 3:
        mask_probability = mask_probability[0]
    binary_mask = (mask_probability > 0.5).astype(np.uint8)

    result.update(
        {
            "mask_probability": mask_probability,
            "binary_mask": binary_mask,
            "polyp_pixel_ratio": float(binary_mask.mean()),
            "metrics": profile.metrics,
        }
    )
    return result


# 7. Detection / Localisation Inference
def run_detection(
    spec: ModelSpec,
    image: Image.Image,
    project_name: str = "polyp_app",
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run RetinaNet and return bounding boxes above the score threshold."""
    device = get_device()
    result: dict[str, Any] = {
        "key": spec.key,
        "display_name": spec.display_name,
        "category": spec.category,
    }

    # ! Important: torchvision detection models normalise internally, so the
    # ! input here is only tensorised (no manual resize / normalisation).
    tensor = transforms.ToTensor()(image.convert("RGB")).to(device)

    with ProfileResource(project_name=project_name, method_name=spec.key) as profile:
        model = load_model(spec)
        with torch.no_grad():
            prediction = model([tensor])[0]

    scores = prediction["scores"].cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    keep = scores >= score_threshold

    result.update(
        {
            "boxes": boxes[keep].tolist(),
            "scores": scores[keep].tolist(),
            "labels": labels[keep].tolist(),
            "metrics": profile.metrics,
        }
    )
    return result
