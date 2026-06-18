# 1. Imports
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw


# 2. Clinical Colour Palette
PRIMARY_COLOR = "#0B5394"
ACCENT_COLOR = "#2F855A"
WARNING_COLOR = "#C53030"


# 3. Segmentation Mask Overlay
def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple = (197, 48, 48),
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a binary segmentation mask onto a copy of the original image."""
    base = image.convert("RGB").resize((mask.shape[1], mask.shape[0]))
    base_array = np.array(base).astype(np.float32)
    blended = base_array.copy()
    blended[mask == 1] = (1 - alpha) * base_array[mask == 1] + alpha * np.array(color)
    return Image.fromarray(blended.astype(np.uint8))


# 4. Bounding Box Drawing
def draw_boxes(
    image: Image.Image, boxes: list, scores: list, color: tuple = (11, 83, 148)
) -> Image.Image:
    """Draw RetinaNet bounding boxes with confidence labels."""
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
        drawer.text((x1, max(0, y1 - 14)), f"{score:.2f}", fill=color)
    return canvas


# 5. Classification Probability Chart
def plot_class_probabilities(
    result: dict[str, Any], class_names: list[str]
) -> go.Figure:
    """Horizontal bar chart of predicted class probabilities for one model."""
    figure = go.Figure(
        go.Bar(
            x=result["probabilities"],
            y=class_names,
            orientation="h",
            marker_color=PRIMARY_COLOR,
        )
    )
    figure.update_layout(
        title=f"{result['display_name']} - Class Probabilities",
        xaxis_title="Probability",
        xaxis_range=[0, 1],
        plot_bgcolor="white",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return figure


# 6. Training History Curves
def plot_history_curves(history: dict[str, list], title: str) -> go.Figure:
    """Plot every numeric series found in a history.json payload."""
    figure = go.Figure()
    for key, values in history.items():
        if isinstance(values, list) and values and isinstance(values[0], (int, float)):
            figure.add_trace(go.Scatter(y=values, mode="lines", name=key))
    figure.update_layout(title=title, xaxis_title="Epoch", plot_bgcolor="white")
    return figure


# 7. Compute Metric Comparison (latency, RAM, emissions, CPU)
def plot_compute_comparison(
    results: list[dict[str, Any]], metric_key: str, title: str, y_label: str
) -> go.Figure:
    """Bar chart comparing one profiler metric across recorded model runs."""
    names = [r["display_name"] for r in results]
    values = [r["metrics"].get(metric_key, 0) for r in results]
    categories = [r.get("category", "") for r in results]

    figure = px.bar(
        x=names,
        y=values,
        color=categories,
        labels={"x": "Model", "y": y_label, "color": "Task"},
        title=title,
    )
    figure.update_layout(plot_bgcolor="white")
    return figure


# 8. Predicted Polyp Area Comparison (segmentation)
def plot_pixel_ratio_comparison(results: list[dict[str, Any]]) -> go.Figure:
    """Bar chart comparing predicted polyp area across segmentation models."""
    names = [r["display_name"] for r in results]
    ratios = [r["polyp_pixel_ratio"] * 100 for r in results]

    figure = px.bar(
        x=names,
        y=ratios,
        labels={"x": "Model", "y": "Predicted Polyp Area (%)"},
        title="Predicted Polyp Area by Model",
        color=ratios,
        color_continuous_scale="Reds",
    )
    figure.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
    return figure


# 9. Checkpoint Size Comparison
def plot_checkpoint_sizes(
    names: list[str], sizes_mb: list[float], title: str
) -> go.Figure:
    """Bar chart comparing on-disk checkpoint footprint across models."""
    figure = px.bar(
        x=names,
        y=sizes_mb,
        labels={"x": "Model", "y": "Checkpoint Size (MB)"},
        title=title,
        color=sizes_mb,
        color_continuous_scale="Blues",
    )
    figure.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
    return figure
