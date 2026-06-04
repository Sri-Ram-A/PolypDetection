from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from IPython.display import display, Image as IPyImage, HTML
import io
import base64
import torch
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# PROJECT PATHS (Root: PolypDetection)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Define paths for each stage


def get_stage_paths(stage_number: int, stage_name: str):
    """
    Get paths for a specific notebook stage.

    Args:
        stage_number: Stage number (1, 2, 3, 4, 5)
        stage_name: Stage name (e.g., 'explore', 'residual-unet')

    Returns:
        dict: Dictionary with stage paths
    """
    stage_dir = NOTEBOOKS_DIR / f"{stage_number}-{stage_name}"
    return {
        "stage_dir": stage_dir,
        "figures_dir": stage_dir / "figures",
        "models_dir": stage_dir / "models",
        "outputs_dir": stage_dir / "outputs",
    }


def get_images_from_dirs(dir_list, valid_exts={".jpg", ".jpeg", ".png"}):
    """
    Takes a list of directories and returns a list of files at the same index from each directory.
    Assumes all directories contain equal number of images with matching order.
    """
    dir_list = [Path(d) for d in dir_list]

    # Get list of files for each directory
    all_files = [
        sorted([f for f in d.iterdir() if f.suffix.lower() in valid_exts])
        for d in dir_list
    ]

    # Check lengths are equal
    lengths = [len(files) for files in all_files]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All directories must have the same number of files, got: {lengths}"
        )

    idx = random.randint(0, lengths[0] - 1)
    return [file_list[idx] for file_list in all_files]


def visualize_images(
    data_dict: dict,
    rows: int = 0,
    cols: int = 0,
    figsize: tuple = (10, 6),
    cmap: str = "gray",
):

    cols = len(data_dict) if rows == 1 else 1
    rows = len(data_dict) if cols == 1 else 1
    plt.figure(figsize=figsize)
    for idx, (title, image) in enumerate(data_dict.items(), start=1):
        plt.subplot(rows, cols, idx)
        # Handle tensor format (C, H, W) → (H, W, C)
        if (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[0] in [1, 3]
        ):
            image = np.moveaxis(image, 0, -1)  # convert (C, H, W) to (H, W, C)
        # Ensure 2D images use colormap
        if image.ndim == 2:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show(images: list | dict, width: int = 1500):

    if isinstance(images, dict):
        for name, image in images.items():
            print(name)
            pil_img = PIL.Image.fromarray(image)
            display(IPyImage(pil_img._repr_png_(), width=width))
    elif isinstance(images, list):
        for image in images:
            pil_img = PIL.Image.fromarray(image)
            display(IPyImage(pil_img._repr_png_(), width=width))


def show_grid(images: list | dict, width: int = 1500, grid: str = "col"):

    def pil_to_bytes(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    if isinstance(images, dict):
        img_list = [(name, PIL.Image.fromarray(img)) for name, img in images.items()]
    elif isinstance(images, list):
        img_list = [(None, PIL.Image.fromarray(img)) for img in images]
    else:
        raise ValueError("images must be a list or dict")

    if grid == "col":
        # show images one under the other
        for name, pil_img in img_list:
            if name:
                print(name)
            display(IPyImage(pil_to_bytes(pil_img), width=width))

    elif grid == "row":
        # show images side by side
        html_imgs = []
        for name, pil_img in img_list:
            img_bytes = pil_to_bytes(pil_img)
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            html_imgs.append(
                f'<div style="display:inline-block; margin:5px; text-align:center">'
                f'<img src="data:image/png;base64,{b64_img}" '
                f'width="{width // len(img_list)}"><br>'
                f"{name if name else ''}</div>"
            )
        display(HTML("".join(html_imgs)))

    else:
        raise ValueError("grid must be 'row' or 'col'")


# ============================================================================
# MODEL UTILITIES
# ============================================================================


def save_model_all_formats(model, base_path: Path, model_name: str = "model"):
    """
    Save model in all PyTorch formats:
    - .pth (state_dict)
    - .pth (complete model - named differently)
    - .pt (TorchScript)

    Args:
        model: PyTorch model to save
        base_path: Directory to save models
        model_name: Base name for the model
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save state dict as .pth
    state_dict_path = base_path / f"{model_name}-state-dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"✓ Saved state_dict: {state_dict_path}")

    # Save complete model as .pth
    complete_path = base_path / f"{model_name}-complete.pth"
    torch.save(model, complete_path)
    print(f"✓ Saved complete model: {complete_path}")

    # Save as TorchScript .pt
    try:
        scripted_model = torch.jit.script(model)
        scripted_path = base_path / f"{model_name}.pt"
        torch.jit.save(scripted_model, scripted_path)
        print(f"✓ Saved TorchScript: {scripted_path}")
    except Exception as e:
        print(f"⚠ Could not save as TorchScript: {e}")


def load_model_from_state_dict(model_class, state_dict_path: Path, device=DEVICE):
    """
    Load model from state_dict.

    Args:
        model_class: The model class to instantiate
        state_dict_path: Path to state_dict file
        device: Device to load model to

    Returns:
        model: Loaded model on specified device
    """
    model = model_class()
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def move_to_device(data, device=DEVICE):
    """
    Move tensors to device (GPU or CPU).
    Works with single tensors, lists, and dicts.

    Args:
        data: Data to move to device
        device: Target device

    Returns:
        Data on the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return type(data)(
            v.to(device) if isinstance(v, torch.Tensor) else v for v in data
        )
    return data


def inspect_model(
    model, input_size, criterion=None, optimizer=None, model_name="Model"
):
    """
    Inspect and display model architecture information.

    Args:
        model: PyTorch model
        input_size: Tuple of input size (batch, channels, height, width)
        criterion: Loss function (optional)
        optimizer: Optimizer (optional)
        model_name: Name of the model for display
    """
    from torchinfo import summary

    print("\n" + "=" * 80)
    print(f"MODEL ARCHITECTURE: {model_name}")
    print("=" * 80)

    # Display model summary using torchinfo
    try:
        model_summary = summary(model, input_size=input_size, verbose=0)
        print(model_summary)
    except Exception as e:
        print(f"Warning: Could not display model summary: {e}")
        print("\nModel Structure:")
        print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Display criterion info
    if criterion is not None:
        print(f"\nLoss Function: {criterion.__class__.__name__}")
        print(f"  {criterion}")

    # Display optimizer info
    if optimizer is not None:
        print(f"\nOptimizer: {optimizer.__class__.__name__}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Parameters: {optimizer.param_groups[0]}")

    # Display device
    device = next(model.parameters()).device
    print(f"\nDevice: {device}")
    print("=" * 80 + "\n")
