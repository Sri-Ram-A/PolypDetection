from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from IPython.display import display, Image as IPyImage, HTML
import io, base64

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
        raise ValueError(f"All directories must have the same number of files, got: {lengths}")

    idx = random.randint(0, lengths[0] - 1)
    return [file_list[idx] for file_list in all_files]

def visualize_images(data_dict: dict, rows: int = 0, cols:int = 0, figsize: tuple = (10, 6), cmap: str = "gray"):

    cols = len(data_dict) if rows == 1 else 1
    rows = len(data_dict) if cols == 1 else 1
    plt.figure(figsize=figsize)
    for idx, (title, image) in enumerate(data_dict.items(), start=1):
        plt.subplot(rows, cols, idx)
        # Handle tensor format (C, H, W) → (H, W, C)
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.moveaxis(image, 0, -1)  # convert (C, H, W) to (H, W, C)
        # Ensure 2D images use colormap
        if image.ndim == 2:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')

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
            if name: print(name)
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
                f'width="{width//len(img_list)}"><br>'
                f'{name if name else ""}</div>'
            )
        display(HTML("".join(html_imgs)))

    else:
        raise ValueError("grid must be 'row' or 'col'")
