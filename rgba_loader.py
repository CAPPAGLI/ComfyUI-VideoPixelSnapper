"""Alpha-preserving still-image loader for ComfyUI pixel-art workflows.

ComfyUI's standard Load Image emits RGB IMAGE plus a separate transparency
MASK. This focused loader additionally returns a true RGBA IMAGE so downstream
Video Pixel Snapper nodes can consume embedded alpha without a separate mask.
"""

import hashlib
import os

import numpy as np
from PIL import Image, ImageOps
import torch

try:
    import folder_paths
    _COMFY_PATHS_AVAILABLE = True
except Exception:  # Unit tests / standalone import.
    folder_paths = None
    _COMFY_PATHS_AVAILABLE = False


_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


def _list_input_images(input_dir):
    """Return recursive input-relative image paths without private APIs.

    ComfyUI 0.33.x does not expose ``folder_paths.get_input_files()`` on every
    build. Calling it from INPUT_TYPES made only this node disappear from the
    Add Node menu while the other mappings remained valid.
    """
    files = []
    if not input_dir or not os.path.isdir(input_dir):
        return files
    for root, _dirs, names in os.walk(input_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() not in _IMAGE_EXTENSIONS:
                continue
            full = os.path.join(root, name)
            relative = os.path.relpath(full, input_dir).replace(os.sep, "/")
            files.append(relative)
    return sorted(files, key=str.lower)


def _load_rgba_path(path, sanitize_hidden_rgb=True):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        array = np.array(image, dtype=np.uint8, copy=True)

    alpha = array[..., 3]
    hidden = alpha == 0
    hidden_count = int(hidden.sum())
    if sanitize_hidden_rgb and hidden_count:
        # RGB beneath alpha=0 is visually irrelevant but leaks back whenever a
        # node/editor drops alpha. Photoshop masks and erasing often leave
        # different white/magenta mattes here, so canonicalize it to black.
        array[..., :3][hidden] = 0

    rgba = torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)
    rgb = rgba[..., :3]
    foreground = rgba[..., 3]
    transparency = 1.0 - foreground
    partial_count = int(((alpha > 0) & (alpha < 255)).sum())
    info = (
        f"Load RGBA Image: {array.shape[1]}x{array.shape[0]} "
        f"hidden_alpha0={hidden_count} partial_alpha={partial_count} "
        f"hidden_rgb={'zeroed' if sanitize_hidden_rgb else 'preserved'}"
    )
    return rgb, rgba, foreground, transparency, info


class VideoPixelSnapperLoadRGBA:
    """Load one PNG/WebP/etc. while retaining alpha inside IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        if _COMFY_PATHS_AVAILABLE:
            files = _list_input_images(folder_paths.get_input_directory())
            return {
                "required": {
                    "image": (sorted(files), {"image_upload": True}),
                    "sanitize_hidden_rgb": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Set RGB to black only where alpha is exactly zero. "
                                   "Visible pixels are unchanged; this prevents Photoshop's "
                                   "hidden white/magenta mattes from reappearing if alpha is dropped."
                    }),
                }
            }
        return {
            "required": {
                "image": ("STRING", {"default": ""}),
                "sanitize_hidden_rgb": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = (
        "image", "transparent_image", "foreground_mask",
        "transparency_mask", "info",
    )
    FUNCTION = "load"
    CATEGORY = "Video Pixel Snapper"
    DESCRIPTION = (
        "Loads a still image as both RGB and true RGBA. Use transparent_image "
        "to let Video Pixel Snapper consume embedded PNG alpha automatically."
    )

    def load(self, image, sanitize_hidden_rgb=True):
        if _COMFY_PATHS_AVAILABLE:
            path = folder_paths.get_annotated_filepath(image)
        else:
            path = image
        if not path or not os.path.isfile(path):
            raise ValueError(f"RGBA image file does not exist: {path!r}")
        return _load_rgba_path(path, bool(sanitize_hidden_rgb))

    @classmethod
    def IS_CHANGED(cls, image, sanitize_hidden_rgb=True):
        if not _COMFY_PATHS_AVAILABLE:
            return float("nan")
        path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"sanitize=1" if sanitize_hidden_rgb else b"sanitize=0")
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, sanitize_hidden_rgb=True):
        if not _COMFY_PATHS_AVAILABLE:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapperLoadRGBA": VideoPixelSnapperLoadRGBA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapperLoadRGBA": "Load RGBA Image (Video Pixel Snapper)",
}
