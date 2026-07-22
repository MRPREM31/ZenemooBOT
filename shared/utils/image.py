"""
==============================================================================
Zenemoo AI - Image Utilities
==============================================================================
Provides reusable image processing operations: Pillow <-> OpenCV transformations,
color space conversions (RGB, BGR, LAB), smart aspect ratio resizing, and compression.
"""

from typing import Tuple, Union
from PIL import Image
import cv2
import numpy as np
from shared.exceptions.image_exception import CorruptImageFileException


def pillow_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Converts PIL Image to OpenCV BGR NumPy array."""
    try:
        rgb_arr = np.array(pil_image.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception as e:
        raise CorruptImageFileException(f"Failed converting PIL to OpenCV array: {e}")


def opencv_to_pillow(cv_image: np.ndarray) -> Image.Image:
    """Converts OpenCV BGR image array to PIL Image."""
    try:
        rgb_arr = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_arr)
    except Exception as e:
        raise CorruptImageFileException(f"Failed converting OpenCV array to PIL: {e}")


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Returns (width, height) of an image file without reading full pixel buffer."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        raise CorruptImageFileException(f"Could not read image dimensions for '{image_path}': {e}")


def resize_aspect_ratio(
    image: np.ndarray,
    max_dim: int = 2048,
) -> np.ndarray:
    """Resizes image keeping aspect ratio if any dimension exceeds max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def smart_downscale_pil(pil_img: Image.Image, max_dim: int = 2048) -> Image.Image:
    """Downscales PIL image maintaining aspect ratio if width or height exceeds max_dim."""
    w, h = pil_img.size
    if max(w, h) <= max_dim:
        return pil_img
    
    scale = max_dim / float(max(w, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

