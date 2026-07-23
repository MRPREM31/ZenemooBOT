"""
==============================================================================
Zenemoo AI - Face Crop, Seamless Mask Blender & Skin Texture Preservation
==============================================================================
Provides independent face cropping with margin padding, Gaussian elliptical mask
blending, and natural skin high-pass texture re-injection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any


def crop_face_with_margin(
    img_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    margin_ratio: float = 0.30,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crops face bounding box (x, y, w, h) with margin_ratio padding.
    Returns (cropped_bgr, (x1, y1, x2, y2)).
    """
    h_img, w_img = img_bgr.shape[:2]
    fx, fy, fw, fh = box

    mx = int(fw * margin_ratio)
    my = int(fh * margin_ratio)

    x1 = max(0, fx - mx)
    y1 = max(0, fy - my)
    x2 = min(w_img, fx + fw + mx)
    y2 = min(h_img, fy + fh + my)

    crop = img_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def preserve_natural_skin_texture(
    orig_crop_bgr: np.ndarray,
    restored_crop_bgr: np.ndarray,
    texture_blend_ratio: float = 0.20,
) -> np.ndarray:
    """
    Extracts high-frequency fine skin texture details from original image crop
    and blends texture_blend_ratio (15-25%) back into restored crop to prevent plastic skin.
    """
    if orig_crop_bgr.shape != restored_crop_bgr.shape:
        orig_crop_bgr = cv2.resize(orig_crop_bgr, (restored_crop_bgr.shape[1], restored_crop_bgr.shape[0]))

    orig_gray = cv2.cvtColor(orig_crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Extract high-frequency detail map
    blur_gray = cv2.GaussianBlur(orig_gray, (5, 5), 0)
    high_pass = orig_gray - blur_gray

    restored_float = restored_crop_bgr.astype(np.float32) / 255.0

    for c in range(3):
        restored_float[:, :, c] += texture_blend_ratio * high_pass

    blended = np.clip(restored_float * 255.0, 0, 255).astype(np.uint8)
    return blended


def blend_restored_face(
    base_bgr: np.ndarray,
    restored_crop_bgr: np.ndarray,
    crop_coords: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Seamlessly blends restored face crop back into base_bgr using a Gaussian
    feathered elliptical alpha mask.
    """
    x1, y1, x2, y2 = crop_coords
    ch = y2 - y1
    cw = x2 - x1

    if (cw, ch) != (restored_crop_bgr.shape[1], restored_crop_bgr.shape[0]):
        restored_crop_bgr = cv2.resize(restored_crop_bgr, (cw, ch))

    # Create smooth Gaussian elliptical alpha mask
    mask = np.zeros((ch, cw), dtype=np.float32)
    center = (cw // 2, ch // 2)
    axes = (max(1, int(cw * 0.42)), max(1, int(ch * 0.42)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    ksize = max(5, int(min(cw, ch) * 0.15) | 1)
    feathered_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)[:, :, np.newaxis]

    base_patch = base_bgr[y1:y2, x1:x2].astype(np.float32)
    restored_patch = restored_crop_bgr.astype(np.float32)

    blended_patch = (restored_patch * feathered_mask) + (base_patch * (1.0 - feathered_mask))
    base_bgr[y1:y2, x1:x2] = np.clip(blended_patch, 0, 255).astype(np.uint8)

    return base_bgr
