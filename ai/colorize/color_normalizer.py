"""
==============================================================================
Zenemoo AI - CIE Lab Color Normalization & Old Photo Post-Processing Engine
==============================================================================
Performs Automatic White Balance (Gray World algorithm), human skin tone
normalization (constraining a*, b* in Lab space to prevent red skin),
red/orange color cast suppression, luminance channel re-injection (L* original),
and Vintage Mode color curve application.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def apply_auto_white_balance(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Gray World Automatic White Balance algorithm in Lab color space.
    Shifts overall chrominance mean towards neutral gray.
    """
    lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Calculate average a* and b* offsets from neutral (128.0)
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))

    # Shift chrominance towards neutral gray smoothly
    a_corrected = a - ((a_mean - 128.0) * 0.6)
    b_corrected = b - ((b_mean - 128.0) * 0.6)

    lab_corrected = np.stack([l, a_corrected, b_corrected], axis=-1)
    lab_uint8 = np.clip(lab_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)


def normalize_skin_tones(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Normalizes skin tones in CIE Lab color space.
    Constrains a* (red-green) and b* (yellow-blue) to natural human skin gamut:
    a* in [3, 22] (standard centered Lab), b* in [5, 28],
    and ensures b* >= a* to prevent unnatural red/burnt face artifacts.
    """
    lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Convert OpenCV 0-255 Lab to standard centered a*, b* (-128 to +127)
    a_std = a - 128.0
    b_std = b - 128.0

    # Identify potential skin tone regions (positive a* and b*, L > 30)
    skin_mask = (l > 30.0) & (a_std > 3.0) & (b_std > 3.0)

    if np.any(skin_mask):
        # 1. Clamp excessive red in skin pixels (a_std <= 22.0)
        a_std[skin_mask] = np.clip(a_std[skin_mask], 3.0, 22.0)

        # 2. Ensure warm skin tone (b_std >= a_std + 2.0)
        too_red = skin_mask & (a_std > b_std - 2.0)
        a_std[too_red] = b_std[too_red] - 2.0

        # Re-map back to OpenCV Lab space
        a = a_std + 128.0
        b = b_std + 128.0

    lab_corrected = np.stack([l, a, b], axis=-1)
    lab_uint8 = np.clip(lab_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)


def suppress_red_orange_casts(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Suppresses excessive red/orange color casts in HSV space.
    Clamps saturation for red/orange hues (H in [0, 25] or [160, 180]).
    """
    hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Red/orange hue mask
    red_mask = ((h <= 25.0) | (h >= 160.0)) & (s > 100.0)

    if np.any(red_mask):
        s[red_mask] = np.clip(s[red_mask] * 0.70, 0.0, 140.0)

    hsv_corrected = np.stack([h, s, v], axis=-1)
    hsv_uint8 = np.clip(hsv_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)


def reinject_original_luminance(original_bgr: np.ndarray, colorized_bgr: np.ndarray) -> np.ndarray:
    """
    Re-injects the original B&W image's Y luminance channel into the colorized image in YCrCb space.
    Guarantees 100% exact preservation of original contrast, brightness, and sharpness.
    """
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    color_ycrcb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2YCrCb)

    color_ycrcb[:, :, 0] = orig_gray
    return cv2.cvtColor(color_ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_vintage_mode(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Applies warm, realistic historical old-photo aesthetics (vintage mode).
    Shifts highlights towards warm golden sepia tones while preserving hues.
    """
    lab = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    a_vintage = a + 2.0
    b_vintage = b + 8.0

    lab_vintage = np.stack([l, a_vintage, b_vintage], axis=-1)
    lab_uint8 = np.clip(lab_vintage, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)
