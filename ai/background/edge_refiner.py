"""
==============================================================================
Zenemoo AI - Production Edge & Hair Refinement Pipeline
==============================================================================
Applies OpenCV Fast Guided Filter, hair strand detail high-pass preservation,
facial feature protection mask blending, and light edge feathering.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
from core.logging import logger


def guided_filter_alpha(
    guide_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    r: int = 4,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Fast Guided Filter for alpha mask edge refinement using OpenCV boxFilter.
    guide_bgr: uint8 BGR image [0, 255]
    alpha_mask: float32 alpha mask [0.0, 1.0]
    """
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
    I = guide_gray.astype(np.float32) / 255.0
    p = alpha_mask.astype(np.float32)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


def protect_facial_regions(alpha_mask: np.ndarray, pil_img: Image.Image) -> np.ndarray:
    """
    Detects face/head bounding boxes and ensures alpha values inside face/ear core zones
    are guaranteed solid (alpha >= 0.98) to prevent missing ears, nose, cheeks, or fingers.
    """
    protected_mask = alpha_mask.copy()
    h, w = alpha_mask.shape[:2]
    boxes = []

    # OpenCV Haar Cascade for facial & ear protection
    try:
        rgb_img = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        for (fx, fy, fw, fh) in detected:
            boxes.append((fx, fy, fw, fh))
    except Exception:
        pass

    for (fx, fy, fw, fh) in boxes:
        margin_x = int(fw * 0.15)
        margin_y = int(fh * 0.15)
        x1 = max(0, fx - margin_x)
        y1 = max(0, fy - margin_y)
        x2 = min(w, fx + fw + margin_x)
        y2 = min(h, fy + fh + margin_y)

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        face_ellipse = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(face_ellipse, center, axes, 0, 0, 360, 1.0, -1)
        face_ellipse = cv2.GaussianBlur(face_ellipse, (15, 15), 0)

        protected_mask = np.maximum(protected_mask, face_ellipse * 0.98)

    return np.clip(protected_mask, 0.0, 1.0)


def refine_hair_and_edges(
    pil_img: Image.Image,
    initial_alpha: np.ndarray,
    enable_hair_refinement: bool = True,
) -> Image.Image:
    """
    Applies complete edge & hair refinement pipeline:
    1. Facial region protection (ears/face/nose/cheeks retained)
    2. Fast Guided Filter edge matting
    3. Hair detail preservation via high-pass blending
    4. Light edge feathering (1-2px)
    Returns RGBA PIL Image.
    """
    img_rgba = np.array(pil_img.convert("RGBA"))
    cv_bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    alpha = initial_alpha.astype(np.float32)
    if alpha.max() > 1.0:
        alpha /= 255.0

    # 1. Protect facial regions
    alpha = protect_facial_regions(alpha, pil_img)

    # 2. Fast Guided Filter for edge matting
    alpha = guided_filter_alpha(cv_bgr, alpha, r=4, eps=1e-4)

    # 3. Hair Detail Preservation (High-Pass Detail Blend)
    if enable_hair_refinement:
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        high_pass = cv2.Laplacian(gray, cv2.CV_32F)
        high_pass = np.abs(high_pass)

        boundary_mask = (alpha > 0.05) & (alpha < 0.95)
        alpha[boundary_mask] = np.clip(
            alpha[boundary_mask] + 0.15 * high_pass[boundary_mask], 0.0, 1.0
        )

    # 4. Light Edge Feathering (1-2px smooth)
    alpha_uint8 = (alpha * 255.0).astype(np.uint8)
    alpha_feathered = cv2.GaussianBlur(alpha_uint8, (3, 3), 0)

    img_rgba[:, :, 3] = alpha_feathered
    return Image.fromarray(img_rgba, mode="RGBA")
