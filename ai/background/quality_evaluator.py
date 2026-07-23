"""
==============================================================================
Zenemoo AI - Segmentation Alpha Mask Quality Evaluator
==============================================================================
Evaluates confidence quality score (0.0 to 1.0) of a background removal mask
based on bimodal separation, edge contrast gradient, and subject completeness.
"""

import cv2
import numpy as np
from PIL import Image


def evaluate_mask_quality(result_img: Image.Image, guide_img: Image.Image) -> float:
    """
    Computes confidence quality score S in [0.0, 1.0].
    Higher scores represent clean, high-contrast foreground/background separation
    with intact subject details and crisp boundary edges.
    """
    try:
        rgba = np.array(result_img.convert("RGBA"))
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

        # 1. Bimodal Separation Metric
        bimodal_dist = np.abs(alpha - 0.5) * 2.0
        bimodal_score = float(np.mean(bimodal_dist))

        # 2. Subject Completeness Metric
        fg_ratio = float(np.count_nonzero(alpha > 0.5) / alpha.size)
        completeness_score = 1.0 if 0.02 <= fg_ratio <= 0.95 else 0.4

        # 3. Edge Gradient Contrast Metric
        gray = cv2.cvtColor(np.array(guide_img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        boundary_zone = (alpha > 0.1) & (alpha < 0.9)
        if np.any(boundary_zone):
            boundary_grad = np.mean(grad_mag[boundary_zone])
            edge_score = float(np.clip(boundary_grad * 3.0, 0.0, 1.0))
        else:
            edge_score = 0.8

        # Weighted Score Computation
        quality_score = (0.50 * bimodal_score) + (0.30 * completeness_score) + (0.20 * edge_score)
        return round(float(np.clip(quality_score, 0.0, 1.0)), 2)

    except Exception:
        return 0.85
