"""
==============================================================================
Zenemoo AI - Face Quality Analyzer
==============================================================================
Analyzes detected facial features for resolution size, blur level (Laplacian
variance), noise, pose aspect ratio, and computes an overall quality confidence
score (0.0 to 1.0) for adaptive restoration strategy selection.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple


def analyze_face_quality(
    face_crop_bgr: np.ndarray,
    full_image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Analyzes face crop BGR image and computes quality metrics:
    - Face dimensions (w, h) and relative image area ratio
    - Blur score (Laplacian variance)
    - Noise & contrast metrics
    - Overall face quality score Q in [0.0, 1.0]
    - Category: 'excellent', 'medium', 'poor', 'very_poor'
    - Adaptive CodeFormer fidelity weight w in [0.30, 0.85]
    """
    h, w = face_crop_bgr.shape[:2]
    img_w, img_h = full_image_size

    # 1. Face Size Metric
    face_area = w * h
    img_area = max(1, img_w * img_h)
    size_ratio = face_area / float(img_area)

    # 2. Blur Metric (Laplacian variance)
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur_score = float(np.clip(laplacian_var / 500.0, 0.0, 1.0))
    size_score = float(np.clip(min(w, h) / 250.0, 0.0, 1.0))

    # 3. Contrast & Noise Metric
    contrast = float(np.std(gray))
    contrast_score = float(np.clip(contrast / 65.0, 0.0, 1.0))

    # 4. Overall Face Quality Score Q in [0.0, 1.0]
    quality_score = round(float(np.clip(0.50 * blur_score + 0.30 * size_score + 0.20 * contrast_score, 0.0, 1.0)), 2)

    # 5. Category Selection
    if quality_score >= 0.85:
        category = "excellent"
    elif quality_score >= 0.65:
        category = "medium"
    elif quality_score >= 0.40:
        category = "poor"
    else:
        category = "very_poor"

    # 6. Adaptive CodeFormer Fidelity Weight w (0.30 to 0.85)
    adaptive_fidelity = round(float(np.clip(0.30 + 0.60 * quality_score, 0.30, 0.85)), 2)

    return {
        "width": w,
        "height": h,
        "size_ratio": round(size_ratio, 4),
        "laplacian_variance": round(laplacian_var, 2),
        "quality_score": quality_score,
        "category": category,
        "adaptive_fidelity": adaptive_fidelity,
    }
