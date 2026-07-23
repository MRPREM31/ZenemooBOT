"""
==============================================================================
Zenemoo AI - Image Classifier for Background Removal Model Selection
==============================================================================
Classifies input image payloads into category types (portrait, product, anime,
general) and maps each to primary and secondary fallback segmentation models.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple


def classify_image_category(pil_img: Image.Image) -> str:
    """
    Classifies image into 'portrait', 'product', 'anime', or 'general'.
    """
    # 1. Check for human faces / person
    try:
        from ai.restore import face_restorer_manager
        if face_restorer_manager.has_faces(pil_img):
            return "portrait"
    except Exception:
        pass

    # 2. Check for anime / illustration characteristics
    img_rgb = np.array(pil_img.convert("RGB"))
    cv_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)

    # Anime images often have strong line art & flat color regions
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.count_nonzero(edges) / float(edges.size)

    small_bgr = cv2.resize(cv_bgr, (64, 64))
    unique_colors = len(np.unique(small_bgr.reshape(-1, 3), axis=0))

    if edge_ratio > 0.08 and unique_colors < 800:
        return "anime"

    # 3. Check for product / object (centered subject with uniform border)
    h, w = gray.shape[:2]
    border_pixels = np.concatenate([
        gray[0:max(1, int(h * 0.08)), :].flatten(),
        gray[min(h - 1, int(h * 0.92)):, :].flatten(),
        gray[:, 0:max(1, int(w * 0.08))].flatten(),
        gray[:, min(w - 1, int(w * 0.92)):].flatten(),
    ])
    border_std = float(np.std(border_pixels))
    if border_std < 25.0:
        return "product"

    return "general"


def select_bg_models(category: str) -> Tuple[str, str]:
    """
    Returns (primary_model_name, fallback_model_name) based on category.
    """
    if category == "portrait":
        # BRIA RMBG 2.0 / U2Net Human Seg for human portraits
        return ("u2net_human_seg", "isnet-general-use")
    elif category == "product":
        # IS-Net for product catalog images
        return ("isnet-general-use", "u2net")
    elif category == "anime":
        # IS-Net Anime / AnimeSeg for anime/cartoons
        return ("isnet-anime", "u2net")
    else:
        # U²-Net for general images
        return ("u2net", "isnet-general-use")
