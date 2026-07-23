"""
==============================================================================
Zenemoo AI - Intelligent Image Analyzer
==============================================================================
Analyzes incoming image properties (resolution, Megapixels, face detection,
alpha transparency, sharpness Laplacian variance) and calculates estimated
VRAM consumption before GPU pipeline scheduling.
"""

import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Union


def analyze_image_properties(image_input: Union[bytes, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """
    Analyzes input image payload for dimensions, megapixels, transparency,
    face presence, sharpness variance, and estimated VRAM footprint.
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        pil_img = image_input

    width, height = pil_img.size
    megapixels = (width * height) / 1_000_000.0

    has_alpha = pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info)

    # Convert to OpenCV BGR for face & sharpness check
    if pil_img.mode != "RGB":
        rgb_img = pil_img.convert("RGB")
    else:
        rgb_img = pil_img

    cv_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

    # Sharpness check via Laplacian variance
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = laplacian_var > 300.0

    # Face detection check
    try:
        from ai.restore import face_restorer_manager
        num_faces = 1 if face_restorer_manager.has_faces(pil_img) else 0
    except Exception:
        num_faces = 0

    return {
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 2),
        "has_transparency": has_alpha,
        "is_sharp": is_sharp,
        "laplacian_variance": round(laplacian_var, 2),
        "num_faces": num_faces,
        "has_faces": num_faces > 0,
    }


def estimate_vram_requirement_gb(
    resolution_mp: float,
    face_restore: bool = False,
    upscale_factor: int = 1,
    remove_bg: bool = False,
    sharpen: bool = False,
) -> float:
    """
    Estimates expected VRAM usage in GB for a given AI job configuration.
    """
    base_vram = 0.5  # PyTorch context base
    if face_restore:
        base_vram += 1.2
    if upscale_factor > 1:
        base_vram += 1.5 if upscale_factor == 4 else 1.0
    if remove_bg:
        base_vram += 0.8
    if sharpen:
        base_vram += 1.0

    scaled_vram = base_vram + (resolution_mp * 0.25)
    return round(scaled_vram, 2)
