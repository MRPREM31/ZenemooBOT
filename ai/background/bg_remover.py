"""
==============================================================================
Zenemoo AI - Background Removal AI Engine (Genuine rembg / U²-Net)
==============================================================================
Executes U²-Net deep learning model inference for automatic background removal,
foreground segmentation, and alpha matting using the official `rembg` package.
"""

import io
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class BackgroundRemoverEngine:
    """Enterprise AI Background Removal Engine using rembg / U2-Net."""

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        """Initializes rembg ONNX session."""
        if self._session is None:
            try:
                import os
                from rembg import new_session
                device = "CUDA" if torch.cuda.is_available() else "CPU"
                logger.info(f"🧠 Initializing rembg U²-Net deep learning session [{self.model_name}] on [{device}]...")
                try:
                    self._session = new_session(self.model_name)
                except Exception as inner_e:
                    logger.warning(f"⚠️ Initial rembg session creation failed ({inner_e}). Checking for corrupted model file...")
                    u2net_path = os.path.expanduser(f"~/.u2net/{self.model_name}.onnx")
                    if os.path.exists(u2net_path):
                        os.remove(u2net_path)
                        logger.info(f"🗑️ Deleted corrupted ONNX file at '{u2net_path}'. Retrying session...")
                    self._session = new_session(self.model_name)
                logger.info(f"✅ Loaded rembg U²-Net model on {device}")
            except Exception as e:
                logger.error(f"Failed initializing rembg session: {e}", exc_info=True)
                raise AIModelException(f"Failed initializing rembg U²-Net model session: {e}", model_name="rembg_u2net")
        return self._session

    def remove_background(
        self,
        image_input: Union[Image.Image, bytes, np.ndarray],
        alpha_matting: bool = True,
        bg_color: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """
        Removes background from image using U²-Net deep learning model.
        Automatically resizes images > 2048px and disables alpha matting for images > 1920px
        to prevent MemoryError on high-resolution inputs.
        Returns PIL.Image.Image in RGBA or RGB format.
        """
        try:
            from shared.utils.image import smart_downscale_pil, opencv_to_pillow

            # 1. Convert input to PIL Image
            if isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGBA")
            elif isinstance(image_input, np.ndarray):
                pil_img = opencv_to_pillow(image_input).convert("RGBA")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGBA")
            else:
                raise ValueError("Unsupported image input format.")

            # 2. Smart Pre-Resize to max 2048px to safeguard VRAM/RAM allocation
            w, h = pil_img.size
            if max(w, h) > 2048:
                logger.info(f"📐 Large image detected ({w}x{h} px). Downscaling to max 2048px working copy for background removal...")
                pil_img = smart_downscale_pil(pil_img, max_dim=2048)
                w, h = pil_img.size

            # 3. Auto-Disable Alpha Matting for images > 1920px to prevent 1.86GB+ RAM spikes
            effective_alpha_matting = alpha_matting
            if max(w, h) > 1920 and alpha_matting:
                logger.info(f"💡 Resolution {w}x{h} px > 1920px. Disabling alpha_matting to prevent MemoryError...")
                effective_alpha_matting = False

            # 4. Execute rembg deep learning neural network inference with retry fallback
            session = self._get_session()
            from rembg import remove

            try:
                logger.info(f"⚡ Executing rembg.remove() U²-Net ({w}x{h} px, alpha_matting={effective_alpha_matting})...")
                result_img = remove(
                    pil_img,
                    session=session,
                    alpha_matting=effective_alpha_matting,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                )
            except (MemoryError, Exception) as mem_err:
                if effective_alpha_matting:
                    logger.warning(f"⚠️ rembg failed with alpha_matting=True ({mem_err}). Retrying with alpha_matting=False & downscaled image...")
                    pil_img = smart_downscale_pil(pil_img, max_dim=1536)
                    result_img = remove(
                        pil_img,
                        session=session,
                        alpha_matting=False,
                    )
                else:
                    raise mem_err

            # 5. Optional Background Color Replacement
            if bg_color is not None:
                background = Image.new("RGBA", result_img.size, (*bg_color, 255))
                background.paste(result_img, (0, 0), result_img)
                return background.convert("RGB")

            return result_img

        except Exception as e:
            logger.error(f"Background removal model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(
                f"Failed rembg background removal inference: {e}",
                model_name="rembg_u2net"
            )


# Singleton Engine Instance
bg_remover_engine = BackgroundRemoverEngine()
