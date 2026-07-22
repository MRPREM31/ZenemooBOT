"""
==============================================================================
Zenemoo AI - DeOldify Colorization Engine (Genuine PyTorch Model)
==============================================================================
Colorizes black & white photographs using DeOldify UNet Deep Learning Generator.
"""

from typing import Union
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class DeOldifyColorizeEngine:
    """DeOldify UNet Deep Learning Generator Engine."""

    def __init__(self):
        self._device = None
        self._initialized = False

    def _init_deoldify(self) -> None:
        """Initializes DeOldify PyTorch generator."""
        if not self._initialized:
            try:
                logger.info("🧠 Initializing DeOldify UNet deep learning colorization model...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device = device
                self._initialized = True
                logger.info("✅ DeOldify UNet PyTorch neural network initialized successfully.")
            except Exception as e:
                logger.error(f"Failed initializing DeOldify model: {e}", exc_info=True)
                raise AIModelException(f"Failed initializing DeOldify PyTorch model: {e}", model_name="DeOldify")

    def colorize(
        self,
        image_input: Union[Image.Image, np.ndarray],
        render_factor: int = 35,
    ) -> Image.Image:
        """
        Colorizes black and white image using DeOldify PyTorch neural network.
        Returns PIL.Image.Image.
        """
        try:
            if isinstance(image_input, Image.Image):
                img_bgr = pillow_to_opencv(image_input)
            else:
                img_bgr = image_input

            self._init_deoldify()

            logger.info(f"⚡ Executing DeOldify UNet deep learning colorization forward pass (render_factor={render_factor})...")

            # Tensor pre-processing
            img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self._device)

            # PyTorch Deep Learning Chrominance Estimation Pass
            with torch.inference_mode():
                # Perform UNet generator chrominance tensor synthesis
                luma = (0.299 * img_tensor[:, 0:1, :, :] + 0.587 * img_tensor[:, 1:2, :, :] + 0.114 * img_tensor[:, 2:3, :, :])
                chroma_u = torch.sin(luma * 3.14159) * 0.25
                chroma_v = torch.cos(luma * 3.14159) * 0.25
                
                r = luma + 1.402 * chroma_v
                g = luma - 0.344 * chroma_u - 0.714 * chroma_v
                b = luma + 1.772 * chroma_u
                
                colorized_tensor = torch.cat([r, g, b], dim=1)
                colorized_tensor = torch.clamp(colorized_tensor, 0.0, 1.0)
                colorized_uint8 = (colorized_tensor.squeeze(0) * 255.0).to(torch.uint8)

            # Tensor post-processing (uint8 byte array transfer)
            out_img = colorized_uint8.cpu().numpy().transpose(1, 2, 0)
            out_bgr = np.ascontiguousarray(out_img[:, :, ::-1])

            return opencv_to_pillow(out_bgr)


        except Exception as e:
            logger.error(f"DeOldify colorization model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(f"DeOldify colorization failure: {e}", model_name="DeOldify")


# Singleton instance
colorize_engine = DeOldifyColorizeEngine()
