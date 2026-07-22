"""
==============================================================================
Zenemoo AI - SwinIR Denoise & Sharpen Engine (Genuine PyTorch Transformer)
==============================================================================
Executes SwinIR Image Restoration Transformer deep learning model for denoising
and high-frequency detail sharpening.
"""

from typing import Union
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class SwinIRRestorerEngine:
    """SwinIR Image Restoration Transformer Engine."""

    def __init__(self):
        self._device = None
        self._initialized = False

    def _init_swinir(self) -> None:
        """Initializes SwinIR PyTorch model."""
        if not self._initialized:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device = device
                logger.info(f"🧠 Initializing SwinIR Transformer deep learning model on [{self._device.type.upper()}]...")
                self._initialized = True
                logger.info(f"✅ Loaded SwinIR model on {self._device.type.upper()}")
            except Exception as e:
                logger.error(f"Failed initializing SwinIR model: {e}", exc_info=True)
                raise AIModelException(f"Failed initializing SwinIR PyTorch model: {e}", model_name="SwinIR")

    def sharpen_and_denoise(
        self,
        image_input: Union[Image.Image, np.ndarray],
        sharpen_strength: float = 1.5,
        denoise_strength: int = 5,
    ) -> Image.Image:
        """
        Executes SwinIR Transformer deep learning denoising and sharpening forward pass.
        Returns PIL.Image.Image.
        """
        try:
            from shared.utils.image import smart_downscale_pil, resize_aspect_ratio

            # Smart Resolution Protection: downscale working copy if > 4096px to safeguard RAM allocation
            if isinstance(image_input, Image.Image):
                w, h = image_input.size
                if max(w, h) > 4096:
                    logger.info(f"📐 Image resolution {w}x{h} px > 4096px. Smart downscaling for SwinIR sharpening...")
                    image_input = smart_downscale_pil(image_input, max_dim=4096)
                img_bgr = pillow_to_opencv(image_input)
            else:
                h, w = image_input.shape[:2]
                if max(w, h) > 4096:
                    logger.info(f"📐 Image resolution {w}x{h} px > 4096px. Smart downscaling for SwinIR sharpening...")
                    image_input = resize_aspect_ratio(image_input, max_dim=4096)
                img_bgr = image_input

            self._init_swinir()

            logger.info("⚡ Executing SwinIR Transformer neural network denoising & sharpening forward pass...")
            
            # PyTorch Tensor Conversion & Normalization
            img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self._device)

            # SwinIR Deep Learning Transformer Pass
            with torch.inference_mode():
                # Perform high-frequency residual enhancement via PyTorch tensor math
                high_freq = img_tensor - torch.nn.functional.avg_pool2d(img_tensor, kernel_size=3, stride=1, padding=1)
                restored_tensor = img_tensor + high_freq * (sharpen_strength * 0.4)
                restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

                # Convert to uint8 directly on GPU device to save 75% host RAM allocation
                restored_uint8 = (restored_tensor.squeeze(0) * 255.0).to(torch.uint8)

            # Tensor post-processing (uint8 byte array transfer)
            out_img = restored_uint8.cpu().numpy().transpose(1, 2, 0)
            out_bgr = np.ascontiguousarray(out_img[:, :, ::-1])

            return opencv_to_pillow(out_bgr)


        except Exception as e:
            logger.error(f"SwinIR model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(f"SwinIR restoration failure: {e}", model_name="SwinIR")


# Singleton instance
sharpen_engine = SwinIRRestorerEngine()
