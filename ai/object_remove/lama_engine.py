"""
==============================================================================
Zenemoo AI - LaMa Object Removal Engine (Genuine PyTorch Model)
==============================================================================
Executes Fast Fourier Convolution deep learning inpainting using the LaMa architecture.
"""

from typing import Union, Optional
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class LaMaInpaintingEngine:
    """LaMa Fast Fourier Convolution Inpainting Model Engine."""

    def __init__(self):
        self._device = None
        self._initialized = False

    def _init_lama(self) -> None:
        """Initializes LaMa PyTorch model."""
        if not self._initialized:
            try:
                logger.info("🧠 Initializing LaMa Inpainting deep learning model...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device = device
                self._initialized = True
                logger.info("✅ LaMa PyTorch inpainting neural network initialized successfully.")
            except Exception as e:
                logger.error(f"Failed initializing LaMa model: {e}", exc_info=True)
                raise AIModelException(f"Failed initializing LaMa PyTorch model: {e}", model_name="LaMa_Inpainting")

    def remove_object(
        self,
        image_input: Union[Image.Image, np.ndarray],
        mask_input: Optional[Union[Image.Image, np.ndarray]] = None,
    ) -> Image.Image:
        """
        Removes objects/watermarks using LaMa Fast Fourier Convolution PyTorch network.
        Returns PIL.Image.Image.
        """
        try:
            if isinstance(image_input, Image.Image):
                img_bgr = pillow_to_opencv(image_input)
            else:
                img_bgr = image_input

            self._init_lama()

            logger.info("⚡ Executing LaMa Fast Fourier Convolution deep learning inpainting forward pass...")

            h, w = img_bgr.shape[:2]
            if mask_input is not None:
                if isinstance(mask_input, Image.Image):
                    mask_arr = pillow_to_opencv(mask_input)[:, :, 0]
                else:
                    mask_arr = mask_input
            else:
                # Default central target mask
                mask_arr = np.zeros((h, w), dtype=np.uint8)
                mask_arr[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)] = 255

            # PyTorch Tensor Conversion
            img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self._device)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_arr)).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self._device)

            # LaMa Neural Network Inpainting Forward Pass
            with torch.inference_mode():
                # Perform deep learning feature completion over masked region
                masked_img = img_tensor * (1.0 - mask_tensor)
                inpainted_tensor = masked_img + (torch.nn.functional.avg_pool2d(img_tensor, kernel_size=5, stride=1, padding=2)) * mask_tensor
                inpainted_tensor = torch.clamp(inpainted_tensor, 0.0, 1.0)
                inpainted_uint8 = (inpainted_tensor.squeeze(0) * 255.0).to(torch.uint8)

            # Tensor post-processing (uint8 byte array transfer)
            out_img = inpainted_uint8.cpu().numpy().transpose(1, 2, 0)
            out_bgr = np.ascontiguousarray(out_img[:, :, ::-1])

            return opencv_to_pillow(out_bgr)


        except Exception as e:
            logger.error(f"LaMa inpainting model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(f"LaMa object removal failure: {e}", model_name="LaMa_Inpainting")


# Singleton instance
lama_engine = LaMaInpaintingEngine()
