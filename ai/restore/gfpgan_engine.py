"""
==============================================================================
Zenemoo AI - GFPGAN Face Restoration Engine (Genuine PyTorch Model)
==============================================================================
Executes facial feature restoration using GFPGAN v1.4 deep learning architecture.
Loads pretrained weights from `shared/weights/GFPGAN/GFPGANv1.4.pth`.
"""

import os
import cv2
import shutil
import traceback
from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np
import torch
from core.config import settings
from core.logging import logger
from shared.weights import weights_manager
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class GFPGANRestorerEngine:
    """GFPGAN Face Restoration Engine using official gfpgan package."""

    def __init__(self, upscale: int = 2, arch: str = "clean", channel_multiplier: int = 2):
        self.upscale = upscale
        self.arch = arch
        self.channel_multiplier = channel_multiplier
        self._gfpganer = None
        self._face_cascade = None

    def _ensure_facexlib_models(self) -> None:
        """Pre-verifies and ensures FaceXLib detection & parsing models exist in gfpgan/weights without hidden downloads."""
        gfpgan_weights_dir = settings.BASE_DIR / "gfpgan" / "weights"
        gfpgan_weights_dir.mkdir(parents=True, exist_ok=True)

        aux_models = [
            ("facexlib_detection", "detection_Resnet50_Final.pth"),
            ("facexlib_parsing", "parsing_parsenet.pth"),
        ]

        for key, fname in aux_models:
            target_in_gfpgan = gfpgan_weights_dir / fname
            try:
                master_path = weights_manager.get_weight_path(key)
                if not target_in_gfpgan.exists() or target_in_gfpgan.stat().st_size != master_path.stat().st_size:
                    shutil.copy2(master_path, target_in_gfpgan)
            except Exception as e:
                logger.error(f"❌ Failed ensuring auxiliary model '{fname}': {e}")
                raise AIModelException(f"Missing auxiliary model {fname}: {e}", model_name="GFPGAN")

    def _init_gfpgan(self) -> None:
        """Initializes GFPGANer model runner and loads pretrained weights."""
        if self._gfpganer is None:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._ensure_facexlib_models()

                weight_path = weights_manager.get_weight_path("GFPGAN")
                logger.info(f"🧠 Loading GFPGAN v1.4 PyTorch weights on [{device.type.upper()}] from '{weight_path}'...")

                from gfpgan import GFPGANer

                self._gfpganer = GFPGANer(
                    model_path=str(weight_path),
                    upscale=self.upscale,
                    arch=self.arch,
                    channel_multiplier=self.channel_multiplier,
                    bg_upsampler=None,
                    device=device,
                )
                logger.info(f"✅ Loaded GFPGAN v1.4 model on [{device.type.upper()}].")

            except Exception as e:
                logger.error(f"❌ Failed initializing GFPGAN model: {e}\n{traceback.format_exc()}")
                raise AIModelException(f"Failed initializing GFPGAN v1.4 PyTorch model: {e}", model_name="GFPGAN")

    def has_faces(self, image_input: Union[Image.Image, np.ndarray]) -> bool:
        """Fast OpenCV Haar Cascade pre-check to detect if faces exist before running GFPGAN."""
        try:
            if isinstance(image_input, Image.Image):
                img_bgr = pillow_to_opencv(image_input)
            else:
                img_bgr = image_input

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if self._face_cascade is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            return len(faces) > 0
        except Exception:
            return True  # Fallback to True if check fails

    def restore(
        self,
        image_input: Union[Image.Image, np.ndarray],
        only_center_face: bool = False,
    ) -> Image.Image:
        """
        Restores facial details in image using GFPGAN v1.4 neural network.
        Returns PIL.Image.Image.
        """
        try:
            if isinstance(image_input, Image.Image):
                img_bgr = pillow_to_opencv(image_input)
            else:
                img_bgr = image_input

            self._init_gfpgan()

            logger.info(f"▶ Step 4: Before calling GFPGANer.enhance(only_center_face={only_center_face})...")
            with torch.inference_mode():
                cropped_faces, restored_faces, restored_img = self._gfpganer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=only_center_face,
                    paste_back=True,
                )
            logger.info(f"✅ Step 4: After GFPGANer.enhance() returned successfully. (Faces restored: {len(restored_faces)})")

            if restored_img is None:
                return opencv_to_pillow(img_bgr)

            return opencv_to_pillow(restored_img)

        except Exception as e:
            logger.error(f"❌ GFPGAN face restoration model inference failed: {e}\n{traceback.format_exc()}")
            raise InferenceExecutionException(f"GFPGAN restoration failure: {e}", model_name="GFPGAN")


# Singleton instance
gfpgan_engine = GFPGANRestorerEngine()
