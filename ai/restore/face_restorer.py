"""
==============================================================================
Zenemoo AI - Unified Face Restorer Manager
==============================================================================
Façade manager coordinating GFPGAN and CodeFormer face restoration models.
"""

from typing import Union
from PIL import Image
import numpy as np
from core.logging import logger
from ai.restore.gfpgan_engine import gfpgan_engine
from ai.restore.codeformer_engine import codeformer_engine


class FaceRestorerManager:
    """Unified Face Restorer Manager."""

    def has_faces(self, image_input: Union[Image.Image, np.ndarray]) -> bool:
        """Delegates fast face detection pre-check to gfpgan_engine."""
        return gfpgan_engine.has_faces(image_input)

    def restore_face(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes],
        model: str = "gfpgan",
        fidelity: float = 0.7,
        only_center_face: bool = False,
    ) -> Image.Image:
        """
        Delegates facial restoration to requested model ('gfpgan' or 'codeformer').
        """
        logger.info(f"🎭 Running face restoration engine [{model.upper()}] (only_center_face={only_center_face})...")

        if model.lower() == "codeformer":
            return codeformer_engine.restore(image_input, fidelity=fidelity)
        else:
            return gfpgan_engine.restore(image_input, only_center_face=only_center_face)


# Singleton instance
face_restorer_manager = FaceRestorerManager()
