import time
import cv2
from typing import Union, List, Tuple
from PIL import Image
import numpy as np
from core.logging import logger
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from ai.restore.gfpgan_engine import gfpgan_engine
from ai.restore.codeformer_engine import codeformer_engine
from ai.restore.face_quality_analyzer import analyze_face_quality
from ai.restore.face_blender import (
    crop_face_with_margin,
    preserve_natural_skin_texture,
    blend_restored_face,
)


from typing import Union, List, Tuple, Any

class FaceRestorerManager:
    """Enterprise Remini-Style Intelligent Face Enhancement Manager."""

    def __init__(self):
        self._face_cascade = None

    def _get_face_cascade(self) -> Any:
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def detect_face_boxes(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects bounding boxes [(x, y, w, h)] for all faces in image."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            cascade = self._get_face_cascade()
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            return [tuple(f) for f in faces]
        except Exception:
            return []

    def has_faces(self, image_input: Union[Image.Image, np.ndarray]) -> bool:
        """Fast face detection pre-check."""
        if isinstance(image_input, Image.Image):
            img_bgr = pillow_to_opencv(image_input)
        else:
            img_bgr = image_input
        boxes = self.detect_face_boxes(img_bgr)
        return len(boxes) > 0

    def restore_face(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes],
        model: str = "auto",
        fidelity: float = 0.7,
        only_center_face: bool = False,
    ) -> Image.Image:
        """
        Executes Remini-Style Intelligent Face Enhancement:
        1. Multi-face bounding box detection
        2. Per-face quality analysis (size, blur, noise, pose, confidence)
        3. Adaptive strategy execution:
           - Excellent (Q >= 0.85): Skip restoration (preserve original sharp identity)
           - Medium (0.65 <= Q < 0.85): GFPGAN (light)
           - Poor (0.40 <= Q < 0.65): CodeFormer with dynamic adaptive fidelity w in [0.5, 0.8]
           - Very Poor (Q < 0.40): GFPGAN pre-pass + CodeFormer post-pass (w = 0.4)
        4. Independent face crop & seamless mask blending
        5. Natural skin texture preservation (re-injects high-pass texture to prevent plastic skin)
        6. Post-restoration quality check & auto-retry fallback
        7. Detailed telemetry logging
        """
        t_start = time.perf_counter()
        if isinstance(image_input, Image.Image):
            base_bgr = pillow_to_opencv(image_input)
        elif isinstance(image_input, np.ndarray):
            base_bgr = image_input.copy()
        else:
            base_bgr = pillow_to_opencv(Image.open(image_input))

        h_img, w_img = base_bgr.shape[:2]
        boxes = self.detect_face_boxes(base_bgr)

        # Fallback to direct model call if no faces detected by Haar cascade or forced non-auto model
        if len(boxes) == 0:
            logger.info("ℹ️ No distinct faces isolated by cascade. Running global face restorer pass...")
            if model.lower() == "codeformer":
                res_pil = codeformer_engine.restore(base_bgr, fidelity=fidelity)
            else:
                res_pil = gfpgan_engine.restore(base_bgr, only_center_face=only_center_face)
            if res_pil.size != (w_img, h_img):
                res_pil = res_pil.resize((w_img, h_img), Image.LANCZOS)
            return res_pil

        logger.info(f"🎭 Remini Face Enhancer isolated {len(boxes)} face(s). Executing intelligent adaptive pipeline...")

        output_bgr = base_bgr.copy()
        strategies_used = []
        initial_qualities = []
        final_qualities = []

        for i, box in enumerate(boxes):
            orig_crop_bgr, crop_coords = crop_face_with_margin(base_bgr, box, margin_ratio=0.30)
            analysis = analyze_face_quality(orig_crop_bgr, (w_img, h_img))
            category = analysis["category"]
            init_q = analysis["quality_score"]
            initial_qualities.append(init_q)
            adaptive_w = analysis["adaptive_fidelity"]

            # Strategy Selection
            if model != "auto" and model in ["gfpgan", "codeformer"]:
                selected_strategy = f"forced ({model})"
                if model == "codeformer":
                    restored_pil = codeformer_engine.restore(orig_crop_bgr, fidelity=fidelity)
                else:
                    restored_pil = gfpgan_engine.restore(orig_crop_bgr)
            elif category == "excellent":
                selected_strategy = "excellent (skipped)"
                restored_pil = opencv_to_pillow(orig_crop_bgr)
            elif category == "medium":
                selected_strategy = "medium (gfpgan)"
                restored_pil = gfpgan_engine.restore(orig_crop_bgr)
            elif category == "poor":
                selected_strategy = f"poor (codeformer w={adaptive_w:.2f})"
                restored_pil = codeformer_engine.restore(orig_crop_bgr, fidelity=adaptive_w)
            else:  # very_poor
                selected_strategy = f"very_poor (gfpgan+codeformer w={adaptive_w:.2f})"
                gfp_pil = gfpgan_engine.restore(orig_crop_bgr)
                restored_pil = codeformer_engine.restore(gfp_pil, fidelity=0.40)

            strategies_used.append(selected_strategy)
            restored_bgr = pillow_to_opencv(restored_pil)

            # Post-Restoration Quality Evaluation
            post_analysis = analyze_face_quality(restored_bgr, (w_img, h_img))
            post_q = post_analysis["quality_score"]

            # Auto-Retry Fallback if quality degraded
            if post_q < init_q and category != "excellent":
                logger.warning(f"⚠️ Face #{i+1} restoration quality dropped ({post_q:.2f} < {init_q:.2f}). Retrying with alternative GFPGAN light pass...")
                retry_pil = gfpgan_engine.restore(orig_crop_bgr)
                retry_bgr = pillow_to_opencv(retry_pil)
                retry_q = analyze_face_quality(retry_bgr, (w_img, h_img))["quality_score"]
                if retry_q > post_q:
                    logger.info(f"✨ Retry succeeded: improved quality score from {post_q:.2f} to {retry_q:.2f}.")
                    restored_bgr = retry_bgr
                    post_q = retry_q

            final_qualities.append(post_q)

            # Natural Skin Texture Preservation (blend 20% original high-pass texture)
            textured_bgr = preserve_natural_skin_texture(orig_crop_bgr, restored_bgr, texture_blend_ratio=0.20)

            # Seamless Mask Blending back into base image
            output_bgr = blend_restored_face(output_bgr, textured_bgr, crop_coords)

        elapsed = round(time.perf_counter() - t_start, 2)
        avg_init = float(np.mean(initial_qualities)) if initial_qualities else 0.0
        avg_final = float(np.mean(final_qualities)) if final_qualities else 0.0

        logger.info(
            f"✨ REMINI FACE ENHANCEMENT COMPLETE | Faces: {len(boxes)} | Strategies: {strategies_used} | "
            f"Processing Time: {elapsed}s | Initial Avg Quality: {avg_init:.2f}/1.00 -> Final Avg Quality: {avg_final:.2f}/1.00"
        )

        return opencv_to_pillow(output_bgr)


# Singleton instance
face_restorer_manager = FaceRestorerManager()
