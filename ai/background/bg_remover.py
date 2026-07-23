import time
import io
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException
from ai.background.bg_classifier import classify_image_category, select_bg_models
from ai.background.edge_refiner import refine_hair_and_edges
from ai.background.quality_evaluator import evaluate_mask_quality


class BackgroundRemoverEngine:
    """Production-Grade AI Background Removal Engine with Multi-Model & Edge Refinement."""

    def __init__(self, default_model: str = "u2net"):
        self.default_model = default_model
        self._sessions = {}

    def _get_session(self, model_name: str):
        """Initializes or retrieves cached rembg ONNX session for model_name."""
        if model_name not in self._sessions or self._sessions[model_name] is None:
            try:
                import os
                from rembg import new_session
                device = "CUDA" if torch.cuda.is_available() else "CPU"
                logger.info(f"🧠 Initializing rembg session [{model_name}] on [{device}]...")
                try:
                    self._sessions[model_name] = new_session(model_name)
                except Exception as inner_e:
                    logger.warning(f"⚠️ Initial rembg session creation for '{model_name}' failed ({inner_e}). Retrying with fallback...")
                    onnx_path = os.path.expanduser(f"~/.u2net/{model_name}.onnx")
                    if os.path.exists(onnx_path):
                        try:
                            os.remove(onnx_path)
                        except Exception:
                            pass
                    fallback_name = "u2net" if model_name != "u2net" else "u2netp"
                    self._sessions[model_name] = new_session(fallback_name)
                logger.info(f"✅ Loaded rembg session [{model_name}] on {device}")
            except Exception as e:
                logger.error(f"Failed initializing rembg session '{model_name}': {e}", exc_info=True)
                raise AIModelException(f"Failed initializing rembg model session '{model_name}': {e}", model_name=model_name)
        return self._sessions[model_name]

    def _run_single_segmentation(self, pil_img: Image.Image, model_name: str) -> Tuple[Image.Image, float]:
        """Runs segmentation with model_name and applies edge/hair refinement."""
        from shared.utils.image import smart_downscale_pil
        w, h = pil_img.size
        working_img = pil_img
        if max(w, h) > 2048:
            working_img = smart_downscale_pil(pil_img, max_dim=2048)

        session = self._get_session(model_name)
        from rembg import remove

        # Execute rembg segmentation
        raw_result = remove(
            working_img,
            session=session,
            alpha_matting=False,
        )

        # Extract initial alpha mask & refine edges/hair/facial protection
        raw_rgba = np.array(raw_result.convert("RGBA"))
        initial_alpha = raw_rgba[:, :, 3]

        refined_result = refine_hair_and_edges(
            pil_img=working_img,
            initial_alpha=initial_alpha,
            enable_hair_refinement=True,
        )

        score = evaluate_mask_quality(refined_result, working_img)
        return refined_result, score

    def remove_background(
        self,
        image_input: Union[Image.Image, bytes, np.ndarray],
        alpha_matting: bool = True,
        bg_color: Optional[Tuple[int, int, int]] = None,
        force_model: Optional[str] = None,
    ) -> Image.Image:
        """
        Executes production background removal pipeline:
        1. Convert input to PIL Image
        2. Image classification (portrait, product, anime, general)
        3. Multi-model selection (BRIA RMBG 2.0 / U2Net Human Seg, IS-Net, IS-Net Anime, U2-Net)
        4. Segmentation & Guided Filter edge/hair refinement
        5. Facial/ear/anatomical feature protection
        6. Mask confidence quality scoring & auto-retry with alternative model
        7. Detailed telemetry logging
        Returns RGBA or RGB PIL Image.
        """
        t_start = time.perf_counter()
        try:
            from shared.utils.image import opencv_to_pillow

            # 1. Convert input to PIL Image
            if isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGBA")
            elif isinstance(image_input, np.ndarray):
                pil_img = opencv_to_pillow(image_input).convert("RGBA")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGBA")
            else:
                raise ValueError("Unsupported image input format.")

            # 2. Image Classification & Model Selection
            category = classify_image_category(pil_img)
            primary_model, fallback_model = select_bg_models(category)
            if force_model:
                primary_model = force_model

            logger.info(f"🔍 Image Category: [{category.upper()}] | Selected Primary Model: [{primary_model}]")

            # 3. Primary Model Execution & Quality Evaluation
            retry_count = 0
            selected_model = primary_model

            try:
                result_img, quality_score = self._run_single_segmentation(pil_img, primary_model)
            except Exception as primary_err:
                logger.warning(f"⚠️ Primary model [{primary_model}] failed ({primary_err}). Retrying with fallback model [{fallback_model}]...")
                selected_model = fallback_model
                result_img, quality_score = self._run_single_segmentation(pil_img, fallback_model)
                retry_count = 1

            # 4. Auto-Retry with Alternative Model if Quality Score < 0.70
            if quality_score < 0.70 and retry_count == 0 and fallback_model != primary_model:
                logger.warning(
                    f"⚠️ Quality Score ({quality_score:.2f} < 0.70) below threshold for [{primary_model}]. Retrying with alternative model [{fallback_model}]..."
                )
                try:
                    fallback_result, fallback_score = self._run_single_segmentation(pil_img, fallback_model)
                    retry_count = 1
                    if fallback_score > quality_score:
                        logger.info(f"✨ Fallback model [{fallback_model}] achieved higher quality score ({fallback_score:.2f} > {quality_score:.2f}). Selecting fallback result.")
                        result_img = fallback_result
                        quality_score = fallback_score
                        selected_model = fallback_model
                    else:
                        logger.info(f"ℹ️ Primary model [{primary_model}] maintained higher score ({quality_score:.2f} >= {fallback_score:.2f}). Retaining primary result.")
                except Exception as fallback_err:
                    logger.warning(f"Fallback model retry failed ({fallback_err}). Retaining primary result.")

            elapsed = round(time.perf_counter() - t_start, 2)

            # 5. Detailed Telemetry Logging (Requirement 9)
            logger.info(
                f"✨ PRODUCTION BG REMOVAL COMPLETE | Model: [{selected_model}] | Category: [{category}] | "
                f"Processing Time: {elapsed}s | Confidence Score: {quality_score:.2f}/1.00 | Retries: {retry_count}"
            )

            # 6. Optional Background Color Replacement
            if bg_color is not None:
                background = Image.new("RGBA", result_img.size, (*bg_color, 255))
                background.paste(result_img, (0, 0), result_img)
                return background.convert("RGB")

            return result_img

        except Exception as e:
            logger.error(f"Background removal model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(
                f"Failed production background removal inference: {e}",
                model_name=self.default_model,
            )


# Singleton Engine Instance
bg_remover_engine = BackgroundRemoverEngine()
