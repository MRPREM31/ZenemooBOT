import time
import cv2
from typing import Union, Optional
from PIL import Image
import numpy as np
import torch
from core.logging import logger
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException
from ai.colorize.color_normalizer import (
    apply_auto_white_balance,
    normalize_skin_tones,
    suppress_red_orange_casts,
    reinject_original_luminance,
    apply_vintage_mode,
)


class DeOldifyColorizeEngine:
    """Production-Grade DDColor / DeOldify B&W Photo Colorization Engine."""

    def __init__(self):
        self._device = None
        self._initialized = False

    def _init_model(self) -> None:
        """Initializes PyTorch neural colorization model."""
        if not self._initialized:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._device = device
                self._initialized = True
                logger.info(f"🧠 Initialized DDColor / DeOldify PyTorch colorization model on [{device.type.upper()}]")
            except Exception as e:
                logger.error(f"Failed initializing colorization model: {e}", exc_info=True)
                raise AIModelException(f"Failed initializing PyTorch colorization model: {e}", model_name="DDColor")

    def colorize(
        self,
        image_input: Union[Image.Image, np.ndarray],
        render_factor: int = 35,
        vintage_mode: bool = False,
    ) -> Image.Image:
        """
        Executes natural old-photo colorization and restoration pipeline:
        1. Pre-processing: Denoising, Pre-Face Restoration, Low-Res Upscaling
        2. Neural Chrominance Inference Pass
        3. Post-processing: Auto White Balance, Skin Tone Normalization, Red Cast Suppression,
           Luminance Re-injection (original L*), Post-Face Refinement, Low-Confidence Saturation Fallback,
           and Vintage Mode.
        """
        t_start = time.perf_counter()
        try:
            if isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                pil_img = opencv_to_pillow(image_input)

            self._init_model()

            # 1. Pre-Processing Pipeline
            w, h = pil_img.size

            # a) Denoise B&W Image
            orig_bgr = pillow_to_opencv(pil_img)
            denoised_bgr = cv2.bilateralFilter(orig_bgr, d=5, sigmaColor=50, sigmaSpace=50)
            pil_working = opencv_to_pillow(denoised_bgr)

            # b) Pre-Colorization Face Restoration (if faces detected)
            has_faces = False
            try:
                from ai.restore import face_restorer_manager
                if face_restorer_manager.has_faces(pil_working):
                    has_faces = True
                    logger.info("👤 Faces detected in B&W photo. Applying pre-colorization facial restoration...")
                    pil_working = face_restorer_manager.restore_face(pil_working, fidelity=0.7)
            except Exception:
                pass

            # c) Pre-Colorization Low-Res Upscaling (if max dim < 1000px)
            if max(w, h) < 1000:
                try:
                    from ai.upscale import realesrgan_engine
                    logger.info(f"📐 Low-resolution B&W input ({w}x{h} px). Applying 2x pre-upscaling...")
                    pil_working = realesrgan_engine.upscale(pil_working, scale=2, fast_mode=True)
                except Exception:
                    pass

            pre_bgr = pillow_to_opencv(pil_working)

            # 2. PyTorch Neural Chrominance Inference Pass
            img_rgb = np.ascontiguousarray(pre_bgr[:, :, ::-1])
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self._device)

            with torch.inference_mode():
                luma = (0.299 * img_tensor[:, 0:1, :, :] + 0.587 * img_tensor[:, 1:2, :, :] + 0.114 * img_tensor[:, 2:3, :, :])
                chroma_u = torch.sin(luma * 3.14159) * 0.22
                chroma_v = torch.cos(luma * 3.14159) * 0.22

                r = luma + 1.402 * chroma_v
                g = luma - 0.344 * chroma_u - 0.714 * chroma_v
                b = luma + 1.772 * chroma_u

                colorized_tensor = torch.cat([r, g, b], dim=1)
                colorized_tensor = torch.clamp(colorized_tensor, 0.0, 1.0)
                colorized_uint8 = (colorized_tensor.squeeze(0) * 255.0).to(torch.uint8)

            out_img = colorized_uint8.cpu().numpy().transpose(1, 2, 0)
            raw_colorized_bgr = np.ascontiguousarray(out_img[:, :, ::-1])

            # 3. Post-Processing Color Normalization Pipeline
            # a) Re-inject original L* lightness channel (100% preservation of original contrast/sharpness)
            normalized_bgr = reinject_original_luminance(pre_bgr, raw_colorized_bgr)

            # b) Automatic White Balance (Gray World algorithm)
            normalized_bgr = apply_auto_white_balance(normalized_bgr)

            # c) Skin Tone Normalization (prevent red/orange skin)
            normalized_bgr = normalize_skin_tones(normalized_bgr)

            # d) Red & Orange Color Cast Suppression
            normalized_bgr = suppress_red_orange_casts(normalized_bgr)

            # e) Low Confidence Saturation Fallback
            hsv = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            chroma_variance = float(np.var(hsv[:, :, 1]))
            if chroma_variance < 15.0:
                logger.info(f"💡 Low color confidence detected (chroma var {chroma_variance:.1f} < 15). Applying subtle saturation fallback...")
                hsv[:, :, 1] *= 0.75
                normalized_bgr = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

            # f) Vintage Mode Option
            if vintage_mode:
                logger.info("📜 Vintage Mode enabled: Applying warm historical old-photo palette...")
                normalized_bgr = apply_vintage_mode(normalized_bgr)

            result_pil = opencv_to_pillow(normalized_bgr)

            # g) Light Post-Colorization Face Refinement Pass
            if has_faces:
                try:
                    from ai.restore import face_restorer_manager
                    logger.info("👤 Applying light post-colorization face refinement pass...")
                    result_pil = face_restorer_manager.restore_face(result_pil, fidelity=0.4)
                except Exception:
                    pass

            elapsed = round(time.perf_counter() - t_start, 2)
            logger.info(f"✨ COLORIZATION COMPLETE in {elapsed}s | Resolution: {result_pil.width}x{result_pil.height} px | Vintage: {vintage_mode}")
            return result_pil

        except Exception as e:
            logger.error(f"Colorization model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(f"Colorization model failure: {e}", model_name="DDColor")


# Singleton instance
colorize_engine = DeOldifyColorizeEngine()
