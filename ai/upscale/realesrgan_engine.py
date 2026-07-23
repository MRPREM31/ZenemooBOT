"""
==============================================================================
Zenemoo AI - Real-ESRGAN Super Resolution Engine (Genuine PyTorch Model)
==============================================================================
Executes 2x and 4x super-resolution image upscaling using Real-ESRGAN deep learning models.
Optimized for RTX 3050 6GB GPU with dynamic VRAM-adaptive tile size calculation,
FP16 half-precision, torch.inference_mode(), and automatic OOM fallback retry loop.
"""

from typing import Union, Tuple
from PIL import Image
import numpy as np
import torch
from core.config import settings
from core.logging import logger
from shared.weights import weights_manager
from shared.utils.gpu import empty_gpu_cache, get_gpu_info
from shared.utils.image import pillow_to_opencv, opencv_to_pillow
from shared.exceptions.ai_exception import InferenceExecutionException, AIModelException


class RealESRGANEngine:
    """Real-ESRGAN Super Resolution Engine using official realesrgan & basicsr packages."""

    def __init__(self):
        self._models = {}

    def _init_model(self, scale: int = 4) -> None:
        """Initializes RealESRGANer model runner once and loads pretrained weights."""
        model_key = "RealESRGAN_x4" if scale == 4 else "RealESRGAN_x2"
        if model_key not in self._models:
            try:
                weight_path = weights_manager.get_weight_path(model_key)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"🧠 Loading Real-ESRGAN {scale}x PyTorch weights on [{device.type.upper()}] from '{weight_path}'...")
                
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                # Standard RealESRGAN RRDBNet architecture parameters
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=scale,
                )

                # Initialize RealESRGANer with FP16 half precision on CUDA
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=str(weight_path),
                    model=model,
                    tile=0,  # Tile size dynamically set per inference call
                    tile_pad=10,
                    pre_pad=0,
                    half=torch.cuda.is_available(),
                    device=device,
                )
                
                self._models[model_key] = upsampler
                logger.info(f"✅ Loaded Real-ESRGAN {scale}x model on {device.type.upper()} (FP16={torch.cuda.is_available()})")
            except Exception as e:
                logger.error(f"Failed initializing Real-ESRGAN model ({model_key}): {e}", exc_info=True)
                raise AIModelException(f"Failed initializing Real-ESRGAN {scale}x PyTorch model: {e}", model_name="Real-ESRGAN")

    def _determine_best_tile_size(self, image_shape: Tuple[int, int]) -> int:
        """Phase 5: Chooses Real-ESRGAN tile size dynamically based on free VRAM."""
        if not torch.cuda.is_available():
            return 512

        from shared.utils.gpu import get_extended_gpu_telemetry
        telemetry = get_extended_gpu_telemetry()
        free_mb = telemetry.get("free_vram_mb", 4000.0)

        # Phase 5 thresholds
        if free_mb > 5000:
            best_tile = 1024
        elif free_mb > 4000:
            best_tile = 768
        elif free_mb > 3000:
            best_tile = 512
        elif free_mb > 2000:
            best_tile = 256
        else:
            best_tile = 256

        return best_tile

    def upscale(
        self,
        image_input: Union[Image.Image, np.ndarray],
        scale: int = 4,
        fast_mode: bool = False,
    ) -> Image.Image:
        """
        Upscales image by scale factor (2 or 4) using Real-ESRGAN neural network.
        Returns PIL.Image.Image.
        """
        try:
            if isinstance(image_input, Image.Image):
                img_bgr = pillow_to_opencv(image_input)
            else:
                img_bgr = image_input

            self._init_model(scale)
            model_key = "RealESRGAN_x4" if scale == 4 else "RealESRGAN_x2"
            upsampler = self._models[model_key]

            h, w = img_bgr.shape[:2]
            mp = (w * h) / 1_000_000.0

            # Dynamic tile calculation & parameters
            tile_size = self._determine_best_tile_size((h, w))
            tile_pad = 6 if fast_mode else 10
            pre_pad = 0
            batch_size = 1
            precision = "FP16 (Half)" if torch.cuda.is_available() else "FP32 (Float)"

            # Set upsampler tile parameters
            upsampler.tile_size = tile_size
            upsampler.tile_pad = tile_pad
            upsampler.pre_pad = pre_pad

            # Compute tile count
            num_h = int(np.ceil(h / tile_size))
            num_w = int(np.ceil(w / tile_size))
            total_tiles = num_h * num_w

            logger.info("=" * 65)
            logger.info(f"📐 Original Image Resolution: {w}x{h} px ({mp:.2f} MP)")
            logger.info(f"🧩 Real-ESRGAN Tile Size: {tile_size}x{tile_size} px | Tile Pad: {tile_pad} | Pre-Pad: {pre_pad} | Batch: {batch_size}")
            logger.info(f"💡 Tiling Rationale: Image ({w}x{h}) is split into {total_tiles} tile(s) ({num_w} horizontal x {num_h} vertical) to prevent GPU VRAM OOM while maximizing RTX 3050 parallel compute efficiency.")
            logger.info("=" * 65)

            # Execution loop with adaptive OOM fallback
            tile_candidates = [tile_size]
            for fallback_tile in [768, 512, 256]:
                if fallback_tile < tile_size:
                    tile_candidates.append(fallback_tile)

            output_bgr = None
            for current_tile in tile_candidates:
                try:
                    upsampler.tile_size = current_tile
                    logger.info(f"⚡ Executing RealESRGANer.enhance() {scale}x forward pass (tile={current_tile})...")
                    
                    with torch.inference_mode():
                        output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
                    
                    if output_bgr is not None:
                        break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                    if "out of memory" in str(err).lower() or isinstance(err, torch.cuda.OutOfMemoryError):
                        empty_gpu_cache()
                        logger.warning(f"⚠️ Real-ESRGAN OOM with tile={current_tile}. Retrying with smaller tile size...")
                        continue
                    else:
                        raise err

            if output_bgr is None:
                raise InferenceExecutionException("Real-ESRGAN enhance returned None image output.", model_name="Real-ESRGAN")

            return opencv_to_pillow(output_bgr)

        except Exception as e:
            logger.error(f"Real-ESRGAN super-resolution model inference failed: {e}", exc_info=True)
            raise InferenceExecutionException(f"Real-ESRGAN upscaling failure: {e}", model_name="Real-ESRGAN")


# Singleton instance
realesrgan_engine = RealESRGANEngine()
