"""
==============================================================================
Zenemoo AI - Smart Output Optimization Engine
==============================================================================
High-performance, quality-preserving image optimization stage.
Executes JPEG (quality 95-98, 4:4:4 chroma subsampling, progressive, optimize=True),
PNG (lossless compress_level=9 + oxipng/optipng if available), and WebP (lossless/quality 95, method=6).
Includes automatic transparency detection and 10MB PNG -> JPEG/WebP fallback guard.
"""

import io
import os
import time
import subprocess
import shutil
from typing import Union, Dict, Any, Tuple, Optional
from PIL import Image
from core.logging import logger
from shared.exceptions.image_exception import ImageProcessingException


class SmartCompressEngine:
    """Enterprise Lossless & High-Fidelity Image Optimization Engine."""

    def _has_transparency(self, pil_img: Image.Image) -> bool:
        """Determines if image contains non-opaque (transparent) pixels."""
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            alpha = pil_img.convert("RGBA").split()[-1]
            min_alpha, _ = alpha.getextrema()
            return min_alpha < 255
        return False

    def _optimize_png_bytes(self, raw_png_bytes: bytes) -> bytes:
        """Applies oxipng / optipng optimization if binary exists on system PATH."""
        oxipng_path = shutil.which("oxipng")
        optipng_path = shutil.which("optipng")

        if oxipng_path:
            try:
                proc = subprocess.run(
                    [oxipng_path, "-o", "6", "-i", "0", "--strip", "safe", "-"],
                    input=raw_png_bytes,
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                if proc.stdout and len(proc.stdout) < len(raw_png_bytes):
                    return proc.stdout
            except Exception as e:
                logger.debug(f"oxipng optimization skipped: {e}")

        elif optipng_path:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(raw_png_bytes)
                    tmp_path = tmp.name
                
                subprocess.run(
                    [optipng_path, "-o5", "-quiet", tmp_path],
                    timeout=5,
                    check=True,
                )
                optimized_bytes = open(tmp_path, "rb").read()
                os.remove(tmp_path)
                if len(optimized_bytes) < len(raw_png_bytes):
                    return optimized_bytes
            except Exception as e:
                logger.debug(f"optipng optimization skipped: {e}")

        return raw_png_bytes

    def optimize_output(
        self,
        image_input: Union[Image.Image, bytes],
        target_format: Optional[str] = None,
        jpeg_quality: int = 95,
        webp_quality: int = 95,
        webp_lossless: bool = False,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Executes smart output optimization while preserving 100% visual quality & resolution.
        Returns (optimized_bytes, telemetry_dict).
        """
        t0 = time.perf_counter()

        try:
            # 1. Normalize input to PIL Image
            if isinstance(image_input, bytes):
                orig_bytes_size = len(image_input)
                pil_img = Image.open(io.BytesIO(image_input))
                orig_fmt = pil_img.format or "RAW"
            else:
                pil_img = image_input
                buf_init = io.BytesIO()
                pil_img.save(buf_init, format=pil_img.format or "PNG")
                orig_bytes_size = len(buf_init.getvalue())
                orig_fmt = pil_img.format or "PNG"

            w, h = pil_img.size
            has_transparency = self._has_transparency(pil_img)

            # 2. Determine Smart Target Format
            if target_format:
                fmt = target_format.upper()
            elif has_transparency:
                fmt = "PNG"
            else:
                fmt = "JPEG"

            # 3. Format Execution Logic
            out_buffer = io.BytesIO()
            opt_fmt = fmt

            if fmt in ["JPEG", "JPG"]:
                opt_fmt = "JPEG"
                # Composite transparent background over white if converting RGBA to JPEG
                if pil_img.mode in ["RGBA", "LA", "P"]:
                    background = Image.new("RGB", pil_img.size, (255, 255, 255))
                    if pil_img.mode == "RGBA":
                        background.paste(pil_img, mask=pil_img.split()[-1])
                    else:
                        background.paste(pil_img.convert("RGBA"), mask=pil_img.convert("RGBA").split()[-1])
                    img_to_save = background
                else:
                    img_to_save = pil_img.convert("RGB")

                # JPEG Optimization Rules: Quality 95-98, optimize=True, progressive=True, 4:4:4 chroma subsampling (subsampling=0)
                quality_clamped = max(95, min(98, jpeg_quality))
                img_to_save.save(
                    out_buffer,
                    format="JPEG",
                    quality=quality_clamped,
                    optimize=True,
                    progressive=True,
                    subsampling=0,  # 4:4:4 chroma subsampling
                )
                output_bytes = out_buffer.getvalue()

            elif fmt == "WEBP":
                opt_fmt = "WEBP"
                img_to_save = pil_img if has_transparency else pil_img.convert("RGB")
                img_to_save.save(
                    out_buffer,
                    format="WEBP",
                    quality=webp_quality,
                    lossless=webp_lossless,
                    method=6,  # Maximum WebP compression effort
                )
                output_bytes = out_buffer.getvalue()

            else:  # PNG
                opt_fmt = "PNG"
                pil_img.save(
                    out_buffer,
                    format="PNG",
                    optimize=True,
                    compress_level=9,
                )
                output_bytes = out_buffer.getvalue()

                # Apply oxipng / optipng lossless optimization if available
                output_bytes = self._optimize_png_bytes(output_bytes)

                # 4. Smart PNG Guard: If PNG > 10 MB and no transparency, convert to JPEG quality 95
                if len(output_bytes) > 10 * 1024 * 1024 and not has_transparency:
                    logger.info(f"💡 PNG size ({len(output_bytes)/(1024*1024):.2f} MB) > 10MB without transparency. Converting to optimized JPEG quality 95...")
                    jpeg_buf = io.BytesIO()
                    img_rgb = pil_img.convert("RGB")
                    img_rgb.save(
                        jpeg_buf,
                        format="JPEG",
                        quality=95,
                        optimize=True,
                        progressive=True,
                        subsampling=0,
                    )
                    output_bytes = jpeg_buf.getvalue()
                    opt_fmt = "JPEG"

            # 5. Compute Metrics Telemetry
            opt_bytes_size = len(output_bytes)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            reduction_pct = round(max(0.0, (1.0 - (opt_bytes_size / max(1, orig_bytes_size))) * 100), 1)

            telemetry = {
                "original_format": orig_fmt,
                "original_size_bytes": orig_bytes_size,
                "original_size_kb": round(orig_bytes_size / 1024, 1),
                "optimized_format": opt_fmt,
                "optimized_size_bytes": opt_bytes_size,
                "optimized_size_kb": round(opt_bytes_size / 1024, 1),
                "compression_ratio_pct": reduction_pct,
                "processing_time_ms": elapsed_ms,
                "resolution": f"{w}x{h}",
            }

            logger.info("=" * 65)
            logger.info("📦 SMART OUTPUT OPTIMIZATION COMPLETE:")
            logger.info(f"• Original Format:  {orig_fmt} ({telemetry['original_size_kb']} KB)")
            logger.info(f"• Optimized Format: {opt_fmt} ({telemetry['optimized_size_kb']} KB)")
            logger.info(f"• Size Reduction:   {reduction_pct}% | Resolution: {w}x{h} px")
            logger.info(f"• Processing Time:  {elapsed_ms} ms")
            logger.info("=" * 65)

            return output_bytes, telemetry

        except Exception as e:
            logger.error(f"Smart output optimization failed: {e}", exc_info=True)
            raise ImageProcessingException(f"Smart output optimization failed: {e}")

    def compress_image(
        self,
        image_input: Union[Image.Image, bytes],
        quality: int = 95,
        target_format: str = "JPEG",
    ) -> bytes:
        """Backwards compatibility helper."""
        out_bytes, _ = self.optimize_output(image_input, target_format=target_format, jpeg_quality=quality)
        return out_bytes


# Singleton instance
compress_engine = SmartCompressEngine()
