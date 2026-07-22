"""
==============================================================================
Zenemoo AI - Model Weights Auto-Downloader
==============================================================================
Manages model weights directory structure and automatically downloads pre-trained
weights from open-source repository releases on first execution if missing.
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict
from core.config import settings
from core.logging import logger
from shared.exceptions.ai_exception import ModelWeightsMissingException

# Model Registry with official open-source URL releases
MODEL_WEIGHTS_REGISTRY: Dict[str, Dict[str, str]] = {
    "GFPGAN": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "subfolder": "GFPGAN",
    },
    "RealESRGAN_x4": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "subfolder": "RealESRGAN",
    },
    "RealESRGAN_x2": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "subfolder": "RealESRGAN",
    },
    "CodeFormer": {
        "filename": "codeformer.pth",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "subfolder": "CodeFormer",
    },
    "facexlib_detection": {
        "filename": "detection_Resnet50_Final.pth",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "subfolder": "facexlib",
        "min_size": 100000000,
    },
    "facexlib_parsing": {
        "filename": "parsing_parsenet.pth",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "subfolder": "facexlib",
        "min_size": 80000000,
    },
}


class WeightsManager:
    """Manages AI model weight downloads and local file paths."""

    def __init__(self):
        self.weights_dir = settings.BASE_DIR / "shared" / "weights"
        self._ensure_folders()

    def _ensure_folders(self) -> None:
        """Ensures weights directory subfolders exist."""
        for model_info in MODEL_WEIGHTS_REGISTRY.values():
            subfolder = self.weights_dir / model_info["subfolder"]
            subfolder.mkdir(parents=True, exist_ok=True)

    def get_weight_path(self, model_key: str) -> Path:
        """Returns path to model weight file, auto-downloading if missing."""
        if model_key not in MODEL_WEIGHTS_REGISTRY:
            raise ModelWeightsMissingException(f"Unknown model key '{model_key}'")

        info = MODEL_WEIGHTS_REGISTRY[model_key]
        target_path = self.weights_dir / info["subfolder"] / info["filename"]

        min_size = info.get("min_size", 0)
        if target_path.exists() and min_size > 0 and target_path.stat().st_size < min_size:
            logger.warning(f"⚠️ Corrupted or incomplete weight file detected for '{model_key}' ({target_path.stat().st_size} bytes < {min_size} bytes). Removing corrupted file...")
            target_path.unlink()

        if not target_path.exists():
            logger.info(f"⬇️ Model weight missing for '{model_key}'. Downloading from {info['url']}...")
            self._download_file(info["url"], target_path)

        return target_path

    def _download_file(self, url: str, destination: Path) -> None:
        """Downloads file with progress logging."""
        temp_path = destination.with_suffix(".tmp")
        try:
            logger.info(f"Downloading model weight file to '{destination}'...")

            def reporthook(count, block_size, total_size):
                if total_size > 0 and count % 50 == 0:
                    percent = int(count * block_size * 100 / total_size)
                    logger.debug(f"Download Progress [{destination.name}]: {percent}%")

            urllib.request.urlretrieve(url, temp_path, reporthook=reporthook)
            urllib.request.urlcleanup()

            try:
                temp_path.replace(destination)
            except OSError:
                import shutil
                shutil.move(str(temp_path), str(destination))

            logger.info(f"✅ Successfully downloaded '{destination.name}'.")
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise ModelWeightsMissingException(
                f"Failed downloading model weights from '{url}': {e}",
                model_name=destination.name,
            )


# Global singleton instance
weights_manager = WeightsManager()
