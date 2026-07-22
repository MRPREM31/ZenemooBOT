"""
==============================================================================
Zenemoo AI - Storage Service Layer
==============================================================================
Provides high-level abstraction for file persistence, supporting local disk
storage and cloud S3 compatibility.
"""

import os
import shutil
from pathlib import Path
from typing import Union, Tuple
from core.config import settings
from core.logging import logger
from shared.exceptions.storage_exception import (
    FileNotFoundStorageException,
    StorageWriteException,
)
from shared.utils.paths import get_upload_path, get_output_path, generate_unique_filename


class StorageService:
    """Enterprise Storage Service for input/output files."""

    def __init__(self):
        settings.ensure_directories()

    def save_upload_bytes(self, file_bytes: bytes, original_filename: str) -> Tuple[str, Path]:
        """Saves incoming upload bytes to uploads directory. Returns (filename, path)."""
        try:
            unique_name = generate_unique_filename(original_filename, prefix="upload")
            target_path = get_upload_path(unique_name)
            
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"💾 Saved upload file '{unique_name}' ({len(file_bytes)} bytes)")
            return unique_name, target_path
        except Exception as e:
            raise StorageWriteException(f"Failed to save upload file bytes: {e}")

    def save_output_image(self, file_bytes: bytes, filename_prefix: str = "enhanced") -> Tuple[str, Path]:
        """Saves processed result image to outputs directory."""
        try:
            unique_name = generate_unique_filename(f"{filename_prefix}.png", prefix=filename_prefix)
            target_path = get_output_path(unique_name)

            with open(target_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"✨ Saved output result image '{unique_name}'")
            return unique_name, target_path
        except Exception as e:
            raise StorageWriteException(f"Failed saving result output image: {e}")

    def get_file_bytes(self, file_path: Union[str, Path]) -> bytes:
        """Reads and returns binary contents of a file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundStorageException(f"File not found: {p}", path=str(p))
        try:
            with open(p, "rb") as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundStorageException(f"Failed reading file '{p}': {e}")


# Singleton instance
storage_service = StorageService()
