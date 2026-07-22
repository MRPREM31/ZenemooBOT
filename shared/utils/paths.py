"""
==============================================================================
Zenemoo AI - Path Resolution & Sanitization Utilities
==============================================================================
Provides safe path generation, filename sanitization, and output path routing.
"""

import os
import uuid
from pathlib import Path
from core.config import settings


def sanitize_filename(filename: str) -> str:
    """Removes potentially dangerous directory traversal characters from filenames."""
    clean_name = os.path.basename(filename)
    clean_name = clean_name.replace(" ", "_").replace("..", "")
    return clean_name


def generate_unique_filename(original_filename: str, prefix: str = "img") -> str:
    """Generates a collision-free UUID filename retaining the original extension."""
    ext = Path(original_filename).suffix.lower() or ".jpg"
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}{ext}"


def get_upload_path(filename: str) -> Path:
    """Returns absolute path in uploads directory."""
    return settings.BASE_DIR / settings.UPLOAD_DIR / sanitize_filename(filename)


def get_output_path(filename: str) -> Path:
    """Returns absolute path in outputs directory."""
    return settings.BASE_DIR / settings.OUTPUT_DIR / sanitize_filename(filename)
