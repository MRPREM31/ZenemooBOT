"""
==============================================================================
Zenemoo AI - File & Upload Validation Utilities
==============================================================================
Validates file extensions, size limits, and inspects magic byte signatures
to prevent uploading spoofed or unreadable files.
"""

import os
from typing import Tuple
from shared.exceptions.image_exception import (
    InvalidImageFormatException,
    ImageSizeExceededException,
)

# Supported MIME / Extension Magic Header signatures
ALLOWED_MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_upload(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 25,
) -> Tuple[bool, str]:
    """
    Validates file size, extension, and magic header signatures.
    Returns (True, format_extension) or raises custom ImageException.
    """
    # 1. Size check
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ImageSizeExceededException(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit ({max_size_mb} MB)."
        )

    # 2. Extension check
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageFormatException(
            f"Extension '{ext}' is not supported. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 3. Magic Number signature check
    is_valid_magic = False
    detected_format = "unknown"
    for signature, fmt in ALLOWED_MAGIC_SIGNATURES.items():
        if file_bytes.startswith(signature):
            is_valid_magic = True
            detected_format = fmt
            break

    if not is_valid_magic:
        raise InvalidImageFormatException(
            "Uploaded file failed binary signature magic header inspection."
        )

    return True, detected_format
