"""
==============================================================================
Zenemoo AI - Unit Tests for Shared Utilities
==============================================================================
Tests file validation, path sanitization, GPU detection, and timer utilities.
"""

import pytest
from PIL import Image
import io
from shared.utils import (
    sanitize_filename,
    generate_unique_filename,
    validate_image_upload,
    get_gpu_info,
    BenchmarkTimer,
)
from shared.exceptions import InvalidImageFormatException, ImageSizeExceededException


def test_sanitize_filename():
    """Tests directory traversal sanitization."""
    unsafe = "../../etc/passwd_image.jpg"
    clean = sanitize_filename(unsafe)
    assert ".." not in clean
    assert clean == "passwd_image.jpg"


def test_generate_unique_filename():
    """Tests UUID filename generation retaining extension."""
    name1 = generate_unique_filename("sample.png", prefix="img")
    name2 = generate_unique_filename("sample.png", prefix="img")
    assert name1 != name2
    assert name1.endswith(".png")
    assert name1.startswith("img_")


def test_validate_image_upload_valid_png():
    """Tests binary magic header validation for valid PNG."""
    # Create valid 10x10 PNG bytes
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    is_valid, fmt = validate_image_upload(png_bytes, "test.png")
    assert is_valid is True
    assert fmt == "png"


def test_validate_image_upload_invalid_format():
    """Tests invalid binary magic signature rejection."""
    junk_bytes = b"NOT_AN_IMAGE_BINARY_DATA"
    with pytest.raises(InvalidImageFormatException):
        validate_image_upload(junk_bytes, "fake.jpg")


def test_gpu_info_retrieval():
    """Tests GPU telemetry structure returns valid keys."""
    gpu = get_gpu_info()
    assert "available" in gpu
    assert "device_name" in gpu


def test_benchmark_timer():
    """Tests timing context manager."""
    with BenchmarkTimer("test_task") as timer:
        sum([i for i in range(10000)])
    assert timer.elapsed_ms >= 0.0
