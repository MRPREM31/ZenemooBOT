"""
==============================================================================
Zenemoo AI - Image Compression Endpoint
==============================================================================
POST /compress
Lossless / Lossy compression optimizer.
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["Image Utilities"])


@router.post("/compress", response_model=ImageProcessResponse)
async def compress_image(
    file: UploadFile = File(...),
    quality: int = Form(85, description="Compression quality (1-100)"),
):
    """Optimizes and compresses image file size."""
    contents = await file.read()
    filename = file.filename or "input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "compress", "quality": quality},
    )
    return result
