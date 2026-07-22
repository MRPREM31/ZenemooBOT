"""
==============================================================================
Zenemoo AI - Background Removal Endpoint
==============================================================================
POST /remove-bg
Removes background using rembg (U²-Net).
"""

from fastapi import APIRouter, UploadFile, File
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Image Enhancement"])


@router.post("/remove-bg", response_model=ImageProcessResponse)
async def remove_background(
    file: UploadFile = File(...),
):
    """Removes image background returning transparent PNG mask."""
    contents = await file.read()
    filename = file.filename or "input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "background_removal"},
    )
    return result
