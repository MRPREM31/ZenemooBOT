"""
==============================================================================
Zenemoo AI - Full Image Enhancement Endpoint
==============================================================================
POST /enhance
Runs complete image enhancement pipeline (Background removal + GFPGAN + Real-ESRGAN + SwinIR + Sharpen + Compress).
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Image Enhancement"])


@router.post("/enhance", response_model=ImageProcessResponse)
async def enhance_image(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    face_restore: bool = Form(True, description="Enable GFPGAN face restoration"),
    upscale_factor: int = Form(4, description="Super-resolution upscale multiplier (2 or 4)"),
):
    """Executes full AI enhancement pipeline on uploaded image."""
    contents = await file.read()
    filename = file.filename or "uploaded_image.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={
            "mode": "full_enhance",
            "face_restore": face_restore,
            "upscale_factor": upscale_factor,
        },
    )
    return result
