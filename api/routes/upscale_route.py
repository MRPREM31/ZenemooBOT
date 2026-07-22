"""
==============================================================================
Zenemoo AI - Image Super Resolution Upscale Endpoint
==============================================================================
POST /upscale
Super resolution upscaling using Real-ESRGAN (2x or 4x scale).
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Image Enhancement"])


@router.post("/upscale", response_model=ImageProcessResponse)
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Form(4, description="Scale factor: 2 or 4"),
):
    """Upscales image resolution by 2x or 4x using Real-ESRGAN."""
    contents = await file.read()
    filename = file.filename or "input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "upscale", "scale": scale},
    )
    return result
