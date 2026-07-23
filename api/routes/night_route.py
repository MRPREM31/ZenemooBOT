"""
==============================================================================
Zenemoo AI - Night Photo Enhance REST Endpoint
==============================================================================
POST /night-enhance
Enhances low-light and night photographs (+EV, noise reduction, white balance, sky boost).
"""

from fastapi import APIRouter, UploadFile, File
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Flagship Studios"])


@router.post("/night-enhance", response_model=ImageProcessResponse)
async def night_enhance(
    file: UploadFile = File(...),
):
    """Enhances low-light and night photographs."""
    contents = await file.read()
    filename = file.filename or "night_input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "night"},
    )
    return result
