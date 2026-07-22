"""
==============================================================================
Zenemoo AI - Black & White Colorization Endpoint
==============================================================================
POST /colorize
Colorizes black and white photos using DeOldify.
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Image Enhancement"])


@router.post("/colorize", response_model=ImageProcessResponse)
async def colorize_image(
    file: UploadFile = File(...),
    render_factor: int = Form(35, description="DeOldify render factor (7-40)"),
):
    """Colorizes legacy black & white photographs."""
    contents = await file.read()
    filename = file.filename or "legacy_bw.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "colorize", "render_factor": render_factor},
    )
    return result
