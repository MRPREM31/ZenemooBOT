"""
==============================================================================
Zenemoo AI - Professional Portrait Studio REST Endpoint
==============================================================================
POST /portrait-studio
Mode-driven portrait enhancement (Professional Headshot, LinkedIn, Instagram, Resume, Beauty, Studio).
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Flagship Studios"])


@router.post("/portrait-studio", response_model=ImageProcessResponse)
async def portrait_studio(
    file: UploadFile = File(...),
    mode: str = Form("linkedin", description="Portrait mode (professional_headshot, linkedin, instagram, resume, beauty, studio)"),
):
    """Enhances portraits according to selected professional mode."""
    contents = await file.read()
    filename = file.filename or "portrait_input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "portrait_studio", "portrait_mode": mode},
    )
    return result
