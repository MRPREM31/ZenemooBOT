"""
==============================================================================
Zenemoo AI - Passport Photo Studio REST Endpoint
==============================================================================
POST /passport-studio
Generates ISO/ICAO compliant passport photos, printable 4x6 grid sheet, and PDF.
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Flagship Studios"])


@router.post("/passport-studio", response_model=ImageProcessResponse)
async def passport_studio(
    file: UploadFile = File(...),
    country: str = Form("india", description="Country specification (india, usa, canada, uk, australia, germany, japan, singapore, uae)"),
    bg_color: str = Form("white", description="Background color (white, blue, light_gray)"),
):
    """Generates passport photos, 4x6 print sheet, and PDF export."""
    contents = await file.read()
    filename = file.filename or "passport_input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={
            "mode": "passport",
            "country": country,
            "bg_color": bg_color,
        },
    )
    return result
