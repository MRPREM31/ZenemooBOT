"""
==============================================================================
Zenemoo AI - Cartoon & Stylize Studio REST Endpoint
==============================================================================
POST /cartoon-studio
Stylizes photos into 13 artistic cartoon styles (Anime, Ghibli, Pixar, Disney, Comic, Sketch, etc.).
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Flagship Studios"])


@router.post("/cartoon-studio", response_model=ImageProcessResponse)
async def cartoon_studio(
    file: UploadFile = File(...),
    style: str = Form("anime", description="Cartoon style (anime, ghibli, pixar, disney, comic, sketch, oil_painting, watercolor, 3d_cartoon, chibi, lego, clay, pixel_art)"),
):
    """Stylizes photos into requested cartoon style."""
    contents = await file.read()
    filename = file.filename or "cartoon_input.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "cartoon", "cartoon_style": style},
    )
    return result
