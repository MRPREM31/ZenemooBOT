"""
==============================================================================
Zenemoo AI - Face Restoration Endpoint
==============================================================================
POST /restore
Face restoration endpoint using GFPGAN / CodeFormer.
"""

from fastapi import APIRouter, UploadFile, File, Form
from services import image_service
from api.schemas import ImageProcessResponse

router = APIRouter(tags=["AI Image Enhancement"])


@router.post("/restore", response_model=ImageProcessResponse)
async def restore_face(
    file: UploadFile = File(...),
    model: str = Form("gfpgan", description="Model choice: 'gfpgan' or 'codeformer'"),
    fidelity: float = Form(0.7, description="CodeFormer fidelity ratio (0.0 to 1.0)"),
):
    """Restores blurry, low-resolution faces in portrait photos."""
    contents = await file.read()
    filename = file.filename or "portrait.jpg"

    result = await image_service.process_image_enhancement(
        file_bytes=contents,
        filename=filename,
        pipeline_options={"mode": "face_restore", "model": model, "fidelity": fidelity},
    )
    return result
