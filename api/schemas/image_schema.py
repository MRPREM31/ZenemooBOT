"""
==============================================================================
Zenemoo AI - Image API Request & Response Pydantic Schemas
==============================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ImageProcessResponse(BaseModel):
    """Standardized response schema returned by image processing endpoints."""
    status: str = Field("success", json_schema_extra={"example": "success"})
    upload_name: str = Field(..., json_schema_extra={"example": "upload_a8f9c2d1.jpg"})
    output_name: str = Field(..., json_schema_extra={"example": "enhanced_72b8d9e0.png"})
    output_path: str = Field(..., json_schema_extra={"example": "outputs/enhanced_72b8d9e0.png"})
    output_url: Optional[str] = Field(None, json_schema_extra={"example": "/outputs/enhanced_72b8d9e0.png"})
    processing_time_ms: float = Field(..., json_schema_extra={"example": 1240.5})
    options: Dict[str, Any] = Field(default_factory=dict)



class UpscaleOptions(BaseModel):
    """Upscaling request options."""
    scale: int = Field(4, description="Scale factor: 2 or 4", json_schema_extra={"example": 4})


class RestoreOptions(BaseModel):
    """Face restoration request options."""
    model: str = Field("gfpgan", description="Face restoration model: gfpgan or codeformer", json_schema_extra={"example": "gfpgan"})
