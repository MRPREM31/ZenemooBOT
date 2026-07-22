"""
==============================================================================
Zenemoo AI - Health & System Status Pydantic Schemas
==============================================================================
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class HealthCheckResponse(BaseModel):
    """Health check endpoint response model."""
    status: str = Field("healthy", json_schema_extra={"example": "healthy"})
    app_name: str = Field("Zenemoo AI", json_schema_extra={"example": "Zenemoo AI"})
    version: str = Field("1.0.0", json_schema_extra={"example": "1.0.0"})
    environment: str = Field("development", json_schema_extra={"example": "development"})
    device: str = Field("cpu", json_schema_extra={"example": "cuda"})
    gpu_info: Dict[str, Any] = Field(default_factory=dict)
