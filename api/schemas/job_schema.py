"""
==============================================================================
Zenemoo AI - Job Status & Progress Pydantic Schemas
==============================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class JobStatusResponse(BaseModel):
    """Async Job status response model."""
    job_id: str = Field(..., json_schema_extra={"example": "a9f2b8c0d1e2"})
    job_type: str = Field(..., json_schema_extra={"example": "full_enhance"})
    status: str = Field(..., json_schema_extra={"example": "completed"})
    progress: int = Field(..., json_schema_extra={"example": 100})
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
