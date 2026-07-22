"""
==============================================================================
Zenemoo AI - API Endpoint Integration Tests
==============================================================================
Tests REST API endpoints using FastAPI TestClient.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from PIL import Image
import io
from core.database import init_db
from api.app import app


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Initializes SQLite test database tables before running API tests."""
    asyncio.run(init_db())


client = TestClient(app)


def _create_sample_png_bytes() -> bytes:
    """Helper creating raw PNG image bytes for upload testing."""
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_check_endpoint():
    """Tests GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "Zenemoo AI"


def test_admin_stats_endpoint():
    """Tests GET /admin/stats endpoint."""
    response = client.get("/admin/stats")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "system" in data
    assert "gpu" in data


def test_enhance_endpoint():
    """Tests POST /enhance endpoint."""
    image_bytes = _create_sample_png_bytes()
    files = {"file": ("test.png", image_bytes, "image/png")}
    data = {"face_restore": "true", "upscale_factor": "4"}

    response = client.post("/enhance", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "output_name" in res


def test_remove_bg_endpoint():
    """Tests POST /remove-bg endpoint."""
    image_bytes = _create_sample_png_bytes()
    files = {"file": ("test.png", image_bytes, "image/png")}

    response = client.post("/remove-bg", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"


def test_upscale_endpoint():
    """Tests POST /upscale endpoint."""
    image_bytes = _create_sample_png_bytes()
    files = {"file": ("test.png", image_bytes, "image/png")}
    data = {"scale": "4"}

    response = client.post("/upscale", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
