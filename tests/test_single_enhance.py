"""
==============================================================================
Zenemoo AI - Single Request Pipeline Test
==============================================================================
"""

import sys
import io
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from fastapi.testclient import TestClient
from api.app import app
from core.logging import logger

client = TestClient(app)

def test_enhance_execution():
    logger.info("=================================================================")
    logger.info("⚡ EXECUTING FASTAPI POST /enhance WITH GENUINE AI INFERENCE ⚡")
    logger.info("=================================================================")

    # 1. Create sample test photo
    img = Image.new("RGB", (256, 256), color=(140, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    sample_bytes = buf.getvalue()

    # Save original test input
    Path("original.png").write_bytes(sample_bytes)

    # 2. Post image to /enhance
    files = {"file": ("original.jpg", sample_bytes, "image/jpeg")}
    data = {"face_restore": "true", "upscale_factor": "4"}

    t0 = time.time()
    response = client.post("/enhance", files=files, data=data)
    elapsed = time.time() - t0

    logger.info(f"Response Status: {response.status_code}")
    res = response.json()
    logger.info(f"Response Payload: {res}")
    logger.info(f"Total Execution Latency: {elapsed:.2f}s")
    
    # Save final output copy
    output_path = res.get("output_path")
    if output_path and Path(output_path).exists():
        final_bytes = Path(output_path).read_bytes()
        Path("final.png").write_bytes(final_bytes)
        logger.info("✅ Saved final output to 'final.png'")


if __name__ == "__main__":
    test_enhance_execution()
