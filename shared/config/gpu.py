"""
Zenemoo AI - GPU Acceleration Configuration
"""

from pydantic_settings import BaseSettings


class GPUConfig(BaseSettings):
    CUDA_DEVICE_ID: int = 0
    ENABLE_FP16: bool = True
    VRAM_USAGE_LIMIT_PERCENT: float = 85.0
    BATCH_SIZE: int = 1
    TILE_SIZE: int = 512
    TILE_PAD: int = 10
