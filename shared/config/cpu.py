"""
Zenemoo AI - CPU Fallback Configuration
"""

from pydantic_settings import BaseSettings


class CPUConfig(BaseSettings):
    NUM_THREADS: int = 4
    ENABLE_OPENMP: bool = True
    TILE_SIZE: int = 256
    TILE_PAD: int = 10
    BATCH_SIZE: int = 1
