"""
==============================================================================
Zenemoo AI - GPU Hardware & VRAM Memory Utilities
==============================================================================
Tracks PyTorch CUDA availability, allocated VRAM, peak memory, system RAM,
and VRAM memory flush utilities.
"""

import gc
import psutil
from typing import Dict, Any


def get_gpu_info() -> Dict[str, Any]:
    """Returns GPU device name, total VRAM, allocated VRAM, peak VRAM, system RAM, and availability status."""
    info = {
        "available": False,
        "device_count": 0,
        "device_name": "CPU Fallback",
        "total_vram_mb": 0.0,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "remaining_vram_mb": 0.0,
        "system_ram_used_mb": 0.0,
        "system_ram_total_mb": 0.0,
    }

    try:
        # System RAM telemetry
        mem = psutil.virtual_memory()
        info["system_ram_used_mb"] = round(mem.used / (1024 * 1024), 2)
        info["system_ram_total_mb"] = round(mem.total / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_memory
            info["total_vram_mb"] = round(total_bytes / (1024 * 1024), 2)
            
            allocated_bytes = torch.cuda.memory_allocated(0)
            info["allocated_mb"] = round(allocated_bytes / (1024 * 1024), 2)
            info["reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            info["max_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
            
            remaining_bytes = max(0, total_bytes - allocated_bytes)
            info["remaining_vram_mb"] = round(remaining_bytes / (1024 * 1024), 2)
    except ImportError:
        pass

    return info


def reset_gpu_peak_memory() -> None:
    """Resets max memory stats tracker in PyTorch CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except ImportError:
        pass


def empty_gpu_cache() -> None:
    """Frees unreferenced PyTorch CUDA cache tensors and runs Python garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
