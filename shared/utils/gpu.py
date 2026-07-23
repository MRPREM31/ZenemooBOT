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


def get_extended_gpu_telemetry() -> Dict[str, Any]:
    """
    Returns extended GPU hardware telemetry including GPU utilization %,
    GPU temperature °C, free VRAM, used VRAM, and total VRAM.
    """
    telemetry = {
        "gpu_utilization_pct": 0.0,
        "temperature_c": 0.0,
        "free_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "used_vram_mb": 0.0,
        "vram_used_pct": 0.0,
        "device_name": "CPU Fallback",
    }

    # 1. PyTorch CUDA memory
    try:
        import torch
        if torch.cuda.is_available():
            telemetry["device_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            telemetry["free_vram_mb"] = round(free_bytes / (1024 * 1024), 2)
            telemetry["total_vram_mb"] = round(total_bytes / (1024 * 1024), 2)
            used_bytes = total_bytes - free_bytes
            telemetry["used_vram_mb"] = round(used_bytes / (1024 * 1024), 2)
            if total_bytes > 0:
                telemetry["vram_used_pct"] = round((used_bytes / total_bytes) * 100.0, 1)
    except Exception:
        pass

    # 2. Try pynvml for utilization & temperature
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        telemetry["gpu_utilization_pct"] = float(util.gpu)
        telemetry["temperature_c"] = float(temp)
        return telemetry
    except Exception:
        pass

    # 3. Fallback to nvidia-smi subprocess query
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split(",")
            if len(parts) >= 2:
                telemetry["gpu_utilization_pct"] = float(parts[0].strip())
                telemetry["temperature_c"] = float(parts[1].strip())
    except Exception:
        pass

    return telemetry


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
