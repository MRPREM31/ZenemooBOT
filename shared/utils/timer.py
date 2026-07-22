"""
==============================================================================
Zenemoo AI - Precision Execution Timing Utilities
==============================================================================
Provides high-precision timers and context managers for tracking AI model inference duration.
"""

import time
from typing import Generator
from contextlib import contextmanager


class BenchmarkTimer:
    """Context manager for measuring execution time in milliseconds."""
    def __init__(self, task_name: str = "Task"):
        self.task_name = task_name
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


@contextmanager
def time_execution(label: str = "Execution") -> Generator[BenchmarkTimer, None, None]:
    """Simple wrapper for time measurement."""
    timer = BenchmarkTimer(label)
    timer.__enter__()
    try:
        yield timer
    finally:
        timer.__exit__(None, None, None)
