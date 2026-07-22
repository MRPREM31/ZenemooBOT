"""
==============================================================================
Zenemoo AI - AI Engine Exception Domain
==============================================================================
Custom exceptions for model loading, weight downloading, PyTorch inference errors,
and GPU CUDA Out Of Memory (OOM) conditions.
"""


class AIModelException(Exception):
    """Base exception for deep learning inference and weight operations."""
    def __init__(self, message: str, model_name: str = ""):
        super().__init__(message)
        self.message = message
        self.model_name = model_name


class ModelWeightsMissingException(AIModelException):
    """Raised when model weights (.pth / .onnx) are missing and auto-download fails."""
    pass


class GPUOutOfMemoryException(AIModelException):
    """Raised when PyTorch encounters CUDA Out Of Memory during model inference."""
    pass


class InferenceExecutionException(AIModelException):
    """Raised when forward pass model execution encounters unexpected tensor errors."""
    pass
