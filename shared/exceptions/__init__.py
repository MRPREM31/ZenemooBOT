"""
Zenemoo AI Custom Exception Package
"""

from .image_exception import (
    ImageProcessingException,
    InvalidImageFormatException,
    ImageSizeExceededException,
    CorruptImageFileException,
)
from .storage_exception import (
    StorageException,
    FileNotFoundStorageException,
    StorageQuotaExceededException,
    StorageWriteException,
)
from .ai_exception import (
    AIModelException,
    ModelWeightsMissingException,
    GPUOutOfMemoryException,
    InferenceExecutionException,
)
from .telegram_exception import (
    TelegramBotException,
    TelegramRateLimitException,
    TelegramBackendCommunicationException,
)

__all__ = [
    "ImageProcessingException",
    "InvalidImageFormatException",
    "ImageSizeExceededException",
    "CorruptImageFileException",
    "StorageException",
    "FileNotFoundStorageException",
    "StorageQuotaExceededException",
    "StorageWriteException",
    "AIModelException",
    "ModelWeightsMissingException",
    "GPUOutOfMemoryException",
    "InferenceExecutionException",
    "TelegramBotException",
    "TelegramRateLimitException",
    "TelegramBackendCommunicationException",
]
