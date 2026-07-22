"""
==============================================================================
Zenemoo AI - Image Exception Domain
==============================================================================
Custom exceptions related to image loading, parsing, resolution, and validation errors.
"""


class ImageProcessingException(Exception):
    """Base exception for all image processing failures."""
    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidImageFormatException(ImageProcessingException):
    """Raised when an uploaded file is not a supported image format or magic number check fails."""
    pass


class ImageSizeExceededException(ImageProcessingException):
    """Raised when an uploaded image exceeds maximum allowed dimensions or megabytes."""
    pass


class CorruptImageFileException(ImageProcessingException):
    """Raised when an image file cannot be opened by Pillow/OpenCV due to corruption."""
    pass
