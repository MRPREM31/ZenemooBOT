"""
==============================================================================
Zenemoo AI - Storage Exception Domain
==============================================================================
Custom exceptions for disk read/write operations, missing files, and S3 errors.
"""


class StorageException(Exception):
    """Base exception for file storage and persistence operations."""
    def __init__(self, message: str, path: str = ""):
        super().__init__(message)
        self.message = message
        self.path = path


class FileNotFoundStorageException(StorageException):
    """Raised when a requested file or asset does not exist on disk/S3."""
    pass


class StorageQuotaExceededException(StorageException):
    """Raised when user or system disk storage limit is reached."""
    pass


class StorageWriteException(StorageException):
    """Raised when file write or directory creation fails."""
    pass
