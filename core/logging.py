"""
==============================================================================
Zenemoo AI - Logging Module
==============================================================================
Provides high-performance, structured logging using Python's logging library
enhanced with Rich color output for terminal debugging and rotating file logs.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler
from core.config import settings


def setup_logging() -> logging.Logger:
    """Configures global loggers for stdout and file output."""
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "zenemoo.log"

    # Base logging level
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Custom Formatter for file output
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler with Rotation (10MB per log file, max 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Force UTF-8 re-encoding for stdout/stderr on Windows legacy terminals
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # Rich Console Handler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=settings.DEBUG,
        show_time=True,
        show_path=True,
    )
    rich_handler.setLevel(log_level)

    # Root Logger Configuration
    logger = logging.getLogger("zenemoo")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers on re-initialization
    if not logger.handlers:
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Global logger instance
logger = setup_logging()
