"""
API Routes Package
"""

from .health_route import router as health_router
from .enhance_route import router as enhance_router
from .restore_route import router as restore_router
from .background_route import router as background_router
from .upscale_route import router as upscale_router
from .compress_route import router as compress_router
from .colorize_route import router as colorize_router
from .auth_route import router as auth_router
from .admin_route import router as admin_router
from .passport_route import router as passport_router
from .night_route import router as night_router
from .portrait_route import router as portrait_router
from .cartoon_route import router as cartoon_router

__all__ = [
    "health_router",
    "enhance_router",
    "restore_router",
    "background_router",
    "upscale_router",
    "compress_router",
    "colorize_router",
    "auth_router",
    "admin_router",
    "passport_router",
    "night_router",
    "portrait_router",
    "cartoon_router",
]
