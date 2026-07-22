"""
==============================================================================
Zenemoo AI - FastAPI Application Factory & Exception Handlers
==============================================================================
Main FastAPI application setup with CORS middleware, static file serving,
OpenAPI metadata tags, and centralized exception handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.config import settings
from core.logging import logger
from core.database import init_db
from shared.exceptions import ImageProcessingException, StorageException, AIModelException
from api.routes import (
    health_router,
    enhance_router,
    restore_router,
    background_router,
    upscale_router,
    compress_router,
    colorize_router,
    auth_router,
    admin_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler initializing database on startup."""
    logger.info("🎬 Initializing database tables on application startup...")
    await init_db()
    yield
    logger.info("🛑 Application shutdown completed.")


def create_app() -> FastAPI:
    """Constructs and configures Zenemoo AI FastAPI backend instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-Ready AI-Powered Image Enhancement Platform API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Static Directories for direct image downloads & Admin Dashboard
    settings.ensure_directories()
    outputs_path = settings.BASE_DIR / settings.OUTPUT_DIR
    web_dashboard_path = settings.BASE_DIR / "clients" / "web"

    app.mount("/outputs", StaticFiles(directory=outputs_path), name="outputs")
    if web_dashboard_path.exists():
        app.mount("/dashboard", StaticFiles(directory=web_dashboard_path, html=True), name="dashboard")

    # Include API Routers
    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(enhance_router)
    app.include_router(restore_router)
    app.include_router(background_router)
    app.include_router(upscale_router)
    app.include_router(compress_router)
    app.include_router(colorize_router)

    # Custom Exception Handlers
    @app.exception_handler(ImageProcessingException)
    async def image_exception_handler(request: Request, exc: ImageProcessingException):
        logger.error(f"Image Exception: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "error_type": exc.__class__.__name__, "message": exc.message},
        )

    @app.exception_handler(StorageException)
    async def storage_exception_handler(request: Request, exc: StorageException):
        logger.error(f"Storage Exception: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error_type": exc.__class__.__name__, "message": exc.message},
        )

    @app.exception_handler(AIModelException)
    async def ai_exception_handler(request: Request, exc: AIModelException):
        logger.error(f"AI Exception: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error_type": exc.__class__.__name__, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.critical(f"Unhandled Server Error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error_type": "InternalServerError", "message": str(exc)},
        )

    return app


app = create_app()
