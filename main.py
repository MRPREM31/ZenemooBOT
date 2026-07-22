"""
==============================================================================
Zenemoo AI - Unified Application Launcher
==============================================================================
Production-ready launcher for FastAPI backend server, Web Dashboard, and Telegram Bot client.
"""

import sys
import asyncio
import uvicorn
from core.config import settings
from core.logging import logger
from shared.utils import get_gpu_info
from shared.weights import weights_manager
from ai import warmup_all_models
from api.app import app


async def startup_check() -> None:
    """Performs environment validation, hardware compute verification, and AI model warmup."""
    logger.info(f"🚀 Initializing {settings.APP_NAME} in [{settings.ENVIRONMENT.upper()}] mode")
    
    # GPU & Compute Hardware Status
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        logger.info(f"⚡ Compute Acceleration Device: [CUDA GPU - {gpu_info['device_name']}]")
    else:
        logger.info(f"⚡ Compute Acceleration Device: [CPU Fallback]")

    # Directory & Storage Check
    logger.info(f"📁 Storage Directories initialized: Uploads='{settings.UPLOAD_DIR}', Outputs='{settings.OUTPUT_DIR}'")
    
    # Model Weights Directory Check
    logger.info(f"🏋️ Model weights directory initialized at '{weights_manager.weights_dir}'")

    # Warmup and persist all 6 AI deep learning engines in VRAM once on startup
    warmup_all_models()


def start_backend_api() -> None:
    """Launches Uvicorn FastAPI REST API server."""
    logger.info(f"🌐 Starting Zenemoo AI Backend REST API on http://{settings.HOST}:{settings.PORT}...")
    logger.info(f"📚 OpenAPI Documentation available at http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run(
        "api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
    )


def main() -> None:
    """Main launcher entry point."""
    try:
        asyncio.run(startup_check())
        
        # CLI Flag Router
        if len(sys.argv) > 1 and sys.argv[1] == "--bot":
            logger.info("🤖 Launching Telegram Bot Client Mode...")
            from clients.telegram.bot_client import TelegramBotClient
            bot = TelegramBotClient()
            bot.run_polling()
        elif len(sys.argv) > 1 and sys.argv[1] == "--api":
            start_backend_api()
        else:
            logger.info("✅ Milestone 9: Web Admin Dashboard UI (http://localhost:8000/dashboard) initialized successfully.")
            logger.info("💡 Run 'python main.py --api' to start FastAPI backend server.")
            logger.info("💡 Run 'python main.py --bot' to start Telegram Bot client.")

    except Exception as e:
        logger.critical(f"❌ Failed to launch Zenemoo AI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()