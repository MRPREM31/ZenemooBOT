"""
==============================================================================
Zenemoo AI - Telegram Bot Client Application Launcher
==============================================================================
Production-ready Telegram Bot Client application setup using `python-telegram-bot` v22+.
Communicates EXCLUSIVELY with the FastAPI backend REST API.
"""

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from core.logging import logger
from clients.telegram.config import bot_settings
from clients.telegram.handlers import (
    start_command,
    help_command,
    about_command,
    settings_command,
    history_command,
    handle_incoming_photo,
    handle_callback_query,
)
from clients.telegram.middlewares import global_error_handler


class TelegramBotClient:
    """Enterprise Telegram Bot Application Wrapper."""

    def __init__(self, token: str = bot_settings.BOT_TOKEN):
        self.token = token
        self.app = None

    def build_application(self) -> Application:
        """Constructs and configures Application handlers and middlewares."""
        if not self.token:
            logger.warning("⚠️ BOT_TOKEN is empty. Telegram bot cannot run without a valid bot token.")

        app = (
            Application.builder()
            .token(self.token)
            .get_updates_read_timeout(120)
            .get_updates_write_timeout(120)
            .get_updates_connection_pool_size(100)
            .build()
        )


        # Register Command Handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("about", about_command))
        app.add_handler(CommandHandler("settings", settings_command))
        app.add_handler(CommandHandler("history", history_command))
        app.add_handler(CommandHandler("enhance", start_command))
        app.add_handler(CommandHandler("removebg", start_command))
        app.add_handler(CommandHandler("upscale", start_command))
        app.add_handler(CommandHandler("restore", start_command))
        app.add_handler(CommandHandler("sharpen", start_command))
        app.add_handler(CommandHandler("compress", start_command))
        app.add_handler(CommandHandler("colorize", start_command))

        # Register Photo & Document Receiver Handlers
        app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_incoming_photo))

        # Register Callback Query Button Handler
        app.add_handler(CallbackQueryHandler(handle_callback_query))

        # Register Global Central Error Handler Middleware
        app.add_error_handler(global_error_handler)

        # Register post_init callback to launch job queue worker
        async def _post_init(application):
            from clients.telegram.services.job_queue_service import job_queue_manager
            job_queue_manager.start_worker(application)

        app.post_init = _post_init

        self.app = app
        return app

    def run_polling(self) -> None:
        """Starts Telegram long-polling event loop."""
        if not self.app:
            self.build_application()
        
        logger.info(f"🤖 Starting Zenemoo AI Telegram Bot Client (Token: {self.token[:8]}...)...")
        logger.info(f"🌐 Backend Target API Endpoint: '{bot_settings.BACKEND_API_URL}'")
        self.app.run_polling(drop_pending_updates=True)


def start_bot() -> None:
    """Helper launcher function."""
    bot = TelegramBotClient()
    bot.run_polling()


if __name__ == "__main__":
    start_bot()
