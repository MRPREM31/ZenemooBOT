"""
==============================================================================
Zenemoo AI - Telegram Bot Callback Query Handler
==============================================================================
Processes inline button interactions. Downloads user photos from Telegram servers,
delegates processing EXCLUSIVELY to the FastAPI backend REST API via `bot_api_client`,
tracks job progress, and returns processed output images.
"""

import time
import io
from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.services.bot_api_client import bot_api_client
from shared.exceptions.telegram_exception import TelegramBackendCommunicationException


# Mapping callback_data to backend API endpoint path
ENDPOINT_MAPPING = {
    "ai_passport": "/passport-studio",
    "ai_night": "/night-enhance",
    "ai_portrait": "/portrait-studio",
    "ai_cartoon": "/cartoon-studio",
    "ai_enhance": "/enhance",
    "ai_removebg": "/remove-bg",
    "ai_restore_gfp": "/restore?model=gfpgan",
    "ai_restore_code": "/restore?model=codeformer",
    "ai_upscale_2x": "/upscale?scale=2",
    "ai_upscale_4x": "/upscale?scale=4",
    "ai_sharpen": "/sharpen",
    "ai_colorize": "/colorize",
    "ai_colorize_vintage": "/colorize?vintage_mode=true",
    "ai_compress": "/compress",
}


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline keyboard callback events."""
    query = update.callback_query
    if not query:
        return

    await query.answer()
    callback_data = query.data

    # 1. System Health Check Button
    if callback_data == "check_health":
        try:
            health_res = await bot_api_client.check_health()
            await query.edit_message_text(
                f"🟢 **Backend Status:** Connected\n"
                f"• Version: {health_res.get('version', '1.0.0')}\n"
                f"• Status: {health_res.get('status', 'healthy')}\n"
                f"• Compute Device: `{health_res.get('device', 'cpu')}`",
                parse_mode="Markdown"
            )
        except Exception as e:
            await query.edit_message_text(
                f"🔴 **Backend API Offline**\n\nCould not reach backend API server: {e}",
                parse_mode="Markdown"
            )
        return

    # 2. Settings / History navigation shortcuts
    if callback_data == "cmd_settings":
        await query.edit_message_text("⚙️ Settings menu (Use /settings)")
        return
    if callback_data == "cmd_history":
        await query.edit_message_text("📜 Processing history (Use /history)")
        return

    # 3. AI Processing Pipeline Actions
    file_id = context.user_data.get("last_photo_file_id")
    if not file_id:
        from clients.telegram.ui.menu_builder import get_feature_action_message
        action_prompt = get_feature_action_message(callback_data)
        await query.message.reply_text(action_prompt, parse_mode="Markdown")
        return

    endpoint = ENDPOINT_MAPPING.get(callback_data, "/enhance")

    # Send initial status message
    from clients.telegram.ui.menu_builder import get_processing_message
    status_msg = await query.message.reply_text(
        get_processing_message("Enqueuing Job", 10),
        parse_mode="Markdown"
    )

    # Import and enqueue job onto non-blocking background queue
    from clients.telegram.services.job_queue_service import job_queue_manager
    
    await job_queue_manager.enqueue_job(
        chat_id=query.message.chat_id,
        file_id=file_id,
        action_name=callback_data,
        endpoint=endpoint,
        status_msg_id=status_msg.message_id,
        context=context,
    )
