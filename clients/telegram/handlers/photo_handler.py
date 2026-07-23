"""
==============================================================================
Zenemoo AI - Telegram Bot Photo Receiver Handler
==============================================================================
Receives incoming photos & documents, caches file details in context user_data,
and presents interactive action buttons for processing pipeline selection.
"""

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.middlewares.rate_limit_middleware import check_rate_limit


async def handle_incoming_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receives photos sent as images or uncompressed documents."""
    if not await check_rate_limit(update, context):
        return

    message = update.message
    if not message:
        return

    # Extract highest resolution photo file
    if message.photo:
        photo_file = message.photo[-1]
        file_id = photo_file.file_id
        file_size_mb = photo_file.file_size / (1024 * 1024) if photo_file.file_size else 0
    elif message.document and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
        file_size_mb = message.document.file_size / (1024 * 1024) if message.document.file_size else 0
    else:
        await message.reply_text("⚠️ Please send a valid image file (JPEG, PNG, WEBP).")
        return

    # Cache file_id in context user data
    context.user_data["last_photo_file_id"] = file_id

    from clients.telegram.ui.menu_builder import get_main_menu_keyboard

    caption_text = (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✨ **Zenemoo AI**\n\n"
        "📥 **Image Received**\n"
        f"• Size: `{file_size_mb:.2f} MB`\n"
        "✔ **Image Analysis Complete**\n\n"
        "Choose an AI enhancement feature below:\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    reply_markup = get_main_menu_keyboard()

    await message.reply_text(caption_text, parse_mode="Markdown", reply_markup=reply_markup)
