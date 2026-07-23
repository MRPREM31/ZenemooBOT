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

    caption_text = (
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "✨ **Zenemoo AI**\n\n"
        "📥 **Image Received**\n"
        f"• Size: `{file_size_mb:.2f} MB`\n"
        "✔ **Image Analysis Complete**\n\n"
        "Select an AI processing engine or Flagship Studio below:\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    keyboard = [
        [
            InlineKeyboardButton("📘 Passport Studio", callback_data="ai_passport"),
            InlineKeyboardButton("🌙 Night Enhance", callback_data="ai_night"),
        ],
        [
            InlineKeyboardButton("🎭 Portrait Studio", callback_data="ai_portrait"),
            InlineKeyboardButton("🎨 Cartoon Studio", callback_data="ai_cartoon"),
        ],
        [
            InlineKeyboardButton("✨ Full AI Enhance", callback_data="ai_enhance"),
            InlineKeyboardButton("🖼️ Remove BG", callback_data="ai_removebg"),
        ],
        [
            InlineKeyboardButton("🔍 2x Upscale", callback_data="ai_upscale_2x"),
            InlineKeyboardButton("🔎 4x Upscale", callback_data="ai_upscale_4x"),
        ],
        [
            InlineKeyboardButton("⚡ Denoise & Sharpen", callback_data="ai_sharpen"),
            InlineKeyboardButton("🎨 Colorize B&W", callback_data="ai_colorize"),
        ],
        [
            InlineKeyboardButton("📜 Vintage Colorize", callback_data="ai_colorize_vintage"),
            InlineKeyboardButton("📦 Smart Compress", callback_data="ai_compress"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(caption_text, parse_mode="Markdown", reply_markup=reply_markup)
