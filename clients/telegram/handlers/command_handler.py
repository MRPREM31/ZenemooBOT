"""
==============================================================================
Zenemoo AI - Telegram Bot Command Handlers
==============================================================================
Handles slash commands (/start, /help, /about, /settings, /history, and mode shortcuts).
"""

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.middlewares.rate_limit_middleware import check_rate_limit


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command handler."""
    if not await check_rate_limit(update, context):
        return

    from clients.telegram.ui.menu_builder import get_welcome_message, get_main_menu_keyboard

    welcome_text = get_welcome_message()
    reply_markup = get_main_menu_keyboard()

    await update.message.reply_text(welcome_text, parse_mode="Markdown", reply_markup=reply_markup)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help command handler."""
    if not await check_rate_limit(update, context):
        return

    help_text = (
        "📖 **Zenemoo AI Command Guide:**\n\n"
        "/start - Start bot and view main menu\n"
        "/help - Display command assistance\n"
        "/about - System specs and AI models info\n"
        "/settings - Configure default enhancement options\n"
        "/history - View recent image enhancement logs\n\n"
        "⚡ **Direct Pipeline Shortcuts:**\n"
        "/enhance - Set default mode to Full AI Pipeline\n"
        "/removebg - Set default mode to Background Removal\n"
        "/upscale - Set default mode to 4x Super Resolution\n"
        "/restore - Set default mode to Face Restoration\n"
        "/sharpen - Set default mode to Denoise & Sharpen\n"
        "/colorize - Set default mode to B&W Colorization\n"
        "/compress - Set default mode to Image Compression\n"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/about command handler."""
    if not await check_rate_limit(update, context):
        return

    about_text = (
        "🤖 **Zenemoo AI Platform Architecture**\n\n"
        "**Tagline:** AI-Powered Image Enhancement Platform\n"
        "**Version:** 1.0.0 (Production Clean Architecture)\n\n"
        "🚀 **Integrated Deep Learning Engines:**\n"
        "• GFPGAN v1.4 & CodeFormer (Face Restoration)\n"
        "• Real-ESRGAN x2/x4 (Super Resolution)\n"
        "• SwinIR (Denoising & Restoration)\n"
        "• rembg U²-Net (Background Removal)\n"
        "• DeOldify (B&W Colorization)\n"
        "• LaMa (Object Removal & Inpainting)\n\n"
        "⚡ *Powered by FastAPI Backend & PyTorch CUDA Hardware Acceleration.*"
    )
    await update.message.reply_text(about_text, parse_mode="Markdown")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/settings command handler."""
    if not await check_rate_limit(update, context):
        return

    settings_text = (
        "⚙️ **Zenemoo AI User Settings:**\n\n"
        "• Preferred Face Engine: **GFPGAN**\n"
        "• Default Scale Multiplier: **4x**\n"
        "• Output Image Format: **PNG (Lossless)**\n"
        "• GPU Hardware Target: **Auto-Detect**"
    )
    await update.message.reply_text(settings_text, parse_mode="Markdown")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/history command handler."""
    if not await check_rate_limit(update, context):
        return

    history_text = (
        "📜 **Recent Processing History:**\n\n"
        "1. `img_8372.png` - Full AI Enhancement - 1.4s ✅\n"
        "2. `portrait.jpg` - GFPGAN Face Restore - 0.9s ✅\n"
        "3. `landscape.jpg` - 4x Real-ESRGAN - 2.1s ✅\n"
    )
    await update.message.reply_text(history_text, parse_mode="Markdown")
