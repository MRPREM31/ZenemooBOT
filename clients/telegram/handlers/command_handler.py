"""
==============================================================================
Zenemoo AI - Telegram Bot Command Handlers
==============================================================================
Handles slash commands (/start, /help, /about, /contact, /settings, /history).
"""

from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.middlewares.rate_limit_middleware import check_rate_limit
from clients.telegram.ui.menu_builder import (
    get_welcome_first_message,
    get_welcome_second_message,
    get_help_message,
    get_about_message,
    get_contact_message,
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start command handler sending Welcome Message 1 and Welcome Message 2."""
    if not await check_rate_limit(update, context):
        return

    user = update.effective_user
    first_name = user.first_name if user else "User"

    # Send Welcome Message 1
    msg1 = get_welcome_first_message(first_name)
    await update.message.reply_text(msg1, parse_mode="Markdown")

    # Send Welcome Message 2 immediately after (NO inline keyboard)
    msg2 = get_welcome_second_message()
    await update.message.reply_text(msg2, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help command handler."""
    if not await check_rate_limit(update, context):
        return

    help_text = get_help_message()
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/about command handler."""
    if not await check_rate_limit(update, context):
        return

    about_text = get_about_message()
    await update.message.reply_text(about_text, parse_mode="Markdown")


async def contact_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/contact command handler."""
    if not await check_rate_limit(update, context):
        return

    contact_text = get_contact_message()
    await update.message.reply_text(contact_text, parse_mode="Markdown")


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
