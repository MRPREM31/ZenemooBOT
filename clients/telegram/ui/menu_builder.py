"""
==============================================================================
Zenemoo AI - Telegram Bot UI Menu Builder & Layout Definitions
==============================================================================
Reusable layout components, inline keyboards, welcome messages, feature prompt
screens, processing progress frames, and completion report templates.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import Dict, Any, Optional


FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "ai_enhance": "Full AI Enhance",
    "ai_portrait": "Portrait Studio",
    "ai_night": "Night Enhance",
    "ai_passport": "Passport Studio",
    "ai_cartoon": "Cartoon Studio",
    "ai_upscale_2x": "2× Upscale",
    "ai_upscale_4x": "4× Upscale",
    "ai_restore_gfp": "Face Restore",
    "ai_restore_code": "Natural Restore",
    "ai_removebg": "Remove Background",
    "ai_sharpen": "Denoise",
    "ai_colorize": "Colorize B&W",
    "ai_colorize_vintage": "Vintage Colorize",
    "ai_compress": "Smart Compress",
}


def get_welcome_message() -> str:
    """Returns the formatted Telegram Welcome Message."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✨ **Welcome to Zenemoo AI**\n\n"
        "**Professional AI Image Enhancement**\n\n"
        "Choose one of the options below to begin.\n\n"
        "🌟 **Full AI Enhance**\n"
        "Automatically analyzes and improves your photo.\n\n"
        "🖼️ **AI Studio**\n"
        "Portrait • Passport • Night • Cartoon\n\n"
        "⚡ **AI Enhancement**\n"
        "Upscale • Face Restore • Natural Restore\n\n"
        "🛠️ **AI Tools**\n"
        "Background Removal • Colorization • Compression\n\n"
        "Powered by **Zenemoo AI**\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Builds the modern, structured Inline Keyboard for Zenemoo AI."""
    keyboard = [
        # Section 1: 🌟 AI Enhance (Full-width flagship button)
        [
            InlineKeyboardButton("✨ Full AI Enhance", callback_data="ai_enhance"),
        ],
        # Section 2: 🖼️ AI Studio (Two-column layout)
        [
            InlineKeyboardButton("👤 Portrait AI", callback_data="ai_portrait"),
            InlineKeyboardButton("🌙 Night Enhance", callback_data="ai_night"),
        ],
        [
            InlineKeyboardButton("🛂 Passport Studio", callback_data="ai_passport"),
            InlineKeyboardButton("🎨 Cartoon Studio", callback_data="ai_cartoon"),
        ],
        # Section 3: ⚡ AI Enhancement (Two-column layout)
        [
            InlineKeyboardButton("🔍 2× Upscale", callback_data="ai_upscale_2x"),
            InlineKeyboardButton("🔎 4× Upscale", callback_data="ai_upscale_4x"),
        ],
        [
            InlineKeyboardButton("🎭 Face Restore", callback_data="ai_restore_gfp"),
            InlineKeyboardButton("🎭 Natural Restore", callback_data="ai_restore_code"),
        ],
        # Section 4: 🛠️ AI Tools (Two-column layout)
        [
            InlineKeyboardButton("🖼️ Remove Background", callback_data="ai_removebg"),
            InlineKeyboardButton("⚡ Denoise", callback_data="ai_sharpen"),
        ],
        [
            InlineKeyboardButton("🎨 Colorize B&W", callback_data="ai_colorize"),
            InlineKeyboardButton("📦 Smart Compress", callback_data="ai_compress"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_feature_action_message(action_name: str) -> str:
    """Returns the professional feature action prompt message."""
    feat_name = FEATURE_DISPLAY_NAMES.get(action_name, action_name.replace("ai_", "").replace("_", " ").title())
    return (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✨ **Zenemoo AI**\n\n"
        "**Feature:**\n"
        f"{feat_name}\n\n"
        "📤 **Upload your image.**\n\n"
        "**Supported:**\n"
        "• JPG\n"
        "• PNG\n"
        "• WEBP\n\n"
        "**Maximum Size:**\n"
        "20 MB\n\n"
        "The AI will automatically analyze and optimize your image.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_processing_message(progress_label: str = "Enhancement Pipeline", pct: int = 50) -> str:
    """Returns the formatted processing screen message."""
    blocks = int(pct / 10)
    bar = "█" * blocks + "░" * (10 - blocks)
    return (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🧠 **Zenemoo AI Processing**\n\n"
        "✔ **Image Received**\n\n"
        "✔ **AI Analysis Complete**\n\n"
        "**Running:**\n\n"
        "• Face Detection\n\n"
        f"• {progress_label}\n\n"
        "• Quality Optimization\n\n"
        f"`{bar}` **{pct}%**\n\n"
        "**Estimated Time**\n\n"
        "10–20 seconds\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_completion_message(
    action_name: str,
    res_w: int = 4000,
    res_h: int = 3000,
    quality_score: int = 97,
    processing_time: float = 11.4,
    direct_link: Optional[str] = None,
) -> str:
    """Returns the formatted completion report screen message."""
    feat_name = FEATURE_DISPLAY_NAMES.get(action_name, action_name.replace("ai_", "").replace("_", " ").title())
    text = (
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✅ **Processing Complete**\n\n"
        "**Feature**\n\n"
        f"{feat_name}\n\n"
        "**Resolution**\n\n"
        f"{res_w} × {res_h}\n\n"
        "**AI Quality Score**\n\n"
        f"{quality_score} / 100\n\n"
        "**Processing Time**\n\n"
        f"{processing_time:.1f} sec\n\n"
        "Thank you for using\n\n"
        "✨ **Zenemoo AI**\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━"
    )
    if direct_link:
        text += f"\n\n🌐 **Direct Image Link:** `{direct_link}`"
    return text
