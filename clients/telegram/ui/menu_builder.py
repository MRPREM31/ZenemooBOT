"""
==============================================================================
Zenemoo AI - Telegram Bot UI Menu Builder & Layout Definitions
==============================================================================
Commercial-grade Telegram UI formatting, onboarding templates, menu keyboards,
progress frames, and completion reports.
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
    "ai_sharpen": "Denoise & Sharpen",
    "ai_colorize": "Colorize B&W",
    "ai_colorize_vintage": "Vintage Colorize",
    "ai_compress": "Smart Compress",
}


def get_welcome_first_message(first_name: str) -> str:
    """Message 1 sent after user clicks /start."""
    name = first_name or "User"
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👋 Welcome, {name}!\n\n"
        "Welcome to **Zenemoo AI** — your professional AI-powered image enhancement platform.\n\n"
        "Transform, restore, upscale, and optimize your photos using advanced AI technology designed for exceptional image quality.\n\n"
        "We're excited to help you create stunning results.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_welcome_second_message() -> str:
    """Message 2 sent immediately following Message 1."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✨ What can Zenemoo AI do?\n\n"
        "Zenemoo AI automatically enhances your images using state-of-the-art AI models optimized for quality and realism.\n\n"
        "📸 Getting Started\n\n"
        "Simply send any image to begin.\n\n"
        "After uploading your image, you'll be able to choose from multiple AI enhancement options tailored to your needs.\n\n"
        "Available AI Features\n\n"
        "✨ Full AI Enhance\n"
        "Automatically analyzes and enhances your image using the most suitable AI models.\n\n"
        "👤 Portrait Studio\n"
        "Professional portrait enhancement with natural facial improvements.\n\n"
        "🌙 Night Enhance\n"
        "Recover details from low-light and nighttime photos.\n\n"
        "🛂 Passport Studio\n"
        "Generate passport-compliant photos with automatic alignment and background correction.\n\n"
        "🎨 Cartoon Studio\n"
        "Transform photos into high-quality artistic cartoon styles.\n\n"
        "🎭 Face Restore\n"
        "Restore blurry or damaged faces.\n\n"
        "🔍 AI Upscale\n"
        "Increase image resolution with intelligent detail enhancement.\n\n"
        "🖼️ Remove Background\n"
        "Create transparent backgrounds with precise edge detection.\n\n"
        "⚡ Denoise & Sharpen\n"
        "Reduce image noise while preserving natural details.\n\n"
        "🎨 Colorize B&W\n"
        "Restore color to black-and-white photographs.\n\n"
        "📦 Smart Compress\n"
        "Reduce file size while maintaining excellent visual quality.\n\n"
        "Need assistance?\n\n"
        "Type\n\n"
        "/help\n\n"
        "to view commands and support information.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Builds the feature selection Inline Keyboard presented ONLY AFTER photo upload."""
    keyboard = [
        [
            InlineKeyboardButton("✨ Full AI Enhance", callback_data="ai_enhance"),
        ],
        [
            InlineKeyboardButton("👤 Portrait Studio", callback_data="ai_portrait"),
            InlineKeyboardButton("🌙 Night Enhance", callback_data="ai_night"),
        ],
        [
            InlineKeyboardButton("🛂 Passport Studio", callback_data="ai_passport"),
            InlineKeyboardButton("🎨 Cartoon Studio", callback_data="ai_cartoon"),
        ],
        [
            InlineKeyboardButton("🔍 2× Upscale", callback_data="ai_upscale_2x"),
            InlineKeyboardButton("🔎 4× Upscale", callback_data="ai_upscale_4x"),
        ],
        [
            InlineKeyboardButton("🎭 Face Restore", callback_data="ai_restore_gfp"),
            InlineKeyboardButton("🎭 Natural Restore", callback_data="ai_restore_code"),
        ],
        [
            InlineKeyboardButton("🖼️ Remove Background", callback_data="ai_removebg"),
            InlineKeyboardButton("⚡ Denoise & Sharpen", callback_data="ai_sharpen"),
        ],
        [
            InlineKeyboardButton("🎨 Colorize B&W", callback_data="ai_colorize"),
            InlineKeyboardButton("📦 Smart Compress", callback_data="ai_compress"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_help_message() -> str:
    """Returns the formatted /help message."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Zenemoo AI Help Center\n\n"
        "Available Commands\n\n"
        "/start\n"
        "Restart the bot\n\n"
        "/help\n"
        "Help and documentation\n\n"
        "/about\n"
        "About Zenemoo AI\n\n"
        "/contact\n"
        "Contact Support\n\n"
        "Supported Formats\n\n"
        "JPG\n\n"
        "PNG\n\n"
        "WEBP\n\n"
        "Maximum Upload Size\n\n"
        "20 MB\n\n"
        "Tips\n\n"
        "• Use high-resolution images whenever possible.\n\n"
        "• Portrait photos produce the best enhancement results.\n\n"
        "• Larger images may require additional processing time.\n\n"
        "Need additional assistance?\n\n"
        "📧 contact@mrprem.in\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_about_message() -> str:
    """Returns the formatted /about message."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "About Zenemoo AI\n\n"
        "Zenemoo AI is a professional AI-powered image enhancement platform built to deliver premium-quality photo restoration, enhancement, upscaling, and creative image transformation.\n\n"
        "Powered by modern deep learning models including:\n\n"
        "• GFPGAN\n\n"
        "• CodeFormer\n\n"
        "• Real-ESRGAN\n\n"
        "• SwinIR\n\n"
        "• U²-Net\n\n"
        "• DDColor\n\n"
        "• LaMa\n\n"
        "Version\n\n"
        "1.0\n\n"
        "Support\n\n"
        "contact@mrprem.in\n\n"
        "Website\n\n"
        "mrprem.in\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_contact_message() -> str:
    """Returns the formatted /contact message."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Zenemoo AI Support\n\n"
        "Email\n\n"
        "contact@mrprem.in\n\n"
        "Website\n\n"
        "mrprem.in\n\n"
        "We're happy to help with:\n\n"
        "• Technical support\n\n"
        "• Bug reports\n\n"
        "• Business inquiries\n\n"
        "• Feature suggestions\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_feature_action_message(action_name: str) -> str:
    """Returns the prompt when user clicks a button without an attached image."""
    feat_name = FEATURE_DISPLAY_NAMES.get(action_name, action_name.replace("ai_", "").replace("_", " ").title())
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_processing_message(pct: int = 20) -> str:
    """Returns the user-friendly progress message without estimated seconds or technical logs."""
    blocks = int(pct / 10)
    bar = "█" * blocks + "░" * (10 - blocks)

    stage_info = {
        20: ("🔍 Analyzing image...", "Checking image quality and detecting faces."),
        30: ("🧠 Preparing AI models...", "Loading enhancement pipeline."),
        50: ("🎯 Restoring image details...", "Applying advanced restoration models."),
        70: ("⚡ Enhancing image quality...", "Optimizing clarity, sharpness, and colors."),
        90: ("✨ Final quality optimization...", "Performing final refinements."),
        100: ("📦 Preparing your enhanced image...", "Finalizing output."),
    }

    status, sub = stage_info.get(pct, ("🧠 Processing image...", "Enhancing photo details."))

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🧠 **Zenemoo AI Processing**\n\n"
        "Image successfully received.\n\n"
        "**Current Status**\n\n"
        f"{status}\n\n"
        f"`{bar}` **{pct}%**\n\n"
        f"{sub}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def get_completion_message(
    action_name: str,
    res_w: int = 5120,
    res_h: int = 3840,
    quality_score: int = 97,
    processing_time: float = 11.4,
) -> str:
    """Returns the formatted completion report message without direct URLs."""
    feat_name = FEATURE_DISPLAY_NAMES.get(action_name, action_name.replace("ai_", "").replace("_", " ").title())
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✅ **Enhancement Complete**\n\n"
        "**Feature**\n\n"
        f"{feat_name}\n\n"
        "**Output Resolution**\n\n"
        f"{res_w} × {res_h}\n\n"
        "**AI Quality Score**\n\n"
        f"{quality_score} / 100\n\n"
        "**Processing Time**\n\n"
        f"{processing_time:.1f} seconds\n\n"
        "Image enhancement completed successfully.\n\n"
        "Thank you for choosing\n\n"
        "✨ **Zenemoo AI**\n\n"
        "Created by\n\n"
        "🌐 **mrprem.in**\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
