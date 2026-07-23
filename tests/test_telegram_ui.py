"""
==============================================================================
Zenemoo AI - Telegram Commercial UI Test Suite
==============================================================================
Validates unified single welcome message onboarding flow, /help, /about,
/contact commands, inline keyboard buttons, progress stages, and completion report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.telegram.ui.menu_builder import (
    get_unified_welcome_message,
    get_main_menu_keyboard,
    get_help_message,
    get_about_message,
    get_contact_message,
    get_feature_action_message,
    get_processing_message,
    get_completion_message,
)


def run_ui_tests():
    print("=" * 80)
    print("🤖 ZENEMOO AI - COMMERCIAL TELEGRAM UI TEST SUITE")
    print("=" * 80)

    # 1. Test Unified Single Welcome Message
    print("\n[1] Testing Unified Single Welcome Message...")
    msg = get_unified_welcome_message("Prem")
    assert "Welcome, Prem!" in msg
    assert "Zenemoo AI" in msg
    assert "📸 Getting Started" in msg
    assert "Simply send any image to begin." in msg
    assert "✨ Available AI Features" in msg
    assert "Full AI Enhance" in msg
    assert "Portrait Studio" in msg
    assert "Night Enhance" in msg
    assert "Passport Studio" in msg
    assert "Cartoon Studio" in msg
    assert "Face Restore" in msg
    assert "AI Upscale" in msg
    assert "Remove Background" in msg
    assert "Denoise & Sharpen" in msg
    assert "Colorize B&W" in msg
    assert "Smart Compress" in msg
    print("  ✔ Unified Single Welcome message format matches specifications.")

    # 2. Test Main Menu Inline Keyboard Layout (Post-Photo Upload)
    print("\n[2] Testing Post-Photo Upload Inline Keyboard Layout...")
    kb = get_main_menu_keyboard()
    rows = kb.inline_keyboard

    assert len(rows[0]) == 1 and rows[0][0].text == "✨ Full AI Enhance"
    assert len(rows[1]) == 2 and rows[1][0].text == "👤 Portrait Studio" and rows[1][1].text == "🌙 Night Enhance"
    assert len(rows[2]) == 2 and rows[2][0].text == "🛂 Passport Studio" and rows[2][1].text == "🎨 Cartoon Studio"
    assert len(rows[3]) == 2 and rows[3][0].text == "🔍 2× Upscale" and rows[3][1].text == "🔎 4× Upscale"
    assert len(rows[4]) == 2 and rows[4][0].text == "🎭 Face Restore" and rows[4][1].text == "🎭 Natural Restore"
    assert len(rows[5]) == 2 and rows[5][0].text == "🖼️ Remove Background" and rows[5][1].text == "⚡ Denoise & Sharpen"
    assert len(rows[6]) == 2 and rows[6][0].text == "🎨 Colorize B&W" and rows[6][1].text == "📦 Smart Compress"
    print("  ✔ Post-photo upload inline keyboard structure matches specifications.")

    # 3. Test /help, /about, and /contact Messages
    print("\n[3] Testing /help, /about, and /contact Commands...")
    help_txt = get_help_message()
    assert "Zenemoo AI Help Center" in help_txt
    assert "Supported Image Formats" in help_txt
    assert "contact@mrprem.in" in help_txt

    about_txt = get_about_message()
    assert "About Zenemoo AI" in about_txt
    assert "mrprem.in" in about_txt

    contact_txt = get_contact_message()
    assert "Zenemoo AI Support" in contact_txt
    assert "contact@mrprem.in" in contact_txt
    print("  ✔ /help, /about, and /contact messages match specifications.")

    # 4. Test User-Friendly Processing Screen
    print("\n[4] Testing User-Friendly Processing Screen...")
    proc_text = get_processing_message(50)
    assert "Zenemoo AI Processing" in proc_text
    assert "Image successfully received." in proc_text
    assert "Restoring image details..." in proc_text
    assert "50%" in proc_text
    assert "seconds" not in proc_text  # Estimated seconds hidden as required
    print("  ✔ Processing screen format matches specifications.")

    # 5. Test Completion Screen
    print("\n[5] Testing Completion Screen...")
    comp_text = get_completion_message("ai_enhance", 5120, 3840, 97, 219.2)
    assert "Enhancement Complete" in comp_text
    assert "Full AI Enhance" in comp_text
    assert "5120 × 3840" in comp_text
    assert "97 / 100" in comp_text
    assert "219.2 seconds" in comp_text
    assert "mrprem.in" in comp_text
    assert "http" not in comp_text  # Direct URLs hidden as required
    print("  ✔ Completion report format matches specifications.")

    print("\n" + "=" * 80)
    print("🏆 COMMERCIAL TELEGRAM UI TEST SUITE PASSED 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_ui_tests()
