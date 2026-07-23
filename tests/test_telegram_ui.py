"""
==============================================================================
Zenemoo AI - Telegram UI & Menu Builder Test Suite
==============================================================================
Validates modern Telegram menu layout, section buttons, action prompts,
processing progress frames, and completion report formatting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients.telegram.ui.menu_builder import (
    get_welcome_message,
    get_main_menu_keyboard,
    get_feature_action_message,
    get_processing_message,
    get_completion_message,
    FEATURE_DISPLAY_NAMES,
)


def run_ui_tests():
    print("=" * 80)
    print("🤖 ZENEMOO AI - TELEGRAM UI REDESIGN TEST SUITE")
    print("=" * 80)

    # 1. Test Welcome Message
    print("\n[1] Testing Welcome Message formatting...")
    welcome_text = get_welcome_message()
    assert "Welcome to Zenemoo AI" in welcome_text
    assert "Full AI Enhance" in welcome_text
    assert "AI Studio" in welcome_text
    assert "AI Enhancement" in welcome_text
    assert "AI Tools" in welcome_text
    print("  ✔ Welcome message format matches specifications.")

    # 2. Test Main Menu Keyboard Layout & Buttons
    print("\n[2] Testing Main Menu Inline Keyboard Layout...")
    kb = get_main_menu_keyboard()
    rows = kb.inline_keyboard

    # Row 1: Section 1 (Full AI Enhance single button)
    assert len(rows[0]) == 1
    assert rows[0][0].text == "✨ Full AI Enhance"
    assert rows[0][0].callback_data == "ai_enhance"

    # Row 2 & 3: Section 2 (AI Studio two-column)
    assert len(rows[1]) == 2
    assert rows[1][0].text == "👤 Portrait AI" and rows[1][0].callback_data == "ai_portrait"
    assert rows[1][1].text == "🌙 Night Enhance" and rows[1][1].callback_data == "ai_night"

    assert len(rows[2]) == 2
    assert rows[2][0].text == "🛂 Passport Studio" and rows[2][0].callback_data == "ai_passport"
    assert rows[2][1].text == "🎨 Cartoon Studio" and rows[2][1].callback_data == "ai_cartoon"

    # Row 4 & 5: Section 3 (AI Enhancement two-column)
    assert len(rows[3]) == 2
    assert rows[3][0].text == "🔍 2× Upscale" and rows[3][0].callback_data == "ai_upscale_2x"
    assert rows[3][1].text == "🔎 4× Upscale" and rows[3][1].callback_data == "ai_upscale_4x"

    assert len(rows[4]) == 2
    assert rows[4][0].text == "🎭 Face Restore" and rows[4][0].callback_data == "ai_restore_gfp"
    assert rows[4][1].text == "🎭 Natural Restore" and rows[4][1].callback_data == "ai_restore_code"

    # Row 6 & 7: Section 4 (AI Tools two-column)
    assert len(rows[5]) == 2
    assert rows[5][0].text == "🖼️ Remove Background" and rows[5][0].callback_data == "ai_removebg"
    assert rows[5][1].text == "⚡ Denoise" and rows[5][1].callback_data == "ai_sharpen"

    assert len(rows[6]) == 2
    assert rows[6][0].text == "🎨 Colorize B&W" and rows[6][0].callback_data == "ai_colorize"
    assert rows[6][1].text == "📦 Smart Compress" and rows[6][1].callback_data == "ai_compress"

    print("  ✔ Inline keyboard section structure & button titles match specifications.")

    # 3. Test Feature Action Prompt Message
    print("\n[3] Testing Feature Action Prompt Message...")
    prompt = get_feature_action_message("ai_portrait")
    assert "Portrait Studio" in prompt
    assert "Upload your image" in prompt
    assert "Maximum Size:" in prompt
    print("  ✔ Feature action prompt screen matches specifications.")

    # 4. Test Processing Screen Message
    print("\n[4] Testing Processing Screen Message...")
    proc_text = get_processing_message("Enhancement Pipeline", 80)
    assert "Zenemoo AI Processing" in proc_text
    assert "Image Received" in proc_text
    assert "AI Analysis Complete" in proc_text
    assert "Enhancement Pipeline" in proc_text
    assert "80%" in proc_text
    print("  ✔ Processing screen format matches specifications.")

    # 5. Test Completion Screen Message
    print("\n[5] Testing Completion Screen Message...")
    comp_text = get_completion_message("ai_portrait", 4000, 3000, 97, 11.4)
    assert "Processing Complete" in comp_text
    assert "Portrait Studio" in comp_text
    assert "4000 × 3000" in comp_text
    assert "97 / 100" in comp_text
    assert "11.4 sec" in comp_text
    print("  ✔ Completion report format matches specifications.")

    print("\n" + "=" * 80)
    print("🏆 TELEGRAM BOT UI REDESIGN TEST SUITE PASSED 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_ui_tests()
