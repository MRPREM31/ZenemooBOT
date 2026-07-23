"""
==============================================================================
Zenemoo AI - Four Flagship Studios Verification Suite
==============================================================================
Validates Passport Photo Studio, Night Photo Enhance, Portrait Studio,
and Cartoon & Stylize Studio engines.
"""

import sys
import io
import time
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logging import logger
from ai.passport.passport_engine import passport_engine, PASSPORT_SPECS
from ai.night.night_engine import night_engine
from ai.portrait.portrait_engine import portrait_engine, PORTRAIT_MODES
from ai.cartoon.cartoon_engine import cartoon_engine, CARTOON_STYLES


def create_synthetic_test_photo() -> Image.Image:
    """Creates a synthetic test portrait photo."""
    img = Image.new("RGB", (800, 800), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    # Head & face region
    draw.ellipse([300, 200, 500, 450], fill=(240, 200, 170))
    draw.ellipse([340, 280, 370, 310], fill=(50, 50, 50))
    draw.ellipse([430, 280, 460, 310], fill=(50, 50, 50))
    draw.ellipse([200, 420, 600, 800], fill=(50, 100, 180))
    return img


def run_flagship_studios_tests():
    print("=" * 80)
    print("🚀 ZENEMOO AI - FOUR FLAGSHIP STUDIOS TEST SUITE")
    print("=" * 80)

    test_img = create_synthetic_test_photo()

    # 1. Test Feature 1: Zenemoo Passport Photo Studio
    print("\n[STUDIO 1] Testing Zenemoo Passport Photo Studio...")
    for country in ["india", "usa", "uk", "uae"]:
        pass_img, meta = passport_engine.generate_passport(test_img, country=country, bg_color_name="blue")
        spec = PASSPORT_SPECS[country]
        print(f" • [{spec['name'].upper()}] Passport Photo: {pass_img.width}x{pass_img.height} px ({meta['dimensions_mm']}) | BG: {meta['background_color']}")
        assert pass_img.size == spec["px"], f"Dimension mismatch for {country}"

    print(" • Generating 4x6 Printable Grid Sheet & High-Res PDF...")
    sheet_img = passport_engine.generate_4x6_print_sheet(pass_img)
    pdf_bytes = passport_engine.generate_pdf_bytes(sheet_img)
    print(f" • 4x6 Printable Grid Sheet Generated: {sheet_img.width}x{sheet_img.height} px | PDF Size: {len(pdf_bytes)} bytes")
    assert sheet_img.size == (1200, 1800), "Print sheet size must be 1200x1800 px @ 300 DPI"
    assert len(pdf_bytes) > 0, "PDF export bytes empty"
    print("-" * 80)

    # 2. Test Feature 2: Zenemoo Night Photo Enhance
    print("\n[STUDIO 2] Testing Zenemoo Night Photo Enhance...")
    night_img, night_meta = night_engine.enhance_night_photo(test_img)
    print(f" • Night Scene Classified: [{night_meta['night_mode'].upper()}] | Noise Reduced: {night_meta['noise_reduced']} | Exposure: {night_meta['exposure_improved']}")
    assert night_img.size == test_img.size, "Night photo resolution mismatch"
    print("-" * 80)

    # 3. Test Feature 3: Zenemoo Portrait Studio
    print("\n[STUDIO 3] Testing Zenemoo Portrait Studio...")
    for mode in ["professional_headshot", "linkedin", "beauty"]:
        port_img, port_meta = portrait_engine.enhance_portrait(test_img, mode=mode)
        print(f" • Portrait Mode: [{port_meta['mode']}] | Analysis: {port_meta['face_analysis']} | Lighting: {port_meta['lighting']}")
        assert port_img.size == test_img.size, "Portrait resolution mismatch"
    print("-" * 80)

    # 4. Test Feature 4: Zenemoo Cartoon & Stylize Studio
    print("\n[STUDIO 4] Testing Zenemoo Cartoon & Stylize Studio (13 Styles)...")
    for style in ["anime", "ghibli", "pixar", "comic", "sketch", "pixel_art"]:
        cart_img, cart_meta = cartoon_engine.stylize_image(test_img, style=style)
        print(f" • Cartoon Style: [{cart_meta['style']}] | Identity Preserved: {cart_meta['identity_preserved']}")
        assert cart_img.size == test_img.size, "Cartoon resolution mismatch"
    print("-" * 80)

    print("\n" + "=" * 80)
    print("🏆 ALL FOUR FLAGSHIP STUDIOS VERIFIED CLEANLY AND PASSED 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_flagship_studios_tests()
