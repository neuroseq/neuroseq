from __future__ import annotations

import io
import random

from PIL import Image

from neuroseq import console_art


def test_welcome_banner_spells_the_greeting_on_one_line() -> None:
    banner = console_art.welcome_banner()
    # A single line of block font, five rows tall.
    assert len(banner) == console_art._GLYPH_ROWS
    assert all(set(line) <= {"#", " "} for line in banner)
    assert any("#" in line for line in banner)
    # Every glyph is 5 columns wide and joined with one space.
    expected_width = len(console_art._BANNER_TEXT) * 6 - 1
    assert all(len(line) == expected_width for line in banner)


def test_brain_image_path_points_at_a_real_bundled_file() -> None:
    path = console_art.brain_image_path()
    assert path.exists()
    assert path.suffix == ".png"


def test_console_brain_has_a_transparent_background() -> None:
    with Image.open(console_art.brain_image_path()) as image:
        assert image.mode == "RGBA"
        alpha = image.getchannel("A")
        assert alpha.getextrema() == (0, 255)
        for corner in (
            (0, 0),
            (image.width - 1, 0),
            (0, image.height - 1),
            (image.width - 1, image.height - 1),
        ):
            # Allow negligible alpha left by background extraction.
            assert alpha.getpixel(corner) <= 1


def test_console_brain_right_half_is_transparent_between_details() -> None:
    with Image.open(console_art.brain_image_path()) as image:
        alpha = image.getchannel("A")
        # Gaps between binary digits and circuit traces must show the console.
        for x, y in ((0.6, 0.6), (0.65, 0.7), (0.7, 0.3), (0.75, 0.7), (0.8, 0.6)):
            assert alpha.getpixel((int(image.width * x), int(image.height * y))) == 0


def test_startup_brain_image_path_points_at_a_different_real_bundled_file() -> None:
    path = console_art.startup_brain_image_path()
    assert path.exists()
    assert path.suffix == ".png"
    assert path != console_art.brain_image_path()


def test_tinted_brain_image_bytes_can_tint_the_startup_image_too() -> None:
    startup_path = console_art.startup_brain_image_path()
    original = Image.open(startup_path).convert("RGBA")
    tinted_bytes = console_art.tinted_brain_image_bytes(random.Random(0), startup_path)
    tinted = Image.open(io.BytesIO(tinted_bytes))

    assert tinted.size == original.size
    assert tinted.split()[3].tobytes() == original.split()[3].tobytes()
    # Sanity check it didn't silently fall back to the console default.
    assert tinted_bytes != console_art.tinted_brain_image_bytes(random.Random(0))


def test_tinted_brain_image_is_a_recolored_copy_of_the_bundled_png() -> None:
    original = Image.open(console_art.brain_image_path()).convert("RGBA")
    tinted_bytes = console_art.tinted_brain_image_bytes(random.Random(0))
    tinted = Image.open(io.BytesIO(tinted_bytes))

    assert tinted.mode == "RGBA"
    assert tinted.size == original.size
    # Transparency (the shape's silhouette) is untouched by the hue shift.
    assert tinted.split()[3].tobytes() == original.split()[3].tobytes()


def test_tinted_brain_image_varies_with_the_rng_but_is_deterministic() -> None:
    first = console_art.tinted_brain_image_bytes(random.Random(1))
    second = console_art.tinted_brain_image_bytes(random.Random(2))
    assert first != second
    assert first == console_art.tinted_brain_image_bytes(random.Random(1))


def test_tinted_brain_image_shift_stays_slight() -> None:
    # A wildly different hue would defeat the point of a "slight" retint -
    # measure the actual hue delta on a real pixel, not just the RNG range.
    original = Image.open(console_art.brain_image_path()).convert("RGBA")
    # The most saturated pixel is the one where a hue shift is most visible.
    original_hue_band, saturation_band, _ = (
        original.convert("RGB").convert("HSV").split()
    )
    saturation_bytes = saturation_band.tobytes()
    peak_index = saturation_bytes.index(max(saturation_bytes))
    x, y = peak_index % original.width, peak_index // original.width
    original_hue = original_hue_band.getpixel((x, y))
    assert isinstance(original_hue, int)

    max_delta = 0
    for seed in range(20):
        tinted_bytes = console_art.tinted_brain_image_bytes(random.Random(seed))
        tinted_hue_band = (
            Image.open(io.BytesIO(tinted_bytes))
            .convert("RGB")
            .convert("HSV")
            .split()[0]
        )
        tinted_hue = tinted_hue_band.getpixel((x, y))
        assert isinstance(tinted_hue, int)
        delta = min(
            (tinted_hue - original_hue) % 256, (original_hue - tinted_hue) % 256
        )
        max_delta = max(max_delta, delta)

    assert 0 < max_delta <= round(console_art._MAX_HUE_SHIFT * 255) + 1
