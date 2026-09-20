from __future__ import annotations

from neuroseq import console_art


def test_welcome_banner_spells_the_greeting_on_one_line() -> None:
    banner = console_art.welcome_banner()
    # A single line of block font, five rows tall.
    assert len(banner) == console_art._GLYPH_ROWS
    assert all(set(line) <= {"#", " "} for line in banner)
    assert any("#" in line for line in banner)
    # Every glyph is 5 columns wide and joined with one space.
    expected_width = len(console_art._BANNER_TEXT) * 6 - 1
    assert max(len(line) for line in banner) == expected_width


def test_brain_image_path_points_at_a_real_bundled_file() -> None:
    path = console_art.brain_image_path()
    assert path.exists()
    assert path.suffix == ".png"
