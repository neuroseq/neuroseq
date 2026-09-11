from __future__ import annotations

import random

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


def test_brain_art_is_ascii_and_rectangular_enough_to_align() -> None:
    art = console_art.brain_art(random.Random(0))
    assert art, "brain art must not be empty"
    assert art[0].strip() and art[-1].strip(), "no blank rows at the edges"
    for line in art:
        assert line.isascii()
        assert len(line) <= console_art._BRAIN_WIDTH
        assert line == line.rstrip()


def test_brain_art_has_two_hemispheres_and_a_stem() -> None:
    art = console_art.brain_art(random.Random(4))
    # The stem hangs off the underside as the last two rows.
    assert art[-1].strip() == "\\|/"
    assert "|" in art[-2]
    # The fold pattern is mirrored, so both hemispheres carry gyri characters.
    body = art[:-2]
    midpoint = console_art._BRAIN_WIDTH // 2
    left = sum(line[:midpoint].count("~") for line in body)
    right = sum(line[midpoint:].count("~") for line in body)
    assert left > 0 and right > 0


def test_brain_art_differs_between_runs() -> None:
    # A different seed must grow a different brain; the same seed must not.
    first = console_art.brain_art(random.Random(1))
    second = console_art.brain_art(random.Random(2))
    assert first != second
    assert first == console_art.brain_art(random.Random(1))


def test_brain_art_is_stable_across_many_seeds() -> None:
    # Guards the silhouette maths against seeds that would produce an empty or
    # broken shape.
    for seed in range(50):
        art = console_art.brain_art(random.Random(seed))
        assert len(art) >= 6
        assert sum(line.count("~") for line in art) > 0


def test_console_splash_stacks_banner_over_brain() -> None:
    splash = console_art.console_splash(random.Random(3))
    lines = splash.split("\n")
    banner = console_art.welcome_banner()
    assert lines[0].strip() == ""
    assert [line.rstrip() for line in lines[1 : 1 + len(banner)]] == banner
    assert splash.endswith("\n")
    assert splash.isascii()
    brain_rows = [line for line in lines if "~" in line]
    assert brain_rows


def test_console_splash_lines_are_equal_length_for_centering() -> None:
    # The console centers each block independently, so lines of unequal length
    # would shear the art. Padding to a common width is what prevents that.
    splash = console_art.console_splash(random.Random(5))
    lines = splash.split("\n")[:-1]  # trailing newline leaves an empty tail
    widths = {len(line) for line in lines}
    assert len(widths) == 1
    assert next(iter(widths)) == max(len(line) for line in console_art.welcome_banner())


def test_console_splash_uses_a_fresh_brain_without_a_seed() -> None:
    assert console_art.console_splash() != console_art.console_splash()
