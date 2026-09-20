"""ASCII wordmark and bundled imagery for the built-in Python console."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

# 5-row block font, only the glyphs the banner needs. Every glyph is 5 columns
# wide and is joined with a single space.
_GLYPH_ROWS = 5
_FONT: dict[str, tuple[str, ...]] = {
    " ": ("     ", "     ", "     ", "     ", "     "),
    "C": (" ####", "#    ", "#    ", "#    ", " ####"),
    "E": ("#####", "#    ", "#### ", "#    ", "#####"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "#  ##", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", " ### "),
    "Q": (" ### ", "#   #", "#   #", "# # #", " ## #"),
    "R": ("#### ", "#   #", "#### ", "#  # ", "#   #"),
    "S": (" ####", "#    ", " ### ", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", " ### "),
    "W": ("#   #", "#   #", "# # #", "## ##", "#   #"),
}

_BANNER_TEXT = "WELCOME TO NEUROSEQ"


def _render_line(text: str) -> list[str]:
    """Render one line of text in the block font."""
    rows = [""] * _GLYPH_ROWS
    for index, character in enumerate(text):
        glyph = _FONT.get(character.upper())
        if glyph is None:
            glyph = _FONT[" "]
        gap = " " if index else ""
        for row in range(_GLYPH_ROWS):
            rows[row] += gap + glyph[row]
    return [row.rstrip() for row in rows]


def welcome_banner() -> list[str]:
    """The 'Welcome to NeuroSeq' wordmark, on a single line, as block-font rows."""
    return _render_line(_BANNER_TEXT)


def brain_image_path() -> Path:
    """Path to the bundled pixel-art brain shown under the welcome banner."""
    resource = resources.files("neuroseq.resources").joinpath("console_brain.png")
    return Path(str(resource))
