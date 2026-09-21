"""ASCII wordmark and bundled imagery for the built-in Python console."""

from __future__ import annotations

import io
import random
from importlib import resources
from pathlib import Path

from PIL import Image

# How far the brain's hue can drift from its bundled pink, as a fraction of
# the full hue circle - kept small so it always reads as a slight retint
# (lavender...pink...peach), never a jarring color change.
_MAX_HUE_SHIFT = 0.15

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
    """Path to the bundled pixel-art brain shown under the console's welcome
    banner."""
    resource = resources.files("neuroseq.resources").joinpath("console_brain.png")
    return Path(str(resource))


def startup_brain_image_path() -> Path:
    """Path to the bundled pixel-art brain flanking the startup screen's
    welcome title."""
    resource = resources.files("neuroseq.resources").joinpath("startup_brain.png")
    return Path(str(resource))


def tinted_brain_image_bytes(
    rng: random.Random | None = None, image_path: Path | None = None
) -> bytes:
    """One of the bundled brain images, hue-shifted by a small random
    amount, as PNG-encoded bytes ready for ``QImage.loadFromData()``.

    Rotating hue (rather than picking new colors outright) shifts every
    shade in the image - outline, fill, and highlight alike - by the same
    amount, so the existing shading/depth is preserved and it just reads as
    the same brain in a slightly different light. Defaults to the console's
    brain; pass ``startup_brain_image_path()`` for the startup screen's one.
    """
    rng = rng if rng is not None else random.Random()
    shift = int(round(rng.uniform(-_MAX_HUE_SHIFT, _MAX_HUE_SHIFT) * 255))
    path = image_path if image_path is not None else brain_image_path()
    with Image.open(path) as source:
        image = source.convert("RGBA")
    red, green, blue, alpha = image.split()
    hue, saturation, value = (
        Image.merge("RGB", (red, green, blue)).convert("HSV").split()
    )
    hue = hue.point(lambda p: (p + shift) % 256)
    tinted_rgb = Image.merge("HSV", (hue, saturation, value)).convert("RGB")
    tinted = Image.merge("RGBA", (*tinted_rgb.split(), alpha))
    buffer = io.BytesIO()
    tinted.save(buffer, format="PNG")
    return buffer.getvalue()
