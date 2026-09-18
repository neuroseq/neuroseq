"""ASCII art for the built-in Python console.

The welcome wordmark is fixed; the brain underneath it is generated fresh on
every launch, so no two sessions open with quite the same one. Everything here
is plain ASCII rendered on a fixed character grid, which the console's
Courier New output widget lines up correctly.
"""

from __future__ import annotations

import math
import random

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

# Character grid the brain is drawn on. Cells are about twice as tall as they
# are wide, which the aspect corrections below account for.
_BRAIN_WIDTH = 46

# Hand-tuned (left, right) column bounds per row for a right-facing sagittal
# (side) profile: frontal pole at the top right, curving back over the vertex
# to the occipital pole at the top left, narrowing at the back for the
# cerebellum and brainstem notch, then the temporal lobe bulging back out at
# the front before tapering to a point underneath. Randomizing this directly
# (rather than a smooth radius-vs-angle formula) is what keeps it reading as
# a brain at this resolution - a couple of concave notches (cerebellum,
# sylvian fissure) are too sharp for a low row count to render smoothly.
_BASE_SPANS: tuple[tuple[int, int], ...] = (
    (19, 30),
    (14, 34),
    (10, 37),
    (7, 40),
    (5, 42),
    (4, 42),
    (4, 39),
    (5, 36),
    (7, 32),
    (9, 35),
    (11, 33),
    (13, 31),
    (16, 28),
    (19, 25),
)


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


def brain_art(rng: random.Random | None = None) -> list[str]:
    """Draw a sagittal (side-view) brain, its sulci, size, and outline all
    varying per call, so no two sessions grow quite the same one.

    The silhouette is a hand-tuned base shape, lightly rescaled and jittered
    row by row for the size/shape variation; the gyri inside it are a fresh
    set of concentric ripples every time, rendered as 0s and 1s for a
    "digital" looking brain.
    """
    rng = rng if rng is not None else random.Random()
    ring_spacing = rng.uniform(5.0, 7.5)
    swirl = rng.uniform(0.9, 1.7)
    lobes = rng.choice((2.0, 3.0, 4.0))
    phase = rng.uniform(0.0, math.tau)

    # Rescale the silhouette (brain "size") and jitter each row's bounds a
    # little (sulci/shape variation) without breaking the smooth left/right
    # progression the outline stitching below relies on.
    scale = rng.uniform(0.92, 1.08)
    mid = (_BRAIN_WIDTH - 1) / 2
    spans = []
    for base_left, base_right in _BASE_SPANS:
        left = mid + (base_left - mid) * scale + rng.uniform(-0.8, 0.8)
        right = mid + (base_right - mid) * scale + rng.uniform(-0.8, 0.8)
        left_column = max(0, min(_BRAIN_WIDTH - 2, round(left)))
        right_column = max(left_column + 1, min(_BRAIN_WIDTH - 1, round(right)))
        spans.append((left_column, right_column))
    height = len(spans)

    half_width = (_BRAIN_WIDTH - 1) / 2
    half_height = (height - 1) / 2
    xs = [(column - half_width) / half_width for column in range(_BRAIN_WIDTH)]
    ys = [(row - half_height) / half_height for row in range(height)]

    def fold(x: float, y: float) -> float:
        # Cells are about twice as tall as they are wide, hence the y stretch.
        adjusted_y = y * 1.1
        shifted_x = x + 0.15
        distance = math.hypot(shifted_x, adjusted_y)
        theta = math.atan2(adjusted_y, shifted_x)
        return math.sin(
            ring_spacing * distance + swirl * math.sin(lobes * theta + phase) + phase
        )

    grid = [[" "] * _BRAIN_WIDTH for _ in range(height)]
    for row, (left, right) in enumerate(spans):
        above = spans[row - 1] if row else None
        below = spans[row + 1] if row + 1 < height else None
        for column in range(left + 1, right):
            x = xs[column]
            value = fold(x, ys[row])
            if abs(value) < 0.58:
                if abs(value) < 0.22:
                    grid[row][column] = rng.choice("01")
                else:
                    grid[row][column] = "0" if value < 0 else "1"
        if above is None or below is None:
            for column in range(left, right + 1):
                grid[row][column] = "_"
            continue
        # Bridge sideways where the silhouette jumps more than one column
        # between rows, so the outline never breaks into floating slashes.
        if left < above[0] - 1:
            for column in range(left + 1, above[0]):
                grid[row][column] = "_"
            grid[row][left] = "/"
        elif left > above[0] + 1:
            for column in range(above[0], left):
                grid[row][column] = "_"
            grid[row][left] = "\\"
        else:
            grid[row][left] = (
                "/" if left < above[0] else "\\" if left > above[0] else "("
            )
        if right > above[1] + 1:
            for column in range(above[1] + 1, right):
                grid[row][column] = "_"
            grid[row][right] = "\\"
        elif right < above[1] - 1:
            for column in range(right + 1, above[1] + 1):
                grid[row][column] = "_"
            grid[row][right] = "/"
        else:
            grid[row][right] = (
                "\\" if right > above[1] else "/" if right < above[1] else ")"
            )

    lines = ["".join(row).rstrip() for row in grid]

    # The brainstem hangs from underneath the cerebellum notch at the back of
    # the last row, not dead center - there's no bilateral fissure to center on.
    last_left, last_right = spans[-1]
    stem_column = round((last_left + last_right) / 2) - 4
    stem_column = max(6, min(_BRAIN_WIDTH - 6, stem_column))
    lines.append(" " * (stem_column - 3) + "\\__ | __/")
    lines.append(" " * (stem_column - 1) + "\\|/")
    return lines


def _center(lines: list[str], width: int) -> list[str]:
    """Indent every line by the same amount, so the art keeps its shape."""
    block_width = max((len(line) for line in lines), default=0)
    indent = " " * max(0, (width - block_width) // 2)
    return [(indent + line).rstrip() for line in lines]


def console_splash(rng: random.Random | None = None) -> str:
    """Full console splash: the wordmark, a freshly grown brain, and a rule.

    Every line is padded to the same width. The console centers the splash by
    giving its text blocks a centered alignment, and Qt centers each line
    independently — so equal-length lines are what keeps the art from
    shearing, however wide the console happens to be.
    """
    banner = welcome_banner()
    width = max(len(line) for line in banner)
    brain = _center(brain_art(rng), width)
    parts = ["", *banner, "", *brain, "", "-" * width, ""]
    return "\n".join(line.ljust(width) for line in parts) + "\n"
