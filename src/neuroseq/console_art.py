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
_BRAIN_HEIGHT = 13


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
    """Draw a brain, with a fold pattern and outline that vary per call.

    The silhouette is only mildly perturbed so it always reads as a brain; the
    gyri are what really change, as concentric ripples mirrored across the
    longitudinal fissure.
    """
    rng = rng if rng is not None else random.Random()
    lumps = [(rng.uniform(0.0, math.tau), rng.uniform(0.02, 0.05), k) for k in (4, 6)]
    ring_spacing = rng.uniform(5.0, 7.5)
    swirl = rng.uniform(0.9, 1.7)
    lobes = rng.choice((2.0, 3.0, 4.0))
    phase = rng.uniform(0.0, math.tau)

    def radius(theta: float) -> float:
        return 1.0 + sum(amp * math.sin(k * theta + p) for p, amp, k in lumps)

    def inside(x: float, y: float) -> bool:
        # Squash vertically, and flatten the underside a little more than the
        # crown, so the silhouette sits somewhere between a circle and a brain.
        adjusted_y = y * 1.05 * (1.25 if y > 0 else 1.0)
        distance = math.hypot(x, adjusted_y)
        if distance == 0.0:
            return True
        return distance <= radius(math.atan2(adjusted_y, x))

    def fold(x: float, y: float) -> float:
        mirrored_x = abs(x) + 0.12
        distance = math.hypot(mirrored_x, y * 1.15)
        theta = math.atan2(y, mirrored_x)
        return math.sin(
            ring_spacing * distance + swirl * math.sin(lobes * theta + phase) + phase
        )

    half_width = (_BRAIN_WIDTH - 1) / 2
    half_height = (_BRAIN_HEIGHT - 1) / 2
    xs = [(column - half_width) / half_width for column in range(_BRAIN_WIDTH)]
    ys = [(row - half_height) / half_height for row in range(_BRAIN_HEIGHT)]

    spans: list[tuple[int, int] | None] = []
    for y in ys:
        columns = [column for column, x in enumerate(xs) if inside(x, y)]
        spans.append((columns[0], columns[-1]) if columns else None)

    grid = [[" "] * _BRAIN_WIDTH for _ in range(_BRAIN_HEIGHT)]
    for row, span in enumerate(spans):
        if span is None:
            continue
        left, right = span
        above = spans[row - 1] if row else None
        below = spans[row + 1] if row + 1 < _BRAIN_HEIGHT else None
        for column in range(left + 1, right):
            x = xs[column]
            if abs(x) < 0.05:
                continue  # the longitudinal fissure between the hemispheres
            value = fold(x, ys[row])
            if abs(value) < 0.58:
                if abs(value) < 0.24:
                    grid[row][column] = "~"
                else:
                    grid[row][column] = "(" if x < 0 else ")"
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

    # Strip the empty grid rows before adding the stem, so it always hangs
    # directly off the underside rather than floating below a gap.
    lines = _strip_blank_edges(["".join(row).rstrip() for row in grid])
    stem_column = _BRAIN_WIDTH // 2
    lines.append(" " * (stem_column - 3) + "\\__ | __/")
    lines.append(" " * (stem_column - 1) + "\\|/")
    return lines


def _strip_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


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
