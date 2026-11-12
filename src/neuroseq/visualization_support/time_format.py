"""Qt-free formatting of time readouts adapted to the visible time range.

Plots keep their time axis in milliseconds. A readout of a recording that is
minutes long (``123456ms``) is hard to read, while one zoomed in on a few
laser pulses needs sub-millisecond detail. ``format_time_ms`` picks the unit
(ms or s) and the number of decimals from the width of the visible range.
"""

from __future__ import annotations

import math

# Visible ranges at least this wide (in ms) are shown in seconds.
SECONDS_THRESHOLD_MS = 2000.0
# Readouts resolve about 1/100 of the visible range...
RESOLUTION_FRACTION = 0.01
# ...but never more finely than this many decimals in the chosen unit.
MAX_DECIMALS = 3


def time_unit_for_span(span_ms: float) -> str:
    """``"s"`` for a visible range of at least 2 s, else ``"ms"``."""
    return "s" if abs(span_ms) >= SECONDS_THRESHOLD_MS else "ms"


def time_decimals_for_span(span_ms: float) -> int:
    """Decimals needed to resolve about 1 % of ``span_ms`` in its unit."""
    unit = time_unit_for_span(span_ms)
    span = abs(span_ms) / (1000.0 if unit == "s" else 1.0)
    resolution = span * RESOLUTION_FRACTION
    if not math.isfinite(resolution) or resolution <= 0:
        return MAX_DECIMALS if unit == "ms" else 0
    return min(MAX_DECIMALS, max(0, math.ceil(-math.log10(resolution) - 1e-9)))


def format_time_ms(time_ms: float, span_ms: float, *, strip_zeros: bool = False) -> str:
    """Format ``time_ms`` with unit and precision adapted to ``span_ms``.

    ``span_ms`` is the width of the visible time range. With ``strip_zeros``
    trailing zeros are dropped (``"0.5 s"`` rather than ``"0.50 s"``), which
    suits axis ticks whose step is not known here; readouts that update while
    the pointer moves keep a fixed width instead.
    """
    unit = time_unit_for_span(span_ms)
    decimals = time_decimals_for_span(span_ms)
    value = float(time_ms) / (1000.0 if unit == "s" else 1.0)
    text = f"{value:.{decimals}f}"
    if strip_zeros and "." in text:
        text = text.rstrip("0").rstrip(".")
    if float(text) == 0.0:
        text = text.lstrip("-")
    return f"{text} {unit}"
