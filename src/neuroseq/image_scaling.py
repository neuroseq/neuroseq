"""Scaling images to 8 bits for display and brightness/contrast. Qt-free.

The Cellpose importer's display image and the brightness/contrast window
(``visualization_support.brightness_contrast``) must use the same scale, so both
call :func:`scale_to_uint8`.
"""

from __future__ import annotations

import numpy as np

#: Fraction of pixels at each end ignored when scaling, as Fiji's Auto ignores
#: histogram bins under 1/5000 of the pixels: a few hot (or dead) pixels, common in
#: two-photon mean images, would otherwise squeeze the tissue into a few grey levels.
CLIP_FRACTION = 1 / 5000


def display_limits(values: np.ndarray) -> tuple[float, float]:
    """The intensities mapped to 0 and 255: the 0.02nd and 99.98th percentiles of
    the finite values, or their minimum and maximum if those coincide."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    # Rounded outwards: an image of fewer than ~5000 pixels keeps its extremes.
    lo = float(np.percentile(finite, 100 * CLIP_FRACTION, method="lower"))
    hi = float(np.percentile(finite, 100 * (1 - CLIP_FRACTION), method="higher"))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    return lo, hi


def scale_to_uint8(values: np.ndarray) -> np.ndarray:
    """``values`` scaled linearly from :func:`display_limits` onto 0-255 (uint8);
    the clipped extremes saturate. A constant image becomes all zeros."""
    array = np.asarray(values, dtype=np.float64)
    lo, hi = display_limits(array)
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(array, nan=lo) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
