"""Fiji-style Brightness/Contrast for the Spatial Footprints FOV image.

The adjustment is display-only: a linear min/max window applied to the 8-bit
FOV (what Fiji's Image > Adjust > Brightness/Contrast does to the display
range before "Apply"), so the stored background image is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.image_scaling import scale_to_uint8

DATA_MIN = 0
DATA_MAX = 255
# Fiji's ContrastAdjuster.autoAdjust constants.
_AUTO_LIMIT_DIVISOR = 10
_AUTO_THRESHOLD_DIVISOR = 5000


@dataclass(frozen=True)
class DisplayRange:
    """Display window over the 0-255 intensity range (Fiji's min/max)."""

    minimum: int = DATA_MIN
    maximum: int = DATA_MAX

    @property
    def is_identity(self) -> bool:
        return self.minimum <= DATA_MIN and self.maximum >= DATA_MAX

    @property
    def brightness(self) -> float:
        """Window centre as a 0-1 fraction, inverted: a lower window shows
        more of the image bright, so 1 is the brightest setting."""
        centre = (self.minimum + self.maximum) / 2
        return 1.0 - (centre - DATA_MIN) / (DATA_MAX - DATA_MIN)

    @property
    def contrast(self) -> float:
        """Window narrowness, 0 (full range, no change) to 1 (1-level window)."""
        width = max(self.maximum - self.minimum, 1)
        return 1.0 - (width - 1) / (DATA_MAX - DATA_MIN)

    @classmethod
    def from_brightness_contrast(
        cls, brightness: float, contrast: float
    ) -> DisplayRange:
        span = DATA_MAX - DATA_MIN
        width = (1.0 - contrast) * (span - 1) + 1
        centre = DATA_MIN + (1.0 - brightness) * span
        minimum = int(round(centre - width / 2))
        maximum = int(round(centre + width / 2))
        return cls.clamped(minimum, maximum)

    @classmethod
    def clamped(cls, minimum: int, maximum: int) -> DisplayRange:
        minimum = max(DATA_MIN, min(int(minimum), DATA_MAX - 1))
        maximum = min(DATA_MAX, max(int(maximum), minimum + 1))
        return cls(minimum, maximum)


def luminance_histogram(image: np.ndarray) -> np.ndarray:
    """256-bin histogram of an 8-bit grey/RGB image (RGB averaged)."""
    grey = image if image.ndim == 2 else image[..., :3].mean(axis=2)
    values = np.clip(np.rint(grey), DATA_MIN, DATA_MAX).astype(np.intp)
    return np.bincount(values.ravel(), minlength=DATA_MAX + 1)[: DATA_MAX + 1]


def auto_display_range(histogram: np.ndarray) -> DisplayRange:
    """Fiji's "Auto" button: clip the dark/bright tails of the histogram.

    Bins above 1/10 of the pixel count (a saturated background spike) are
    ignored, and the window starts at the first/last bin holding more than
    1/5000 of the pixels - so a few hot pixels don't stretch the range.
    """
    total = int(histogram.sum())
    if total == 0:
        return DisplayRange()
    limit = total // _AUTO_LIMIT_DIVISOR
    threshold = total // _AUTO_THRESHOLD_DIVISOR
    counts = np.where(histogram > limit, 0, histogram)
    above = np.flatnonzero(counts > threshold)
    if above.size == 0 or above[0] >= above[-1]:
        return DisplayRange()
    return DisplayRange.clamped(int(above[0]), int(above[-1]))


#: Fiji's Process > Enhance Contrast default: this percentage of pixels saturates.
ENHANCE_SATURATED_PERCENT = 0.35


def enhanced_display_range(
    histogram: np.ndarray, saturated_percent: float = ENHANCE_SATURATED_PERCENT
) -> DisplayRange:
    """Fiji's Enhance Contrast: the window that saturates ``saturated_percent`` of
    the pixels, half at each end. Unlike :func:`auto_display_range` it always
    stretches an image that does not already fill the range up to that point."""
    total = int(histogram.sum())
    if total == 0:
        return DisplayRange()
    cumulative = np.cumsum(histogram)
    tail = total * saturated_percent / 200.0
    low = int(np.searchsorted(cumulative, tail, side="right"))
    high = int(np.searchsorted(cumulative, total - tail, side="left"))
    if high <= low:
        return DisplayRange()
    return DisplayRange.clamped(low, high)


def build_lut(display_range: DisplayRange) -> np.ndarray:
    """uint8 lookup table mapping the display window onto 0-255."""
    levels = np.arange(DATA_MAX + 1, dtype=np.float32)
    span = float(display_range.maximum - display_range.minimum)
    scaled = (levels - display_range.minimum) / span
    return np.clip(scaled * DATA_MAX, 0, DATA_MAX).astype(np.uint8)


def to_8bit(image: np.ndarray) -> np.ndarray:
    """The 8-bit image the display window works on: an 8-bit image as it is (as
    the Cellpose importer shows it), anything else scaled like the importer's
    display image (see :func:`neuroseq.image_scaling.scale_to_uint8`)."""
    if np.asarray(image).dtype == np.uint8:
        return np.asarray(image)
    return scale_to_uint8(image)


def apply_display_range(image: np.ndarray, display_range: DisplayRange) -> np.ndarray:
    """Return `image` (uint8) with the display window applied."""
    if display_range.is_identity:
        return image
    return build_lut(display_range)[image]


def adjust_fov_and_overlay(
    background: np.ndarray | None,
    overlay: np.ndarray,
    segmentation: np.ndarray | None,
    display_range: DisplayRange,
    cell_opacity: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """(FOV background, overlay) with ``display_range`` applied to the FOV only, so
    cell colors keep their exact legend colors.

    The overlay is the FOV blended with cell color inside segmented pixels (see
    CellPoseImporter.generate_overlay; ``segmentation`` uses -1 for background), so
    the FOV's change is added back at the weight it carries there: 1 outside cells,
    ``1 - cell_opacity`` inside. Returned unchanged when there is nothing to adjust.
    """
    if (
        display_range.is_identity
        or background is None
        or segmentation is None
        or background.shape != overlay.shape
        or background.dtype != np.uint8
    ):
        return background, overlay
    adjusted_background = apply_display_range(background, display_range)
    weight = np.where(np.asarray(segmentation) >= 0, 1.0 - cell_opacity, 1.0).astype(
        np.float32
    )[..., np.newaxis]
    delta = adjusted_background.astype(np.float32) - background.astype(np.float32)
    adjusted_overlay = np.clip(overlay.astype(np.float32) + delta * weight, 0, 255)
    return adjusted_background, adjusted_overlay.astype(np.uint8)


class _HistogramView(QtWidgets.QWidget):
    """Log-scaled intensity histogram with the display window drawn over it,
    like the plot at the top of Fiji's B&C window."""

    HEIGHT = 110

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._histogram = np.zeros(DATA_MAX + 1, dtype=np.int64)
        self._range = DisplayRange()
        self.setFixedHeight(self.HEIGHT)
        self.setMinimumWidth(256)

    def set_histogram(self, histogram: np.ndarray) -> None:
        self._histogram = histogram
        self.update()

    def set_range(self, display_range: DisplayRange) -> None:
        self._range = display_range
        self.update()

    def _x_of(self, level: float) -> float:
        return level / DATA_MAX * (self.width() - 1)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        painter = QtGui.QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        peak = float(self._histogram.max()) if self._histogram.size else 0.0
        if peak > 0:
            heights = np.log1p(self._histogram) / np.log1p(peak)
            painter.setPen(QtGui.QPen(palette.mid().color()))
            bar_width = self.width() / (DATA_MAX + 1)
            for level, fraction in enumerate(heights):
                bar_height = int(fraction * (self.height() - 4))
                if bar_height > 0:
                    x = int(level * bar_width)
                    painter.drawLine(
                        x, self.height() - 1, x, self.height() - bar_height
                    )
        # The LUT diagonal between the window ends, as in Fiji.
        line_pen = QtGui.QPen(QtGui.QColor(220, 60, 60))
        line_pen.setWidth(2)
        painter.setPen(line_pen)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        left = self._x_of(self._range.minimum)
        right = self._x_of(self._range.maximum)
        painter.drawLine(
            QtCore.QPointF(0, self.height() - 1),
            QtCore.QPointF(left, self.height() - 1),
        )
        painter.drawLine(
            QtCore.QPointF(left, self.height() - 1), QtCore.QPointF(right, 1)
        )
        painter.drawLine(QtCore.QPointF(right, 1), QtCore.QPointF(self.width(), 1))
        painter.end()


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Non-modal Brightness/Contrast window: histogram, Minimum/Maximum/
    Brightness/Contrast sliders, and Auto/Reset buttons. Emits
    `range_changed` live as the window moves."""

    range_changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Brightness/Contrast")
        self._range = DisplayRange()
        self._histogram = np.zeros(DATA_MAX + 1, dtype=np.int64)
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        self.histogram_view = _HistogramView(self)
        layout.addWidget(self.histogram_view)
        self.range_label = QtWidgets.QLabel()
        self.range_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.range_label)

        self.minimum_slider = self._add_slider(layout, "Minimum", DATA_MAX - 1)
        self.maximum_slider = self._add_slider(layout, "Maximum", DATA_MAX)
        self.brightness_slider = self._add_slider(layout, "Brightness", 1000)
        self.contrast_slider = self._add_slider(layout, "Contrast", 1000)
        self.minimum_slider.valueChanged.connect(self._on_window_edited)
        self.maximum_slider.valueChanged.connect(self._on_window_edited)
        self.brightness_slider.valueChanged.connect(self._on_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self._on_brightness_contrast)

        buttons = QtWidgets.QHBoxLayout()
        self.auto_button = QtWidgets.QPushButton("Auto")
        self.auto_button.clicked.connect(self.auto)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        buttons.addWidget(self.auto_button)
        buttons.addWidget(self.reset_button)
        layout.addLayout(buttons)
        self._sync_controls()

    def _add_slider(
        self, layout: QtWidgets.QVBoxLayout, name: str, maximum: int
    ) -> QtWidgets.QSlider:
        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setFixedWidth(70)
        row.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, maximum)
        row.addWidget(slider)
        layout.addLayout(row)
        return slider

    @property
    def display_range(self) -> DisplayRange:
        return self._range

    def set_image(self, image: np.ndarray | None) -> None:
        """Point the histogram at a (new) FOV image; keeps the window."""
        if image is None:
            self._histogram = np.zeros(DATA_MAX + 1, dtype=np.int64)
        else:
            self._histogram = luminance_histogram(image)
        self.histogram_view.set_histogram(self._histogram)

    def set_display_range(self, display_range: DisplayRange, *, emit: bool = True):
        self._range = display_range
        self._sync_controls()
        if emit:
            self.range_changed.emit(display_range)

    def auto(self) -> None:
        self.set_display_range(auto_display_range(self._histogram))

    def reset(self) -> None:
        self.set_display_range(DisplayRange())

    def _on_window_edited(self) -> None:
        if self._updating:
            return
        self.set_display_range(
            DisplayRange.clamped(
                self.minimum_slider.value(), self.maximum_slider.value()
            )
        )

    def _on_brightness_contrast(self) -> None:
        if self._updating:
            return
        self.set_display_range(
            DisplayRange.from_brightness_contrast(
                self.brightness_slider.value() / 1000,
                self.contrast_slider.value() / 1000,
            )
        )

    def _sync_controls(self) -> None:
        self._updating = True
        try:
            self.minimum_slider.setValue(self._range.minimum)
            self.maximum_slider.setValue(self._range.maximum)
            self.brightness_slider.setValue(round(self._range.brightness * 1000))
            self.contrast_slider.setValue(round(self._range.contrast * 1000))
            self.histogram_view.set_range(self._range)
            self.range_label.setText(f"{self._range.minimum} - {self._range.maximum}")
        finally:
            self._updating = False
