"""Adjust brightness/contrast of an image before Cellpose segments it (#611)."""

from __future__ import annotations

import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.visualization_support.brightness_contrast import BrightnessContrastDialog
from neuroseq.visualization_support.brightness_contrast import DisplayRange
from neuroseq.visualization_support.brightness_contrast import apply_display_range
from neuroseq.visualization_support.brightness_contrast import to_8bit

PREVIEW_SIZE = 512


class SegmentationBrightnessDialog(QtWidgets.QDialog):
    """Preview of the image to segment with the Spatial Footprints panel's
    Brightness/Contrast controls. Accepting segments the adjusted image."""

    def __init__(
        self, image: np.ndarray, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust brightness before segmentation")
        self._image_8bit = to_8bit(image)

        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(256, 256)
        self.controls = BrightnessContrastDialog(self)
        # Embedded as a panel of this dialog rather than its own window.
        self.controls.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.controls.set_image(self._image_8bit)
        self.controls.range_changed.connect(self._update_preview)

        hint = QtWidgets.QLabel(
            "Cellpose segments the image as shown. Reset keeps the original image."
        )
        hint.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.segment_button = buttons.addButton(
            "Segment", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        content = QtWidgets.QHBoxLayout()
        content.addWidget(self.preview, 1)
        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.controls)
        side.addWidget(hint)
        side.addStretch(1)
        content.addLayout(side)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(content)
        layout.addWidget(buttons)
        self._update_preview(self.controls.display_range)

    @property
    def display_range(self) -> DisplayRange:
        return self.controls.display_range

    def _update_preview(self, display_range: DisplayRange) -> None:
        image = np.ascontiguousarray(
            apply_display_range(self._image_8bit, display_range)
        )
        if image.ndim == 3:
            image = np.ascontiguousarray(image[..., :3].mean(axis=2).astype(np.uint8))
        height, width = image.shape
        qimage = QtGui.QImage(
            image.data, width, height, width, QtGui.QImage.Format.Format_Grayscale8
        ).copy()
        self.preview.setPixmap(
            QtGui.QPixmap.fromImage(qimage).scaled(
                PREVIEW_SIZE,
                PREVIEW_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
