from __future__ import annotations

import logging
from collections.abc import Iterable

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.utils import jet_colors_for_neurons


class OverlayPreviewLabel(QtWidgets.QLabel):
    """Label that displays cell IDs as tooltips when hovering over an overlay image."""

    def __init__(self, parent_widget: CellOverlayWidget) -> None:
        super().__init__("Load files to see overlay preview")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(200)
        self.setStyleSheet("QLabel { border: 1px solid gray; }")
        self.setMouseTracking(True)
        self.parent_widget = parent_widget

        self.setStyleSheet(
            """
            QLabel { border: 1px solid gray; }
            QToolTip {
                background-color: yellow;
                color: black;
                border: 1px solid black;
                padding: 4px;
                border-radius: 3px;
            }
        """
        )

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)

        segmentation = self.parent_widget.segmentation
        if segmentation is None or self.pixmap() is None:
            self.setToolTip("")
            return

        pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        label_size = self.size()

        if label_size.width() == 0 or label_size.height() == 0:
            self.setToolTip("")
            return

        scale_x = pixmap_rect.width() / label_size.width()
        scale_y = pixmap_rect.height() / label_size.height()
        scale = max(scale_x, scale_y)

        displayed_width = int(pixmap_rect.width() / max(scale, 1e-9))
        displayed_height = int(pixmap_rect.height() / max(scale, 1e-9))

        offset_x = (label_size.width() - displayed_width) // 2
        offset_y = (label_size.height() - displayed_height) // 2

        magnified_img_x = int((pos.x() - offset_x) * scale)
        magnified_img_y = int((pos.y() - offset_y) * scale)

        if self.parent_widget.current_magnification <= 0:
            self.setToolTip("")
            return

        original_img_x = int(magnified_img_x / self.parent_widget.current_magnification)
        original_img_y = int(magnified_img_y / self.parent_widget.current_magnification)

        if (
            0 <= original_img_x < segmentation.shape[1]
            and 0 <= original_img_y < segmentation.shape[0]
        ):
            cell_id = segmentation[original_img_y, original_img_x]
            if cell_id > 0:
                self.setToolTip(f"Cell ID: {cell_id}")
            else:
                self.setToolTip("Background")
        else:
            self.setToolTip("")


class CellOverlayWidget(QtWidgets.QGroupBox):
    """Reusable widget that shows a cell overlay image, magnification toggle, and legend."""

    def __init__(self, title: str = "Color-coded Preview") -> None:
        super().__init__(title)
        self.logger = logging.getLogger(__name__)
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.legend_data: list[tuple[int, tuple[int, int, int]]] = []
        self.current_magnification: int = 1
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.magnified_checkbox = QtWidgets.QCheckBox(
            "5x magnified view with cell numbers"
        )
        self.magnified_checkbox.toggled.connect(self._display_overlay_preview)
        layout.addWidget(self.magnified_checkbox)

        self.preview_label = OverlayPreviewLabel(self)
        self.preview_scroll = QtWidgets.QScrollArea()
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setMinimumHeight(300)
        self.preview_scroll.setMaximumHeight(500)
        layout.addWidget(self.preview_scroll)

        self.legend_scroll = QtWidgets.QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setMaximumHeight(140)
        self.legend_widget = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QGridLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend_layout.setSpacing(6)
        self.legend_scroll.setWidget(self.legend_widget)
        layout.addWidget(self.legend_scroll)

    def set_overlay(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
    ) -> None:
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.legend_data = list(legend_data or [])
        self._display_overlay_preview()
        self._update_legend()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.legend_data = []
        self.preview_label.setPixmap(QtGui.QPixmap())
        self.preview_label.setText(message)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._clear_legend()

    def _display_overlay_preview(self) -> None:
        if self.overlay_image is None:
            self.preview_label.setPixmap(QtGui.QPixmap())
            return

        if self.magnified_checkbox.isChecked():
            image_data = self._create_magnified_overlay()
            self.current_magnification = 5
        else:
            image_data = self._create_base_magnified_overlay()
            self.current_magnification = 2

        height, width, _ = image_data.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            image_data.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.preview_label.resize(pixmap.size())
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setScaledContents(False)
        self.preview_label.setText("")

    def _create_base_magnified_overlay(self) -> np.ndarray:
        """Create a 2x magnified overlay for better visibility."""
        if self.overlay_image is None:
            return np.array([]).reshape(0, 0, 3)

        original_image = self.overlay_image
        height, width, channels = original_image.shape
        magnified_image = np.zeros(
            (height * 2, width * 2, channels), dtype=original_image.dtype
        )

        for y in range(height):
            for x in range(width):
                pixel_value = original_image[y, x]
                magnified_image[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2] = pixel_value

        try:
            from PIL import Image

            debug_image = Image.fromarray(magnified_image)
            debug_image.save("latest_cellpose_2.png")
        except Exception:
            pass

        return magnified_image

    def _create_magnified_overlay(self) -> np.ndarray:
        """Create a 5x magnified overlay with cell numbers."""
        if self.overlay_image is None:
            return np.array([]).reshape(0, 0, 3)

        original_image = self.overlay_image
        original_segmentation = self.segmentation
        height, width, channels = original_image.shape

        magnified_image = np.zeros(
            (height * 5, width * 5, channels), dtype=original_image.dtype
        )

        if original_segmentation is None:
            for y in range(height):
                for x in range(width):
                    pixel_value = original_image[y, x]
                    magnified_image[y * 5 : y * 5 + 5, x * 5 : x * 5 + 5] = pixel_value
            return magnified_image

        magnified_seg = np.zeros(
            (height * 5, width * 5), dtype=original_segmentation.dtype
        )

        for y in range(height):
            for x in range(width):
                pixel_value = original_image[y, x]
                seg_value = original_segmentation[y, x]
                magnified_image[y * 5 : y * 5 + 5, x * 5 : x * 5 + 5] = pixel_value
                magnified_seg[y * 5 : y * 5 + 5, x * 5 : x * 5 + 5] = seg_value

        try:
            from PIL import Image
            from PIL import ImageDraw
            from PIL import ImageFont

            pil_image = Image.fromarray(magnified_image)
            draw = ImageDraw.Draw(pil_image)

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16
                )
            except OSError:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except OSError:
                    font = ImageFont.load_default()

            unique_ids = np.unique(magnified_seg)
            for cell_id in unique_ids:
                if cell_id == 0:
                    continue
                mask = magnified_seg == cell_id
                if not np.any(mask):
                    continue
                y_coords, x_coords = np.where(mask)
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                text = str(cell_id)
                for dx, dy in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]:
                    draw.text(
                        (center_x + dx, center_y + dy),
                        text,
                        font=font,
                        fill=(255, 255, 255, 255),
                    )
                draw.text((center_x, center_y), text, font=font, fill=(0, 0, 0, 128))

            result_image = np.array(pil_image)
            try:
                debug_image = Image.fromarray(result_image)
                debug_image.save("latest_cellpose_5.png")
            except Exception:
                pass

            return result_image
        except ImportError:
            try:
                from PIL import Image

                debug_image = Image.fromarray(magnified_image)
                debug_image.save("latest_cellpose_5.png")
            except Exception:
                pass
            return magnified_image

    def _clear_legend(self) -> None:
        for i in reversed(range(self.legend_layout.count())):
            child = self.legend_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

    def _update_legend(self) -> None:
        self._clear_legend()
        if self.segmentation is None:
            return

        legend = (
            self.legend_data
            if self.legend_data
            else self._compute_default_legend(self.segmentation)
        )
        if not legend:
            return

        cols = 4
        for i, (cell_id, rgb) in enumerate(legend):
            row = i // cols
            col = i % cols
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item_widget)
            item_layout.setContentsMargins(2, 2, 2, 2)
            item_layout.setSpacing(4)

            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(16, 16)
            color_label.setStyleSheet(
                f"QLabel {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid black; }}"
            )
            text_label = QtWidgets.QLabel(f"Cell {cell_id}")
            text_label.setFixedHeight(16)

            item_layout.addWidget(color_label)
            item_layout.addWidget(text_label)
            item_layout.addStretch()

            self.legend_layout.addWidget(item_widget, row, col)

    def _compute_default_legend(
        self, segmentation: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int]]]:
        unique_ids = np.unique(segmentation)
        neuron_ids = unique_ids[unique_ids > 0]
        if neuron_ids.size == 0:
            return []
        zero_based = neuron_ids - neuron_ids.min()
        colors = jet_colors_for_neurons(zero_based)
        legend = []
        for cell_id, color in zip(neuron_ids, colors, strict=True):
            legend.append((int(cell_id), (int(color[0]), int(color[1]), int(color[2]))))
        return legend
