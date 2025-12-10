from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.utils import neuron_indices_to_colors

feature_flag_text = True  # todo: remove.


# todo: this is a bit slow to render, not sure right now why. Need to test: initial zoom.
class OverlayView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Suppress subplot title text by providing an empty name
        self.figure: fpl.Figure = fpl.Figure(
            canvas_kwargs={"parent": self}, names=[None]
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.widget = self.figure.show()

        # Hide any fastplotlib toolbars/controls
        if hasattr(self.widget, "toolbar"):
            self.widget.toolbar.hide()
        if hasattr(self.widget, "_toolbar"):
            self.widget._toolbar.hide()
        if hasattr(self.widget, "menubar"):
            self.widget.menubar.hide()
        if hasattr(self.widget, "_menubar"):
            self.widget._menubar.hide()

        layout.addWidget(self.widget)
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.overlay_graphic = None
        self.segmentation_graphic = None
        self.text = None
        self.setLayout(layout)
        self.subplot = self.figure[0, 0]
        self.subplot.frame.plane.visible = False
        self.subplot.frame.resize_handle.visible = False
        self.subplot.title.text = ""
        self.setMinimumSize(QtCore.QSize(200, 200))
        self.resize(800, 800)
        self.logger = logging.getLogger(__name__)
        self.mouse_pos: np.ndarray | None = None
        self.base_camera_state: dict | None = None
        self.zoom_factor = 2.5
        self._overlay_shape: tuple[int, int] | None = None
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")

    def add_overlay_data(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
    ) -> None:
        previous_shape = self._overlay_shape
        previous_camera_state = self._capture_view_state()
        self.subplot.clear()
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.overlay_graphic = None
        self.segmentation_graphic = None
        self.text = None
        self.base_camera_state = None
        self._overlay_shape = (
            (overlay_image.shape[0], overlay_image.shape[1])
            if overlay_image is not None
            else None
        )

        if overlay_image is None:
            return

        legend_lookup: dict[int, tuple[float, float, float]] = {}
        if legend_data:
            legend_lookup = {
                int(cell_id): (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                for cell_id, rgb in legend_data
            }

        # keep the overlay view simple: hide axes and grids
        self.subplot.axes.visible = False
        self.subplot.grids = False
        self.subplot.toolbar = False
        self.subplot.camera.maintain_aspect = True

        self.overlay_graphic = self.subplot.add_image(
            overlay_image,
            interpolation="nearest",
        )

        if segmentation is not None:
            unique_ids = np.unique(segmentation)
            neuron_ids = unique_ids[
                unique_ids > 0
            ]  # exclude background, same as legend

            if len(neuron_ids) > 0:
                zero_based_ids = neuron_ids - neuron_ids.min()
                id_colors = neuron_indices_to_colors(zero_based_ids)

                rgb = np.zeros((*segmentation.shape, 4))
                for i, neuron_id in enumerate(neuron_ids):
                    mask = segmentation == neuron_id
                    color = legend_lookup.get(int(neuron_id), tuple(id_colors[i][:3]))
                    rgb[mask, :3] = color

                rgb[:, :, 3] = 0.4  # alpha for all cells
                rgb[segmentation == 0, 3] = 0  # transparent background
            else:
                rgb = np.zeros((*segmentation.shape, 4))
            self.segmentation_graphic = self.subplot.add_image(
                rgb, interpolation="nearest"
            )

            unique_ids = np.unique(segmentation)
            if feature_flag_text:
                for cell_id in unique_ids:
                    if cell_id == 0:
                        continue
                    mask = segmentation == cell_id
                    if not np.any(mask):
                        continue
                    y_coords, x_coords = np.where(mask)
                    center_y = float(np.mean(y_coords))
                    center_x = float(np.mean(x_coords))
                    self.subplot.add_text(
                        f"{cell_id}",
                        font_size=5,
                        offset=(center_x, center_y, 200.0),
                        face_color=(0.0, 0.0, 0.0),
                        outline_color=(1.0, 1.0, 1.0),
                        outline_thickness=3.0,
                        center=True,
                        screen_space=False,
                    )

            self.text = self.subplot.add_text(
                "",
                font_size=24,
                offset=(8.0, 4.0, 200.0),
                face_color=(0.0, 0.0, 0.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=3.0,
                center=False,
                anchor="bottom-right",
                screen_space=True,
            )

        self._set_base_camera_state(
            position=(
                previous_camera_state.get("position")
                if (
                    previous_camera_state is not None
                    and previous_shape == self._overlay_shape
                )
                else None
            )
        )
        if (
            previous_camera_state is not None
            and previous_shape == self._overlay_shape
            and self.base_camera_state is not None
        ):
            previous_width = float(previous_camera_state.get("width", 0.0))
            if previous_width > 0:
                base_width = float(self.base_camera_state.get("width", 0.0))
                self.zoom_factor = max(base_width / previous_width, 0.1)
        self._apply_zoom(self.zoom_factor, force=True)

    def _capture_view_state(self) -> dict | None:
        """Return the current camera state for restoring pan/zoom."""
        if self.overlay_image is None:
            return None

        cam_state = self.subplot.camera.get_state()
        if not cam_state:
            return None

        position = cam_state.get("position", (0.0, 0.0, 0.0))
        # guarantee 3-tuple for safety
        position = (
            float(position[0]),
            float(position[1]),
            float(position[2]) if len(position) > 2 else 0.0,
        )

        return {
            "width": float(cam_state.get("width", 0.0)),
            "height": float(cam_state.get("height", 0.0)),
            "position": position,
            "maintain_aspect": bool(cam_state.get("maintain_aspect", True)),
        }

    def _set_base_camera_state(
        self, position: tuple[float, float, float] | None = None
    ) -> None:
        """Store a base camera state centered on the image (or provided position)."""
        if self.overlay_image is None:
            self.base_camera_state = None
            return

        height, width = self.overlay_image.shape[:2]
        cam_state = self.subplot.camera.get_state()
        camera_z = float(cam_state.get("position", (0.0, 0.0, 0.0))[2])
        maintain_aspect = bool(cam_state.get("maintain_aspect", True))

        center_x = float(position[0]) if position is not None else float(width) / 2.0
        center_y = float(position[1]) if position is not None else float(height) / 2.0
        center_x = float(np.clip(center_x, 0, width))
        center_y = float(np.clip(center_y, 0, height))

        self.base_camera_state = {
            "width": float(width),
            "height": float(height),
            "position": (center_x, center_y, camera_z),
            "maintain_aspect": maintain_aspect,
        }
        self.subplot.axes.visible = False
        self.subplot.grids = False
        self.subplot.toolbar = False

    def _reset_camera(self) -> None:
        """Fit the camera to the image and apply the current zoom factor."""
        if self.overlay_image is None:
            self.base_camera_state = None
            self._overlay_shape = None
            return

        self._set_base_camera_state()
        self._apply_zoom(self.zoom_factor, force=True)

    def set_zoom_factor(self, zoom_factor: float) -> None:
        """Zoom the camera while keeping pixelated rendering."""
        self._apply_zoom(zoom_factor)

    def _apply_zoom(self, zoom_factor: float, force: bool = False) -> None:
        if self.base_camera_state is None:
            return

        zoom = max(zoom_factor, 0.1)
        if not force and math.isclose(zoom, self.zoom_factor, rel_tol=1e-3):
            return

        self.zoom_factor = zoom
        cam_state = dict(self.base_camera_state)
        cam_state["maintain_aspect"] = True
        cam_state["width"] = self.base_camera_state["width"] / zoom
        cam_state["height"] = self.base_camera_state["height"] / zoom
        cam_state["position"] = (
            self.base_camera_state["position"][0],
            self.base_camera_state["position"][1],
            self.subplot.camera.get_state().get("position", (0.0, 0.0, 0.0))[2],
        )
        self.subplot.camera.set_state(cam_state)
        self.figure.renderer.request_draw()

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        if self.text is None or self.segmentation is None:
            return
        self.mouse_pos = self.subplot.map_screen_to_world(ev)
        if self.mouse_pos is None:
            self.text.visible = False
            return
        x = int(self.mouse_pos[0])
        y = int(self.mouse_pos[1])
        if (
            x < 0
            or y < 0
            or x >= self.segmentation.shape[1]
            or y >= self.segmentation.shape[0]
        ):
            self.text.visible = False
            return
        neuron_index = self.segmentation[y, x]
        if neuron_index == 0:
            self.text.visible = False
            return
        self.text.visible = True
        self.text.offset = (self.mouse_pos[0], self.mouse_pos[1], 200)
        self.text.text = f"{neuron_index}"

    def closeEvent(self, event):
        if self.figure is not None:
            self.figure.close()
            self.figure = None
        super().closeEvent(event)


class CellOverlayWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Color-coded Preview") -> None:
        super().__init__(title)
        self.logger = logging.getLogger(__name__)
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.legend_data: list[tuple[int, tuple[int, int, int]]] = []
        self.original_cellpose_ids: np.ndarray | None = None
        self.stimulus_to_cellpose_mapping: dict[int, int] | None = None
        self.show_stimulus_ids = True  # Default to stimulus IDs
        self._tracked_window: QtWidgets.QWidget | None = None
        self._window_handle = None
        self._last_target_size: tuple[int, int] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.overlay_view = OverlayView()
        self.overlay_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.overlay_view.setMinimumHeight(600)
        layout.addWidget(self.overlay_view)

        # Add toggle checkbox for ID display mode
        self.id_toggle_checkbox = QtWidgets.QCheckBox(
            "Show stimulus IDs (default: original cellpose IDs)"
        )
        self.id_toggle_checkbox.setChecked(self.show_stimulus_ids)
        self.id_toggle_checkbox.toggled.connect(self._on_id_toggle_changed)
        layout.addWidget(self.id_toggle_checkbox)

        self.legend_scroll = QtWidgets.QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setMaximumHeight(140)
        self.legend_widget = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QGridLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend_layout.setSpacing(6)
        self.legend_scroll.setWidget(self.legend_widget)
        self._apply_legend_theme()
        layout.addWidget(self.legend_scroll)
        self._update_overlay_scale()

    def _on_id_toggle_changed(self, checked: bool) -> None:
        """Handle toggle between stimulus IDs and original cellpose IDs"""
        self.show_stimulus_ids = checked
        self._update_overlay_display()

    def _update_overlay_display(self) -> None:
        """Update the overlay display with current ID mode"""
        display_segmentation = self._get_display_segmentation()
        display_legend = self._get_display_legend(display_segmentation)
        self.overlay_view.add_overlay_data(
            self.overlay_image, display_segmentation, display_legend
        )
        self._update_legend(display_legend)

    def _get_display_segmentation(self) -> np.ndarray | None:
        """Get segmentation array with appropriate IDs for current display mode"""
        if self.segmentation is None:
            return None

        if not self.show_stimulus_ids or self.stimulus_to_cellpose_mapping is None:
            # Show original cellpose IDs (convert 0-based back to 1-based)
            display_seg = self.segmentation.copy()
            return display_seg
        else:
            # Show stimulus IDs based on mapping (with stims)
            display_seg = np.zeros_like(self.segmentation)
            # Create reverse mapping from cellpose ID to stimulus ID
            cellpose_to_stimulus = {
                v: k for k, v in self.stimulus_to_cellpose_mapping.items()
            }

            for cellpose_id, stimulus_id in cellpose_to_stimulus.items():
                mask = self.segmentation == cellpose_id
                display_seg[mask] = stimulus_id

            return display_seg

    def _get_display_legend(
        self, display_segmentation: np.ndarray | None
    ) -> list[tuple[int, tuple[int, int, int]]]:
        """Build legend entries that match the currently displayed IDs and colors."""
        if display_segmentation is None:
            return []

        # If no legend data was provided, fall back to computing from the display data
        if not self.legend_data:
            return self._compute_default_legend(display_segmentation)

        # When showing stimulus IDs, remap legend entries to stimulus IDs but keep colors
        if self.show_stimulus_ids and self.stimulus_to_cellpose_mapping:
            cellpose_to_stimulus = {
                v: k for k, v in self.stimulus_to_cellpose_mapping.items()
            }
            remapped: list[tuple[int, tuple[int, int, int]]] = []
            for cellpose_id, rgb in self.legend_data:
                stim_id = cellpose_to_stimulus.get(cellpose_id)
                if stim_id is None:
                    continue
                remapped.append((int(stim_id), rgb))

            if remapped:
                remapped.sort(key=lambda item: item[0])
                return remapped

        return self.legend_data

    def set_overlay(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
        original_cellpose_ids: np.ndarray | None = None,
        stimulus_to_cellpose_mapping: dict[int, int] | None = None,
    ) -> None:
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.legend_data = list(legend_data or [])
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_to_cellpose_mapping = stimulus_to_cellpose_mapping

        # Show/hide toggle checkbox based on whether mapping data is available
        has_mapping = (
            original_cellpose_ids is not None
            and stimulus_to_cellpose_mapping is not None
        )
        self.id_toggle_checkbox.setVisible(has_mapping)

        self._update_overlay_display()
        self._update_overlay_scale()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.legend_data = []
        self.overlay_view.add_overlay_data(None, None)
        self._clear_legend()
        self._update_overlay_scale()

    def _clear_legend(self) -> None:
        for i in reversed(range(self.legend_layout.count())):
            child = self.legend_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

    def _update_legend(
        self, legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None
    ) -> None:
        self._clear_legend()
        if self.segmentation is None:
            return

        legend = (
            list(legend_data)
            if legend_data is not None
            else (
                self.legend_data
                if self.legend_data
                else self._compute_default_legend(self.segmentation)
            )
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
            color_label.setPixmap(self._color_chip_pixmap(rgb))
            text_label = QtWidgets.QLabel(f"Cell {cell_id}")
            text_label.setFixedHeight(16)
            text_label.setStyleSheet(
                "QLabel { color: black; background-color: transparent; }"
            )

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
        colors = neuron_indices_to_colors(zero_based)
        legend = []
        for cell_id, color in zip(neuron_ids, colors, strict=True):
            rgb_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            legend.append((int(cell_id), rgb_255))
        return legend

    def _apply_legend_theme(self) -> None:
        """Ensure legend backgrounds/text stay readable against overlays."""
        self.legend_widget.setStyleSheet("color: black;")
        self.legend_scroll.setStyleSheet("QScrollArea { border: none; }")
        viewport = self.legend_scroll.viewport()
        if viewport is not None:
            viewport.setStyleSheet("")

    def _color_chip_pixmap(self, rgb: tuple[int, int, int]) -> QtGui.QPixmap:
        """Create a color chip showing 50% color over black."""
        size = 16
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        overlay_color = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), 128)
        painter.fillRect(0, 0, size, size, overlay_color)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.drawRect(0, 0, size - 1, size - 1)
        painter.end()
        return pixmap

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_window_tracking()
        self._update_overlay_scale()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._tracked_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.ScreenChangeInternal,
        ):
            self._update_overlay_scale()
        return super().eventFilter(obj, event)

    def _get_current_screen(self):
        window = self.window()
        if window is not None:
            handle = window.windowHandle()
            if handle is not None and handle.screen() is not None:
                return handle.screen()
        return QtWidgets.QApplication.primaryScreen()

    def _ensure_window_tracking(self) -> None:
        """Listen for window moves/resizes to recompute scaling."""
        window = self.window()
        if window is None or window is self._tracked_window:
            return

        if self._tracked_window is not None:
            self._tracked_window.removeEventFilter(self)
        self._tracked_window = window
        self._tracked_window.installEventFilter(self)

        handle = window.windowHandle()
        if handle is None:
            return

        if self._window_handle is not None and self._window_handle is not handle:
            try:
                self._window_handle.screenChanged.disconnect(self._update_overlay_scale)
            except Exception:
                pass
        self._window_handle = handle
        try:
            handle.screenChanged.connect(self._update_overlay_scale)
        except Exception:
            # May already be connected; ignore duplicate connection errors
            pass

    def _update_overlay_scale(self, *_args) -> None:
        """Scale overlay view to 70% of the current window height with minimum 600px."""
        self._ensure_window_tracking()
        window = self.window()
        if window is not None:
            available_height = window.height()
        else:
            screen = self._get_current_screen()
            if screen is None:
                return
            available_height = screen.availableGeometry().height()

        target_height = max(600, int(available_height * 0.7))
        aspect_ratio = 1.0
        if self.overlay_image is not None and self.overlay_image.shape[0] > 0:
            aspect_ratio = max(
                float(self.overlay_image.shape[1]) / float(self.overlay_image.shape[0]),
                0.1,
            )
        target_width = max(1, int(target_height * aspect_ratio))
        target_size = (target_width, target_height)
        if self._last_target_size == target_size:
            return

        self._last_target_size = target_size
        self.overlay_view.setFixedSize(target_width, target_height)
        self.overlay_view.widget.setFixedSize(target_width, target_height)
        self.overlay_view.updateGeometry()
        self.overlay_view.widget.updateGeometry()
