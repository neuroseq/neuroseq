from __future__ import annotations

import contextlib
import logging

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from qtpy import QtCore
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
        self.stim_to_cellpose: dict[int, int] = {}
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
        stim_to_cellpose: dict[int, int] | None = None,
    ) -> None:
        cam_state = self.subplot.camera.__getstate__()
        self.subplot.clear()
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.stim_to_cellpose = stim_to_cellpose or {}
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
                    rgb[mask, :3] = tuple(id_colors[i][:3])

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
                    stim_id = int(cell_id)
                    cellpose_id = self.stim_to_cellpose.get(stim_id, stim_id)
                    self.subplot.add_text(
                        f"{stim_id}",
                        font_size=6,
                        offset=(center_x, center_y, 200.0),
                        face_color=(1.0, 1.0, 1.0),
                        outline_color=(0.0, 0.0, 0.0),
                        outline_thickness=2.0,
                        center=True,
                        screen_space=False,
                    )
                    self.subplot.add_text(
                        f"({cellpose_id})",
                        font_size=4,
                        offset=(
                            center_x + (7.2 if cellpose_id < 100 else 9.7),
                            center_y,
                            200.0,
                        ),  # support up to 1k cells in the image preview in theory
                        face_color=(0.8, 0.8, 0.8),
                        outline_color=(0.0, 0.0, 0.0),
                        outline_thickness=1.0,
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
        self.subplot.camera.__setstate__(cam_state)  # won't work

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
        self.original_cellpose_ids: np.ndarray | None = None
        self.stimulus_to_cellpose_mapping: dict[int, int] | None = None
        self.show_stimulus_ids = True  # Always show stimulus IDs when available
        self._tracked_window: QtWidgets.QWidget | None = None
        self._window_handle = None
        self._last_target_size: tuple[int, int] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        self.overlay_view = OverlayView()
        self.overlay_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.overlay_view.setMinimumHeight(600)
        layout.addWidget(self.overlay_view)
        self._update_overlay_scale()

    def _update_overlay_display(self) -> None:
        """Update the overlay display with current ID mode"""
        display_segmentation = self._get_display_segmentation()
        self.overlay_view.add_overlay_data(
            self.overlay_image,
            display_segmentation,
            self.stimulus_to_cellpose_mapping,
        )

    def _get_display_segmentation(self) -> np.ndarray | None:
        """Get segmentation array with appropriate IDs for current display mode"""
        if self.segmentation is None:
            return None

        if not self.show_stimulus_ids or self.stimulus_to_cellpose_mapping is None:
            display_seg = self.segmentation.copy()
            return display_seg

        display_seg = np.zeros_like(self.segmentation)
        cellpose_to_stimulus = {
            v: k for k, v in self.stimulus_to_cellpose_mapping.items()
        }

        for cellpose_id, stimulus_id in cellpose_to_stimulus.items():
            mask = self.segmentation == cellpose_id
            display_seg[mask] = stimulus_id

        return display_seg

    def set_overlay(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        original_cellpose_ids: np.ndarray | None = None,
        stimulus_to_cellpose_mapping: dict[int, int] | None = None,
    ) -> None:
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_to_cellpose_mapping = stimulus_to_cellpose_mapping

        self._update_overlay_display()
        self._update_overlay_scale()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.overlay_view.add_overlay_data(None, None)
        self._update_overlay_scale()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_window_tracking()
        self._update_overlay_scale()

    def eventFilter(self, obj, event) -> bool:
        if (
            obj is self._tracked_window
            and hasattr(event, "type")
            and event.type()
            in (
                QtCore.QEvent.Type.Resize,
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.ScreenChangeInternal,
            )
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
            with contextlib.suppress(Exception):
                self._window_handle.screenChanged.disconnect(self._update_overlay_scale)
        self._window_handle = handle
        with contextlib.suppress(Exception):
            # May already be connected; ignore duplicate connection errors
            handle.screenChanged.connect(self._update_overlay_scale)

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
