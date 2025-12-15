from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.fpl_widget import FPLWidget
from neuroseq.utils import neuron_indices_to_colors


@dataclass
class _CameraState:
    width: float
    height: float
    position: tuple[float, float, float]


class OverlayView(FPLWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.hide_controls()
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.text = None
        self.stim_to_cellpose: dict[int, int] = {}
        self.base_camera_state: _CameraState | None = None
        self.zoom_factor = 2.5
        self._configure_subplot()
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.setMinimumSize(QtCore.QSize(200, 200))
        # keep renderer/swapchain in sync with widget size
        self._sync_canvas_size(self.size())

    def _configure_subplot(self) -> None:
        self.subplot.frame.plane.visible = False
        self.subplot.frame.resize_handle.visible = False
        self.subplot.title.text = ""
        self.subplot.axes.visible = False
        self.subplot.grids = False
        self.subplot.toolbar = False
        self.subplot.camera.maintain_aspect = True

    def add_overlay_data(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        stim_to_cellpose: dict[int, int] | None = None,
    ) -> None:
        self.subplot.clear()
        self._configure_subplot()
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.stim_to_cellpose = stim_to_cellpose or {}
        self.text = None
        self.base_camera_state = None

        if overlay_image is None:
            self.figure.renderer.request_draw()
            return

        self.subplot.add_image(overlay_image, interpolation="nearest")

        if segmentation is not None:
            color_image = self._build_segmentation_colors(segmentation)
            self.subplot.add_image(color_image, interpolation="nearest")
            self._add_cell_labels(segmentation)
            self.text = self._create_hover_label()

        self._reset_camera()

    def _build_segmentation_colors(self, segmentation: np.ndarray) -> np.ndarray:
        rgb = np.zeros((*segmentation.shape, 4))
        neuron_ids = np.unique(segmentation)
        neuron_ids = neuron_ids[neuron_ids > 0]
        if neuron_ids.size == 0:
            return rgb

        zero_based_ids = neuron_ids - neuron_ids.min()
        id_colors = neuron_indices_to_colors(zero_based_ids)

        for i, neuron_id in enumerate(neuron_ids):
            mask = segmentation == neuron_id
            rgb[mask, :3] = tuple(id_colors[i][:3])

        rgb[:, :, 3] = 0.4
        rgb[segmentation == 0, 3] = 0
        return rgb

    def _add_cell_labels(self, segmentation: np.ndarray) -> None:
        for cell_id in np.unique(segmentation):
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
                ),
                face_color=(0.8, 0.8, 0.8),
                outline_color=(0.0, 0.0, 0.0),
                outline_thickness=1.0,
                center=True,
                screen_space=False,
            )

    def _create_hover_label(self):
        return self.subplot.add_text(
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

    def _reset_camera(self) -> None:
        if self.overlay_image is None:
            self.base_camera_state = None
            return

        height, width = self.overlay_image.shape[:2]
        cam_state = self.subplot.camera.get_state()
        camera_z = float(cam_state.get("position", (0.0, 0.0, 0.0))[2])
        self.base_camera_state = _CameraState(
            width=float(width),
            height=float(height),
            position=(float(width) / 2.0, float(height) / 2.0, camera_z),
        )
        self._apply_zoom(self.zoom_factor, force=True)

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._apply_zoom(zoom_factor)

    def _apply_zoom(self, zoom_factor: float, force: bool = False) -> None:
        if self.base_camera_state is None:
            return

        zoom = float(zoom_factor)
        if not force and math.isclose(self.zoom_factor, zoom):
            return

        self.zoom_factor = zoom
        cam_state: dict[str, float | tuple[float, float, float]] = {
            "width": self.base_camera_state.width / zoom,
            "height": self.base_camera_state.height / zoom,
            "position": (
                self.base_camera_state.position[0],
                self.base_camera_state.position[1],
                self.base_camera_state.position[2] * zoom,
            ),
        }
        self.subplot.camera.set_state(cam_state)
        self.figure.renderer.request_draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_canvas_size(event.size())

    def _sync_canvas_size(self, size: QtCore.QSize) -> None:
        width = max(1, int(size.width()))
        height = max(1, int(size.height()))
        with contextlib.suppress(Exception):
            self.figure.canvas.set_logical_size(width, height)
        self.figure.renderer.request_draw()

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        if self.text is None or self.segmentation is None:
            return

        mouse_pos = self.subplot.map_screen_to_world(ev)
        if mouse_pos is None:
            self.text.visible = False
            return

        x = int(mouse_pos[0])
        y = int(mouse_pos[1])
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
        self.text.offset = (mouse_pos[0], mouse_pos[1], 200)
        self.text.text = f"{neuron_index}"


class CellOverlayWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Color-coded Preview") -> None:
        super().__init__(title)
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.original_cellpose_ids: np.ndarray | None = None
        self.stimulus_to_cellpose_mapping: dict[int, int] | None = None
        self._tracked_window: QtWidgets.QWidget | None = None
        self._window_handle = None
        self._has_overlay = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        self.overlay_view = OverlayView()
        self.overlay_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.overlay_view.setMinimumHeight(200)
        layout.addWidget(self.overlay_view)
        # let layouts manage sizing; avoid manual fixed canvas sizes

    def _update_overlay_display(self) -> None:
        display_segmentation = self._get_display_segmentation()
        self.overlay_view.add_overlay_data(
            self.overlay_image,
            display_segmentation,
            self.stimulus_to_cellpose_mapping,
        )

    def _get_display_segmentation(self) -> np.ndarray | None:
        if self.segmentation is None:
            return None

        if self.stimulus_to_cellpose_mapping is None:
            return self.segmentation.copy()

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

        self._has_overlay = overlay_image is not None and segmentation is not None
        self._update_overlay_display()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.stimulus_to_cellpose_mapping = None
        self._has_overlay = False
        self.overlay_view.add_overlay_data(None, None)
        # keep placeholder compact until data is loaded; no fixed sizing

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_window_tracking()
        # sizing handled by layout

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
        """Sizing is handled by Qt layouts; no manual scaling required."""
        return
