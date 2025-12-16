from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TypedDict

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.fpl_widget import FPLWidget
from neuroseq.utils import neuron_indices_to_colors


class CameraState(TypedDict):
    width: float
    height: float
    position: tuple[float, float, float]
    maintain_aspect: bool


class OverlayView(FPLWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.hide_controls()
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.text = None
        self.stim_to_cellpose: dict[int, int] = {}
        self.base_camera_state: CameraState | None = None
        self.zoom_factor = 2.5
        self._configure_subplot()
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.setMinimumSize(QtCore.QSize(200, 200))

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
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
        stim_to_cellpose: dict[int, int] | None = None,
    ) -> None:
        previous_shape = (
            self.overlay_image.shape[:2] if self.overlay_image is not None else None
        )
        previous_camera_state = self.subplot.camera.get_state()
        previous_base_state = self.base_camera_state

        self.subplot.clear()
        self._configure_subplot()
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.text = None
        self.base_camera_state = None
        self.stim_to_cellpose = stim_to_cellpose or {}

        if overlay_image is None:
            self.figure.renderer.request_draw()
            return

        self.subplot.add_image(overlay_image, interpolation="nearest")

        if segmentation is not None:
            color_image = self._build_segmentation_colors(segmentation, legend_data)
            self.subplot.add_image(color_image, interpolation="nearest")
            self._add_cell_labels(segmentation, self.stim_to_cellpose)
            self.text = self._create_hover_label()

        same_shape = (
            previous_shape is not None
            and overlay_image is not None
            and previous_shape == overlay_image.shape[:2]
        )
        if same_shape and previous_camera_state is not None and previous_base_state:
            self.base_camera_state = previous_base_state
            self.subplot.camera.set_state(previous_camera_state)
            self.figure.renderer.request_draw()
        else:
            self._reset_camera()

    def _build_segmentation_colors(
        self,
        segmentation: np.ndarray,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None,
    ) -> np.ndarray:
        rgb = np.zeros((*segmentation.shape, 4))
        neuron_ids = np.unique(segmentation)
        neuron_ids = neuron_ids[neuron_ids > 0]
        if neuron_ids.size == 0:
            return rgb

        legend_lookup = {
            int(cell_id): (rgb_val[0] / 255.0, rgb_val[1] / 255.0, rgb_val[2] / 255.0)
            for cell_id, rgb_val in (legend_data or [])
        }
        zero_based = neuron_ids - neuron_ids.min()
        id_colors = neuron_indices_to_colors(zero_based)

        for idx, neuron_id in enumerate(neuron_ids):
            mask = segmentation == neuron_id
            color = legend_lookup.get(int(neuron_id), tuple(id_colors[idx][:3]))
            rgb[mask, :3] = color

        rgb[:, :, 3] = 0.4
        rgb[segmentation < 0, 3] = 0
        return rgb

    def _add_cell_labels(
        self, segmentation: np.ndarray, stim_to_cellpose: dict[int, int]
    ) -> None:
        reverse_mapping = {v: k for k, v in stim_to_cellpose.items()}
        for cell_id in np.unique(segmentation):
            if cell_id < 0:
                continue

            mask = segmentation == cell_id
            if not np.any(mask):
                continue

            y_coords, x_coords = np.where(mask)
            center_y = float(np.mean(y_coords))
            center_x = float(np.mean(x_coords))
            stim_id = reverse_mapping.get(cell_id, cell_id)
            cellpose_id = stim_to_cellpose.get(cell_id, cell_id)

            # Primary stim ID label
            self.subplot.add_text(
                f"{stim_id}",
                font_size=4,
                offset=(center_x, center_y, 200.0),
                face_color=(0.0, 0.0, 0.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=3.0,
                center=True,
                screen_space=False,
            )
            # Secondary cellpose ID label
            self.subplot.add_text(
                f"{cellpose_id}",
                font_size=3,
                offset=(center_x + 5.0, center_y, 200.0),
                face_color=(0.3, 0.3, 0.3),
                outline_color=(0.8, 0.8, 0.8),
                outline_thickness=0.4,  # and deliberately not centered.
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
        self.base_camera_state = {
            "width": float(width),
            "height": float(height),
            "position": (float(width) / 2.0, float(height) / 2.0, camera_z),
            "maintain_aspect": True,
        }
        self._apply_zoom(self.zoom_factor, force=True)

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._apply_zoom(zoom_factor)

    def _apply_zoom(self, zoom_factor: float, force: bool = False) -> None:
        base_state = self.base_camera_state
        if base_state is None:
            return

        zoom = zoom_factor
        if not force:
            zoom = max(0.1, min(zoom_factor, 10.0))

        self.zoom_factor = zoom
        cam_state: CameraState = {
            "width": base_state["width"] / zoom,
            "height": base_state["height"] / zoom,
            "position": (
                base_state["position"][0],
                base_state["position"][1],
                base_state["position"][2] * zoom,
            ),
            "maintain_aspect": base_state["maintain_aspect"],
        }
        self.subplot.camera.set_state(cam_state)
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
        if neuron_index < 0:
            self.text.visible = False
            return

        reverse_mapping = {v: k for k, v in self.stim_to_cellpose.items()}
        stim_id = reverse_mapping.get(neuron_index, neuron_index)

        self.text.visible = True
        self.text.offset = (mouse_pos[0], mouse_pos[1], 200)
        self.text.text = f"{stim_id}"


class CellOverlayWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Color-coded Preview") -> None:
        super().__init__(title)
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.legend_data: list[tuple[int, tuple[int, int, int]]] = []
        self.original_cellpose_ids: np.ndarray | None = None
        self.stimulus_to_cellpose_mapping: dict[int, int] | None = None
        self.show_stimulus_ids = True  # delet enot hceck box removed?
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
        display_segmentation = self._get_display_segmentation()
        display_legend = self._get_display_legend(display_segmentation)
        self.overlay_view.add_overlay_data(
            self.overlay_image,
            display_segmentation,
            display_legend,
            self.stimulus_to_cellpose_mapping,
        )

    def _get_display_segmentation(self) -> np.ndarray | None:
        if self.segmentation is None:
            return None

        if not self.show_stimulus_ids or self.stimulus_to_cellpose_mapping is None:
            return self.segmentation.copy()

        display_seg = np.full_like(self.segmentation, -1)
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
        if display_segmentation is None:
            return []

        if not self.legend_data:
            return self._compute_segementation_colors(display_segmentation)

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

        return list(self.legend_data)

    def set_overlay(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        original_cellpose_ids: np.ndarray | None = None,
        stimulus_to_cellpose_mapping: dict[int, int] | None = None,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
    ) -> None:
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.legend_data = list(legend_data or [])
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_to_cellpose_mapping = stimulus_to_cellpose_mapping

        has_mapping = (
            original_cellpose_ids is not None
            and stimulus_to_cellpose_mapping is not None
        )
        self._update_overlay_display()
        self._update_overlay_scale()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.legend_data = []
        self.original_cellpose_ids = None
        self.stimulus_to_cellpose_mapping = None
        self.overlay_view.add_overlay_data(None, None)
        self._update_overlay_scale()

    def _compute_segementation_colors(
        self, segmentation: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int]]]:
        unique_ids = np.unique(segmentation)
        neuron_ids = [cell_id for cell_id in unique_ids if cell_id >= 0]
        if not neuron_ids:
            return []

        zero_based = np.array(neuron_ids) - min(neuron_ids)
        id_colors = neuron_indices_to_colors(zero_based)

        legend: list[tuple[int, tuple[int, int, int]]] = []
        for cell_id, color in zip(neuron_ids, id_colors, strict=False):
            rgb_255 = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            legend.append((int(cell_id), rgb_255))
        return legend

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
            handle.screenChanged.connect(self._update_overlay_scale)

    def _update_overlay_scale(self, *_args) -> None:
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
