from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from typing import Protocol
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


Color = tuple[float, float, float] | tuple[float, float, float, float]


class TextGraphic(Protocol):
    text: str
    font_size: float
    offset: tuple[float, float, float]
    face_color: Color
    outline_color: Color
    outline_thickness: float
    visible: bool


class LabelGraphics(TypedDict):
    stim: TextGraphic
    cellpose: TextGraphic
    stim_strike: TextGraphic
    cellpose_strike: TextGraphic
    stim_bold: list[TextGraphic]
    cellpose_bold: list[TextGraphic]
    all: list[TextGraphic]


class OverlayView(FPLWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.hide_controls()
        self.overlay_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.text = None
        self.stim_to_cellpose: dict[int, int] = {}
        self.show_cellpose_ids = True
        self.show_stimulus_ids = False
        self.highlight_mappings = True
        self._image_graphic = None
        self._segmentation_graphic = None
        self._label_centers: dict[int, tuple[float, float]] = {}
        self._label_graphics: dict[int, LabelGraphics] = {}
        self._label_segmentation_id: int | None = None
        self._segmentation_shape: tuple[int, int] | None = None
        self.base_camera_state: CameraState | None = None
        self.zoom_factor = 1
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
        show_segmentation: bool = True,
        show_cellpose_ids: bool = True,
        show_stimulus_ids: bool = False,
        highlight_mappings: bool = True,
        label_segmentation: np.ndarray | None = None,
    ) -> None:
        previous_shape = (
            self.overlay_image.shape[:2] if self.overlay_image is not None else None
        )
        previous_camera_state = self.subplot.camera.get_state()
        previous_base_state = self.base_camera_state

        self.overlay_image = overlay_image
        segmentation_for_display = segmentation
        self.segmentation = (
            label_segmentation if label_segmentation is not None else segmentation
        )
        self.stim_to_cellpose = stim_to_cellpose or {}
        self.show_cellpose_ids = show_cellpose_ids
        self.show_stimulus_ids = show_stimulus_ids
        self.highlight_mappings = highlight_mappings

        if overlay_image is None:
            self._reset_graphics(clear_subplot=True)
            self.figure.renderer.request_draw()
            return

        shape_changed = (
            previous_shape is None or previous_shape != overlay_image.shape[:2]
        )
        if shape_changed or self._image_graphic is None:
            self._reset_graphics(clear_subplot=True)
            self._configure_subplot()
            self._image_graphic = self.subplot.add_image(
                overlay_image, interpolation="nearest"
            )
        else:
            self._image_graphic.data = overlay_image

        if segmentation_for_display is not None:
            if show_segmentation:
                color_image = self._build_segmentation_colors(
                    segmentation_for_display, legend_data
                )
                if (
                    self._segmentation_graphic is None
                    or self._segmentation_shape != segmentation_for_display.shape[:2]
                ):
                    if self._segmentation_graphic is not None:
                        with contextlib.suppress(Exception):
                            self.subplot.remove_graphic(self._segmentation_graphic)
                    self._segmentation_graphic = self.subplot.add_image(
                        color_image, interpolation="nearest"
                    )
                    self._segmentation_shape = segmentation_for_display.shape[:2]
                else:
                    self._segmentation_graphic.data = color_image
                if self._segmentation_graphic is not None:
                    self._segmentation_graphic.visible = True
            elif self._segmentation_graphic is not None:
                self._segmentation_graphic.visible = False

            self._ensure_label_graphics()
            self._update_label_graphics(
                self.stim_to_cellpose, show_cellpose_ids, show_stimulus_ids
            )
            if self.text is None:
                self.text = self._create_hover_label()
        else:
            if self._segmentation_graphic is not None:
                self._segmentation_graphic.visible = False
            self._hide_label_graphics()

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

    def _reset_graphics(self, *, clear_subplot: bool = False) -> None:
        if clear_subplot and getattr(self.subplot, "renderer", None) is not None:
            self.subplot.clear()
        self._image_graphic = None
        self._segmentation_graphic = None
        self._segmentation_shape = None
        self.text = None
        self.base_camera_state = None
        self._clear_label_graphics()

    def _clear_label_graphics(self) -> None:
        for graphic_set in self._label_graphics.values():
            for graphic in graphic_set["all"]:
                with contextlib.suppress(Exception):
                    self.subplot.remove_graphic(graphic)
        self._label_graphics = {}
        self._label_centers = {}
        self._label_segmentation_id = None

    def _hide_label_graphics(self) -> None:
        for graphic_set in self._label_graphics.values():
            for graphic in graphic_set["all"]:
                graphic.visible = False

    def _ensure_label_graphics(self) -> None:
        if self.segmentation is None:
            self._hide_label_graphics()
            return

        segmentation_id = id(self.segmentation)
        if segmentation_id == self._label_segmentation_id:
            return

        self._clear_label_graphics()
        self._label_segmentation_id = segmentation_id
        self._label_centers = self._compute_label_centers(self.segmentation)

        for cell_id, (center_x, center_y) in self._label_centers.items():
            base_offset = (center_x, center_y, 200.0)
            cellpose = self.subplot.add_text(
                "",
                font_size=1,
                offset=base_offset,
                face_color=(0.25, 0.25, 0.25),
                outline_color=(0.9, 0.9, 0.9),
                outline_thickness=3.0,
                center=False,
                screen_space=False,
            )
            stim = self.subplot.add_text(
                "",
                font_size=1,
                offset=base_offset,
                face_color=(0.0, 0.0, 0.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=0.6,
                center=False,
                screen_space=False,
            )
            stim_strike = self.subplot.add_text(
                "",
                font_size=1,
                offset=base_offset,
                face_color=(0.0, 0.0, 0.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=0.6,
                center=False,
                screen_space=False,
            )
            stim_bold = [
                self.subplot.add_text(
                    "",
                    font_size=1,
                    offset=base_offset,
                    face_color=(0.0, 0.0, 0.0),
                    outline_color=(1.0, 1.0, 1.0),
                    outline_thickness=0.6,
                    center=False,
                    screen_space=False,
                )
                for _ in range(3)
            ]
            cellpose_strike = self.subplot.add_text(
                "",
                font_size=1,
                offset=base_offset,
                face_color=(0.25, 0.25, 0.25),
                outline_color=(0.9, 0.9, 0.9),
                outline_thickness=3.0,
                center=False,
                screen_space=False,
            )
            cellpose_bold = [
                self.subplot.add_text(
                    "",
                    font_size=1,
                    offset=base_offset,
                    face_color=(0.25, 0.25, 0.25),
                    outline_color=(0.9, 0.9, 0.9),
                    outline_thickness=3.0,
                    center=False,
                    screen_space=False,
                )
                for _ in range(3)
            ]
            all_graphics: list[TextGraphic] = [
                cellpose,
                stim,
                stim_strike,
                cellpose_strike,
                *stim_bold,
                *cellpose_bold,
            ]
            for graphic in all_graphics:
                graphic.visible = False

            self._label_graphics[cell_id] = {
                "stim": stim,
                "cellpose": cellpose,
                "stim_strike": stim_strike,
                "cellpose_strike": cellpose_strike,
                "stim_bold": stim_bold,
                "cellpose_bold": cellpose_bold,
                "all": all_graphics,
            }

    def _compute_label_centers(
        self, segmentation: np.ndarray
    ) -> dict[int, tuple[float, float]]:
        centers: dict[int, tuple[float, float]] = {}
        unique_ids = np.unique(segmentation)
        for cell_id in unique_ids:
            if cell_id < 0:
                continue
            mask = segmentation == cell_id
            if not np.any(mask):
                continue
            y_coords, x_coords = np.where(mask)
            centers[int(cell_id)] = (float(np.mean(x_coords)), float(np.mean(y_coords)))
        return centers

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
        self,
        segmentation: np.ndarray,
        stim_to_cellpose: dict[int, int],
        show_cellpose_ids: bool,
        show_stimulus_ids: bool,
    ) -> None:
        self._ensure_label_graphics()
        self._update_label_graphics(
            stim_to_cellpose, show_cellpose_ids, show_stimulus_ids
        )

    def _update_label_graphics(
        self,
        stim_to_cellpose: dict[int, int],
        show_cellpose_ids: bool,
        show_stimulus_ids: bool,
    ) -> None:
        if not show_cellpose_ids and not show_stimulus_ids:
            self._hide_label_graphics()
            return

        reverse_mapping = {v: k for k, v in stim_to_cellpose.items()}

        for cell_id, graphics in self._label_graphics.items():
            center = self._label_centers.get(cell_id)
            if center is None:
                continue
            center_x, center_y = center
            stim_id = reverse_mapping.get(cell_id)
            cellpose_id = cell_id
            stim_display_id = stim_id if stim_id is not None else cellpose_id
            mapped_elsewhere = (
                cellpose_id in stim_to_cellpose
                and stim_to_cellpose.get(cellpose_id) != cellpose_id
            )
            default_mapping = stim_id is None and stim_display_id == cellpose_id
            cross_out_default = default_mapping and mapped_elsewhere
            highlight = (
                self.highlight_mappings
                and stim_id is not None
                and bool(stim_to_cellpose)
            )
            outdated_cellpose = False
            if self.highlight_mappings and bool(stim_to_cellpose):
                mapped_cellpose = stim_to_cellpose.get(cellpose_id)
                outdated_cellpose = (
                    mapped_cellpose is not None and mapped_cellpose != cellpose_id
                )
            mismatch = stim_id is not None and stim_id != cellpose_id
            cellpose_visible = show_cellpose_ids
            stim_visible = show_stimulus_ids
            if stim_visible and cellpose_visible and stim_id == cellpose_id:
                cellpose_visible = False

            base_offset = (center_x, center_y, 200.0)
            secondary_offset = (center_x + 7.25, center_y, 200.0)
            stim_offset = base_offset
            cellpose_offset = secondary_offset if stim_visible else base_offset
            stim_font_size = 5.0 if cellpose_visible else 6.0
            cellpose_font_size = 3.0 if stim_visible else 5.0
            size_scale = 1.35 if highlight else 1.0
            stim_font_size *= size_scale
            cellpose_font_size *= size_scale
            if mismatch:
                stim_offset = (
                    center_x - (stim_font_size * 0.3),
                    center_y,
                    200.0,
                )

            if cellpose_visible:
                cellpose_text = f"{cellpose_id}"
                cellpose_face: Color
                cellpose_outline: Color
                if outdated_cellpose:
                    cellpose_face = (0.31, 0.31, 0.31, 0.6)
                    cellpose_outline = (0.8, 0.8, 0.8, 0.6)
                    cellpose_outline_thickness = 2.0
                else:
                    cellpose_face = (
                        (0.3, 0.6, 0.95) if highlight else (0.25, 0.25, 0.25)
                    )
                    cellpose_outline = (0.9, 0.9, 0.9)
                    cellpose_outline_thickness = 5.0 if mismatch else 3.0
                cellpose_graphic = graphics["cellpose"]
                cellpose_graphic.text = cellpose_text
                cellpose_graphic.font_size = cellpose_font_size
                cellpose_graphic.offset = cellpose_offset
                cellpose_graphic.face_color = cellpose_face
                cellpose_graphic.outline_color = cellpose_outline
                cellpose_graphic.outline_thickness = cellpose_outline_thickness
                cellpose_graphic.visible = True
                self._update_strike_graphic(
                    graphics["cellpose_strike"],
                    cellpose_font_size,
                    cellpose_offset,
                    cellpose_face,
                    cellpose_outline,
                    cellpose_outline_thickness,
                    cross_out_default and not stim_visible,
                )
                self._update_bold_graphics(
                    graphics["cellpose_bold"],
                    cellpose_text,
                    cellpose_font_size,
                    cellpose_offset,
                    cellpose_face,
                    cellpose_outline,
                    cellpose_outline_thickness,
                    mismatch,
                )
            else:
                graphics["cellpose"].visible = False
                graphics["cellpose_strike"].visible = False
                for graphic in graphics["cellpose_bold"]:
                    graphic.visible = False

            if stim_visible:
                stim_text = f"{stim_display_id}"
                stim_face = (0.25, 0.55, 0.9) if highlight else (0.0, 0.0, 0.0)
                stim_outline = (1.0, 1.0, 1.0)
                stim_outline_thickness = 1.2 if mismatch else 0.6
                stim_graphic = graphics["stim"]
                stim_graphic.text = stim_text
                stim_graphic.font_size = stim_font_size
                stim_graphic.offset = stim_offset
                stim_graphic.face_color = stim_face
                stim_graphic.outline_color = stim_outline
                stim_graphic.outline_thickness = stim_outline_thickness
                stim_graphic.visible = True
                self._update_strike_graphic(
                    graphics["stim_strike"],
                    stim_font_size,
                    stim_offset,
                    stim_face,
                    stim_outline,
                    stim_outline_thickness,
                    cross_out_default,
                )
                self._update_bold_graphics(
                    graphics["stim_bold"],
                    stim_text,
                    stim_font_size,
                    stim_offset,
                    stim_face,
                    stim_outline,
                    stim_outline_thickness,
                    False,
                )
            else:
                graphics["stim"].visible = False
                graphics["stim_strike"].visible = False
                for graphic in graphics["stim_bold"]:
                    graphic.visible = False

    def _update_bold_graphics(
        self,
        graphics: list[TextGraphic],
        text: str,
        font_size: float,
        offset: tuple[float, float, float],
        face_color: Color,
        outline_color: Color,
        outline_thickness: float,
        visible: bool,
    ) -> None:
        if not graphics:
            return
        bold_offsets = (
            (offset[0] + (font_size * 0.18), offset[1], offset[2]),
            (offset[0] + (font_size * 0.08), offset[1] + (font_size * 0.08), offset[2]),
            (offset[0] + (font_size * 0.2), offset[1] + (font_size * 0.02), offset[2]),
        )
        for graphic, bold_offset in zip(graphics, bold_offsets, strict=False):
            graphic.text = text
            graphic.font_size = font_size
            graphic.offset = bold_offset
            graphic.face_color = face_color
            graphic.outline_color = outline_color
            graphic.outline_thickness = outline_thickness
            graphic.visible = visible

    def _update_strike_graphic(
        self,
        graphic: TextGraphic,
        font_size: float,
        offset: tuple[float, float, float],
        face_color: Color,
        outline_color: Color,
        outline_thickness: float,
        visible: bool,
    ) -> None:
        graphic.text = "\u2014"
        graphic.font_size = font_size
        graphic.offset = offset
        graphic.face_color = face_color
        graphic.outline_color = outline_color
        graphic.outline_thickness = outline_thickness
        graphic.visible = visible

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
            zoom = max(1, min(zoom_factor, 10.0))

        self.zoom_factor = int(zoom)
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

        if not self.show_cellpose_ids and not self.show_stimulus_ids:
            self.text.visible = False
            return

        reverse_mapping = {v: k for k, v in self.stim_to_cellpose.items()}
        stim_id = reverse_mapping.get(neuron_index)
        cellpose_id = int(neuron_index)
        stim_display_id = stim_id if stim_id is not None else cellpose_id
        cellpose_visible = self.show_cellpose_ids
        stim_visible = self.show_stimulus_ids

        if cellpose_visible and stim_visible:
            hover_text = f"{stim_display_id} / {cellpose_id}"
        elif stim_visible:
            hover_text = f"{stim_display_id}"
        elif cellpose_visible:
            hover_text = f"{cellpose_id}"
        else:
            self.text.visible = False
            return

        self.text.visible = True
        self.text.offset = (mouse_pos[0], mouse_pos[1], 200)
        self.text.text = hover_text


class CellOverlayWidget(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Color-coded Preview") -> None:
        super().__init__(title)
        self.logger = logging.getLogger(__name__)
        self.overlay_image: np.ndarray | None = None
        self.base_image: np.ndarray | None = None
        self.segmentation: np.ndarray | None = None
        self.legend_data: list[tuple[int, tuple[int, int, int]]] = []
        self.original_cellpose_ids: np.ndarray | None = None
        self.stimulus_to_cellpose_mapping: dict[int, int] | None = None
        self.show_cellpose_ids = True
        self.show_stimulus_ids = False
        self.show_segmentation = True
        self.highlight_mappings = True
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
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_view.setMinimumHeight(600)
        layout.addWidget(self.overlay_view)
        self._update_overlay_scale()

    def _update_overlay_display(self) -> None:
        display_segmentation = self._get_display_segmentation()
        display_legend = (
            []
            if not self.show_segmentation
            else self._get_display_legend(display_segmentation)
        )
        if self.show_segmentation or self.base_image is None:
            image_to_show = self.overlay_image
        else:
            image_to_show = self.base_image
        self.overlay_view.add_overlay_data(
            image_to_show,
            display_segmentation,
            display_legend,
            self.stimulus_to_cellpose_mapping,
            show_segmentation=self.show_segmentation,
            show_cellpose_ids=self.show_cellpose_ids,
            show_stimulus_ids=self.show_stimulus_ids,
            highlight_mappings=self.highlight_mappings,
            label_segmentation=self.segmentation,
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
        base_image: np.ndarray | None = None,
        highlight_mappings: bool | None = None,
    ) -> None:
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.legend_data = list(legend_data or [])
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_to_cellpose_mapping = stimulus_to_cellpose_mapping
        self.base_image = base_image
        if highlight_mappings is not None:
            self.highlight_mappings = highlight_mappings

        self._update_overlay_display()
        self._update_overlay_scale()

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.base_image = None
        self.segmentation = None
        self.legend_data = []
        self.original_cellpose_ids = None
        self.stimulus_to_cellpose_mapping = None
        self.highlight_mappings = True
        self.overlay_view.add_overlay_data(
            None,
            None,
            show_segmentation=self.show_segmentation,
            show_cellpose_ids=self.show_cellpose_ids,
            show_stimulus_ids=self.show_stimulus_ids,
            highlight_mappings=self.highlight_mappings,
        )
        self._update_overlay_scale()

    def set_display_options(
        self,
        *,
        show_cellpose_ids: bool | None = None,
        show_stimulus_ids: bool | None = None,
        show_segmentation: bool | None = None,
    ) -> None:
        if show_cellpose_ids is not None:
            self.show_cellpose_ids = show_cellpose_ids
        if show_stimulus_ids is not None:
            self.show_stimulus_ids = show_stimulus_ids
        if show_segmentation is not None:
            self.show_segmentation = show_segmentation
        self._update_overlay_display()

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
        margins = self.contentsMargins()
        available_height = self.height() - margins.top() - margins.bottom()
        available_width = self.width() - margins.left() - margins.right()
        logging.info(
            "Upon render: height %s - top %s - bottom %s = %s; "
            "width %s - left %s - right %s = %s",
            self.height(),
            margins.top(),
            margins.bottom(),
            available_height,
            self.width(),
            margins.left(),
            margins.right(),
            available_width,
        )
        if available_height <= 0:
            window = self.window()
            if window is not None:
                available_height = window.height()
                available_width = window.width()
            else:
                screen = self._get_current_screen()
                if screen is None:
                    return
                available_height = screen.availableGeometry().height()
                available_width = screen.availableGeometry().width()

        target_height = int(available_height * 0.95)
        target_width = int(available_width * 0.98)
        aspect_ratio = 1.0
        if self.overlay_image is not None and self.overlay_image.shape[0] > 0:
            aspect_ratio = max(
                float(self.overlay_image.shape[1]) / float(self.overlay_image.shape[0]),
                0.1,
            )

        if self.overlay_image is not None and self.overlay_image.shape[0] > 0:
            width_from_height = int(target_height * aspect_ratio)
            height_from_width = int(target_width / aspect_ratio)
            if width_from_height <= target_width:
                final_width = max(1, width_from_height)
                final_height = max(1, target_height)
            else:
                final_width = max(1, target_width)
                final_height = max(1, height_from_width)
        else:
            final_width = max(1, target_width)
            final_height = max(1, target_height)

        target_size = (final_width, final_height)
        logging.info(
            "Upon render: height %s / width %s. Target width %s ",
            available_height,
            available_width,
            final_width,
        )

        if self._last_target_size == target_size:
            return

        self._last_target_size = target_size
        self.overlay_view.setFixedSize(final_width, final_height)
        self.overlay_view.widget.setFixedSize(final_width, final_height)
        self.overlay_view.updateGeometry()
        self.overlay_view.widget.updateGeometry()
