from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol
from typing import TypedDict

import brian2.numpy_ as np
import pygfx
from matplotlib.path import Path as MplPath
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.aoi_shapes import circle_world_handles
from neuroseq.aoi_shapes import circle_world_polyline
from neuroseq.aoi_shapes import square_user_corners
from neuroseq.aoi_shapes import square_world_corners
from neuroseq.aoi_shapes import square_world_polyline
from neuroseq.aoi_shapes import world_to_user_point
from neuroseq.fpl_widget import FullScreenFPLWidget
from neuroseq.importer_cellpose import AOIRole
from neuroseq.importer_cellpose import AreasOfInterestConfig
from neuroseq.importer_cellpose import CircleAOI
from neuroseq.importer_cellpose import SquareAOI
from neuroseq.importer_cellpose import clone_aoi_config
from neuroseq.utils import neuron_indices_to_colors


class CameraState(TypedDict):
    width: float
    height: float
    position: tuple[float, float, float]
    maintain_aspect: bool


Color = tuple[float, float, float] | tuple[float, float, float, float]
STRIPED_CELL_GREY = (0.55, 0.55, 0.55)
STRIPED_CELL_WHITE = (1.0, 1.0, 1.0)
STRIPED_CELL_ALPHA = 1.0
STRIPED_CELL_STRIPE_WIDTH = 4
LASSO_LINE_COLOR = (1.0, 1.0, 0.0, 0.9)


class TextGraphic(Protocol):
    text: str
    font_size: float
    offset: tuple[float, float, float]
    face_color: Color
    outline_color: Color
    outline_thickness: float
    visible: bool


class VisibleGraphic(Protocol):
    visible: bool


class ImageGraphic(Protocol):
    data: np.ndarray
    visible: bool


class LineGraphic(Protocol):
    data: np.ndarray
    colors: Color
    thickness: float
    visible: bool


class ScatterGraphic(Protocol):
    data: np.ndarray
    colors: Color
    edge_colors: Color
    sizes: float | np.ndarray
    visible: bool


class LabelGraphics(TypedDict):
    stim: TextGraphic
    stim_dash: TextGraphic
    cellpose: TextGraphic
    all: list[VisibleGraphic]


@dataclass
class _AOIDragState:
    shape: str
    part: str
    anchor_world: tuple[float, float]
    original_config: AreasOfInterestConfig


def compute_segmentation_legend(
    segmentation: np.ndarray,
) -> list[tuple[int, tuple[int, int, int]]]:
    """Assign each segmented cell a colour keyed by its raw Cellpose id."""
    unique_ids = np.unique(segmentation)
    neuron_ids = [cell_id for cell_id in unique_ids if cell_id >= 0]
    if not neuron_ids:
        return []

    zero_based = np.array(neuron_ids) - min(neuron_ids)
    id_colors = neuron_indices_to_colors(zero_based)
    return [
        (
            int(cell_id),
            (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)),
        )
        for cell_id, c in zip(neuron_ids, id_colors, strict=False)
    ]


class OverlayView(FullScreenFPLWidget):
    neuron_clicked = QtCore.Signal(int)
    neuron_lasso_selected = QtCore.Signal(list)

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
        self.striped_cell_ids: set[int] = set()
        self.striped_label_ids: set[int] = set()
        self._image_graphic: ImageGraphic | None = None
        self._segmentation_graphic: ImageGraphic | None = None
        self._label_centers: dict[int, tuple[float, float]] = {}
        self._label_graphics: dict[int, LabelGraphics] = {}
        self._label_segmentation_id: int | None = None
        self._segmentation_shape: tuple[int, int] | None = None
        self.base_camera_state: CameraState | None = None
        self.zoom_factor = 1
        self._aoi_config: AreasOfInterestConfig | None = None
        self._aoi_visible = True
        self._square_line: LineGraphic | None = None
        self._square_handles: list[ScatterGraphic] = []
        self._circle_line: LineGraphic | None = None
        self._circle_handles: list[ScatterGraphic] = []
        self._drag_state: _AOIDragState | None = None
        self.aoi_editable = True
        self.enable_cell_selection = False
        self._cell_click_down_world: tuple[float, float] | None = None
        self.lasso_mode = False
        self._lasso_points: list[tuple[float, float]] = []
        self._lasso_line: LineGraphic | None = None
        self.on_geometry_changed: Callable[[AreasOfInterestConfig], None] | None = None
        self._configure_subplot()
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")
        self.figure.renderer.add_event_handler(self._aoi_pointer_down, "pointer_down")
        self.figure.renderer.add_event_handler(self._aoi_pointer_up, "pointer_up")
        self.figure.renderer.add_event_handler(
            self._cell_selection_pointer_down, "pointer_down"
        )
        self.figure.renderer.add_event_handler(
            self._cell_selection_pointer_up, "pointer_up"
        )
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
        striped_cell_ids: Iterable[int] | None = None,
        striped_label_ids: Iterable[int] | None = None,
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
        self.striped_cell_ids = {int(cell_id) for cell_id in (striped_cell_ids or [])}
        self.striped_label_ids = {int(cell_id) for cell_id in (striped_label_ids or [])}

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
                    segmentation_for_display,
                    legend_data,
                    self.striped_cell_ids,
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
        self.set_aoi_shapes(self._aoi_config, self._aoi_visible)

    def _reset_graphics(self, *, clear_subplot: bool = False) -> None:
        if clear_subplot and getattr(self.subplot, "renderer", None) is not None:
            self.subplot.clear()
        self._image_graphic = None
        self._segmentation_graphic = None
        self._segmentation_shape = None
        self.text = None
        self.base_camera_state = None
        self._clear_label_graphics()
        self._clear_aoi_graphics()

    def set_aoi_shapes(
        self,
        config: AreasOfInterestConfig | None,
        visible: bool,
    ) -> None:
        self._aoi_config = clone_aoi_config(config)
        self._aoi_visible = visible
        if self.overlay_image is None or self._aoi_config is None or not visible:
            self._hide_aoi_graphics()
            self.figure.renderer.request_draw()
            return

        image_height = self.overlay_image.shape[0]
        self._update_square_graphics(self._aoi_config.square, image_height)
        self._update_circle_graphics(self._aoi_config.circle, image_height)
        self.figure.renderer.request_draw()

    def _clear_aoi_graphics(self) -> None:
        graphics: list[VisibleGraphic] = []
        if self._square_line is not None:
            graphics.append(self._square_line)
        if self._circle_line is not None:
            graphics.append(self._circle_line)
        graphics.extend(self._square_handles)
        graphics.extend(self._circle_handles)
        for graphic in graphics:
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(graphic)
        self._square_line = None
        self._circle_line = None
        self._square_handles = []
        self._circle_handles = []

    def _hide_aoi_graphics(self) -> None:
        graphics: list[VisibleGraphic] = []
        if self._square_line is not None:
            graphics.append(self._square_line)
        if self._circle_line is not None:
            graphics.append(self._circle_line)
        graphics.extend(self._square_handles)
        graphics.extend(self._circle_handles)
        for graphic in graphics:
            graphic.visible = False

    def _shape_color(self, role: AOIRole) -> tuple[float, float, float, float]:
        if role == AOIRole.EXCLUSION:
            return (0.95, 0.3, 0.3, 0.9)
        return (0.3, 0.85, 0.3, 0.9)

    def _update_square_graphics(self, square: SquareAOI, image_height: int) -> None:
        if self._square_line is None:
            self._square_line = self.subplot.add_line(
                square_world_polyline(square, image_height),
                colors=self._shape_color(square.role),
                thickness=2.0,
                size_space="model",
            )
        else:
            self._square_line.data = square_world_polyline(square, image_height)
            self._square_line.colors = self._shape_color(square.role)
            self._square_line.thickness = 2.0
        self._square_line.visible = square.enabled

        corner_points = square_world_corners(square, image_height)
        while len(self._square_handles) < 4:
            handle = self.subplot.add_scatter(
                np.asarray([[0.0, 0.0, 220.0]], dtype=np.float32),
                colors=self._shape_color(square.role),
                edge_colors=(1.0, 1.0, 1.0, 1.0),
                sizes=8,
                size_space="screen",
                uniform_size=True,
                uniform_color=True,
            )
            self._square_handles.append(handle)
        for idx, handle in enumerate(self._square_handles):
            handle.data = corner_points[idx : idx + 1]
            handle.colors = self._shape_color(square.role)
            handle.visible = square.enabled

    def _update_circle_graphics(self, circle: CircleAOI, image_height: int) -> None:
        if self._circle_line is None:
            self._circle_line = self.subplot.add_line(
                circle_world_polyline(circle, image_height),
                colors=self._shape_color(circle.role),
                thickness=2.0,
                size_space="model",
            )
        else:
            self._circle_line.data = circle_world_polyline(circle, image_height)
            self._circle_line.colors = self._shape_color(circle.role)
            self._circle_line.thickness = 2.0
        self._circle_line.visible = circle.enabled

        handle_points = circle_world_handles(circle, image_height)
        while len(self._circle_handles) < 4:
            handle = self.subplot.add_scatter(
                np.asarray([[0.0, 0.0, 220.0]], dtype=np.float32),
                colors=self._shape_color(circle.role),
                edge_colors=(1.0, 1.0, 1.0, 1.0),
                sizes=8,
                size_space="screen",
                uniform_size=True,
                uniform_color=True,
            )
            self._circle_handles.append(handle)
        for idx, handle in enumerate(self._circle_handles):
            handle.data = handle_points[idx : idx + 1]
            handle.colors = self._shape_color(circle.role)
            handle.visible = circle.enabled

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
            stim_dash = self.subplot.add_text(
                "",
                font_size=1,
                offset=base_offset,
                face_color=(0.0, 0.0, 0.0, 1.0),
                outline_color=(1.0, 1.0, 1.0),
                outline_thickness=0.6,
                center=False,
                screen_space=False,
            )
            all_graphics: list[VisibleGraphic] = [
                cellpose,
                stim,
                stim_dash,
            ]
            for graphic in all_graphics:
                graphic.visible = False

            self._label_graphics[cell_id] = {
                "stim": stim,
                "stim_dash": stim_dash,
                "cellpose": cellpose,
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
        striped_cell_ids: set[int] | None = None,
    ) -> np.ndarray:
        rgb = np.zeros((*segmentation.shape, 4), dtype=np.float32)
        neuron_ids = np.unique(segmentation)
        neuron_ids = neuron_ids[neuron_ids >= 0]
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
        if striped_cell_ids:
            y_grid, x_grid = np.indices(segmentation.shape)
            stripe_mask = ((x_grid + y_grid) // STRIPED_CELL_STRIPE_WIDTH) % 2 == 0
            for neuron_id in striped_cell_ids:
                mask = segmentation == neuron_id
                if not np.any(mask):
                    continue
                rgb[mask & stripe_mask, :3] = STRIPED_CELL_GREY
                rgb[mask & ~stripe_mask, :3] = STRIPED_CELL_WHITE
                rgb[mask, 3] = STRIPED_CELL_ALPHA
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

            striped_label = cell_id in self.striped_label_ids
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
                if striped_label:
                    cellpose_face = (0.0, 0.0, 0.0, 1.0)
                    cellpose_outline = (1.0, 1.0, 1.0, 1.0)
                    cellpose_outline_thickness = 1.0
                elif outdated_cellpose:
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
            else:
                graphics["cellpose"].visible = False

            if stim_visible:
                stim_text = f"{stim_display_id}"
                stim_face: Color = (0.25, 0.55, 0.9) if highlight else (0.0, 0.0, 0.0)
                stim_outline: Color = (1.0, 1.0, 1.0)
                stim_outline_thickness = 1.2 if mismatch else 0.6
                if striped_label:
                    stim_face = (0.0, 0.0, 0.0, 1.0)
                    stim_outline = (1.0, 1.0, 1.0, 1.0)
                    stim_outline_thickness = 1.0
                elif cross_out_default:
                    stim_face = (*stim_face, 0.25)
                    stim_outline = (*stim_outline, 0.25)
                stim_graphic = graphics["stim"]
                stim_graphic.text = stim_text
                stim_graphic.font_size = stim_font_size
                stim_graphic.offset = stim_offset
                stim_graphic.face_color = stim_face
                stim_graphic.outline_color = stim_outline
                stim_graphic.outline_thickness = stim_outline_thickness
                stim_graphic.visible = True
                stim_dash = graphics["stim_dash"]
                stim_dash.text = "-" * max(1, len(stim_text))
                stim_dash.font_size = stim_font_size
                stim_dash.offset = (
                    stim_offset[0],
                    stim_offset[1] + (stim_font_size * 0.15),
                    stim_offset[2],
                )
                stim_dash.face_color = (0.0, 0.0, 0.0, 1.0)
                stim_dash.outline_color = (1.0, 1.0, 1.0, 1.0)
                stim_dash.outline_thickness = max(1.0, stim_outline_thickness * 2.0)
                stim_dash.visible = cross_out_default
            else:
                graphics["stim"].visible = False
                graphics["stim_dash"].visible = False

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

    def _aoi_pointer_down(self, ev: pygfx.PointerEvent) -> None:
        if (
            self._aoi_config is None
            or not self._aoi_visible
            or self.overlay_image is None
            or not self.aoi_editable
        ):
            return

        mouse_pos = self.subplot.map_screen_to_world(ev)
        if mouse_pos is None:
            return

        hit = self._aoi_hit_test(float(mouse_pos[0]), float(mouse_pos[1]))
        if hit is None:
            return

        original_config = clone_aoi_config(self._aoi_config)
        assert original_config is not None
        self._drag_state = _AOIDragState(
            shape=hit[0],
            part=hit[1],
            anchor_world=(float(mouse_pos[0]), float(mouse_pos[1])),
            original_config=original_config,
        )
        self.subplot.controller.enabled = False
        if self.text is not None:
            self.text.visible = False
        ev.stop_propagation()

    def _aoi_pointer_up(self, ev: pygfx.PointerEvent) -> None:
        if self._drag_state is None:
            return
        self._drag_state = None
        self.subplot.controller.enabled = True
        if self._aoi_config is not None and self.on_geometry_changed is not None:
            config = clone_aoi_config(self._aoi_config)
            assert config is not None
            self.on_geometry_changed(config)
        ev.stop_propagation()

    def _cell_selection_pointer_down(self, ev: pygfx.PointerEvent) -> None:
        if not self.enable_cell_selection or self.segmentation is None:
            return
        mouse_pos = self.subplot.map_screen_to_world(ev)
        if mouse_pos is None:
            return
        if self.lasso_mode:
            self._lasso_points = [(float(mouse_pos[0]), float(mouse_pos[1]))]
            self.subplot.controller.enabled = False
            return
        self._cell_click_down_world = (float(mouse_pos[0]), float(mouse_pos[1]))

    def _cell_selection_pointer_up(self, ev: pygfx.PointerEvent) -> None:
        if not self.enable_cell_selection:
            return
        if self.lasso_mode:
            points = self._lasso_points
            self._lasso_points = []
            self._clear_lasso_line()
            self.subplot.controller.enabled = True
            if len(points) >= 3:
                cellpose_ids = self._cells_inside_polygon(points)
                if cellpose_ids:
                    self.neuron_lasso_selected.emit(cellpose_ids)
            return
        if self._cell_click_down_world is None:
            return
        down_x, down_y = self._cell_click_down_world
        self._cell_click_down_world = None
        mouse_pos = self.subplot.map_screen_to_world(ev)
        if mouse_pos is None:
            return
        if (
            math.hypot(float(mouse_pos[0]) - down_x, float(mouse_pos[1]) - down_y)
            > self._world_tolerance()
        ):
            return
        cellpose_id = self._segmentation_id_at(float(mouse_pos[0]), float(mouse_pos[1]))
        if cellpose_id is not None:
            self.neuron_clicked.emit(cellpose_id)

    def _cells_inside_polygon(self, points: list[tuple[float, float]]) -> list[int]:
        if not self._label_centers:
            return []
        polygon = MplPath(points)
        cell_ids, centers = zip(*self._label_centers.items(), strict=False)
        inside = polygon.contains_points(np.asarray(centers))
        return [
            cell_id
            for cell_id, is_inside in zip(cell_ids, inside, strict=False)
            if is_inside
        ]

    def _update_lasso_line(self) -> None:
        if len(self._lasso_points) < 2:
            return
        closed_points = [*self._lasso_points, self._lasso_points[0]]
        data = np.asarray([(x, y, 220.0) for x, y in closed_points], dtype=np.float32)
        # The line grows by one point per pointer_move; fastplotlib's LineGraphic
        # buffer is fixed-size at creation and can't be resized via `.data = ...`
        # (unlike the AOI square/circle lines, which always keep a constant point
        # count), so recreate the graphic each time instead of updating in place.
        if self._lasso_line is not None:
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(self._lasso_line)
        self._lasso_line = self.subplot.add_line(
            data,
            colors=LASSO_LINE_COLOR,
            thickness=2.0,
            size_space="model",
        )
        self._lasso_line.visible = True
        self.figure.renderer.request_draw()

    def _clear_lasso_line(self) -> None:
        if self._lasso_line is not None:
            with contextlib.suppress(Exception):
                self.subplot.remove_graphic(self._lasso_line)
            self._lasso_line = None
        self.figure.renderer.request_draw()

    def _segmentation_id_at(self, world_x: float, world_y: float) -> int | None:
        if self.segmentation is None:
            return None
        x, y = int(world_x), int(world_y)
        if (
            x < 0
            or y < 0
            or y >= self.segmentation.shape[0]
            or x >= self.segmentation.shape[1]
        ):
            return None
        cell_id = int(self.segmentation[y, x])
        return cell_id if cell_id >= 0 else None

    def _aoi_hit_test(self, world_x: float, world_y: float) -> tuple[str, str] | None:
        if self._aoi_config is None or self.overlay_image is None:
            return None

        image_height = self.overlay_image.shape[0]
        tolerance = self._world_tolerance()
        square_corners = square_world_corners(self._aoi_config.square, image_height)
        if self._aoi_config.square.enabled:
            for idx, point in enumerate(square_corners):
                if np.hypot(world_x - point[0], world_y - point[1]) <= tolerance:
                    return ("square", f"corner_{idx}")

        circle_handles = circle_world_handles(self._aoi_config.circle, image_height)
        if self._aoi_config.circle.enabled:
            for idx, point in enumerate(circle_handles):
                if np.hypot(world_x - point[0], world_y - point[1]) <= tolerance:
                    return ("circle", f"handle_{idx}")

        user_x, user_y = world_to_user_point(world_x, world_y, image_height)
        if self._aoi_config.circle.enabled and self._point_in_circle_body(
            self._aoi_config.circle, user_x, user_y
        ):
            return ("circle", "body")
        if self._aoi_config.square.enabled and self._point_in_square_body(
            self._aoi_config.square, user_x, user_y
        ):
            return ("square", "body")

        return None

    def _world_tolerance(self) -> float:
        camera_state = self.subplot.camera.get_state()
        camera_width = float(camera_state.get("width", 1.0))
        return max(2.0, (camera_width / max(1, self.width())) * 10.0)

    def _point_in_square_body(self, square: SquareAOI, ux: float, uy: float) -> bool:
        return (
            square.enabled
            and square.x <= ux <= square.x + square.side
            and square.y <= uy <= square.y + square.side
        )

    def _point_in_circle_body(self, circle: CircleAOI, ux: float, uy: float) -> bool:
        if not circle.enabled:
            return False
        radius = circle.diameter / 2.0
        return (ux - circle.cx) ** 2 + (uy - circle.cy) ** 2 <= radius * radius

    def _apply_drag(self, world_x: float, world_y: float) -> None:
        if (
            self._drag_state is None
            or self._aoi_config is None
            or self.overlay_image is None
        ):
            return

        original = self._drag_state.original_config
        delta_world_x = world_x - self._drag_state.anchor_world[0]
        delta_world_y = world_y - self._drag_state.anchor_world[1]
        delta_user_x = delta_world_x
        delta_user_y = -delta_world_y
        image_height, image_width = self.overlay_image.shape[:2]

        if self._drag_state.shape == "square":
            updated_square = self._drag_square(
                original.square,
                self._drag_state.part,
                delta_user_x,
                delta_user_y,
                world_x,
                world_y,
                image_width,
                image_height,
            )
            self._aoi_config.square = updated_square
        elif self._drag_state.shape == "circle":
            updated_circle = self._drag_circle(
                original.circle,
                self._drag_state.part,
                delta_user_x,
                delta_user_y,
                world_x,
                world_y,
                image_width,
                image_height,
            )
            self._aoi_config.circle = updated_circle

        self.set_aoi_shapes(self._aoi_config, self._aoi_visible)
        if self.on_geometry_changed is not None:
            config = clone_aoi_config(self._aoi_config)
            assert config is not None
            self.on_geometry_changed(config)

    def _drag_square(
        self,
        original: SquareAOI,
        part: str,
        delta_user_x: float,
        delta_user_y: float,
        world_x: float,
        world_y: float,
        image_width: int,
        image_height: int,
    ) -> SquareAOI:
        if part == "body":
            side = max(4.0, float(original.side))
            x = min(
                max(0.0, float(original.x) + delta_user_x), max(0.0, image_width - side)
            )
            y = min(
                max(0.0, float(original.y) + delta_user_y),
                max(0.0, image_height - side),
            )
            return SquareAOI(original.enabled, original.role, x, y, side)

        handle_index = int(part.rsplit("_", 1)[1])
        corners = square_user_corners(original)
        opposite_corner = corners[(handle_index + 2) % 4]
        handle_user = world_to_user_point(world_x, world_y, image_height)
        dx = handle_user[0] - opposite_corner[0]
        dy = handle_user[1] - opposite_corner[1]
        side = max(4.0, abs(dx), abs(dy))
        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0
        moved_x = opposite_corner[0] + (sign_x * side)
        moved_y = opposite_corner[1] + (sign_y * side)
        x = min(opposite_corner[0], moved_x)
        y = min(opposite_corner[1], moved_y)
        x = min(max(0.0, x), max(0.0, image_width - side))
        y = min(max(0.0, y), max(0.0, image_height - side))
        return SquareAOI(original.enabled, original.role, x, y, side)

    def _drag_circle(
        self,
        original: CircleAOI,
        part: str,
        delta_user_x: float,
        delta_user_y: float,
        world_x: float,
        world_y: float,
        image_width: int,
        image_height: int,
    ) -> CircleAOI:
        if part == "body":
            radius = max(2.0, float(original.diameter) / 2.0)
            cx = min(
                max(radius, float(original.cx) + delta_user_x),
                max(radius, image_width - radius),
            )
            cy = min(
                max(radius, float(original.cy) + delta_user_y),
                max(radius, image_height - radius),
            )
            return CircleAOI(original.enabled, original.role, cx, cy, radius * 2.0)

        handle_user = world_to_user_point(world_x, world_y, image_height)
        radius = math.dist((float(original.cx), float(original.cy)), handle_user)
        radius = max(2.0, radius)
        radius = min(
            radius,
            float(original.cx),
            float(image_width - original.cx),
            float(original.cy),
            float(image_height - original.cy),
        )
        return CircleAOI(
            original.enabled,
            original.role,
            float(original.cx),
            float(original.cy),
            radius * 2.0,
        )

    def pointer_move(self, ev: pygfx.PointerEvent) -> None:
        if self._drag_state is not None:
            mouse_pos = self.subplot.map_screen_to_world(ev)
            if mouse_pos is None:
                return
            self._apply_drag(float(mouse_pos[0]), float(mouse_pos[1]))
            ev.stop_propagation()
            return

        if self.lasso_mode and self._lasso_points:
            mouse_pos = self.subplot.map_screen_to_world(ev)
            if mouse_pos is None:
                return
            self._lasso_points.append((float(mouse_pos[0]), float(mouse_pos[1])))
            self._update_lasso_line()
            return

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
        cellpose_visible = self.show_cellpose_ids
        stim_visible = self.show_stimulus_ids and stim_id is not None

        mapped_elsewhere = (  # this is correct: if this current cellpose ID (The original ID) has a stim ID in
            # stim_to_cellpose, and it's not this one, a remapping has happened.
            cellpose_id in self.stim_to_cellpose
            and self.stim_to_cellpose.get(cellpose_id) != cellpose_id
        )
        overwritten = (
            self.show_stimulus_ids
            and cellpose_visible
            and stim_id is None
            and mapped_elsewhere
        )

        if overwritten:
            hover_text = f"Overwritten | ({cellpose_id})"
        elif cellpose_visible and stim_visible:
            hover_text = f"{stim_id}:{cellpose_id}"
        elif stim_visible:
            hover_text = f"{stim_id}"
        elif cellpose_visible:
            hover_text = f"{cellpose_id}"
        else:
            self.text.visible = False
            return

        self.text.visible = True
        self.text.offset = (mouse_pos[0], mouse_pos[1], 200)
        self.text.text = hover_text


class CellOverlayWidget(QtWidgets.QGroupBox):
    aoi_geometry_changed = QtCore.Signal(object)
    neuron_clicked = QtCore.Signal(int)
    neuron_lasso_selected = QtCore.Signal(list)

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
        self.striped_cell_ids: set[int] = set()
        self._aoi_config: AreasOfInterestConfig | None = None
        self._show_aoi = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        self.overlay_view = OverlayView(self)
        self.overlay_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_view.on_geometry_changed = lambda config: (
            self.aoi_geometry_changed.emit(config)
        )
        self.overlay_view.neuron_clicked.connect(self.neuron_clicked.emit)
        self.overlay_view.neuron_lasso_selected.connect(self.neuron_lasso_selected.emit)
        layout.addWidget(self.overlay_view)
        self.logger.info(
            "overlay ui: widget=%sx%s view=%sx%s",
            self.width(),
            self.height(),
            self.overlay_view.width(),
            self.overlay_view.height(),
        )

    def _update_overlay_display(self) -> None:
        display_segmentation = self._get_display_segmentation()
        display_legend = (
            []
            if not self.show_segmentation
            else self._get_display_legend(display_segmentation)
        )
        display_striped_cell_ids = self._get_display_striped_cell_ids()
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
            striped_cell_ids=display_striped_cell_ids,
            striped_label_ids=self.striped_cell_ids,
        )
        self.overlay_view.set_aoi_shapes(self._aoi_config, self._show_aoi)

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

    def _get_display_striped_cell_ids(self) -> set[int]:
        if not self.striped_cell_ids:
            return set()

        if not self.show_stimulus_ids or self.stimulus_to_cellpose_mapping is None:
            return set(self.striped_cell_ids)

        cellpose_to_stimulus = {
            v: k for k, v in self.stimulus_to_cellpose_mapping.items()
        }
        return {
            int(stimulus_id)
            for cellpose_id in self.striped_cell_ids
            if (stimulus_id := cellpose_to_stimulus.get(cellpose_id)) is not None
        }

    def set_overlay(
        self,
        overlay_image: np.ndarray | None,
        segmentation: np.ndarray | None,
        original_cellpose_ids: np.ndarray | None = None,
        stimulus_to_cellpose_mapping: dict[int, int] | None = None,
        legend_data: Iterable[tuple[int, tuple[int, int, int]]] | None = None,
        base_image: np.ndarray | None = None,
        highlight_mappings: bool | None = None,
        striped_cell_ids: Iterable[int] | None = None,
    ) -> None:
        self.logger.info(
            "set_overlay: image=%s seg=%s widget=%sx%s show_seg=%s show_cellpose=%s show_stim=%s",
            None if overlay_image is None else overlay_image.shape,
            None if segmentation is None else segmentation.shape,
            self.width(),
            self.height(),
            self.show_segmentation,
            self.show_cellpose_ids,
            self.show_stimulus_ids,
        )
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.legend_data = list(legend_data or [])
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_to_cellpose_mapping = stimulus_to_cellpose_mapping
        self.base_image = base_image
        self.striped_cell_ids = {int(cell_id) for cell_id in (striped_cell_ids or [])}
        if highlight_mappings is not None:
            self.highlight_mappings = highlight_mappings

        self._update_overlay_display()

    def set_aoi_overlay(
        self,
        config: AreasOfInterestConfig | None,
        *,
        show: bool,
    ) -> None:
        self._aoi_config = clone_aoi_config(config)
        self._show_aoi = show
        self.overlay_view.set_aoi_shapes(self._aoi_config, self._show_aoi)

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.base_image = None
        self.segmentation = None
        self.legend_data = []
        self.original_cellpose_ids = None
        self.stimulus_to_cellpose_mapping = None
        self.highlight_mappings = True
        self.striped_cell_ids = set()
        self._aoi_config = None
        self._show_aoi = False
        self.overlay_view.add_overlay_data(
            None,
            None,
            show_segmentation=self.show_segmentation,
            show_cellpose_ids=self.show_cellpose_ids,
            show_stimulus_ids=self.show_stimulus_ids,
            highlight_mappings=self.highlight_mappings,
        )
        self.logger.info(
            "set_empty_message: widget=%sx%s view=%sx%s",
            self.width(),
            self.height(),
            self.overlay_view.width(),
            self.overlay_view.height(),
        )

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

    def set_cell_selection_enabled(self, enabled: bool) -> None:
        self.overlay_view.enable_cell_selection = enabled

    def set_lasso_mode_enabled(self, enabled: bool) -> None:
        self.overlay_view.lasso_mode = enabled

    def set_aoi_editable(self, editable: bool) -> None:
        """Whether the AOI square/circle (if shown) can be dragged to reshape
        them, vs. displayed purely as a read-only reference."""
        self.overlay_view.aoi_editable = editable

    def _compute_segementation_colors(
        self, segmentation: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int]]]:
        return compute_segmentation_legend(segmentation)

    def closeEvent(self, event):
        # explicitly close all fastplotlib figures to avoid renderer trying to access them after they have been destroyed
        overlay_view = getattr(self, "overlay_view", None)
        if overlay_view is not None:
            overlay_view.close()
        super().closeEvent(event)
