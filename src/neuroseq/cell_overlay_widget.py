from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import brian2.numpy_ as np
import fastplotlib as fpl
import pygfx
from cmap import Colormap
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.utils import QtABCMeta
from neuroseq.utils import jet_colors_for_neurons

feature_flag_text = True  # todo: remove.


# todo: this is a bit slow to render, not sure right now why. Need to test: initial zoom.
class OverlayView(QtWidgets.QWidget, metaclass=QtABCMeta):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure: fpl.Figure = fpl.Figure(canvas_kwargs={"parent": self})
        layout = QtWidgets.QVBoxLayout(self)
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
        self.setMinimumSize(QtCore.QSize(200, 200))
        self.resize(800, 800)
        self.widget.setMinimumHeight(1000)
        self.logger = logging.getLogger(__name__)
        self.mouse_pos: np.ndarray | None = None
        self.base_camera_state: dict | None = None
        self.zoom_factor = 2.5
        self.figure.renderer.add_event_handler(self.pointer_move, "pointer_move")

    def add_overlay_data(
        self, overlay_image: np.ndarray | None, segmentation: np.ndarray | None
    ) -> None:
        self.subplot.clear()
        self.overlay_image = overlay_image
        self.segmentation = segmentation
        self.overlay_graphic = None
        self.segmentation_graphic = None
        self.text = None
        self.base_camera_state = None

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
            cmap = Colormap("jet")
            norm = float(np.max(segmentation))
            if norm == 0.0:
                norm = 1.0
            rgb = np.array(
                [[np.asarray(cmap(v / norm)) for v in row] for row in segmentation]
            )
            rgb[:, :, 3] = 0.4
            rgb[segmentation == 0, 3] = 0
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

        self._reset_camera()

    def _reset_camera(self) -> None:
        """Fit the camera to the image and apply the current zoom factor."""
        if self.overlay_image is None:
            self.base_camera_state = None
            return

        height, width = self.overlay_image.shape[:2]
        cam_state = self.subplot.camera.get_state()
        camera_z = float(cam_state.get("position", (0.0, 0.0, 0.0))[2])
        maintain_aspect = bool(cam_state.get("maintain_aspect", True))
        self.base_camera_state = {
            "width": float(width),
            "height": float(height),
            "position": (float(width) / 2.0, float(height) / 2.0, camera_z),
            "maintain_aspect": maintain_aspect,
        }
        self.subplot.axes.visible = False
        self.subplot.grids = False
        self.subplot.toolbar = False
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
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.overlay_view = OverlayView()
        self.overlay_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
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
        layout.addWidget(self.legend_scroll)

    def _on_id_toggle_changed(self, checked: bool) -> None:
        """Handle toggle between stimulus IDs and original cellpose IDs"""
        self.show_stimulus_ids = checked
        self._update_overlay_display()

    def _update_overlay_display(self) -> None:
        """Update the overlay display with current ID mode"""
        display_segmentation = self._get_display_segmentation()
        self.overlay_view.add_overlay_data(self.overlay_image, display_segmentation)
        self._update_legend()

    def _get_display_segmentation(self) -> np.ndarray | None:
        """Get segmentation array with appropriate IDs for current display mode"""
        if self.segmentation is None:
            return None

        if not self.show_stimulus_ids or self.stimulus_to_cellpose_mapping is None:
            # Show original cellpose IDs (convert 0-based back to 1-based)
            display_seg = self.segmentation.copy()
            display_seg[display_seg > 0] = display_seg[display_seg > 0] + 1
            return display_seg
        else:
            # Show stimulus IDs based on mapping
            display_seg = np.zeros_like(self.segmentation)
            # Create reverse mapping from cellpose ID to stimulus ID
            cellpose_to_stimulus = {
                v: k for k, v in self.stimulus_to_cellpose_mapping.items()
            }

            for cellpose_id, stimulus_id in cellpose_to_stimulus.items():
                # cellpose_id is 1-based from mapping, but segmentation is 0-based
                mask = self.segmentation == (cellpose_id - 1)
                display_seg[mask] = stimulus_id

            return display_seg

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

    def set_empty_message(self, message: str) -> None:
        self.overlay_image = None
        self.segmentation = None
        self.legend_data = []
        self.overlay_view.add_overlay_data(None, None)
        self._clear_legend()

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
