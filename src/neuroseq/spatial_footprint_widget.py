from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from scipy.ndimage import binary_erosion

from neuroseq.cell_overlay_widget import STRIPED_CELL_GREY
from neuroseq.cell_overlay_widget import STRIPED_CELL_STRIPE_WIDTH
from neuroseq.cell_overlay_widget import STRIPED_CELL_WHITE
from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cell_overlay_widget import compute_segmentation_legend
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spatial_footprint_state import SpatialFootprintState

SELECTION_OUTLINE_RED = (0.9, 0.15, 0.15)
SELECTION_OUTLINE_BLACK = (0.0, 0.0, 0.0)
SELECTION_OUTLINE_STRIPE_WIDTH = 2
SELECTION_OUTLINE_THICKNESS = 2


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show imported cell spatial footprints with click-to-grey state.

    A view over a `SpatialFootprintState`. Multiple instances can share one
    state (pass the same object as `state` to each) and will stay in sync -
    a mutation via any one of them refreshes all of them. If `state` is
    omitted, this instance gets its own private state, matching the
    behavior of a single, non-shared panel.
    """

    import_requested = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        state: SpatialFootprintState | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state if state is not None else SpatialFootprintState()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        selection_tool_layout = QtWidgets.QHBoxLayout()
        self.click_selection_radio = QtWidgets.QRadioButton("Click")
        self.click_selection_radio.setChecked(True)
        self.lasso_selection_radio = QtWidgets.QRadioButton("Lasso")
        self._selection_tool_group = QtWidgets.QButtonGroup(self)
        self._selection_tool_group.addButton(self.click_selection_radio)
        self._selection_tool_group.addButton(self.lasso_selection_radio)
        selection_tool_layout.addWidget(self.click_selection_radio)
        selection_tool_layout.addWidget(self.lasso_selection_radio)
        selection_tool_layout.addStretch(1)
        layout.addLayout(selection_tool_layout)

        top_bar_layout = QtWidgets.QHBoxLayout()
        self.exclusion_mode_radio = QtWidgets.QRadioButton("Exclusion")
        self.exclusion_mode_radio.setChecked(True)
        self.preferential_mode_radio = QtWidgets.QRadioButton("Preferential selection")
        self._cell_mode_group = QtWidgets.QButtonGroup(self)
        self._cell_mode_group.addButton(self.exclusion_mode_radio)
        self._cell_mode_group.addButton(self.preferential_mode_radio)
        self.preferential_mode_radio.toggled.connect(self._on_cell_mode_changed)
        top_bar_layout.addWidget(self.exclusion_mode_radio)
        top_bar_layout.addWidget(self.preferential_mode_radio)
        top_bar_layout.addStretch(1)
        self.apply_selection_button = QtWidgets.QPushButton("Apply")
        self.apply_selection_button.setEnabled(False)
        self.apply_selection_button.clicked.connect(self._apply_preferential_selection)
        top_bar_layout.addWidget(self.apply_selection_button)
        top_bar_layout.addStretch(1)
        self.clear_selection_button = QtWidgets.QPushButton("Clear cell selection")
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.clicked.connect(self.clear_all_greyed)
        top_bar_layout.addWidget(self.clear_selection_button)
        layout.addLayout(top_bar_layout)

        self.empty_state_widget = QtWidgets.QWidget(self)
        empty_layout = QtWidgets.QVBoxLayout(self.empty_state_widget)
        empty_layout.addStretch(1)
        empty_label = QtWidgets.QLabel("No cell segmentation data loaded.")
        empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_label)
        self.import_button = QtWidgets.QPushButton("Import Cell Segmentation")
        self.import_button.clicked.connect(self.import_requested.emit)
        empty_layout.addWidget(
            self.import_button,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        empty_layout.addStretch(1)
        layout.addWidget(self.empty_state_widget)

        self.overlay_widget = CellOverlayWidget(title="Spatial Footprints")
        self.overlay_widget.set_display_options(
            show_cellpose_ids=False,
            show_stimulus_ids=True,
        )
        self.overlay_widget.set_cell_selection_enabled(True)
        # AOI shapes are shown here purely as a reference of what produced the
        # current mapping - editing them happens in the Cell Segmentation
        # import dialog, not here.
        self.overlay_widget.set_aoi_editable(False)
        self.overlay_widget.neuron_clicked.connect(self._on_neuron_clicked)
        self.overlay_widget.neuron_lasso_selected.connect(
            self._on_neuron_lasso_selected
        )
        self.lasso_selection_radio.toggled.connect(
            self.overlay_widget.set_lasso_mode_enabled
        )
        layout.addWidget(self.overlay_widget)

        self._base_legend: list[tuple[int, tuple[int, int, int]]] = []
        self._state.data_changed.connect(self._on_state_data_changed)
        self._state.greyed_changed.connect(self._on_state_greyed_changed)
        self._state.preferential_selection_changed.connect(
            self._on_state_preferential_changed
        )
        self._state.neuron_greyed_out_changed.connect(
            self.neuron_greyed_out_changed.emit
        )
        self._on_state_data_changed()

    @property
    def _cellpose_data(self) -> CellposeImportResult | None:
        return self._state.cellpose_data

    @property
    def _greyed_cellpose_ids(self) -> set[int]:
        return self._state.greyed_cellpose_ids

    @property
    def _preferential_selected_cellpose_ids(self) -> set[int]:
        return self._state.preferential_selected_cellpose_ids

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._state.set_data(cellpose_data)

    def _on_state_data_changed(self) -> None:
        cellpose_data = self._state.cellpose_data
        self._update_clear_selection_button()
        self._update_apply_button()
        if cellpose_data is None:
            self._base_legend = []
            self.overlay_widget.set_empty_message("No cell spatial footprints loaded.")
            self.overlay_widget.setVisible(False)
            self.empty_state_widget.setVisible(True)
            self.overlay_widget.set_aoi_overlay(None, show=False)
            return

        self.empty_state_widget.setVisible(False)
        self.overlay_widget.setVisible(True)
        self._base_legend = compute_segmentation_legend(
            cellpose_data.segmentation_array
        )
        self._refresh_overlay()
        aoi_config = cellpose_data.aoi_config
        self.overlay_widget.set_aoi_overlay(
            aoi_config, show=bool(aoi_config is not None and aoi_config.show_shapes)
        )

    def _on_state_greyed_changed(self) -> None:
        self._update_clear_selection_button()
        self._refresh_overlay()

    def _on_state_preferential_changed(self) -> None:
        self._update_clear_selection_button()
        self._update_apply_button()
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if self._cellpose_data is None:
            return
        self.overlay_widget.set_overlay(
            self._composed_overlay_image(),
            self._cellpose_data.segmentation_array,
            self._cellpose_data.original_cellpose_ids,
            self._cellpose_data.stimulus_to_cellpose_mapping,
            legend_data=self._base_legend,
            striped_cell_ids=self._greyed_cellpose_ids,
        )

    def _composed_overlay_image(self) -> np.ndarray:
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        if (
            not self._greyed_cellpose_ids
            and not self._preferential_selected_cellpose_ids
        ):
            return cellpose_data.overlay_image

        image = np.array(cellpose_data.overlay_image, copy=True)
        segmentation = cellpose_data.segmentation_array
        self._paint_striped(
            image,
            segmentation,
            self._greyed_cellpose_ids,
            STRIPED_CELL_GREY,
            STRIPED_CELL_WHITE,
        )
        self._paint_selection_outline(
            image, segmentation, self._preferential_selected_cellpose_ids
        )
        return image

    def _paint_striped(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        cellpose_ids: set[int],
        color_a: tuple[float, float, float],
        color_b: tuple[float, float, float],
    ) -> None:
        if not cellpose_ids:
            return
        y_grid, x_grid = np.indices(segmentation.shape)
        stripe_mask = ((x_grid + y_grid) // STRIPED_CELL_STRIPE_WIDTH) % 2 == 0
        value_a = self._image_color_value(image, color_a)
        value_b = self._image_color_value(image, color_b)

        for cellpose_id in cellpose_ids:
            mask = segmentation == cellpose_id
            if not np.any(mask):
                continue
            if image.ndim == 2:
                image[mask & stripe_mask] = value_a[0]
                image[mask & ~stripe_mask] = value_b[0]
                continue

            n_channels = min(image.shape[2], 3)
            image[mask & stripe_mask, :n_channels] = value_a[:n_channels]
            image[mask & ~stripe_mask, :n_channels] = value_b[:n_channels]

    def _paint_selection_outline(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        cellpose_ids: set[int],
    ) -> None:
        """Paint a red/black dashed ring around each cell's boundary, marking it as a
        pending (not-yet-applied) preferential selection."""
        if not cellpose_ids:
            return
        y_grid, x_grid = np.indices(segmentation.shape)
        stripe_mask = ((x_grid + y_grid) // SELECTION_OUTLINE_STRIPE_WIDTH) % 2 == 0
        red = self._image_color_value(image, SELECTION_OUTLINE_RED)
        black = self._image_color_value(image, SELECTION_OUTLINE_BLACK)

        for cellpose_id in cellpose_ids:
            mask = segmentation == cellpose_id
            if not np.any(mask):
                continue
            eroded = binary_erosion(mask, iterations=SELECTION_OUTLINE_THICKNESS)
            boundary = mask & ~eroded
            if image.ndim == 2:
                image[boundary & stripe_mask] = red[0]
                image[boundary & ~stripe_mask] = black[0]
                continue

            n_channels = min(image.shape[2], 3)
            image[boundary & stripe_mask, :n_channels] = red[:n_channels]
            image[boundary & ~stripe_mask, :n_channels] = black[:n_channels]

    @staticmethod
    def _image_color_value(
        image: np.ndarray,
        color: tuple[float, float, float],
    ) -> np.ndarray:
        if np.issubdtype(image.dtype, np.integer):
            scale = float(np.iinfo(image.dtype).max)
        else:
            max_value = float(np.nanmax(image)) if image.size else 1.0
            scale = 255.0 if max_value > 1.0 else 1.0
        values = np.asarray(color, dtype=np.float32) * scale
        values = np.clip(values, 0.0, scale)
        return values.astype(image.dtype, copy=False)

    def _update_clear_selection_button(self) -> None:
        self.clear_selection_button.setEnabled(
            bool(self._greyed_cellpose_ids)
            or bool(self._preferential_selected_cellpose_ids)
        )

    def _update_apply_button(self) -> None:
        self.apply_selection_button.setEnabled(
            self.preferential_mode_radio.isChecked()
            and bool(self._preferential_selected_cellpose_ids)
        )

    def clear_all_greyed(self) -> None:
        self._state.clear_all()

    def _on_cell_mode_changed(self, preferential_checked: bool) -> None:
        # NOTE: this clears the *shared* pending preferential selection when
        # this view's own mode radio switches away from preferential. With
        # multiple views sharing one state, switching one view's mode could
        # discard another view's in-progress preferential picks - an accepted
        # simplification for now, since the common case is a single view.
        if not preferential_checked:
            self._state.clear_preferential_selection()
        self._update_apply_button()

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
        if self.preferential_mode_radio.isChecked():
            self._state.toggle_preferential_selection(cellpose_id)
        else:
            self._state.toggle_greyed(cellpose_id)

    def _on_neuron_lasso_selected(self, cellpose_ids: list[int]) -> None:
        if self.preferential_mode_radio.isChecked():
            self._state.add_preferential_selection(cellpose_ids)
        else:
            self._state.add_greyed(cellpose_ids)

    def _apply_preferential_selection(self) -> None:
        self._state.apply_preferential_selection()

    def close(self) -> bool:
        overlay_widget = getattr(self, "overlay_widget", None)
        if overlay_widget is not None:
            overlay_widget.close()
        return super().close()
