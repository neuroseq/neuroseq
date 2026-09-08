from __future__ import annotations

from collections.abc import Iterable

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

SELECTION_OUTLINE_RED = (0.9, 0.15, 0.15)
SELECTION_OUTLINE_BLACK = (0.0, 0.0, 0.0)
SELECTION_OUTLINE_STRIPE_WIDTH = 2
SELECTION_OUTLINE_THICKNESS = 2


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show imported cell spatial footprints with click-to-grey state."""

    import_requested = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
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

        self._cellpose_data: CellposeImportResult | None = None
        self._base_legend: list[tuple[int, tuple[int, int, int]]] = []
        self._greyed_cellpose_ids: set[int] = set()
        self._preferential_selected_cellpose_ids: set[int] = set()
        self.set_data(None)

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._cellpose_data = cellpose_data
        self._greyed_cellpose_ids = set()
        self._preferential_selected_cellpose_ids = set()
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

    def _cellpose_to_stim_id(self, cellpose_id: int) -> int:
        mapping = (
            self._cellpose_data.stimulus_to_cellpose_mapping
            if self._cellpose_data is not None
            else None
        )
        cellpose_to_stim = {v: k for k, v in (mapping or {}).items()}
        return cellpose_to_stim.get(cellpose_id, cellpose_id)

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
        """Un-grey every currently greyed cell and drop any pending (unapplied)
        preferential selection.

        Emits the same neuron_greyed_out_changed(stim_id, False) signal a
        click would for each greyed cell, so anything listening (trace/stim rows)
        stays in sync without duplicating that propagation logic here. Pending
        preferential picks were never greyed, so no signal is emitted for those.
        """
        if (
            not self._greyed_cellpose_ids
            and not self._preferential_selected_cellpose_ids
        ):
            return
        cellpose_ids = list(self._greyed_cellpose_ids)
        self._greyed_cellpose_ids.clear()
        self._preferential_selected_cellpose_ids.clear()
        self._update_clear_selection_button()
        self._update_apply_button()
        self._refresh_overlay()
        for cellpose_id in cellpose_ids:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), False
            )

    def _on_cell_mode_changed(self, preferential_checked: bool) -> None:
        if not preferential_checked and self._preferential_selected_cellpose_ids:
            self._preferential_selected_cellpose_ids.clear()
            self._update_clear_selection_button()
            self._refresh_overlay()
        self._update_apply_button()

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
        if self.preferential_mode_radio.isChecked():
            self._toggle_preferential_selection(cellpose_id)
        else:
            self._toggle_greyed(cellpose_id)

    def _toggle_greyed(self, cellpose_id: int) -> None:
        if cellpose_id in self._greyed_cellpose_ids:
            self._greyed_cellpose_ids.discard(cellpose_id)
            greyed = False
        else:
            self._greyed_cellpose_ids.add(cellpose_id)
            greyed = True
        self._update_clear_selection_button()
        self._refresh_overlay()
        self.neuron_greyed_out_changed.emit(
            self._cellpose_to_stim_id(cellpose_id),
            greyed,
        )

    def _toggle_preferential_selection(self, cellpose_id: int) -> None:
        if cellpose_id in self._preferential_selected_cellpose_ids:
            self._preferential_selected_cellpose_ids.discard(cellpose_id)
        else:
            self._preferential_selected_cellpose_ids.add(cellpose_id)
        self._update_clear_selection_button()
        self._update_apply_button()
        self._refresh_overlay()

    def _on_neuron_lasso_selected(self, cellpose_ids: list[int]) -> None:
        if self.preferential_mode_radio.isChecked():
            self._add_preferential_selection(cellpose_ids)
        else:
            self._add_greyed(cellpose_ids)

    def _add_greyed(self, cellpose_ids: Iterable[int]) -> None:
        """Add every id to the greyed set (additive - never un-greys an id already
        in the lasso, unlike the single-click toggle)."""
        new_ids = set(cellpose_ids) - self._greyed_cellpose_ids
        if not new_ids:
            return
        self._greyed_cellpose_ids |= new_ids
        self._update_clear_selection_button()
        self._refresh_overlay()
        for cellpose_id in new_ids:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), True
            )

    def _add_preferential_selection(self, cellpose_ids: Iterable[int]) -> None:
        """Add every id to the pending preferential selection (additive, same
        rationale as _add_greyed)."""
        new_ids = set(cellpose_ids) - self._preferential_selected_cellpose_ids
        if not new_ids:
            return
        self._preferential_selected_cellpose_ids |= new_ids
        self._update_clear_selection_button()
        self._update_apply_button()
        self._refresh_overlay()

    def _apply_preferential_selection(self) -> None:
        """Grey out every cell not marked as preferentially selected, and un-grey
        any previously-greyed cell that was marked - an absolute "only the
        selected cells stay visible" reset."""
        if self._cellpose_data is None or not self._preferential_selected_cellpose_ids:
            return
        all_ids = {
            int(cellpose_id)
            for cellpose_id in np.unique(self._cellpose_data.segmentation_array)
            if cellpose_id >= 0
        }
        old_greyed = self._greyed_cellpose_ids
        new_greyed = all_ids - self._preferential_selected_cellpose_ids
        self._preferential_selected_cellpose_ids = set()
        self._greyed_cellpose_ids = new_greyed
        self._update_clear_selection_button()
        self._update_apply_button()
        self._refresh_overlay()
        for cellpose_id in new_greyed - old_greyed:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), True
            )
        for cellpose_id in old_greyed - new_greyed:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), False
            )

    def close(self) -> bool:
        overlay_widget = getattr(self, "overlay_widget", None)
        if overlay_widget is not None:
            overlay_widget.close()
        return super().close()
