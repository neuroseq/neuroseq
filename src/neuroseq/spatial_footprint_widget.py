from __future__ import annotations

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.cell_overlay_widget import STRIPED_CELL_GREY
from neuroseq.cell_overlay_widget import STRIPED_CELL_STRIPE_WIDTH
from neuroseq.cell_overlay_widget import STRIPED_CELL_WHITE
from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cell_overlay_widget import compute_segmentation_legend
from neuroseq.importer_cellpose import CellposeImportResult


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show imported cell spatial footprints with click-to-grey state."""

    import_requested = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top_bar_layout = QtWidgets.QHBoxLayout()
        top_bar_layout.addStretch(1)
        self.clear_selection_button = QtWidgets.QPushButton("Clear cell selection")
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.clicked.connect(self.clear_all_greyed)
        top_bar_layout.addWidget(self.clear_selection_button)
        top_bar_layout.addStretch(1)
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
        self.overlay_widget.neuron_clicked.connect(self._on_neuron_clicked)
        layout.addWidget(self.overlay_widget)

        self._cellpose_data: CellposeImportResult | None = None
        self._base_legend: list[tuple[int, tuple[int, int, int]]] = []
        self._greyed_cellpose_ids: set[int] = set()
        self.set_data(None)

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._cellpose_data = cellpose_data
        self._greyed_cellpose_ids = set()
        self._update_clear_selection_button()
        if cellpose_data is None:
            self._base_legend = []
            self.overlay_widget.set_empty_message("No cell spatial footprints loaded.")
            self.overlay_widget.setVisible(False)
            self.empty_state_widget.setVisible(True)
            return

        self.empty_state_widget.setVisible(False)
        self.overlay_widget.setVisible(True)
        self._base_legend = compute_segmentation_legend(
            cellpose_data.segmentation_array
        )
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if self._cellpose_data is None:
            return
        self.overlay_widget.set_overlay(
            self._greyed_overlay_image(),
            self._cellpose_data.segmentation_array,
            self._cellpose_data.original_cellpose_ids,
            self._cellpose_data.stimulus_to_cellpose_mapping,
            legend_data=self._base_legend,
            striped_cell_ids=self._greyed_cellpose_ids,
        )

    def _greyed_overlay_image(self) -> np.ndarray:
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        if not self._greyed_cellpose_ids:
            return cellpose_data.overlay_image

        image = np.array(cellpose_data.overlay_image, copy=True)
        segmentation = cellpose_data.segmentation_array
        y_grid, x_grid = np.indices(segmentation.shape)
        stripe_mask = ((x_grid + y_grid) // STRIPED_CELL_STRIPE_WIDTH) % 2 == 0
        grey = self._image_color_value(image, STRIPED_CELL_GREY)
        white = self._image_color_value(image, STRIPED_CELL_WHITE)

        for cellpose_id in self._greyed_cellpose_ids:
            mask = segmentation == cellpose_id
            if not np.any(mask):
                continue
            if image.ndim == 2:
                image[mask & stripe_mask] = grey[0]
                image[mask & ~stripe_mask] = white[0]
                continue

            n_channels = min(image.shape[2], 3)
            image[mask & stripe_mask, :n_channels] = grey[:n_channels]
            image[mask & ~stripe_mask, :n_channels] = white[:n_channels]

        return image

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
        self.clear_selection_button.setEnabled(bool(self._greyed_cellpose_ids))

    def clear_all_greyed(self) -> None:
        """Un-grey every currently greyed cell.

        Emits the same neuron_greyed_out_changed(stim_id, False) signal a
        click would for each one, so anything listening (trace/stim rows)
        stays in sync without duplicating that propagation logic here.
        """
        if not self._greyed_cellpose_ids:
            return
        cellpose_ids = list(self._greyed_cellpose_ids)
        self._greyed_cellpose_ids.clear()
        self._update_clear_selection_button()
        self._refresh_overlay()
        for cellpose_id in cellpose_ids:
            self.neuron_greyed_out_changed.emit(
                self._cellpose_to_stim_id(cellpose_id), False
            )

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
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

    def close(self) -> bool:
        overlay_widget = getattr(self, "overlay_widget", None)
        if overlay_widget is not None:
            overlay_widget.close()
        return super().close()
