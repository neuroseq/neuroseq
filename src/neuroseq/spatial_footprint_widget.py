from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

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
        # The striped/grey look is drawn entirely by the (opaque, for striped
        # cells) segmentation colour layer in OverlayView; it always renders on
        # top of this base image, so there is nothing to gain from also
        # recolouring the base image itself.
        self.overlay_widget.set_overlay(
            self._cellpose_data.overlay_image,
            self._cellpose_data.segmentation_array,
            self._cellpose_data.original_cellpose_ids,
            self._cellpose_data.stimulus_to_cellpose_mapping,
            legend_data=self._base_legend,
            striped_cell_ids=self._greyed_cellpose_ids,
        )

    def _cellpose_to_stim_id(self, cellpose_id: int) -> int:
        mapping = (
            self._cellpose_data.stimulus_to_cellpose_mapping
            if self._cellpose_data is not None
            else None
        )
        cellpose_to_stim = {v: k for k, v in (mapping or {}).items()}
        return cellpose_to_stim.get(cellpose_id, cellpose_id)

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
        if cellpose_id in self._greyed_cellpose_ids:
            self._greyed_cellpose_ids.discard(cellpose_id)
            greyed = False
        else:
            self._greyed_cellpose_ids.add(cellpose_id)
            greyed = True
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
