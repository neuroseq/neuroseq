from __future__ import annotations

import contextlib
from collections.abc import Callable

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from scipy.ndimage import binary_erosion

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cell_overlay_widget import compute_segmentation_legend
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.spatial_footprint_state import SpatialFootprintState

SELECTION_OUTLINE_RED = (0.9, 0.15, 0.15)
SELECTION_OUTLINE_BLACK = (0.0, 0.0, 0.0)
SELECTION_OUTLINE_STRIPE_WIDTH = 2
SELECTION_OUTLINE_THICKNESS = 2


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show imported cell spatial footprints with click-to-select state.

    Clicking a cell selects it (marked with a red/black ring); every cell
    not currently selected is hidden from the trace/stim panels, live, with
    no separate apply step - a ring is the only visual change to a cell
    itself, its normal segmentation color is otherwise left alone. An empty
    selection means no filtering is in effect. Dragging (rather than a
    plain click) draws a lasso that selects every enclosed cell; clicking on
    empty space between cells clears the whole selection.

    A view over a `SpatialFootprintState`. Multiple instances can share one
    state (pass the same object as `state` to each) and will stay in sync -
    a mutation via any one of them refreshes all of them. If `state` is
    omitted, this instance gets its own private state, matching the
    behavior of a single, non-shared panel.
    """

    import_requested = QtCore.Signal()
    trim_to_mapped_neurons_requested = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)
    neurons_greyed_out_changed = QtCore.Signal(list, list)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        state: SpatialFootprintState | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state if state is not None else SpatialFootprintState()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top_bar_layout = QtWidgets.QHBoxLayout()
        self.trim_to_mapped_neurons_button = QtWidgets.QPushButton(
            "Trim to Mapped Neurons…"
        )
        self.trim_to_mapped_neurons_button.setEnabled(False)
        self.trim_to_mapped_neurons_button.setToolTip(
            "Permanently keep only the neurons with a Cellpose cell mapped to "
            "them, removing the rest from every stim array and experiment in "
            "this project."
        )
        self.trim_to_mapped_neurons_button.clicked.connect(
            self.trim_to_mapped_neurons_requested.emit
        )
        top_bar_layout.addWidget(self.trim_to_mapped_neurons_button)
        self.busy_indicator = QtWidgets.QProgressBar()
        self.busy_indicator.setRange(0, 0)  # indeterminate: no per-step progress
        self.busy_indicator.setTextVisible(False)
        self.busy_indicator.setMaximumWidth(120)
        self.busy_indicator.setVisible(False)
        top_bar_layout.addWidget(self.busy_indicator)
        top_bar_layout.addStretch(1)
        self.clear_selection_button = QtWidgets.QPushButton("Clear cell selection")
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.clicked.connect(self.clear_selection)
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
        self.overlay_widget.background_clicked.connect(self.clear_selection)
        layout.addWidget(self.overlay_widget)

        self._base_legend: list[tuple[int, tuple[int, int, int]]] = []
        # Keep the connection handles (not just the callables) so close() can
        # sever them precisely: a signal-to-signal connection like the third
        # one below can't be disconnected by passing the callable again (e.g.
        # `signal.disconnect(other_signal.emit)` raises "is not connected"
        # even though the connection is live) - only disconnecting the exact
        # handle .connect() returned works.
        self._state_connections = [
            self._state.data_changed.connect(self._on_state_data_changed),
            self._state.selection_changed.connect(self._on_state_selection_changed),
            self._state.neuron_greyed_out_changed.connect(
                self.neuron_greyed_out_changed.emit
            ),
            self._state.neurons_greyed_out_changed.connect(
                self.neurons_greyed_out_changed.emit
            ),
        ]
        self._on_state_data_changed()

    @property
    def _cellpose_data(self) -> CellposeImportResult | None:
        return self._state.cellpose_data

    @property
    def _greyed_cellpose_ids(self) -> set[int]:
        return self._state.greyed_cellpose_ids

    @property
    def _selected_cellpose_ids(self) -> set[int]:
        return self._state.selected_cellpose_ids

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._state.set_data(cellpose_data)

    def _on_state_data_changed(self) -> None:
        cellpose_data = self._state.cellpose_data
        self._update_clear_selection_button()
        self.trim_to_mapped_neurons_button.setEnabled(
            bool(
                cellpose_data is not None and cellpose_data.stimulus_to_cellpose_mapping
            )
        )
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

    def _on_state_selection_changed(self) -> None:
        self._update_clear_selection_button()
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
        )

    def _composed_overlay_image(self) -> np.ndarray:
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        if not self._selected_cellpose_ids:
            return cellpose_data.overlay_image

        image = np.array(cellpose_data.overlay_image, copy=True)
        segmentation = cellpose_data.segmentation_array
        self._paint_selection_ring(image, segmentation, self._selected_cellpose_ids)
        return image

    def _paint_selection_ring(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        cellpose_ids: set[int],
    ) -> None:
        """Paint a red/black dashed ring around each selected cell's
        boundary - the only visual change a selected cell gets; cells that
        are hidden as a result stay their normal color."""
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
        self.clear_selection_button.setEnabled(bool(self._selected_cellpose_ids))

    def clear_selection(self) -> None:
        self._with_busy_indicator(self._state.clear_selection)

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
        self._with_busy_indicator(lambda: self._state.toggle_selected(cellpose_id))

    def _on_neuron_lasso_selected(self, cellpose_ids: list[int]) -> None:
        self._with_busy_indicator(lambda: self._state.add_selected(cellpose_ids))

    def _with_busy_indicator(self, mutate: Callable[[], None]) -> None:
        """Run a selection-changing state mutation with a busy indicator
        shown for its duration.

        Selecting/deselecting a cell synchronously refreshes every view
        sharing this state (recomposing the overlay image, updating the
        trace/stim panels) - with a large segmentation or many neurons this
        can take a noticeable moment, during which Qt would otherwise not
        repaint at all (a single blocking call). Showing the indicator and
        forcing one repaint via processEvents() before starting the mutation
        is what makes it actually visible instead of flashing at the very
        end alongside everything else.
        """
        self.busy_indicator.setVisible(True)
        QtWidgets.QApplication.processEvents()
        try:
            mutate()
        finally:
            self.busy_indicator.setVisible(False)

    def close(self) -> bool:
        # A detached view shares self._state with every other view (docked
        # or detached) bound to it. Qt is supposed to auto-disconnect a
        # destroyed receiver, but this widget's underlying render canvas can
        # be torn down (via overlay_widget.close() below) well before its
        # Python/Qt object is actually deleted, leaving the connection live
        # in between. A later mutation from another view synchronously
        # invokes this (now GPU-resource-less) view's slot and segfaults
        # touching its dead canvas - disconnect explicitly instead of
        # relying on that timing.
        for connection in self._state_connections:
            with contextlib.suppress(TypeError, RuntimeError):
                self._state.disconnect(connection)  # type: ignore[call-arg]
        self._state_connections = []
        overlay_widget = getattr(self, "overlay_widget", None)
        if overlay_widget is not None:
            overlay_widget.close()
        return super().close()
