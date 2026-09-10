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
UNSELECTED_CELL_BRIGHTNESS = 0.5
HOVERED_CELL_BRIGHTNESS_BOOST = 1.8
ACTIVE_STIM_BRIGHTNESS_BOOST = 1.8


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show imported cell spatial footprints with click-to-select state.

    Clicking a cell selects it (marked with a red/black ring); every cell
    not currently selected is hidden from the trace/stim panels, live, with
    no separate apply step, and is dimmed in this view so the selection
    stands out. An empty selection means no filtering is in effect (nothing
    dimmed). Dragging (rather than a plain click) draws a lasso that selects
    every enclosed cell; clicking on empty space between cells clears the
    whole selection. Hovering a row in the Trace/Stim panels highlights that
    same neuron here, brighter than a normal selected cell, so its location
    in the FOV is easy to spot. Every cell currently receiving nonzero
    stimulation at the time slider's position is brightened the same way,
    driven by gui.py from the active stim array - not an interaction with
    this panel.

    A view over a `SpatialFootprintState`. Multiple instances can share one
    state (pass the same object as `state` to each) and will stay in sync -
    a mutation via any one of them refreshes all of them. If `state` is
    omitted, this instance gets its own private state, matching the
    behavior of a single, non-shared panel.
    """

    import_requested = QtCore.Signal()
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
        self.busy_indicator = QtWidgets.QProgressBar()
        self.busy_indicator.setRange(0, 0)  # indeterminate: no per-step progress
        self.busy_indicator.setTextVisible(False)
        self.busy_indicator.setMaximumWidth(120)
        self.busy_indicator.setVisible(False)
        top_bar_layout.addWidget(self.busy_indicator)
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

        self.overlay_widget = CellOverlayWidget("")
        self.overlay_widget.set_display_options(
            show_cellpose_ids=False,
            show_stimulus_ids=True,
        )
        self.overlay_widget.set_cell_selection_enabled(True)
        # Right click (drag) only, so left click/drag is free to just pan
        # the view like everywhere else - the Cell Segmentation import
        # dialog is unaffected, and keeps selecting on any button.
        self.overlay_widget.set_cell_selection_button(2)
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

        # Same three display toggles as the Cell Segmentation import dialog
        # (cellpose_import_dialog.py), so this docked/detached panel offers
        # the same at-a-glance controls. Only meaningful once there's a
        # segmentation to toggle, so the whole row is hidden until then.
        checkbox_layout = QtWidgets.QHBoxLayout()
        checkbox_layout.addStretch(1)
        self.show_labels_checkbox = QtWidgets.QCheckBox("Show labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self._on_display_toggle_changed)
        checkbox_layout.addWidget(self.show_labels_checkbox)
        self.show_segmentation_checkbox = QtWidgets.QCheckBox("Show segmentation")
        self.show_segmentation_checkbox.setChecked(True)
        self.show_segmentation_checkbox.toggled.connect(self._on_display_toggle_changed)
        checkbox_layout.addWidget(self.show_segmentation_checkbox)
        self.show_shapes_checkbox = QtWidgets.QCheckBox("Show shapes")
        self.show_shapes_checkbox.setChecked(True)
        self.show_shapes_checkbox.setEnabled(False)
        self.show_shapes_checkbox.toggled.connect(self._on_display_toggle_changed)
        checkbox_layout.addWidget(self.show_shapes_checkbox)
        checkbox_layout.addStretch(1)
        self.checkbox_container = QtWidgets.QWidget()
        self.checkbox_container.setLayout(checkbox_layout)
        self.checkbox_container.setVisible(False)
        layout.addWidget(self.checkbox_container)

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
            self._state.hovered_stim_id_changed.connect(self._on_state_hovered_changed),
            self._state.active_stim_ids_changed.connect(
                self._on_state_active_stim_ids_changed
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
        if cellpose_data is None:
            self._base_legend = []
            self.overlay_widget.set_empty_message("No cell spatial footprints loaded.")
            self.overlay_widget.setVisible(False)
            self.empty_state_widget.setVisible(True)
            self.checkbox_container.setVisible(False)
            self.show_shapes_checkbox.setEnabled(False)
            self.overlay_widget.set_aoi_overlay(None, show=False)
            return

        self.empty_state_widget.setVisible(False)
        self.overlay_widget.setVisible(True)
        self.checkbox_container.setVisible(True)
        self._base_legend = compute_segmentation_legend(
            cellpose_data.segmentation_array
        )
        self._refresh_overlay()
        aoi_config = cellpose_data.aoi_config
        self.show_shapes_checkbox.setEnabled(aoi_config is not None)
        self.show_shapes_checkbox.setChecked(
            bool(aoi_config is not None and aoi_config.show_shapes)
        )
        self._update_aoi_overlay_visibility()

    def _on_display_toggle_changed(self) -> None:
        """Sync overlay display options with the checkboxes, matching the
        Cell Segmentation import dialog's own checkbox row."""
        self.overlay_widget.set_display_options(
            show_cellpose_ids=False,
            show_stimulus_ids=self.show_labels_checkbox.isChecked(),
            show_segmentation=self.show_segmentation_checkbox.isChecked(),
        )
        self._update_aoi_overlay_visibility()

    def _update_aoi_overlay_visibility(self) -> None:
        cellpose_data = self._cellpose_data
        aoi_config = cellpose_data.aoi_config if cellpose_data is not None else None
        self.overlay_widget.set_aoi_overlay(
            aoi_config,
            show=bool(aoi_config is not None and self.show_shapes_checkbox.isChecked()),
        )

    def _on_state_selection_changed(self) -> None:
        self._refresh_overlay()

    def _on_state_hovered_changed(self) -> None:
        # A hover can fire once per pointer-move event in the Trace/Stim
        # panels - far too often to pay for a full legend/label rebuild
        # (_refresh_overlay/add_overlay_data), so this only recolors the
        # existing overlay image in place.
        if self._cellpose_data is None:
            return
        self.overlay_widget.set_overlay_image(self._composed_overlay_image())

    def _on_state_active_stim_ids_changed(self) -> None:
        # Driven by the time slider, which can move once per playback tick
        # (25fps) - same lightweight, image-only update as hovering.
        if self._cellpose_data is None:
            return
        self.overlay_widget.set_overlay_image(self._composed_overlay_image())

    def _refresh_overlay(self) -> None:
        if self._cellpose_data is None:
            return
        self.overlay_widget.set_overlay(
            self._composed_overlay_image(),
            self._cellpose_data.segmentation_array,
            self._cellpose_data.original_cellpose_ids,
            self._cellpose_data.stimulus_to_cellpose_mapping,
            legend_data=self._base_legend,
            # Without this, unchecking "Show segmentation" only hides an
            # additional color graphic - the composed image itself already
            # has cell colors baked in (CellPoseImporter.generate_overlay),
            # so there'd be nothing visibly different to fall back to.
            base_image=self._cellpose_data.background_image,
        )

    def _composed_overlay_image(self) -> np.ndarray:
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        hovered_cellpose_id = self._state.hovered_cellpose_id
        active_cellpose_ids = self._state.active_cellpose_ids
        if (
            not self._selected_cellpose_ids
            and hovered_cellpose_id is None
            and not active_cellpose_ids
        ):
            return cellpose_data.overlay_image

        image = np.array(cellpose_data.overlay_image, copy=True)
        segmentation = cellpose_data.segmentation_array
        if self._selected_cellpose_ids:
            self._dim_unselected_cells(image, segmentation, self._selected_cellpose_ids)
        for cellpose_id in active_cellpose_ids:
            self._boost_cell_brightness(
                image,
                cellpose_data.overlay_image,
                segmentation,
                cellpose_id,
                ACTIVE_STIM_BRIGHTNESS_BOOST,
            )
        if hovered_cellpose_id is not None:
            self._boost_cell_brightness(
                image,
                cellpose_data.overlay_image,
                segmentation,
                hovered_cellpose_id,
                HOVERED_CELL_BRIGHTNESS_BOOST,
            )
        if self._selected_cellpose_ids:
            self._paint_selection_ring(image, segmentation, self._selected_cellpose_ids)
        return image

    @staticmethod
    def _dim_unselected_cells(
        image: np.ndarray,
        segmentation: np.ndarray,
        selected_cellpose_ids: set[int],
    ) -> None:
        """Darken every segmented cell that is not currently selected, so
        the selected cell(s) stand out against the rest. Background pixels
        (not part of any cell, negative ids) are left untouched."""
        dim_mask = (segmentation >= 0) & ~np.isin(
            segmentation, np.array(list(selected_cellpose_ids))
        )
        if not np.any(dim_mask):
            return
        image[dim_mask] = (
            image[dim_mask].astype(np.float32) * UNSELECTED_CELL_BRIGHTNESS
        ).astype(image.dtype, copy=False)

    @staticmethod
    def _boost_cell_brightness(
        image: np.ndarray,
        original_image: np.ndarray,
        segmentation: np.ndarray,
        cellpose_id: int,
        boost: float,
    ) -> None:
        """Boost one cell's brightness beyond even a normal selected cell's,
        so it's unmistakable in the FOV - regardless of whether it's
        currently dimmed (unselected) or not, computed from the pristine
        `original_image` rather than the already-dimmed `image` so boosting
        a dimmed cell always restores it to full brightness and beyond.
        Used both for the row currently hovered in the Trace/Stim panels
        and for rows currently receiving nonzero stimulation at the time
        slider's position."""
        mask = segmentation == cellpose_id
        if not np.any(mask):
            return
        if np.issubdtype(image.dtype, np.integer):
            max_value = float(np.iinfo(image.dtype).max)
        else:
            max_value = float(np.nanmax(original_image)) if original_image.size else 1.0
            max_value = 255.0 if max_value > 1.0 else 1.0
        boosted = original_image[mask].astype(np.float32) * boost
        image[mask] = np.clip(boosted, 0.0, max_value).astype(image.dtype, copy=False)

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
