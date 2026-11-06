from __future__ import annotations

import contextlib

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtWidgets
from scipy.ndimage import binary_erosion

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.qt_helpers import StableSizeTabWidget
from neuroseq.visualization.calcium_video_widget import CalciumVideoWidget
from neuroseq.visualization.cell_overlay_widget import CellOverlayWidget
from neuroseq.visualization.cell_overlay_widget import compute_segmentation_legend
from neuroseq.visualization.fpl_widget import FPLWidget
from neuroseq.visualization_support.brightness_contrast import BrightnessContrastDialog
from neuroseq.visualization_support.brightness_contrast import DisplayRange
from neuroseq.visualization_support.brightness_contrast import adjust_fov_and_overlay
from neuroseq.visualization_support.cellpose_ids import cellpose_to_stim_id
from neuroseq.visualization_support.spatial_footprint_state import SpatialFootprintState

SELECTION_OUTLINE_RED = (0.9, 0.15, 0.15)
SELECTION_OUTLINE_BLACK = (0.0, 0.0, 0.0)
SELECTION_OUTLINE_STRIPE_WIDTH = 2
SELECTION_OUTLINE_THICKNESS = 2
UNSELECTED_CELL_BRIGHTNESS = 0.5
HOVERED_CELL_BRIGHTNESS_BOOST = 1.8
ACTIVE_STIM_BRIGHTNESS_BOOST = 1.8
# CellPoseImporter.generate_overlay's default blend of cell color over the FOV.
OVERLAY_CELL_OPACITY = 0.5


def carry_image_view(
    source: FPLWidget,
    source_shape: tuple[int, int],
    target: FPLWidget,
    target_shape: tuple[int, int],
) -> None:
    """Show the part of `target`'s image that `source` shows of its own.

    The images cover the same field of view, possibly at different
    resolutions (live previews are downsampled), so the view is scaled by the
    resolution ratio. Images with a different aspect ratio are left alone.
    """
    (source_rows, source_cols), (target_rows, target_cols) = (
        source_shape,
        target_shape,
    )
    scale_x = target_cols / source_cols
    scale_y = target_rows / source_rows
    # Downsampling rounds up, e.g. 511 columns -> 256, so allow ~1%.
    if abs(scale_x - scale_y) > 0.01 * max(scale_x, scale_y):
        return
    state = source.subplot.camera.get_state()
    x, y = state["position"][:2]
    z = target.subplot.camera.get_state()["position"][2]
    # Pixel centres sit at integers, so an image spans -0.5 .. size - 0.5.
    target.subplot.camera.set_state(
        {
            "position": ((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5, z),
            "width": state["width"] * scale_x,
            "height": state["height"] * scale_y,
            "zoom": state["zoom"],
        }
    )
    target.figure.renderer.request_draw()


class _PanelLayout(QtWidgets.QVBoxLayout):
    """Expose widgets nested in the cell tab for legacy layout introspection."""

    def indexOf(  # type: ignore[override]
        self, widget: QtWidgets.QWidget | QtWidgets.QLayoutItem | None
    ) -> int:
        direct_index = super().indexOf(widget)
        if not isinstance(widget, QtWidgets.QWidget):
            return direct_index
        if direct_index >= 0:
            return direct_index
        for index in range(super().count()):
            item = super().itemAt(index)
            if item is None:
                continue
            nested_layout = item.layout()
            nested_widget = item.widget()
            if nested_layout is None and nested_widget is not None:
                nested_layout = nested_widget.layout()
            if isinstance(nested_widget, QtWidgets.QTabWidget):
                for page_index in range(nested_widget.count()):
                    page = nested_widget.widget(page_index)
                    page_layout = page.layout() if page is not None else None
                    if page_layout is not None and page_layout.indexOf(widget) >= 0:
                        return index
            if nested_layout is not None and nested_layout.indexOf(widget) >= 0:
                return index
        return -1


class SpatialFootprintWidget(QtWidgets.QWidget):
    """Show cell spatial footprints and calcium video in a tabbed panel.

    A plain left click on a cell selects it (marked with a red/black ring);
    every cell not currently selected is hidden from the trace/stim panels,
    live, with no separate apply step, and is dimmed in this view so the
    selection stands out. An empty selection means no filtering is in
    effect (nothing dimmed). Left click-and-hold never selects - it just
    pans the view, like everywhere else, unless Shift is held, in which
    case it draws a lasso instead (same as a plain right click-and-hold,
    with no Shift needed). A lasso toggles every enclosed cell: one already
    selected is deselected, one that isn't is added - so lassoing over a
    mix of selected and unselected cells is not purely additive. Clicking
    on empty space (left) clears the whole selection. Hovering a row in the
    Trace/Stim panels highlights that same neuron here, brighter than a normal
    selected cell, so its location in the FOV is easy to spot. Every cell
    currently receiving nonzero stimulation at the time slider's position
    is brightened the same way, driven by gui.py from the active stim
    array - not an interaction with this panel.

    The first tab shows imported cell spatial footprints; the second tab
    contains the raw calcium video and its stimulation overlay. A view over a
    `SpatialFootprintState`. Multiple instances can share one
    state (pass the same object as `state` to each) and will stay in sync -
    a mutation via any one of them refreshes all of them. If `state` is
    omitted, this instance gets its own private state, matching the
    behavior of a single, non-shared panel.
    """

    import_requested = QtCore.Signal()
    neuron_greyed_out_changed = QtCore.Signal(int, bool)
    neurons_greyed_out_changed = QtCore.Signal(list, list)
    # Right click on a cell: show this neuron's trace (issue #603).
    neuron_focus_requested = QtCore.Signal(int)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        state: SpatialFootprintState | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state if state is not None else SpatialFootprintState()
        root_layout = _PanelLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = StableSizeTabWidget(self)
        self.tabs.setDocumentMode(True)
        cell_tab = QtWidgets.QWidget(self.tabs)
        layout = QtWidgets.QVBoxLayout(cell_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        # Matches CalciumVideoWidget's own layout (see FPLWidget) - without
        # this, Qt's default inter-widget spacing opened a gap above the
        # checkbox row that the Video tab's bottom bar doesn't have.
        layout.setSpacing(0)

        # Same three display toggles as the Cell Segmentation import dialog
        # (cellpose_import_dialog.py), so this docked/detached panel offers
        # the same at-a-glance controls. Only meaningful once there's a
        # segmentation to toggle, so the whole row is hidden until then.
        checkbox_layout = QtWidgets.QHBoxLayout()
        # Matches CalciumVideoWidget's bottom action bar margins exactly, so
        # both tabs' bottom bars look identical.
        checkbox_layout.setContentsMargins(4, 4, 4, 4)
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
        self.show_targets_checkbox = QtWidgets.QCheckBox("Show targets")
        self.show_targets_checkbox.setChecked(True)
        self.show_targets_checkbox.setToolTip(
            "Mark, with a red cross, every neuron targeted by the sequence "
            "whose import was confirmed by the stimulation setup. Not live illumination."
        )
        self.show_targets_checkbox.toggled.connect(
            self._update_target_overlay_visibility
        )
        checkbox_layout.addWidget(self.show_targets_checkbox)
        self.brightness_contrast_button = QtWidgets.QPushButton("Brightness/Contrast…")
        self.brightness_contrast_button.setToolTip(
            "Adjust the display brightness/contrast of the FOV image "
            "(display only - the image data is not changed)"
        )
        self.brightness_contrast_button.clicked.connect(
            self._show_brightness_contrast_dialog
        )
        checkbox_layout.addWidget(self.brightness_contrast_button)
        checkbox_layout.addStretch(1)
        self.checkbox_container = QtWidgets.QWidget()
        self.checkbox_container.setLayout(checkbox_layout)
        self.checkbox_container.setVisible(False)
        # Same fixed height and size policy as CalciumVideoWidget's bottom
        # action bar (Save video.../Remove imaging data), so the Cell
        # segmentation and Video tabs both end with a bar of identical size
        # - keeps StableSizeTabWidget's per-page size hints matched instead
        # of merely close, for the smoothest possible switch between them.
        self.checkbox_container.setFixedHeight(44)
        self.checkbox_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

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
        # Left click selects a cell; left click-and-hold just pans the view
        # like everywhere else (a left drag is never a lasso here). Right
        # click-and-hold is the lasso instead. The Cell Segmentation import
        # dialog is unaffected and keeps responding to any button for both.
        self.overlay_widget.set_cell_selection_buttons(click_button=1, lasso_button=2)
        # AOI shapes are shown here purely as a reference of what produced the
        # current mapping - editing them happens in the Cell Segmentation
        # import dialog, not here.
        self.overlay_widget.set_aoi_editable(False)
        self.overlay_widget.neuron_clicked.connect(self._on_neuron_clicked)
        self.overlay_widget.neuron_lasso_selected.connect(
            self._on_neuron_lasso_selected
        )
        self.overlay_widget.background_clicked.connect(self.clear_selection)
        self.overlay_widget.neuron_right_clicked.connect(self._on_neuron_right_clicked)
        layout.addWidget(self.overlay_widget)
        # At the bottom, like CalciumVideoWidget's action bar - see the
        # checkbox_container size comment above.
        layout.addWidget(self.checkbox_container)

        self._base_legend: list[tuple[int, tuple[int, int, int]]] = []
        self._brightness_contrast_dialog: BrightnessContrastDialog | None = None
        self._adjusted_cache: tuple[tuple, np.ndarray | None, np.ndarray] | None = None
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
            self._state.display_range_changed.connect(
                self._on_state_display_range_changed
            ),
            self._state.targeted_stim_ids_changed.connect(
                self._update_target_overlay_visibility
            ),
        ]
        self.calcium_video_widget = CalciumVideoWidget(parent=self.tabs)
        self.tabs.addTab(cell_tab, "Cell segmentation")
        self.tabs.addTab(self.calcium_video_widget, "Imaging")
        # checkbox_container now lives inside cell_tab's own layout (added
        # above), so self.tabs (a StableSizeTabWidget) is root_layout's only
        # child and already reports a size hint that's stable across tabs -
        # no need for SpatialFootprintWidget to override its own.
        root_layout.addWidget(self.tabs)
        # Both tabs show the same field of view: switching tabs keeps the
        # zoom and pan of the tab being left.
        self._view_tab = self.tabs.currentIndex()
        self.tabs.currentChanged.connect(self._carry_view_to_tab)
        self._on_state_data_changed()

    def apply_theme(self, theme: str) -> None:
        """Forward the theme to both imaging tabs."""
        self.overlay_widget.apply_theme(theme)
        self.calcium_video_widget.apply_theme(theme)

    @property
    def _cellpose_data(self) -> CellposeImportResult | None:
        return self._state.cellpose_data

    @property
    def _greyed_cellpose_ids(self) -> set[int]:
        return self._state.greyed_cellpose_ids

    @property
    def _selected_cellpose_ids(self) -> set[int]:
        return self._state.selected_cellpose_ids

    def _tab_image_view(self, index: int) -> tuple[FPLWidget, tuple[int, int]] | None:
        """The canvas of tab `index` and the (rows, cols) of its image."""
        if self.tabs.widget(index) is self.calcium_video_widget:
            video = self.calcium_video_widget
            graphic = video._frame_graphic
            if graphic is None or video.figure is None:
                return None
            return video, graphic.data.value.shape[:2]
        overlay = self.overlay_widget.overlay_view
        if overlay.overlay_image is None or overlay.figure is None:
            return None
        return overlay, overlay.overlay_image.shape[:2]

    def _carry_view_to_tab(self, index: int) -> None:
        previous, self._view_tab = self._view_tab, index
        source = self._tab_image_view(previous)
        target = self._tab_image_view(index)
        if previous != index and source is not None and target is not None:
            carry_image_view(*source, *target)

    def set_data(self, cellpose_data: CellposeImportResult | None) -> None:
        self._state.set_data(cellpose_data)

    def _on_state_data_changed(self) -> None:
        cellpose_data = self._state.cellpose_data
        if cellpose_data is None:
            self._base_legend = []
            self.overlay_widget.set_empty_message("No cell spatial footprints loaded.")
            self._update_overlay_visibility()
            self._update_empty_state_visibility()
            self._update_checkbox_visibility()
            self.show_shapes_checkbox.setEnabled(False)
            self.overlay_widget.set_aoi_overlay(None, show=False)
            self._update_target_overlay_visibility()
            return

        if self._brightness_contrast_dialog is not None:
            self._brightness_contrast_dialog.set_image(cellpose_data.background_image)
            self._brightness_contrast_dialog.set_display_range(
                self._display_range, emit=False
            )
        self._update_empty_state_visibility()
        self._update_overlay_visibility()
        self._update_checkbox_visibility()
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
        self._update_target_overlay_visibility()

    def _update_checkbox_visibility(self) -> None:
        # No tabs.currentIndex() check needed: checkbox_container lives
        # inside the Cell segmentation page now, so the tab widget already
        # hides it on screen whenever the Video tab is current. Gating this
        # on the current tab too would make it (and therefore cell_tab's own
        # size hint) flicker between two different sizes depending on
        # whichever tab happens to be showing - exactly the bounce
        # StableSizeTabWidget exists to prevent.
        self.checkbox_container.setVisible(self._cellpose_data is not None)

    def _update_empty_state_visibility(self) -> None:
        self.empty_state_widget.setVisible(self._cellpose_data is None)

    def _update_overlay_visibility(self) -> None:
        self.overlay_widget.setVisible(self._cellpose_data is not None)

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

    def _update_target_overlay_visibility(self) -> None:
        self.overlay_widget.set_target_overlay(
            self._state.targeted_cellpose_ids,
            show=self.show_targets_checkbox.isChecked(),
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
            base_image=self._adjusted_images()[0],
        )

    def _show_brightness_contrast_dialog(self) -> None:
        dialog = self._brightness_contrast_dialog
        if dialog is None:
            dialog = BrightnessContrastDialog(self)
            dialog.range_changed.connect(self._on_dialog_range_changed)
            self._brightness_contrast_dialog = dialog
        cellpose_data = self._cellpose_data
        dialog.set_image(
            None if cellpose_data is None else cellpose_data.background_image
        )
        dialog.set_display_range(self._display_range, emit=False)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @property
    def _display_range(self) -> DisplayRange:
        stored = self._state.display_range
        return DisplayRange() if stored is None else DisplayRange.clamped(*stored)

    def _on_dialog_range_changed(self, display_range: DisplayRange) -> None:
        # Stored on the shared state (and so the project), which re-renders
        # every view bound to it - including this one.
        self._state.set_display_range(
            None
            if display_range.is_identity
            else (display_range.minimum, display_range.maximum)
        )

    def _on_state_display_range_changed(self) -> None:
        if self._cellpose_data is None:
            return
        if self._brightness_contrast_dialog is not None:
            self._brightness_contrast_dialog.set_display_range(
                self._display_range, emit=False
            )
        # Image-only updates: no label/legend rebuild per slider tick.
        background, _ = self._adjusted_images()
        self.overlay_widget.set_overlay_image(self._composed_overlay_image())
        self.overlay_widget.set_base_image(background)

    def _adjusted_images(self) -> tuple[np.ndarray | None, np.ndarray]:
        """(FOV background, overlay) with the brightness/contrast window
        applied to the FOV only - cell colors keep their exact legend colors.

        The overlay is the FOV blended with cell color inside segmented
        pixels (see CellPoseImporter.generate_overlay), so the FOV's change
        is added back at the weight it carries there: 1 outside cells,
        1 - OVERLAY_CELL_OPACITY inside.
        """
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        background = cellpose_data.background_image
        overlay = cellpose_data.overlay_image
        key = (id(background), id(overlay), self._display_range)
        if self._adjusted_cache is not None and self._adjusted_cache[0] == key:
            return self._adjusted_cache[1], self._adjusted_cache[2]
        adjusted_background, adjusted_overlay = adjust_fov_and_overlay(
            background,
            overlay,
            cellpose_data.segmentation_array,
            self._display_range,
            OVERLAY_CELL_OPACITY,
        )
        if adjusted_background is not background:
            assert adjusted_background is not None
            self._adjusted_cache = (key, adjusted_background, adjusted_overlay)
        return adjusted_background, adjusted_overlay

    def _composed_overlay_image(self) -> np.ndarray:
        cellpose_data = self._cellpose_data
        assert cellpose_data is not None
        hovered_cellpose_id = self._state.hovered_cellpose_id
        active_cellpose_ids = self._state.active_cellpose_ids
        base_overlay = self._adjusted_images()[1]
        if (
            not self._selected_cellpose_ids
            and hovered_cellpose_id is None
            and not active_cellpose_ids
        ):
            return base_overlay

        image = np.array(base_overlay, copy=True)
        segmentation = cellpose_data.segmentation_array
        if self._selected_cellpose_ids:
            self._dim_unselected_cells(image, segmentation, self._selected_cellpose_ids)
        for cellpose_id in active_cellpose_ids:
            self._boost_cell_brightness(
                image,
                base_overlay,
                segmentation,
                cellpose_id,
                ACTIVE_STIM_BRIGHTNESS_BOOST,
            )
        if hovered_cellpose_id is not None:
            self._boost_cell_brightness(
                image,
                base_overlay,
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
        self._state.clear_selection()

    def _on_neuron_clicked(self, cellpose_id: int) -> None:
        self._state.toggle_selected(cellpose_id)

    def _on_neuron_right_clicked(self, cellpose_id: int) -> None:
        neuron_id = cellpose_to_stim_id(cellpose_id, self._cellpose_data)
        if neuron_id is not None:
            self.neuron_focus_requested.emit(neuron_id)

    def _on_neuron_lasso_selected(self, cellpose_ids: list[int]) -> None:
        self._state.toggle_selected_many(cellpose_ids)

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
        if self._brightness_contrast_dialog is not None:
            self._brightness_contrast_dialog.close()
        overlay_widget = getattr(self, "overlay_widget", None)
        if overlay_widget is not None:
            overlay_widget.close()
        video_widget = getattr(self, "calcium_video_widget", None)
        if video_widget is not None:
            video_widget.close()
        return super().close()
