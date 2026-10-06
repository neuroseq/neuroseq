from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.cellpose_annotation import CELLPOSE_MODEL_NAMES
from neuroseq.importer_cellpose import AOIRole
from neuroseq.importer_cellpose import AreasOfInterestConfig
from neuroseq.importer_cellpose import CellPoseImporter
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import CircleAOI
from neuroseq.importer_cellpose import SquareAOI
from neuroseq.importer_cellpose import clone_aoi_config
from neuroseq.importer_cellpose import clone_cellpose_result
from neuroseq.importer_cellpose import default_aoi_config
from neuroseq.importer_cellpose import generate_aoi_stimulus_mapping
from neuroseq.importer_cellpose import generate_linear_stimulus_mapping
from neuroseq.importer_cellpose import point_in_circle
from neuroseq.importer_cellpose import point_in_square
from neuroseq.mapping_validation import ensure_unique_ids
from neuroseq.utils import get_primary_available_geometry
from neuroseq.utils import is_vertical_screen

CUSTOM_AOI_STRATEGY = "Custom with areas of interest"
AOI_PIXELS_MODE = "Coordinates (pixels)"
AOI_PERCENT_MODE = "% of FOV"


class CellposeImportDialog(QtWidgets.QDialog):
    """Dialog for importing Cellpose segmentation data with PNG/TIFF image overlay."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        existing_data: CellposeImportResult | None = None,
        delete_handler: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._imported_data: CellposeImportResult | None = None
        self.image_path: Path | None = None
        self.segmentation_path: Path | None = None
        self._base_segmentation_data: np.ndarray | None = None
        self.importer = CellPoseImporter()
        self.overlay_widget = CellOverlayWidget("")
        self.overlay_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        # Use a defensive copy so the preview can't mutate the stored instance
        self.existing_data = clone_cellpose_result(existing_data)
        self.delete_handler = delete_handler
        self.file_selection_container: QtWidgets.QWidget | None = None
        self.main_content_container: QtWidgets.QWidget | None = None
        self.left_container: QtWidgets.QWidget | None = None
        self.right_container: QtWidgets.QWidget | None = None
        self.left_layout: QtWidgets.QVBoxLayout | None = None
        self.checkbox_container: QtWidgets.QWidget | None = None
        self.mapping_group: QtWidgets.QGroupBox | None = None
        self.main_content_layout: QtWidgets.QBoxLayout | None = None
        self._vertical_layout = False
        self._is_adjusting_position = False
        self._current_screen: QtGui.QScreen | None = None
        self._suppress_aoi_signals = False
        self._aoi_initialized_shape: tuple[int, int] | None = None
        # Cellpose ids in the order they were clicked while manual-click
        # assignment is active - not persisted (see _refresh_manual_mapping_text).
        self._manual_click_order: list[int] = []
        self._setup_ui()
        self._load_existing_data_if_available()

    def _fit_preview_to_screen(self, screen: QtGui.QScreen | None = None) -> None:
        available = self._get_available_geometry(screen)
        if available is None:
            return
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMaximized)
        width = min(1100, int(available.width() * 0.75))
        height = min(760, int(available.height() * 0.75))
        geometry = QtCore.QRect(0, 0, width, height)
        geometry.moveCenter(available.center())
        self.setGeometry(geometry)

    def _get_available_geometry(
        self, screen: QtGui.QScreen | None = None
    ) -> QtCore.QRect | None:
        if screen is not None:
            return screen.availableGeometry()
        return get_primary_available_geometry()

    def _get_screen_for_window(self) -> QtGui.QScreen | None:
        window = self.windowHandle()
        screen = window.screen() if window is not None else None
        if screen is None:
            frame = self.frameGeometry()
            screen = QtGui.QGuiApplication.screenAt(frame.center())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        return screen

    def _should_use_vertical_layout(self, screen: QtGui.QScreen | None = None) -> bool:
        return is_vertical_screen(self._get_available_geometry(screen))

    def _setup_ui(self) -> None:
        self.setWindowTitle("Cell segmentation")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        self.file_selection_container = QtWidgets.QWidget()
        selection_layout = QtWidgets.QVBoxLayout(self.file_selection_container)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        # External GUI controls are added to the bottom of the dialog below.
        self.open_fiji_button = QtWidgets.QPushButton("Open Fiji")
        self.open_fiji_button.setAutoDefault(False)
        self.open_fiji_button.clicked.connect(self._open_in_fiji)

        self.segmentation_run_group = QtWidgets.QGroupBox(
            "Select file (single or multiframe TIFF, or PNG)"
        )
        cellpose_button_layout = QtWidgets.QHBoxLayout(self.segmentation_run_group)
        cellpose_button_layout.addWidget(QtWidgets.QLabel("Model:"))
        self.cellpose_model_combo = QtWidgets.QComboBox()
        self.cellpose_model_combo.addItems(CELLPOSE_MODEL_NAMES)
        self.cellpose_model_combo.setToolTip(
            "Cellpose segmentation model used for one-button segmentation. "
            "cpsam_v2 offers higher sensitivity than cpsam on low "
            "signal-to-noise fields of view, at a similar runtime cost."
        )
        cellpose_button_layout.addWidget(self.cellpose_model_combo)
        cellpose_button_layout.addStretch()
        self.annotate_tiff_button = QtWidgets.QPushButton(
            "Select file for segmentation…"
        )
        self.annotate_tiff_button.setToolTip(
            "Select a mean-fluorescence TIFF or a PNG snapshot and "
            "immediately segment it with the Cellpose model chosen above. "
            "Results are saved beside the selected file."
        )
        self.annotate_tiff_button.clicked.connect(self._annotate_mean_tiff)
        cellpose_button_layout.addWidget(self.annotate_tiff_button)
        selection_layout.addWidget(self.segmentation_run_group)

        self.run_cellpose_button = QtWidgets.QPushButton("Open Cellpose")
        self.run_cellpose_button.setAutoDefault(False)
        self.run_cellpose_button.clicked.connect(self._run_cellpose)

        # Create horizontal layout for file selection and info
        top_horizontal_layout = QtWidgets.QHBoxLayout()

        # File selection group
        self.file_group = QtWidgets.QGroupBox(
            "or load cell segmentation from the files"
        )
        file_layout = QtWidgets.QFormLayout(self.file_group)

        self.image_path_label = QtWidgets.QLabel("No image selected")
        self.image_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_image_button = QtWidgets.QPushButton("Browse…")
        browse_image_button.clicked.connect(self._browse_image_file)
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.addWidget(self.image_path_label, stretch=1)
        image_layout.addWidget(browse_image_button)
        file_layout.addRow("Image (PNG/TIF):", image_layout)

        self.segmentation_path_label = QtWidgets.QLabel("No segmentation file selected")
        self.segmentation_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_seg_button = QtWidgets.QPushButton("Browse…")
        browse_seg_button.clicked.connect(self._browse_segmentation_file)
        seg_layout = QtWidgets.QHBoxLayout()
        seg_layout.addWidget(self.segmentation_path_label, stretch=1)
        seg_layout.addWidget(browse_seg_button)
        file_layout.addRow("Segmentation CSV/NPY:", seg_layout)
        self.file_group.setMinimumHeight(
            file_layout.sizeHint().height() + 50
        )  # Ensure it does nto get clipped vertically

        top_horizontal_layout.addWidget(self.file_group, stretch=2)

        # Data information group
        self.info_group = QtWidgets.QGroupBox("Data Information")
        info_layout = QtWidgets.QVBoxLayout(self.info_group)

        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        self.info_group.hide()

        top_horizontal_layout.addWidget(self.info_group, stretch=1)

        selection_layout.addLayout(top_horizontal_layout)
        layout.addWidget(self.file_selection_container)

        # Main content (hidden until both files are selected)
        self.main_content_container = QtWidgets.QWidget()
        self._vertical_layout = self._should_use_vertical_layout()
        direction = (
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if self._vertical_layout
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        self.main_content_layout = QtWidgets.QBoxLayout(
            direction,
            self.main_content_container,
        )
        self.main_content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_content_layout.setSpacing(16)

        # Add ID mapping options
        self.mapping_group = QtWidgets.QGroupBox("ID Mapping Options")
        mapping_layout = QtWidgets.QFormLayout(self.mapping_group)

        # Strategy selection
        strategy_layout = QtWidgets.QHBoxLayout()
        self.mapping_buttons = QtWidgets.QButtonGroup(self)
        for option in ("Linear", "Custom", CUSTOM_AOI_STRATEGY):
            button = QtWidgets.QRadioButton(option)
            self.mapping_buttons.addButton(button)
            strategy_layout.addWidget(button)
        strategy_layout.addStretch(1)
        strategy_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        for button in self.mapping_buttons.buttons():
            if button.text() == CUSTOM_AOI_STRATEGY:
                button.setChecked(True)
                break
        mapping_layout.addRow("Mapping strategy:", strategy_layout)

        # Linear offset input
        self.linear_offset_spin = QtWidgets.QSpinBox()
        self.linear_offset_spin.setRange(0, 1000000)
        self.linear_offset_spin.setValue(0)
        self.linear_offset_spin.valueChanged.connect(self._on_linear_offset_changed)
        self.linear_offset_label = QtWidgets.QLabel("Linear First-cellpose-ID offset:")
        mapping_layout.addRow(self.linear_offset_label, self.linear_offset_spin)

        # Custom mapping input
        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setMinimumWidth(260)
        self.custom_mapping_line.setMinimumHeight(
            36
        )  # make it larger, ideally we'd make font larger too.
        self.custom_mapping_line.setPlaceholderText(
            "e.g., 1:12,2:3 (inputkey:cellposeId)"
        )
        self.custom_mapping_line.textChanged.connect(
            self._on_custom_mapping_text_changed
        )
        self.custom_mapping_line.setVisible(False)
        self.custom_mapping_row_index = mapping_layout.rowCount()
        self.custom_mapping_label = QtWidgets.QLabel("Custom mapping (input:cellpose):")
        self.custom_mapping_label.setStyleSheet("QLabel { color: #3b7dd8; }")
        self.custom_mapping_label.setVisible(False)
        mapping_layout.addRow(self.custom_mapping_label, self.custom_mapping_line)

        # Manual click assignment - an alternative way to fill in the same
        # custom mapping text field above: click cells in the viewer instead
        # of typing ids by hand. Stimulus ids are assigned in click order;
        # clicking an already-clicked cell again undoes it. Every cell not
        # clicked gets the remaining ids afterward, in ascending cellpose-id
        # order, so every segmented cell always ends up mapped to something.
        manual_click_layout = QtWidgets.QHBoxLayout()
        self.manual_click_checkbox = QtWidgets.QCheckBox("Assign by clicking cells")
        self.manual_click_checkbox.setToolTip(
            "Click cells in the viewer to assign input ids in click order "
            "(click again to undo). Unclicked cells get the remaining ids "
            "automatically, in ascending cellpose-id order."
        )
        self.manual_click_checkbox.toggled.connect(self._on_manual_click_toggled)
        manual_click_layout.addWidget(self.manual_click_checkbox)
        self.manual_click_clear_button = QtWidgets.QPushButton("Clear")
        self.manual_click_clear_button.setAutoDefault(False)
        self.manual_click_clear_button.clicked.connect(self._clear_manual_click_order)
        manual_click_layout.addWidget(self.manual_click_clear_button)
        # Read-only reference while clicking cells in order: lets the user
        # see where the Areas of Interest sit (as configured under the
        # "Custom with areas of interest" strategy) without switching to it.
        self.show_aoi_in_custom_checkbox = QtWidgets.QCheckBox("Show areas of interest")
        self.show_aoi_in_custom_checkbox.setToolTip(
            "Draw the areas of interest on the viewer as a reference while "
            "choosing the cell order. They are not editable here and do not "
            'affect the mapping - set them under "Custom with areas of '
            'interest".'
        )
        self.show_aoi_in_custom_checkbox.toggled.connect(
            self._update_aoi_overlay_visibility
        )
        manual_click_layout.addWidget(self.show_aoi_in_custom_checkbox)
        manual_click_layout.addStretch(1)
        self.manual_click_row_widget = QtWidgets.QWidget()
        self.manual_click_row_widget.setLayout(manual_click_layout)
        self.manual_click_row_widget.setVisible(False)
        mapping_layout.addRow(self.manual_click_row_widget)

        self.aoi_group = QtWidgets.QGroupBox("Areas of Interest")
        aoi_layout = QtWidgets.QVBoxLayout(self.aoi_group)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Position and size:"))
        self.aoi_coordinate_mode_combo = QtWidgets.QComboBox()
        self.aoi_coordinate_mode_combo.addItems((AOI_PIXELS_MODE, AOI_PERCENT_MODE))
        self.aoi_coordinate_mode_combo.setToolTip(
            "Percentage mode uses x/y as the shape centre; sizes are relative "
            "to the shorter FOV dimension."
        )
        mode_layout.addWidget(self.aoi_coordinate_mode_combo)
        mode_layout.addStretch(1)
        aoi_layout.addLayout(mode_layout)
        self._aoi_display_mode = "pixels"

        self.square_enabled_cb = QtWidgets.QCheckBox("Square")
        self.square_enabled_cb.setChecked(True)
        self.square_role_buttons = QtWidgets.QButtonGroup(self)
        square_pref_radio = QtWidgets.QRadioButton("preferential")
        square_excl_radio = QtWidgets.QRadioButton("exclusion")
        self.square_role_buttons.addButton(square_pref_radio)
        self.square_role_buttons.addButton(square_excl_radio)
        square_pref_radio.setChecked(True)
        square_header = QtWidgets.QHBoxLayout()
        square_header.addWidget(self.square_enabled_cb)
        square_header.addWidget(square_pref_radio)
        square_header.addWidget(square_excl_radio)
        square_header.addStretch(1)
        aoi_layout.addLayout(square_header)
        square_values = QtWidgets.QHBoxLayout()
        self.square_x_spin = self._make_aoi_spinbox()
        self.square_y_spin = self._make_aoi_spinbox()
        self.square_side_spin = self._make_aoi_spinbox()
        square_values.addWidget(QtWidgets.QLabel("Center x:"))
        square_values.addWidget(self.square_x_spin)
        square_values.addWidget(QtWidgets.QLabel("Center y:"))
        square_values.addWidget(self.square_y_spin)
        square_values.addWidget(QtWidgets.QLabel("side:"))
        square_values.addWidget(self.square_side_spin)
        aoi_layout.addLayout(square_values)

        self.circle_enabled_cb = QtWidgets.QCheckBox("Circle")
        self.circle_enabled_cb.setChecked(True)
        self.circle_role_buttons = QtWidgets.QButtonGroup(self)
        circle_pref_radio = QtWidgets.QRadioButton("preferential")
        circle_excl_radio = QtWidgets.QRadioButton("exclusion")
        self.circle_role_buttons.addButton(circle_pref_radio)
        self.circle_role_buttons.addButton(circle_excl_radio)
        circle_excl_radio.setChecked(True)
        circle_header = QtWidgets.QHBoxLayout()
        circle_header.addWidget(self.circle_enabled_cb)
        circle_header.addWidget(circle_pref_radio)
        circle_header.addWidget(circle_excl_radio)
        circle_header.addStretch(1)
        aoi_layout.addLayout(circle_header)
        circle_values = QtWidgets.QHBoxLayout()
        self.circle_cx_spin = self._make_aoi_spinbox()
        self.circle_cy_spin = self._make_aoi_spinbox()
        self.circle_diameter_spin = self._make_aoi_spinbox()
        circle_values.addWidget(QtWidgets.QLabel("Center x:"))
        circle_values.addWidget(self.circle_cx_spin)
        circle_values.addWidget(QtWidgets.QLabel("Center y:"))
        circle_values.addWidget(self.circle_cy_spin)
        circle_values.addWidget(QtWidgets.QLabel("diameter:"))
        circle_values.addWidget(self.circle_diameter_spin)
        aoi_layout.addLayout(circle_values)
        self._configure_aoi_spinboxes("pixels")
        self.aoi_group.setVisible(False)
        mapping_layout.addRow(self.aoi_group)

        # Connect radio buttons to show/hide custom mapping input
        for abstract_button in self.mapping_buttons.buttons():
            if hasattr(abstract_button, "toggled"):
                abstract_button.toggled.connect(self._on_mapping_changed)
        self._last_mapping_strategy = CUSTOM_AOI_STRATEGY

        # Place overlay first so the visual is top-most
        self.left_container = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(8)

        self.overlay_widget.setMinimumHeight(400)
        self.left_layout.addWidget(self.overlay_widget, stretch=1)

        self.checkbox_container = QtWidgets.QWidget()
        checkbox_layout = QtWidgets.QHBoxLayout(self.checkbox_container)
        checkbox_layout.setContentsMargins(4, 0, 4, 0)
        self.show_labels_checkbox = QtWidgets.QCheckBox("Show labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.setEnabled(True)
        self.show_labels_checkbox.toggled.connect(self._on_display_toggle_changed)
        checkbox_layout.addWidget(self.show_labels_checkbox)
        self.show_segmentation_checkbox = QtWidgets.QCheckBox("Show segmentation")
        self.show_segmentation_checkbox.setChecked(True)
        self.show_segmentation_checkbox.setEnabled(True)
        self.show_segmentation_checkbox.toggled.connect(self._on_display_toggle_changed)
        checkbox_layout.addWidget(self.show_segmentation_checkbox)
        self.show_shapes_checkbox = QtWidgets.QCheckBox("Show shapes")
        self.show_shapes_checkbox.setChecked(True)
        self.show_shapes_checkbox.setVisible(False)
        self.show_shapes_checkbox.toggled.connect(self._update_aoi_overlay_visibility)
        checkbox_layout.addWidget(self.show_shapes_checkbox)
        checkbox_layout.addStretch(1)
        self.left_layout.addWidget(self.checkbox_container)
        self.overlay_widget.aoi_geometry_changed.connect(
            self._on_overlay_aoi_geometry_changed
        )
        self.overlay_widget.neuron_clicked.connect(self._on_manual_click_cell)
        self._undo_aoi_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self._undo_aoi_shortcut.activated.connect(self._undo_aoi_edit)
        self._redo_aoi_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+Z"), self
        )
        self._redo_aoi_shortcut.activated.connect(self._redo_aoi_edit)

        self.main_content_layout.addWidget(self.left_container, stretch=3)
        self.right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(self.right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.mapping_group)
        guidance_path = (
            Path(__file__).resolve().parents[2]
            / "example_data"
            / "cellpose_guidance.png"
        )
        self.guidance_label = QtWidgets.QLabel()
        self.guidance_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if guidance_path.exists():
            self.guidance_label.setPixmap(QtGui.QPixmap(str(guidance_path)))
        else:
            self.guidance_label.setVisible(False)
        right_layout.addWidget(self.guidance_label)
        self.guidance_label.setVisible(self._get_mapping_strategy() == "Custom")
        self.aoi_summary_browser = QtWidgets.QTextBrowser()
        self.aoi_summary_browser.setReadOnly(True)
        self.aoi_summary_browser.setOpenExternalLinks(False)
        self.aoi_summary_browser.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.aoi_summary_browser.setVisible(False)
        right_layout.addWidget(self.aoi_summary_browser, stretch=1)

        button_layout = QtWidgets.QHBoxLayout()

        self.delete_button = QtWidgets.QPushButton("Delete Cellpose Data")
        self.delete_button.setEnabled(self.existing_data is not None)
        self.delete_button.clicked.connect(self._delete_existing_data)
        button_layout.addWidget(self.delete_button)

        button_layout.addStretch()

        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.setDefault(True)
        self.import_button.clicked.connect(self._import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        right_layout.addLayout(button_layout)
        self.main_content_layout.addWidget(self.right_container, stretch=2)
        layout.addWidget(self.main_content_container)
        self.main_content_container.setVisible(False)

        self.external_gui_group = QtWidgets.QGroupBox("External GUIs")
        external_gui_layout = QtWidgets.QHBoxLayout(self.external_gui_group)
        external_gui_layout.addStretch(1)
        external_gui_layout.addWidget(self.open_fiji_button)
        external_gui_layout.addWidget(self.run_cellpose_button)
        layout.addWidget(self.external_gui_group)

        self._connect_aoi_controls()
        self._set_square_controls_enabled(self.square_enabled_cb.isChecked())
        self._set_circle_controls_enabled(self.circle_enabled_cb.isChecked())
        self._on_mapping_changed()
        self._set_ui_loaded_state(False)
        self._on_display_toggle_changed()

    def _set_ui_loaded_state(self, loaded: bool) -> None:
        """Toggle between pre-selection and post-selection UI, also used to help process in order expanding the preview
        widget of the overlay only when the dialog is in a state we known is maximised (otherwise, the fastplotlib
        full size widget(a custom class) can only be expanded to the dialog boundaries)"""
        if self.file_selection_container is not None:
            self.file_selection_container.setVisible(not loaded)
        if self.main_content_container is not None:
            self.main_content_container.setVisible(loaded)
            if loaded:
                self._apply_screen_layout(self._get_screen_for_window())
                self.logger.info(
                    "cellpose ui loaded: window=%sx%s main=%sx%s",
                    self.width(),
                    self.height(),
                    self.main_content_container.width(),
                    self.main_content_container.height(),
                )
                if self.import_button.isEnabled():
                    self.import_button.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            else:
                self._resize_for_file_selection()
        if not loaded and hasattr(self, "import_button"):
            self.import_button.setEnabled(False)

    def _resize_for_file_selection(self) -> None:
        """Restore the compact, pre-import dialog size."""
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMaximized)
        if self.layout() is not None:
            self.layout().activate()
        self.adjustSize()

    def _make_aoi_spinbox(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1_000_000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        return spin

    def _connect_aoi_controls(self) -> None:
        for spin in (
            self.square_x_spin,
            self.square_y_spin,
            self.square_side_spin,
            self.circle_cx_spin,
            self.circle_cy_spin,
            self.circle_diameter_spin,
        ):
            spin.valueChanged.connect(self._regenerate_aoi_mapping)
        self.aoi_coordinate_mode_combo.currentIndexChanged.connect(
            self._on_aoi_coordinate_mode_changed
        )

        self.square_enabled_cb.toggled.connect(self._set_square_controls_enabled)
        self.square_enabled_cb.toggled.connect(self._regenerate_aoi_mapping)
        self.circle_enabled_cb.toggled.connect(self._set_circle_controls_enabled)
        self.circle_enabled_cb.toggled.connect(self._regenerate_aoi_mapping)

        for button in self.square_role_buttons.buttons():
            button.toggled.connect(self._regenerate_aoi_mapping)
        for button in self.circle_role_buttons.buttons():
            button.toggled.connect(self._regenerate_aoi_mapping)

    def _configure_aoi_spinboxes(self, mode: str) -> None:
        percent = mode == "percent"
        for spin in (
            self.square_x_spin,
            self.square_y_spin,
            self.square_side_spin,
            self.circle_cx_spin,
            self.circle_cy_spin,
            self.circle_diameter_spin,
        ):
            spin.setRange(0.0, 100.0 if percent else 1_000_000.0)
            spin.setSuffix(" %" if percent else "")

    def _set_square_controls_enabled(self, enabled: bool) -> None:
        for button in self.square_role_buttons.buttons():
            button.setEnabled(enabled)
        for spin in (
            self.square_x_spin,
            self.square_y_spin,
            self.square_side_spin,
        ):
            spin.setEnabled(enabled)

    def _set_circle_controls_enabled(self, enabled: bool) -> None:
        for button in self.circle_role_buttons.buttons():
            button.setEnabled(enabled)
        for spin in (
            self.circle_cx_spin,
            self.circle_cy_spin,
            self.circle_diameter_spin,
        ):
            spin.setEnabled(enabled)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if (
            self.main_content_container is not None
            and self.main_content_container.isVisible()
        ):
            self._apply_screen_layout(self._get_screen_for_window())
            self.logger.info(
                "cellpose showEvent: window=%sx%s main=%sx%s",
                self.width(),
                self.height(),
                self.main_content_container.width(),
                self.main_content_container.height(),
            )
        else:
            self._resize_for_file_selection()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._handle_screen_change()

    def _handle_screen_change(self) -> None:
        if self._is_adjusting_position:
            return
        screen = self._get_screen_for_window()
        if screen is None:
            return
        if screen != self._current_screen:
            self._current_screen = screen
            self._apply_screen_layout(screen)
            return
        self._ensure_on_screen(screen)

    def _apply_screen_layout(self, screen: QtGui.QScreen | None) -> None:
        """
        Ensure the dialog layout and size match the target screen orientation
        and resync the overlay preview after moving between monitors.
        OCcurs as user crosses screen boundaries, which keeps to max their pointer y position from the top, so they can snap the window.
        """
        if screen is None:
            return
        self._current_screen = screen
        desired_vertical = self._should_use_vertical_layout(screen)
        if self.main_content_layout is not None and isinstance(
            self.main_content_layout, QtWidgets.QBoxLayout
        ):
            direction = (
                QtWidgets.QBoxLayout.Direction.TopToBottom
                if desired_vertical
                else QtWidgets.QBoxLayout.Direction.LeftToRight
            )
            self.main_content_layout.setDirection(direction)
        self._vertical_layout = desired_vertical
        self._is_adjusting_position = True
        self._fit_preview_to_screen(screen)
        self._is_adjusting_position = False
        self._sync_overlay_size()

    def _ensure_on_screen(self, screen: QtGui.QScreen | None = None) -> None:
        if self._is_adjusting_position:
            return
        if screen is None:
            screen = self._get_screen_for_window()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        max_height = max(0, available.height() - 150)
        if frame.height() > max_height:
            self.resize(frame.width(), max_height)
            frame = self.frameGeometry()
        if available.contains(frame):
            return
        new_x = min(max(frame.x(), available.left()), available.right() - frame.width())
        new_y = min(
            max(frame.y(), available.top()), available.bottom() - frame.height()
        )
        if new_x == frame.x() and new_y == frame.y():
            return
        self._is_adjusting_position = True
        self.move(new_x, new_y)
        self._is_adjusting_position = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_overlay_size()

    def _sync_overlay_size(self) -> None:
        if (
            self.main_content_container is None
            or not self.main_content_container.isVisible()
        ):
            return
        if (
            self.left_container is None
            or self.right_container is None
            or self.checkbox_container is None
            or self.left_layout is None
        ):
            return
        main_rect = self.main_content_container.contentsRect()
        if main_rect.width() < 100 or main_rect.height() < 100:
            self.logger.info(
                "cellpose dialog resize: skip tiny main rect=%sx%s",
                main_rect.width(),
                main_rect.height(),
            )
            return
        if self._vertical_layout:
            self.left_container.setMaximumWidth(main_rect.width())
            self.right_container.setMaximumWidth(main_rect.width())
            left_width = self.left_container.contentsRect().width()
        else:
            left_width = int(main_rect.width() * 0.6)
            right_width = max(1, main_rect.width() - left_width)
            self.left_container.setMaximumWidth(left_width)
            self.right_container.setMaximumWidth(right_width)
        left_rect = self.left_container.contentsRect()
        available_height = (
            left_rect.height()
            - self.checkbox_container.height()
            - self.left_layout.spacing()
        )
        square_size = min(left_width, available_height)
        if square_size > 0:
            self.overlay_widget.setFixedSize(square_size, square_size)
        else:
            self.logger.info(
                "cellpose dialog resize: skip square=%s left_rect=%sx%s checkbox=%s",
                square_size,
                left_rect.width(),
                left_rect.height(),
                self.checkbox_container.height(),
            )
            return
        self.logger.info(
            "cellpose dialog resize: main=%sx%s left=%sx%s right=%sx%s square=%s",
            main_rect.width(),
            main_rect.height(),
            left_rect.width(),
            left_rect.height(),
            self.right_container.contentsRect().width(),
            self.right_container.contentsRect().height(),
            square_size,
        )

    def _reset_importer_state(self, clear_existing_data: bool = False) -> None:
        """Return dialog to initial state after an importer failure."""
        self._imported_data = None
        self._manual_click_order = []
        self.manual_click_checkbox.setChecked(False)
        if clear_existing_data:
            self.existing_data = None
            if hasattr(self, "delete_button"):
                self.delete_button.setEnabled(False)

        self.importer = CellPoseImporter()
        self.image_path = None
        self.segmentation_path = None
        self._base_segmentation_data = None
        self._aoi_initialized_shape = None

        self.image_path_label.setText("No image selected")
        self.image_path_label.setStyleSheet("QLabel { color: gray; }")
        self.segmentation_path_label.setText("No segmentation file selected")
        self.segmentation_path_label.setStyleSheet("QLabel { color: gray; }")
        # self.custom_mapping_line.clear()
        self.linear_offset_spin.setValue(0)
        self.aoi_summary_browser.clear()
        self.aoi_summary_browser.setVisible(False)
        for button in self.mapping_buttons.buttons():
            if button.text() == CUSTOM_AOI_STRATEGY:
                button.setChecked(True)
                break
        self.overlay_widget.set_aoi_overlay(None, show=False)

        self._on_display_toggle_changed()

        self.overlay_widget.set_overlay(None, None, None, None, None, None)
        self._hide_info()
        self._set_ui_loaded_state(False)

    def _apply_existing_mapping(
        self, mapping: dict[int, int], mapping_strategy: str, linear_offset: int
    ) -> None:
        target_mode = mapping_strategy or "Linear"
        if target_mode in ("Custom", CUSTOM_AOI_STRATEGY):
            self.custom_mapping_line.setText(self._mapping_to_text(mapping))
        if target_mode == "Linear":
            self.linear_offset_spin.setValue(int(linear_offset or 0))
        for button in self.mapping_buttons.buttons():
            if button.text() == target_mode:
                button.setChecked(True)
                break

    def _on_mapping_changed(self) -> None:
        """Handle toggle between mapping strategies"""
        mapping_strategy = self._get_mapping_strategy()
        is_aoi = mapping_strategy == CUSTOM_AOI_STRATEGY
        is_any_custom = mapping_strategy in ("Custom", CUSTOM_AOI_STRATEGY)
        self.manual_click_row_widget.setVisible(mapping_strategy == "Custom")
        if mapping_strategy != "Custom" and self.manual_click_checkbox.isChecked():
            self.manual_click_checkbox.setChecked(False)
        self.custom_mapping_line.setVisible(is_any_custom)
        self.custom_mapping_label.setVisible(is_any_custom)
        self.custom_mapping_line.setReadOnly(
            is_aoi or self.manual_click_checkbox.isChecked()
        )
        self.linear_offset_spin.setVisible(not is_any_custom)
        self.linear_offset_label.setVisible(not is_any_custom)
        self.aoi_group.setVisible(is_aoi)
        self.show_shapes_checkbox.setVisible(is_aoi)
        if hasattr(self, "guidance_label"):
            self.guidance_label.setVisible(is_any_custom)
        if not is_aoi:
            self.aoi_summary_browser.setVisible(False)
        if is_aoi:
            self._regenerate_aoi_mapping()
        else:
            self._update_aoi_overlay_visibility()

        # Update preview if data is loaded
        if (
            self.importer.segmentation_data is not None
            and self.importer.overlay_image is not None
        ):
            self._update_preview_mapping()
        self._last_mapping_strategy = mapping_strategy

    def _on_manual_click_toggled(self, checked: bool) -> None:
        self.overlay_widget.set_cell_selection_enabled(checked)
        self.custom_mapping_line.setReadOnly(
            checked or self._get_mapping_strategy() == CUSTOM_AOI_STRATEGY
        )
        if checked:
            self._refresh_manual_mapping_text()

    def _on_manual_click_cell(self, cellpose_id: int) -> None:
        if (
            not self.manual_click_checkbox.isChecked()
            or self._get_mapping_strategy() != "Custom"
        ):
            return
        if cellpose_id in self._manual_click_order:
            self._manual_click_order.remove(cellpose_id)
        else:
            self._manual_click_order.append(cellpose_id)
        self._refresh_manual_mapping_text()

    def _clear_manual_click_order(self) -> None:
        self._manual_click_order = []
        if self.manual_click_checkbox.isChecked():
            self._refresh_manual_mapping_text()

    def _refresh_manual_mapping_text(self) -> None:
        """Fill in the custom mapping text field from the current click
        order, plus every unclicked cell in ascending cellpose-id order -
        every segmented cell ends up mapped to something, unlike a
        hand-typed Custom mapping which may deliberately leave cells out.

        Written directly rather than through _mapping_to_text(), which
        drops stim_id == cellpose_id pairs for a shorter display - those
        pairs are just as real here (a cell coincidentally keeping its own
        id) and dropping them would silently leave that cell unmapped.
        """
        original_ids = self.importer._original_cellpose_ids
        if original_ids is None:
            return
        clicked = [int(cellpose_id) for cellpose_id in self._manual_click_order]
        remaining = sorted(
            int(cellpose_id)
            for cellpose_id in original_ids
            if int(cellpose_id) not in clicked
        )
        ordered_cellpose_ids = clicked + remaining
        text = ",".join(
            f"{stim_id}:{cellpose_id}"
            for stim_id, cellpose_id in enumerate(ordered_cellpose_ids)
        )
        self.custom_mapping_line.setText(text)

    def _on_custom_mapping_text_changed(self, _text: str) -> None:
        """Refresh preview live while editing custom mapping."""
        if self._get_mapping_strategy() != "Custom":
            return
        if self.importer.segmentation_data is None:
            return
        self._update_preview_mapping()

    def _on_linear_offset_changed(self, _value: int) -> None:
        """Refresh preview when linear offset changes."""
        if self._get_mapping_strategy() != "Linear":
            return
        if self.importer.segmentation_data is None:
            return
        self._update_preview_mapping()

    def _on_display_toggle_changed(self) -> None:
        """Sync overlay display options with the checkboxes."""
        show_labels = (
            self.show_labels_checkbox.isChecked()
            if hasattr(self, "show_labels_checkbox")
            else True
        )
        show_segmentation = (
            self.show_segmentation_checkbox.isChecked()
            if hasattr(self, "show_segmentation_checkbox")
            else True
        )
        self.overlay_widget.set_display_options(
            show_cellpose_ids=show_labels,
            show_stimulus_ids=show_labels,
            show_segmentation=show_segmentation,
        )
        self._update_aoi_overlay_visibility()

    def _get_mapping_strategy(self) -> str:
        """Get the currently selected mapping strategy"""
        checked = self.mapping_buttons.checkedButton()
        return checked.text() if checked is not None else "Linear"

    def _get_linear_offset(self) -> int:
        """Return the selected linear offset (defaults to 0)."""
        return (
            int(self.linear_offset_spin.value())
            if hasattr(self, "linear_offset_spin")
            else 0
        )

    def _role_from_button_group(self, group: QtWidgets.QButtonGroup) -> AOIRole:
        checked = group.checkedButton()
        if checked is not None and checked.text() == AOIRole.EXCLUSION.value:
            return AOIRole.EXCLUSION
        return AOIRole.PREFERENTIAL

    def _set_role_button_group(
        self, group: QtWidgets.QButtonGroup, role: AOIRole
    ) -> None:
        for button in group.buttons():
            if button.text() == role.value:
                button.setChecked(True)
                return

    def _current_aoi_config(self) -> AreasOfInterestConfig | None:
        if self.importer.image_data is None:
            return None
        return AreasOfInterestConfig(
            square=self._current_square_aoi(),
            circle=self._current_circle_aoi(),
            show_shapes=self.show_shapes_checkbox.isChecked(),
            coordinate_mode=self._aoi_display_mode,
        )

    def _aoi_dimensions(self) -> tuple[float, float, float]:
        if self.importer.image_data is None:
            return 1.0, 1.0, 1.0
        height, width = self.importer.image_data.shape[:2]
        return float(width), float(height), float(min(width, height))

    def _current_square_aoi(self):
        x = float(self.square_x_spin.value())
        y = float(self.square_y_spin.value())
        side = float(self.square_side_spin.value())
        if self._aoi_display_mode == "percent":
            width, height, short = self._aoi_dimensions()
            side = side * short / 100.0
            x = x * width / 100.0
            y = y * height / 100.0
        x -= side / 2.0
        y -= side / 2.0
        return SquareAOI(
            enabled=self.square_enabled_cb.isChecked(),
            role=self._role_from_button_group(self.square_role_buttons),
            x=x,
            y=y,
            side=side,
        )

    def _current_circle_aoi(self):
        cx = float(self.circle_cx_spin.value())
        cy = float(self.circle_cy_spin.value())
        diameter = float(self.circle_diameter_spin.value())
        if self._aoi_display_mode == "percent":
            width, height, short = self._aoi_dimensions()
            cx = cx * width / 100.0
            cy = cy * height / 100.0
            diameter = diameter * short / 100.0
        return CircleAOI(
            enabled=self.circle_enabled_cb.isChecked(),
            role=self._role_from_button_group(self.circle_role_buttons),
            cx=cx,
            cy=cy,
            diameter=diameter,
        )

    def _on_aoi_coordinate_mode_changed(self, index: int) -> None:
        new_mode = "percent" if index == 1 else "pixels"
        if new_mode == self._aoi_display_mode:
            return
        config = self._current_aoi_config()
        self._aoi_display_mode = new_mode
        self._configure_aoi_spinboxes(new_mode)
        if config is not None:
            config.coordinate_mode = new_mode
            self._set_aoi_controls(config)
        self._regenerate_aoi_mapping()

    def _set_aoi_controls(self, config: AreasOfInterestConfig) -> None:
        self._suppress_aoi_signals = True
        try:
            mode = getattr(config, "coordinate_mode", "pixels")
            self._aoi_display_mode = mode
            self._configure_aoi_spinboxes(mode)
            self.aoi_coordinate_mode_combo.setCurrentIndex(
                1 if mode == "percent" else 0
            )
            width, height, short = self._aoi_dimensions()
            self.square_enabled_cb.setChecked(config.square.enabled)
            self._set_role_button_group(self.square_role_buttons, config.square.role)
            if mode == "percent":
                self.square_x_spin.setValue(
                    (config.square.x + config.square.side / 2.0) * 100.0 / width
                )
                self.square_y_spin.setValue(
                    (config.square.y + config.square.side / 2.0) * 100.0 / height
                )
                self.square_side_spin.setValue(config.square.side * 100.0 / short)
            else:
                self.square_x_spin.setValue(
                    float(config.square.x + config.square.side / 2.0)
                )
                self.square_y_spin.setValue(
                    float(config.square.y + config.square.side / 2.0)
                )
                self.square_side_spin.setValue(float(config.square.side))

            self.circle_enabled_cb.setChecked(config.circle.enabled)
            self._set_role_button_group(self.circle_role_buttons, config.circle.role)
            if mode == "percent":
                self.circle_cx_spin.setValue(config.circle.cx * 100.0 / width)
                self.circle_cy_spin.setValue(config.circle.cy * 100.0 / height)
                self.circle_diameter_spin.setValue(
                    config.circle.diameter * 100.0 / short
                )
            else:
                self.circle_cx_spin.setValue(float(config.circle.cx))
                self.circle_cy_spin.setValue(float(config.circle.cy))
                self.circle_diameter_spin.setValue(float(config.circle.diameter))

            self.show_shapes_checkbox.setChecked(bool(config.show_shapes))
            self._set_square_controls_enabled(config.square.enabled)
            self._set_circle_controls_enabled(config.circle.enabled)
        finally:
            self._suppress_aoi_signals = False

    def _ensure_aoi_defaults(self, force: bool = False) -> None:
        if self.importer.image_data is None:
            return
        shape = tuple(self.importer.image_data.shape[:2])
        if not force and self._aoi_initialized_shape == shape:
            return
        self._set_aoi_controls(default_aoi_config(shape))
        self._aoi_initialized_shape = shape

    def _compute_aoi_mapping(self) -> dict[int, int]:
        config = self._current_aoi_config()
        if config is None or self.importer.image_data is None:
            return {}
        return generate_aoi_stimulus_mapping(
            self.importer._original_cellpose_ids,
            self.overlay_widget.overlay_view._label_centers,
            config,
            self.importer.image_data.shape[0],
        )

    def _mapping_to_text(self, mapping: dict[int, int]) -> str:
        filtered_pairs = [
            (stim_id, cellpose_id)
            for stim_id, cellpose_id in mapping.items()
            if stim_id != cellpose_id
        ]
        filtered_pairs.sort(key=lambda pair: pair[0])
        return ",".join(
            f"{stim_id}:{cellpose_id}" for stim_id, cellpose_id in filtered_pairs
        )

    def _parse_custom_mapping(self, mapping_str: str) -> dict[int, int]:
        """Parse custom mapping string like '1:12,2:3' into {1: 12, 2: 3}"""
        mapping: dict[int, int] = {}
        stim_keys: list[int] = []
        if not mapping_str.strip():
            return mapping

        try:
            pairs = mapping_str.split(",")
            for pair in pairs:
                pair = pair.strip()
                if not pair or ":" not in pair:
                    continue
                stim_key_str, cellpose_id_str = pair.split(":", 1)
                stim_key_str = stim_key_str.strip()
                cellpose_id_str = cellpose_id_str.strip()
                if not stim_key_str or not cellpose_id_str:
                    continue
                stim_key = int(stim_key_str)
                cellpose_id = int(cellpose_id_str)
                stim_keys.append(stim_key)
                mapping[stim_key] = cellpose_id
            ensure_unique_ids(
                stim_keys,
                "Custom mapping would assign input ID {value} more than once.",
            )
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid custom mapping format: {exc}") from exc

        return mapping

    def _role_html(self, role: AOIRole) -> str:
        color = "#b42318" if role == AOIRole.EXCLUSION else "#1a7f37"
        return f"<span style='color:{color}; font-weight:600'>{role.value}</span>"

    def _format_aoi_indices(
        self,
        *,
        shape_name: str,
        enabled: bool,
        role: AOIRole,
        members: list[tuple[int, int]],
    ) -> str:
        status = "" if enabled else " <span style='color:#666'>(disabled)</span>"
        if members:
            stim_ids = ", ".join(str(stim_id) for stim_id, _ in members)
            cellpose_ids = ", ".join(str(cellpose_id) for _, cellpose_id in members)
        else:
            stim_ids = "none"
            cellpose_ids = "none"
        return (
            f"<p><b>{shape_name}</b> {self._role_html(role)}{status}<br>"
            f"Input IDs: {stim_ids}<br>"
            f"Cellpose IDs: {cellpose_ids}</p>"
        )

    def _update_aoi_summary(self, mapping: dict[int, int] | None = None) -> None:
        if self._get_mapping_strategy() != CUSTOM_AOI_STRATEGY:
            self.aoi_summary_browser.clear()
            self.aoi_summary_browser.setVisible(False)
            return

        config = self._current_aoi_config()
        if config is None or self.importer.image_data is None:
            self.aoi_summary_browser.clear()
            self.aoi_summary_browser.setVisible(False)
            return

        if mapping is None:
            mapping = self._compute_aoi_mapping()

        image_height = self.importer.image_data.shape[0]
        reverse_mapping = {
            cellpose_id: stim_id for stim_id, cellpose_id in mapping.items()
        }
        square_members: list[tuple[int, int]] = []
        circle_members: list[tuple[int, int]] = []
        for cellpose_id, (
            world_x,
            world_y,
        ) in self.overlay_widget.overlay_view._label_centers.items():
            user_x = float(world_x)
            user_y = float(image_height - world_y)
            stim_id = reverse_mapping.get(int(cellpose_id))
            if stim_id is None:
                continue
            if config.square.enabled and point_in_square(user_x, user_y, config.square):
                square_members.append((int(stim_id), int(cellpose_id)))
            if config.circle.enabled and point_in_circle(user_x, user_y, config.circle):
                circle_members.append((int(stim_id), int(cellpose_id)))

        square_members.sort()
        circle_members.sort()
        html = (
            "<div>"
            "<p><b>Areas of interest summary</b></p>"
            f"{self._format_aoi_indices(shape_name='Square', enabled=config.square.enabled, role=config.square.role, members=square_members)}"
            f"{self._format_aoi_indices(shape_name='Circle', enabled=config.circle.enabled, role=config.circle.role, members=circle_members)}"
            "<p style='color:#666'>If shapes overlap, exclusion cells still receive the highest input IDs.</p>"
            "</div>"
        )
        self.aoi_summary_browser.setHtml(html)
        self.aoi_summary_browser.setVisible(True)

    def _regenerate_aoi_mapping(self, *_args) -> None:
        if self._suppress_aoi_signals:
            return
        if self._get_mapping_strategy() != CUSTOM_AOI_STRATEGY:
            return

        config = self._current_aoi_config()
        if config is None:
            return

        mapping = self._compute_aoi_mapping()
        self.custom_mapping_line.blockSignals(True)
        try:
            self.custom_mapping_line.setText(self._mapping_to_text(mapping))
        finally:
            self.custom_mapping_line.blockSignals(False)
        self.overlay_widget.set_aoi_overlay(
            config,
            show=self.show_shapes_checkbox.isChecked(),
        )
        self._update_aoi_summary(mapping)
        self._update_preview_mapping()

    def _update_aoi_overlay_visibility(self, *_args) -> None:
        strategy = self._get_mapping_strategy()
        is_aoi = strategy == CUSTOM_AOI_STRATEGY
        # Under plain Custom the shapes are only a reference for the click
        # order: drawn on request, and never draggable (a drag there would
        # change nothing, since the mapping doesn't use them).
        is_reference = (
            strategy == "Custom" and self.show_aoi_in_custom_checkbox.isChecked()
        )
        if is_reference:
            self._ensure_aoi_defaults()
        self.overlay_widget.set_aoi_editable(is_aoi)
        self.overlay_widget.set_aoi_overlay(
            self._current_aoi_config() if (is_aoi or is_reference) else None,
            show=(is_aoi and self.show_shapes_checkbox.isChecked()) or is_reference,
        )

    def _on_overlay_aoi_geometry_changed(self, config: AreasOfInterestConfig) -> None:
        if self._get_mapping_strategy() != CUSTOM_AOI_STRATEGY:
            return
        self._set_aoi_controls(config)
        self._regenerate_aoi_mapping()

    def _undo_aoi_edit(self) -> None:
        """Revert the last shape move/resize done by dragging (issue #531)."""
        self.overlay_widget.overlay_view.undo_aoi_edit()

    def _redo_aoi_edit(self) -> None:
        self.overlay_widget.overlay_view.redo_aoi_edit()

    def _generate_stimulus_mapping(self, strategy: str) -> dict[int, int]:
        """Generate stimulus-to-cellpose mapping based on strategy"""
        if self.importer._original_cellpose_ids is None:
            return {}

        original_ids = self.importer._original_cellpose_ids

        if strategy == "Linear":
            offset = self._get_linear_offset()
            return generate_linear_stimulus_mapping(original_ids, offset)
        elif strategy == "Custom":
            # Parse custom mapping string
            return self._parse_custom_mapping(self.custom_mapping_line.text())
        elif strategy == CUSTOM_AOI_STRATEGY:
            return self._compute_aoi_mapping()
        else:
            return {}

    def _update_preview_mapping(self) -> None:
        """Update the preview with current mapping strategy"""
        try:
            mapping_strategy = self._get_mapping_strategy()
            preview_mapping = self._generate_stimulus_mapping(mapping_strategy)

            self.overlay_widget.set_overlay(
                self.importer.overlay_image,
                self.importer.segmentation_data,
                self.importer._original_cellpose_ids,
                preview_mapping,
                highlight_mappings=(
                    mapping_strategy in ("Custom", CUSTOM_AOI_STRATEGY)
                ),
                base_image=self.importer.image_data,
            )
            self._update_aoi_overlay_visibility()
        except Exception as exc:
            self.logger.warning("Error updating preview mapping: %s", exc)

    def _apply_segmentation_transformations(self) -> None:
        if self._base_segmentation_data is None:
            return
        seg = np.array(self._base_segmentation_data, copy=True)
        self.importer.segmentation_data = seg
        self.importer.generate_overlay()

    def _browse_image_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select image file",
            "",
            "Image files (*.png *.tif *.tiff);;PNG files (*.png);;TIFF files (*.tif *.tiff);;All files (*)",
        )
        if filename:
            self._load_image_file(filename)

    def _browse_segmentation_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select segmentation file",
            "",
            "Segmentation files (*.csv *.npy);;CSV files (*.csv);;NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self._load_segmentation_file(filename)

    def _annotate_mean_tiff(self) -> None:
        """Select and immediately annotate a mean-fluorescence TIFF or a
        single PNG snapshot with the Cellpose model chosen in the one-button
        segmentation pane."""
        filename, _ = getopenfilename(
            self,
            "Select image for segmentation",
            "",
            "Image files (*.tif *.tiff *.png);;TIFF files (*.tif *.tiff);;"
            "PNG files (*.png);;All files (*)",
        )
        if not filename:
            return
        if importlib.util.find_spec("cellpose") is None:
            QtWidgets.QMessageBox.information(
                self,
                "Install Cellpose",
                "Cellpose is not installed in the current Python environment.",
            )
            return

        from neuroseq.cellpose_annotation import TIFF_SUFFIXES

        frame_count = 1
        if Path(filename).suffix.lower() in TIFF_SUFFIXES:
            try:
                from neuroseq.cellpose_annotation import tiff_frame_count

                frame_count = tiff_frame_count(filename)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid TIFF file",
                    f"Could not inspect {filename}:\n{exc}",
                )
                return
        if frame_count > 1:
            sampled_count = len(range(0, min(frame_count, 300), 2))
            QtWidgets.QMessageBox.information(
                self,
                "Multi-frame TIFF selected",
                f"This TIFF contains {frame_count} frames. NeuroSeq will use "
                f"every second frame from the first {min(frame_count, 300)} "
                f"frames ({sampled_count} frames total), calculate their "
                "average, and process that average with Cellpose.",
            )

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.annotate_tiff_button.setEnabled(False)
        try:
            from neuroseq.cellpose_annotation import annotate_mean_fluorescence_tiff

            segmentation_path, mask_path = annotate_mean_fluorescence_tiff(
                filename, model_name=self.cellpose_model_combo.currentText()
            )
            self._load_image_file(filename)
            self._load_segmentation_file(str(segmentation_path))
            QtWidgets.QMessageBox.information(
                self,
                "Cell annotation complete",
                "Cellpose annotation was saved as:\n"
                f"{segmentation_path.name}\n{mask_path.name}",
            )
        except Exception as exc:
            self.logger.exception("Cellpose annotation failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Cellpose annotation failed",
                f"Could not annotate {filename}:\n{exc}",
            )
        finally:
            self.annotate_tiff_button.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()

    def _open_in_fiji(self) -> None:
        """Launch Fiji/ImageJ, optionally opening the currently selected image."""
        from neuroseq.fiji import resolve_fiji_path

        fiji_path = resolve_fiji_path(self)
        if not fiji_path:
            return

        args: list[str] = []
        if self.image_path is not None and self.image_path.exists():
            args.append(str(self.image_path))

        try:
            started = QtCore.QProcess.startDetached(fiji_path, args)
            if not started:
                raise RuntimeError(
                    f"QProcess.startDetached({fiji_path!r}) returned False"
                )
        except Exception as exc:
            self.logger.exception("Failed to launch Fiji")
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to launch Fiji",
                f"Could not launch Fiji:\n{exc}",
            )

    def _run_cellpose(self) -> None:
        """Launch the cellpose application from the current Python environment."""
        if importlib.util.find_spec("cellpose") is None:
            QtWidgets.QMessageBox.information(
                self,
                "Install Cellpose",
                "Cellpose is not installed in the current Python environment, please install it.\n\n",
            )
            return
        try:
            # Use subprocess to run cellpose as a GUI application in the background
            # The -m cellpose flag runs cellpose as a module, which includes the GUI
            subprocess.Popen(
                [sys.executable, "-m", "cellpose"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # If PyTorch has no GPU backend, Cellpose's "use GPU" box is disabled;
            # tell the user why and how to fix it (once per session).
            from neuroseq.gpu import warn_if_gpu_unavailable

            warn_if_gpu_unavailable(self)
            # todo: Ask Andrey if we can remove this commented code.
            # QtWidgets.QMessageBox.information(
            #    self,
            #    "Cellpose Launched",
            #    "Cellpose GUI has been launched in the current environment.",
            # )
        except Exception as exc:
            self.logger.exception("Failed to launch cellpose")
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to launch Cellpose",
                f"Could not launch Cellpose:\n{exc}\n\n"
                "Make sure cellpose is installed in the current Python environment.",
            )

    def _load_image_file(self, filename: str) -> None:
        try:
            self.image_path = Path(filename)
            self.image_path_label.setText(self.image_path.name)
            self.image_path_label.setStyleSheet("QLabel { color: black; }")

            self.importer.load_image_file(filename)
            self._check_files_loaded()
        except Exception as exc:
            self.logger.exception("Error loading image file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading Image", f"Could not load {filename}:\n{exc}"
            )
            self._reset_importer_state()

    def _load_segmentation_file(self, filename: str) -> None:
        try:
            self.segmentation_path = Path(filename)
            self.segmentation_path_label.setText(self.segmentation_path.name)
            self.segmentation_path_label.setStyleSheet("QLabel { color: black; }")
            # A new segmentation has a different set of cellpose ids - any
            # click order recorded against the old one is meaningless now.
            self._manual_click_order = []
            self.manual_click_checkbox.setChecked(False)

            self.importer.load_segmentation_file(filename)
            self._base_segmentation_data = np.array(
                self.importer.segmentation_data, copy=True
            )
            self._check_files_loaded()
        except Exception as exc:
            self.logger.exception("Error loading segmentation file")
            QtWidgets.QMessageBox.critical(
                self, "Error Loading Segmentation", f"Could not load {filename}:\n{exc}"
            )
            self._reset_importer_state()

    def _check_files_loaded(self) -> None:
        if self.importer.image_data is None or self.importer.segmentation_data is None:
            self.import_button.setEnabled(False)
            self._set_ui_loaded_state(False)
            return

        try:
            self.importer.validate_dimensions()
            if self._base_segmentation_data is None:
                self._base_segmentation_data = np.array(
                    self.importer.segmentation_data, copy=True
                )
            self._apply_segmentation_transformations()
            self._ensure_aoi_defaults()
            self.import_button.setEnabled(True)
            self._set_ui_loaded_state(True)
            # Generate preview mapping to show in overlay
            mapping_strategy = self._get_mapping_strategy()
            preview_mapping = (
                self._generate_stimulus_mapping(mapping_strategy)
                if mapping_strategy
                else {}
            )

            self.logger.info(
                "cellpose preview: image=%s seg=%s overlay=%sx%s mapping=%s",
                None
                if self.importer.image_data is None
                else self.importer.image_data.shape,
                None
                if self.importer.segmentation_data is None
                else self.importer.segmentation_data.shape,
                self.overlay_widget.width(),
                self.overlay_widget.height(),
                mapping_strategy,
            )
            self.overlay_widget.set_overlay(
                self.importer.overlay_image,
                self.importer.segmentation_data,
                self.importer._original_cellpose_ids,
                preview_mapping,
                highlight_mappings=(
                    mapping_strategy in ("Custom", CUSTOM_AOI_STRATEGY)
                ),
                base_image=self.importer.image_data,
            )
            if mapping_strategy == CUSTOM_AOI_STRATEGY:
                self._regenerate_aoi_mapping()
            else:
                self._update_aoi_overlay_visibility()
            self._hide_info()
        except Exception as exc:
            self.logger.exception("Error preparing cellpose preview")
            title = (
                "Dimension Mismatch" if isinstance(exc, ValueError) else "Import Error"
            )
            QtWidgets.QMessageBox.critical(self, title, str(exc))
            self._reset_importer_state()

    def _update_info(self, error: str | None = None) -> None:
        info = self.importer.get_info()
        if not info and not error:
            self._hide_info()
            return

        info_lines = []
        if error:
            info_lines.append(f"Error: {error}")
        if info:
            info_lines.extend(
                [
                    f"Image shape: {info['image_shape']}",
                    f"Segmentation shape: {info['segmentation_shape']}",
                    f"Unique segment IDs: {info['unique_ids']}",
                    f"Background pixels (ID=0): {info['background_pixels']}",
                    f"Detected neurons: {info['n_neurons']}",
                ]
            )
            if "neuron_id_range" in info:
                min_id, max_id = info["neuron_id_range"]
                info_lines.append(f"Neuron ID range: {min_id}-{max_id}")

        self.info_text.setPlainText("\n".join(info_lines))
        self.info_group.show()

    def _hide_info(self) -> None:
        self.info_text.clear()
        self.info_group.hide()

    def _import_data(self) -> None:
        if (
            self.importer.segmentation_data is None
            or self.importer.overlay_image is None
            or self.image_path is None
        ):
            return

        try:
            # Generate stimulus-to-cellpose mapping based on selected strategy
            mapping_strategy = self._get_mapping_strategy()
            stimulus_mapping = self._generate_stimulus_mapping(mapping_strategy)
            if mapping_strategy == CUSTOM_AOI_STRATEGY:
                self._update_aoi_summary(stimulus_mapping)
            if mapping_strategy == "Custom":
                original_ids = (
                    []
                    if self.importer._original_cellpose_ids is None
                    else [
                        int(cellpose_id)
                        for cellpose_id in self.importer._original_cellpose_ids
                    ]
                )
                mapped_cellpose_ids = set(stimulus_mapping.values())
                mapped_stim_ids = set(stimulus_mapping.keys())
                for cellpose_id in original_ids:
                    if (
                        cellpose_id in mapped_cellpose_ids
                        or cellpose_id in mapped_stim_ids
                    ):
                        continue
                    stimulus_mapping[cellpose_id] = cellpose_id
            stored_mapping = (
                stimulus_mapping
                if mapping_strategy in ("Custom", CUSTOM_AOI_STRATEGY)
                else None
            )
            stored_aoi_config = (
                clone_aoi_config(self._current_aoi_config())
                if mapping_strategy == CUSTOM_AOI_STRATEGY
                else None
            )

            self._imported_data = self.importer.create_result(
                self.image_path,
                stored_mapping,
                mapping_strategy=mapping_strategy,
                linear_offset=self._get_linear_offset(),
                aoi_config=stored_aoi_config,
                segmentation_path=self.segmentation_path,
            )

            info = self.importer.get_info()
            unique_ids = [uid for uid in info["unique_ids"] if uid > 0]

            # Create success message with mapping info
            success_msg = (
                f"Successfully imported Cellpose data.\n"
                f"Image: {info['image_shape']}\n"
                f"Detected {len(unique_ids)} neurons\n"
                f"Mapping strategy: {mapping_strategy}"
            )
            linear_offset = self._get_linear_offset()
            if mapping_strategy == "Linear" and linear_offset:
                success_msg += f" (offset {linear_offset})"

            if mapping_strategy in ("Custom", CUSTOM_AOI_STRATEGY) and stimulus_mapping:
                success_msg += f"\nInput mapping: {dict(list(stimulus_mapping.items())[:5])}{'...' if len(stimulus_mapping) > 5 else ''}"

            QtWidgets.QMessageBox.information(self, "Import Successful", success_msg)
            self.accept()

        except Exception as exc:
            self.logger.exception("Error importing Cellpose data")
            QtWidgets.QMessageBox.critical(
                self, "Import Error", f"Error importing Cellpose data:\n{exc}"
            )
            self._reset_importer_state()

    def get_imported_data(self) -> CellposeImportResult | None:
        """Return the last successfully imported data payload."""
        return clone_cellpose_result(self._imported_data)

    def _load_existing_data_if_available(self) -> None:
        """Load existing cellpose data if available"""
        if self.existing_data is None:
            return

        try:
            # Set up file paths
            self.image_path = self.existing_data.image_path
            self.image_path_label.setText(self.image_path.name)
            self.image_path_label.setStyleSheet("QLabel { color: blue; }")
            self.segmentation_path = getattr(
                self.existing_data, "segmentation_path", None
            )
            if self.segmentation_path is not None:
                self.segmentation_path_label.setText(self.segmentation_path.name)
                self.segmentation_path_label.setStyleSheet("QLabel { color: blue; }")

            # Load the existing image and segmentation data
            self.importer.restore_from_result(self.existing_data)
            self._base_segmentation_data = (
                None
                if self.importer.segmentation_data is None
                else np.array(self.importer.segmentation_data, copy=True)
            )
            self._apply_segmentation_transformations()
            self._ensure_aoi_defaults(force=True)

            mapping = self.existing_data.stimulus_to_cellpose_mapping or {}
            mapping_strategy = getattr(self.existing_data, "mapping_strategy", "")
            if not mapping_strategy:
                mapping_strategy = "Custom" if mapping else "Linear"
            linear_offset = getattr(self.existing_data, "linear_offset", 0)
            aoi_config = clone_aoi_config(
                getattr(self.existing_data, "aoi_config", None)
            )
            if mapping_strategy == CUSTOM_AOI_STRATEGY:
                if aoi_config is None and self.importer.image_data is not None:
                    aoi_config = default_aoi_config(self.importer.image_data.shape[:2])
                if aoi_config is not None:
                    self._set_aoi_controls(aoi_config)
            self._apply_existing_mapping(mapping, mapping_strategy, linear_offset)
            self.logger.info(
                "cellpose restore: strategy=%s mapping=%s linear_offset=%s",
                mapping_strategy,
                "custom" if mapping else "none",
                linear_offset,
            )

            # Update UI
            self.segmentation_path_label.setText("(from existing data)")
            self.segmentation_path_label.setStyleSheet("QLabel { color: blue; }")
            if hasattr(self, "delete_button"):
                self.delete_button.setEnabled(True)
            self.import_button.setEnabled(True)

            self._set_ui_loaded_state(True)
            # Update preview
            self._update_preview_mapping()
            self._update_info()

        except Exception as exc:
            self.logger.warning("Failed to load existing cellpose data: %s", exc)
            self._reset_importer_state(clear_existing_data=True)

    def _delete_existing_data(self) -> None:
        """Delete stored cellpose data after confirmation."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Cellpose Data",
            "Are you sure you want to delete all cellpose data? This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if self.delete_handler is not None:
                self.delete_handler()
            self._reset_importer_state(clear_existing_data=True)
