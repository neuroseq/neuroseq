from __future__ import annotations

import logging
from pathlib import Path

from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.importer_cellpose import CellPoseImporter
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import clone_cellpose_result


class CellposeImportDialog(QtWidgets.QDialog):
    """Dialog for importing Cellpose segmentation data with PNG/TIFF image overlay."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        existing_data: CellposeImportResult | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._imported_data: CellposeImportResult | None = None
        self.image_path: Path | None = None
        self.segmentation_path: Path | None = None
        self.importer = CellPoseImporter()
        self.overlay_widget = CellOverlayWidget("")
        # Use a defensive copy so the preview can't mutate the stored instance
        self.existing_data = clone_cellpose_result(existing_data)
        self.file_selection_container: QtWidgets.QWidget | None = None
        self.main_content_container: QtWidgets.QWidget | None = None
        self._setup_ui()
        self._load_existing_data_if_available()

    def _maximise_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        available = screen.availableGeometry()
        # Use the maximum available window size
        self.setGeometry(available)
        self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowMaximized)

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Cellpose Data")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        self.file_selection_container = QtWidgets.QWidget()
        selection_layout = QtWidgets.QVBoxLayout(self.file_selection_container)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        instructions = QtWidgets.QLabel(
            "Select a PNG/TIFF image and corresponding CSV or NPY segmentation file. "
            "The segmentation file should contain pixel-by-pixel segmentation IDs."
        )
        instructions.setWordWrap(True)
        selection_layout.addWidget(instructions)

        # Create horizontal layout for file selection and info
        top_horizontal_layout = QtWidgets.QHBoxLayout()

        # File selection group
        file_group = QtWidgets.QGroupBox("File Selection")
        file_layout = QtWidgets.QFormLayout(file_group)

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

        top_horizontal_layout.addWidget(file_group, stretch=2)

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
        main_content_layout = QtWidgets.QVBoxLayout(self.main_content_container)
        main_content_layout.setContentsMargins(0, 0, 0, 0)

        # Add ID mapping options
        mapping_group = QtWidgets.QGroupBox("ID Mapping Options")
        mapping_layout = QtWidgets.QFormLayout(mapping_group)

        # Strategy selection
        strategy_layout = QtWidgets.QHBoxLayout()
        self.mapping_buttons = QtWidgets.QButtonGroup(self)
        for option in ("Linear", "Custom"):
            button = QtWidgets.QRadioButton(option)
            self.mapping_buttons.addButton(button)
            strategy_layout.addWidget(button)
        strategy_layout.addStretch(1)
        strategy_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        if self.mapping_buttons.buttons():
            first_button = self.mapping_buttons.buttons()[0]
            first_button.setChecked(True)
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
        self.custom_mapping_line.setPlaceholderText(
            "e.g., 1:12,2:3 (stimkey:cellposeId)"
        )
        self.custom_mapping_line.textChanged.connect(
            self._on_custom_mapping_text_changed
        )
        self.custom_mapping_line.setVisible(False)
        self.custom_mapping_row_index = mapping_layout.rowCount()
        self.custom_mapping_label = QtWidgets.QLabel("Custom mapping:")
        self.custom_mapping_label.setVisible(False)
        mapping_layout.addRow(self.custom_mapping_label, self.custom_mapping_line)

        # Connect radio buttons to show/hide custom mapping input
        for abstract_button in self.mapping_buttons.buttons():
            if hasattr(abstract_button, "toggled"):
                abstract_button.toggled.connect(self._on_mapping_changed)

        # Place overlay first so the visual is top-most
        self.overlay_widget.setMinimumHeight(400)
        main_content_layout.addWidget(self.overlay_widget)

        checkbox_layout = QtWidgets.QHBoxLayout()
        checkbox_layout.setContentsMargins(4, 0, 4, 0)
        self.show_cellpose_checkbox = QtWidgets.QCheckBox("Show Cellpose IDs")
        self.show_cellpose_checkbox.setChecked(True)
        self.show_stimulus_checkbox = QtWidgets.QCheckBox("Show Stimulus IDs")
        self.show_stimulus_checkbox.setChecked(False)
        self.show_segmentation_checkbox = QtWidgets.QCheckBox("Show segmentation")
        self.show_segmentation_checkbox.setChecked(True)
        for checkbox in (
            self.show_cellpose_checkbox,
            self.show_stimulus_checkbox,
            self.show_segmentation_checkbox,
        ):
            checkbox.toggled.connect(self._on_display_toggle_changed)
            checkbox_layout.addWidget(checkbox)
        checkbox_layout.addStretch(1)
        main_content_layout.addLayout(checkbox_layout)
        main_content_layout.addWidget(mapping_group)

        button_layout = QtWidgets.QHBoxLayout()

        self.clear_button = QtWidgets.QPushButton("Clear Data")
        self.clear_button.clicked.connect(self._clear_existing_data)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()

        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        main_content_layout.addLayout(button_layout)
        layout.addWidget(self.main_content_container)
        self.main_content_container.setVisible(False)
        self._set_ui_loaded_state(False)
        self._on_display_toggle_changed()

    def _set_ui_loaded_state(self, loaded: bool) -> None:
        """Toggle between pre-selection and post-selection UI."""
        if self.file_selection_container is not None:
            self.file_selection_container.setVisible(not loaded)
        if self.main_content_container is not None:
            self.main_content_container.setVisible(loaded)
            if loaded:
                self._maximise_screen()
        if not loaded and hasattr(self, "import_button"):
            self.import_button.setEnabled(False)

    def _reset_importer_state(self, clear_existing_data: bool = False) -> None:
        """Return dialog to initial state after an importer failure."""
        self._imported_data = None
        if clear_existing_data:
            self.existing_data = None

        self.importer = CellPoseImporter()
        self.image_path = None
        self.segmentation_path = None

        self.image_path_label.setText("No image selected")
        self.image_path_label.setStyleSheet("QLabel { color: gray; }")
        self.segmentation_path_label.setText("No segmentation file selected")
        self.segmentation_path_label.setStyleSheet("QLabel { color: gray; }")
        self.custom_mapping_line.clear()
        self.linear_offset_spin.setValue(0)

        for button in self.mapping_buttons.buttons():
            if button.text() == "Linear":
                button.setChecked(True)
                break

        self.show_cellpose_checkbox.setChecked(True)
        self.show_stimulus_checkbox.setChecked(False)
        self.show_segmentation_checkbox.setChecked(True)
        self._on_display_toggle_changed()

        self.overlay_widget.set_overlay(None, None, None, None, None, None)
        self._hide_info()
        self._set_ui_loaded_state(False)

    def _on_mapping_changed(self) -> None:
        """Handle toggle between mapping strategies"""
        mapping_strategy = self._get_mapping_strategy()
        is_custom = mapping_strategy == "Custom"
        self.custom_mapping_line.setVisible(is_custom)
        self.custom_mapping_label.setVisible(is_custom)
        self.linear_offset_spin.setVisible(not is_custom)
        self.linear_offset_label.setVisible(not is_custom)

        if is_custom and not self.show_stimulus_checkbox.isChecked():
            self.show_stimulus_checkbox.setChecked(True)

        # Update preview if data is loaded
        if (
            self.importer.segmentation_data is not None
            and self.importer.overlay_image is not None
        ):
            self._update_preview_mapping()

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
        self.overlay_widget.set_display_options(
            show_cellpose_ids=self.show_cellpose_checkbox.isChecked(),
            show_stimulus_ids=self.show_stimulus_checkbox.isChecked(),
            show_segmentation=self.show_segmentation_checkbox.isChecked(),
        )

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

    def _parse_custom_mapping(self, mapping_str: str) -> dict[int, int]:
        """Parse custom mapping string like '1:12,2:3' into {1: 12, 2: 3}"""
        mapping: dict[int, int] = {}
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
                mapping[stim_key] = cellpose_id
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid custom mapping format: {exc}") from exc

        return mapping

    def _generate_stimulus_mapping(self, strategy: str) -> dict[int, int]:
        """Generate stimulus-to-cellpose mapping based on strategy"""
        if self.importer._original_cellpose_ids is None:
            return {}

        original_ids = self.importer._original_cellpose_ids

        if strategy == "Linear":
            offset = self._get_linear_offset()
            # Map stimulus ID = cellpose_id - offset (leaving gaps if IDs are missing)
            mapping: dict[int, int] = {}
            for cellpose_id in original_ids:
                if cellpose_id < offset:
                    continue  # should start with (if offset 5) cellpose ID 5 and map to stim ID 1.
                stim_id = int(cellpose_id - offset)
                mapping[stim_id] = int(
                    cellpose_id
                )  # not sure why we see "1:1", "2:2", but then conspiciously from 5, "5:1". Need to solve.
                # think cell-stim are swapped in ordering on map vs this calculation.
            return mapping
        elif strategy == "Custom":
            # Parse custom mapping string
            return self._parse_custom_mapping(self.custom_mapping_line.text())
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
                base_image=self.importer.image_data,
            )
        except Exception as exc:
            self.logger.warning("Error updating preview mapping: %s", exc)

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

            self.importer.load_segmentation_file(filename)
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
            self.importer.generate_overlay()
            self.import_button.setEnabled(True)
            self._set_ui_loaded_state(True)
            # Generate preview mapping to show in overlay
            mapping_strategy = self._get_mapping_strategy()
            preview_mapping = (
                self._generate_stimulus_mapping(mapping_strategy)
                if mapping_strategy
                else {}
            )

            self.overlay_widget.set_overlay(
                self.importer.overlay_image,
                self.importer.segmentation_data,
                self.importer._original_cellpose_ids,
                preview_mapping,
                base_image=self.importer.image_data,
            )
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

            self._imported_data = self.importer.create_result(
                self.image_path, stimulus_mapping
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

            if stimulus_mapping:
                success_msg += f"\nStimulus mapping: {dict(list(stimulus_mapping.items())[:5])}{'...' if len(stimulus_mapping) > 5 else ''}"

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

            # Load the existing image and segmentation data
            self.importer.restore_from_result(self.existing_data)

            # Pre-fill mapping if available, prefer Linear when mapping is identity
            mapping = self.existing_data.stimulus_to_cellpose_mapping or {}
            if mapping:
                mapping_items = list(mapping.items())
                is_identity = all(
                    stim_id == cellpose_id for stim_id, cellpose_id in mapping_items
                )

                if not is_identity:
                    mapping_str = ",".join(
                        [
                            f"{stim_id}:{cellpose_id}"
                            for stim_id, cellpose_id in mapping_items
                        ]
                    )
                    self.custom_mapping_line.setText(mapping_str)
                else:
                    self.custom_mapping_line.clear()
                    self.linear_offset_spin.setValue(0)

                target_mode = "Linear" if is_identity else "Custom"
                for button in self.mapping_buttons.buttons():
                    if button.text() == target_mode:
                        button.setChecked(True)
                        break

            # Update UI
            self.segmentation_path_label.setText("(from existing data)")
            self.segmentation_path_label.setStyleSheet("QLabel { color: blue; }")
            # Clear button is now always visible
            self.import_button.setEnabled(True)

            # Update preview
            self._update_preview_mapping()
            self._update_info()
            self._set_ui_loaded_state(True)

        except Exception as exc:
            self.logger.warning("Failed to load existing cellpose data: %s", exc)
            self._reset_importer_state(clear_existing_data=True)

    def _clear_existing_data(self) -> None:
        """Clear cellpose data after confirmation"""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Data",
            "Are you sure you want to clear all cellpose data? This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._reset_importer_state(clear_existing_data=True)
