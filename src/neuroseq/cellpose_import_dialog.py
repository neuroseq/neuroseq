from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.importer_cellpose import CellPoseImporter
from neuroseq.importer_cellpose import CellposeImportResult


class CellposeMappingPreviewDialog(QtWidgets.QDialog):
    """Dialog to preview custom cellpose mapping before import."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        overlay_image: np.ndarray | None = None,
        segmentation_data: np.ndarray | None = None,
        legend_data: list[tuple[int, tuple[int, int, int]]] | None = None,
        original_cellpose_ids: np.ndarray | None = None,
        stimulus_mapping: dict[int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.overlay_image = overlay_image
        self.segmentation_data = segmentation_data
        self.legend_data = legend_data or []
        self.original_cellpose_ids = original_cellpose_ids
        self.stimulus_mapping = stimulus_mapping
        self.result: str | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Custom Mapping Preview")
        self.setModal(True)

        # Set dialog size to be large enough for preview
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width()
        screen_height = screen.availableGeometry().height()
        target_width = max(1000, int(0.8 * screen_width))
        target_height = max(700, int(0.8 * screen_height))
        self.resize(target_width, target_height)

        layout = QtWidgets.QVBoxLayout(self)

        # Preview widget
        self.preview_widget = CellOverlayWidget("Custom Mapping Preview")
        self.preview_widget.set_overlay(
            self.overlay_image,
            self.segmentation_data,
            self.legend_data,
            self.original_cellpose_ids,
            self.stimulus_mapping,
        )
        layout.addWidget(self.preview_widget)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.edit_button = QtWidgets.QPushButton("Edit further")
        self.edit_button.clicked.connect(self._edit_further)
        button_layout.addWidget(self.edit_button)

        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.clicked.connect(self._import_confirmed)
        self.import_button.setDefault(True)
        button_layout.addWidget(self.import_button)

        layout.addLayout(button_layout)

    def _edit_further(self) -> None:
        self.result = "edit"
        self.reject()

    def _import_confirmed(self) -> None:
        self.result = "import"
        self.accept()

    def get_result(self) -> str | None:
        return self.result


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
        self.overlay_widget = CellOverlayWidget("Color-coded Preview")
        self.existing_data = existing_data
        self._setup_ui()
        self._load_existing_data_if_available()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Cellpose Data")
        self.setModal(True)

        # Set width to at least 1200px or 90% of screen width, whichever is larger
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = screen.availableGeometry().width()
        target_width = max(1200, int(0.9 * screen_width))
        self.resize(target_width, 600)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Select a PNG/TIFF image and corresponding CSV or NPY segmentation file. "
            "The segmentation file should contain pixel-by-pixel segmentation IDs."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

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

        layout.addLayout(top_horizontal_layout)

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

        # Custom mapping input
        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setPlaceholderText(
            "e.g., 1:12,2:3 (stimkey:cellposeId)"
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

        layout.addWidget(mapping_group)

        # Limit preview height to make room for other controls
        self.overlay_widget.setMinimumHeight(400)
        layout.addWidget(self.overlay_widget)

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

        layout.addLayout(button_layout)

    def _on_mapping_changed(self) -> None:
        """Handle toggle between mapping strategies"""
        mapping_strategy = self._get_mapping_strategy()
        is_custom = mapping_strategy == "Custom"
        self.custom_mapping_line.setVisible(is_custom)
        self.custom_mapping_label.setVisible(is_custom)

        # Update preview if data is loaded
        if (
            self.importer.segmentation_data is not None
            and self.importer.overlay_image is not None
        ):
            self._update_preview_mapping()

    def _get_mapping_strategy(self) -> str:
        """Get the currently selected mapping strategy"""
        checked = self.mapping_buttons.checkedButton()
        return checked.text() if checked is not None else "Linear"

    def _parse_custom_mapping(self, mapping_str: str) -> dict[int, int]:
        """Parse custom mapping string like '1:12,2:3' into {1: 12, 2: 3}"""
        mapping: dict[int, int] = {}
        if not mapping_str.strip():
            return mapping

        try:
            pairs = mapping_str.split(",")
            for pair in pairs:
                pair = pair.strip()
                if ":" not in pair:
                    continue
                stim_key_str, cellpose_id_str = pair.split(":", 1)
                stim_key = int(stim_key_str.strip())
                cellpose_id = int(cellpose_id_str.strip())
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
            # Map stimulus IDs 1,2,3... to cellpose IDs in order
            return {
                i + 1: int(cellpose_id) for i, cellpose_id in enumerate(original_ids)
            }
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
                self.importer.get_legend_data(),
                self.importer._original_cellpose_ids,
                preview_mapping,
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
            self._update_info(error=str(exc))

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
            self._update_info(error=str(exc))

    def _check_files_loaded(self) -> None:
        if self.importer.image_data is None or self.importer.segmentation_data is None:
            self.import_button.setEnabled(False)
            return

        try:
            self.importer.validate_dimensions()
            self.importer.generate_overlay()
            self.import_button.setEnabled(True)
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
                self.importer.get_legend_data(),
                self.importer._original_cellpose_ids,
                preview_mapping,
            )
            self._hide_info()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Dimension Mismatch", str(exc))
            self._update_info(error=str(exc))
            self.import_button.setEnabled(False)

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

            # Show preview of custom mapping if Custom is selected
            if mapping_strategy == "Custom" and stimulus_mapping:
                preview_dialog = CellposeMappingPreviewDialog(
                    self,
                    self.importer.overlay_image,
                    self.importer.segmentation_data,
                    self.importer.get_legend_data(),
                    self.importer._original_cellpose_ids,
                    stimulus_mapping,
                )
                preview_dialog.exec()

                result = preview_dialog.get_result()
                if result == "edit":
                    # User wants to edit further - return to dialog
                    return
                elif result != "import":
                    # User cancelled or closed dialog
                    return

            self._imported_data = self.importer.create_result(
                self.image_path, stimulus_mapping
            )

            info = self.importer.get_info()
            unique_ids = [uid for uid in info["unique_ids"] if uid > 0]

            # Create success message with mapping info
            success_msg = (
                f"Successfully imported Cellpose data.\n"
                f"Image: {info['image_shape']}\n"
                f"Detected {info['n_neurons']} neurons with IDs: {unique_ids}\n"
                f"Mapping strategy: {mapping_strategy}"
            )

            if stimulus_mapping:
                success_msg += f"\nStimulus mapping: {dict(list(stimulus_mapping.items())[:5])}{'...' if len(stimulus_mapping) > 5 else ''}"

            QtWidgets.QMessageBox.information(self, "Import Successful", success_msg)
            self.accept()

        except Exception as exc:
            self.logger.exception("Error importing Cellpose data")
            QtWidgets.QMessageBox.critical(
                self, "Import Error", f"Error importing Cellpose data:\n{exc}"
            )

    def get_imported_data(self) -> CellposeImportResult | None:
        """Return the last successfully imported data payload."""
        return self._imported_data

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

            # Pre-fill custom mapping if available
            if self.existing_data.stimulus_to_cellpose_mapping:
                mapping_str = ",".join(
                    [
                        f"{stim_id}:{cellpose_id}"
                        for stim_id, cellpose_id in self.existing_data.stimulus_to_cellpose_mapping.items()
                    ]
                )
                self.custom_mapping_line.setText(mapping_str)

                # Set to Custom mapping mode
                for button in self.mapping_buttons.buttons():
                    if button.text() == "Custom":
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

        except Exception as exc:
            self.logger.warning("Failed to load existing cellpose data: %s", exc)

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
            # Reset all data
            self.existing_data = None
            self.image_path = None
            self.segmentation_path = None
            self.importer = CellPoseImporter()
            self._imported_data = None

            # Reset UI
            self.image_path_label.setText("No image selected")
            self.image_path_label.setStyleSheet("QLabel { color: gray; }")
            self.segmentation_path_label.setText("No segmentation file selected")
            self.segmentation_path_label.setStyleSheet("QLabel { color: gray; }")
            self.custom_mapping_line.clear()

            # Reset to Linear mapping
            for button in self.mapping_buttons.buttons():
                if button.text() == "Linear":
                    button.setChecked(True)
                    break

            # Clear button remains visible
            self.import_button.setEnabled(False)

            # Clear preview
            self.overlay_widget.set_overlay(None, None)
            self._hide_info()
