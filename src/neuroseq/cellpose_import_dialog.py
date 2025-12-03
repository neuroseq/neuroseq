from __future__ import annotations

import logging
from pathlib import Path

from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.importer_cellpose import CellPoseImporter
from neuroseq.importer_cellpose import CellposeImportResult


class CellposeImportDialog(QtWidgets.QDialog):
    """Dialog for importing Cellpose segmentation data with PNG/TIFF image overlay."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._imported_data: CellposeImportResult | None = None
        self.image_path: Path | None = None
        self.segmentation_path: Path | None = None
        self.importer = CellPoseImporter()
        self.overlay_widget = CellOverlayWidget("Color-coded Preview")
        self._setup_ui()

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
        info_group = QtWidgets.QGroupBox("Data Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)

        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)

        top_horizontal_layout.addWidget(info_group, stretch=1)

        layout.addLayout(top_horizontal_layout)

        # Limit preview height to make room for other controls
        self.overlay_widget.setMaximumHeight(300)
        layout.addWidget(self.overlay_widget)

        button_layout = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

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

    def _check_files_loaded(self) -> None:
        if self.importer.image_data is None or self.importer.segmentation_data is None:
            self.import_button.setEnabled(False)
            return

        try:
            self.importer.validate_dimensions()
            self.importer.generate_overlay()
            self.import_button.setEnabled(True)
            self.overlay_widget.set_overlay(
                self.importer.overlay_image,
                self.importer.segmentation_data,
                self.importer.get_legend_data(),
            )
            self._update_info()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Dimension Mismatch", str(exc))
            self.import_button.setEnabled(False)

    def _update_info(self) -> None:
        info = self.importer.get_info()
        if not info:
            self.info_text.clear()
            return

        info_lines = [
            f"Image shape: {info['image_shape']}",
            f"Segmentation shape: {info['segmentation_shape']}",
            f"Unique segment IDs: {info['unique_ids']}",
            f"Background pixels (ID=0): {info['background_pixels']}",
            f"Detected neurons: {info['n_neurons']}",
        ]

        if "neuron_id_range" in info:
            min_id, max_id = info["neuron_id_range"]
            info_lines.append(f"Neuron ID range: {min_id}-{max_id}")

        self.info_text.setPlainText("\n".join(info_lines))

    def _import_data(self) -> None:
        if (
            self.importer.segmentation_data is None
            or self.importer.overlay_image is None
            or self.image_path is None
        ):
            return

        try:
            self._imported_data = self.importer.create_result(self.image_path)

            info = self.importer.get_info()
            unique_ids = [uid for uid in info["unique_ids"] if uid > 0]

            QtWidgets.QMessageBox.information(
                self,
                "Import Successful",
                f"Successfully imported Cellpose data.\n"
                f"Image: {info['image_shape']}\n"
                f"Detected {info['n_neurons']} neurons with IDs: {unique_ids}",
            )
            self.accept()

        except Exception as exc:
            self.logger.exception("Error importing Cellpose data")
            QtWidgets.QMessageBox.critical(
                self, "Import Error", f"Error importing Cellpose data:\n{exc}"
            )

    def get_imported_data(self) -> CellposeImportResult | None:
        """Return the last successfully imported data payload."""
        return self._imported_data
