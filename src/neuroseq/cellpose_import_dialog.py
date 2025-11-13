from __future__ import annotations

import logging
from pathlib import Path

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.importer import CellPoseImporter
from neuroseq.importer import CellposeImportResult


class CellposeImportDialog(QtWidgets.QDialog):
    """Dialog for importing Cellpose segmentation data with PNG/TIFF image overlay."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._imported_data: CellposeImportResult | None = None
        self.image_path: Path | None = None
        self.segmentation_path: Path | None = None
        self.importer = CellPoseImporter()
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Cellpose Data")
        self.setModal(True)
        self.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Select a PNG/TIFF image and corresponding CSV segmentation file. "
            "The segmentation file should contain pixel-by-pixel segmentation IDs."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

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
        file_layout.addRow("Segmentation CSV:", seg_layout)

        layout.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("Color-coded Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_label = QtWidgets.QLabel("Load files to see overlay preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("QLabel { border: 1px solid gray; }")
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group)

        info_group = QtWidgets.QGroupBox("Data Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)

        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)

        legend_group = QtWidgets.QGroupBox("Cell Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)

        self.legend_scroll = QtWidgets.QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setMaximumHeight(120)
        self.legend_widget = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QGridLayout(self.legend_widget)
        self.legend_scroll.setWidget(self.legend_widget)
        legend_layout.addWidget(self.legend_scroll)

        layout.addWidget(legend_group)

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
            self, "Select segmentation CSV file", "", "CSV files (*.csv);;All files (*)"
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
            self._display_overlay_preview()
            self._update_info()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Dimension Mismatch", str(exc))
            self.import_button.setEnabled(False)

    def _display_overlay_preview(self) -> None:
        if self.importer.overlay_image is None:
            return

        # Convert to QPixmap for display
        height, width, channel = self.importer.overlay_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            self.importer.overlay_image.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )

        # Scale to fit preview label
        label_size = self.preview_label.size()
        pixmap = QtGui.QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        self.preview_label.setPixmap(scaled_pixmap)
        self._update_legend()

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

    def _update_legend(self) -> None:
        """Update the cell legend with colored squares for each cell ID."""
        # Clear existing legend
        for i in reversed(range(self.legend_layout.count())):
            child = self.legend_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        legend_data = self.importer.get_legend_data()
        if not legend_data:
            return

        # Add legend entries in a grid (4 columns)
        cols = 4
        for i, (cell_id, rgb) in enumerate(legend_data):
            row = i // cols
            col = i % cols

            # Create horizontal layout for this legend item
            item_widget = QtWidgets.QWidget()
            item_layout = QtWidgets.QHBoxLayout(item_widget)
            item_layout.setContentsMargins(2, 2, 2, 2)
            item_layout.setSpacing(4)

            # Create colored square (16x16 px)
            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(16, 16)
            color_label.setStyleSheet(
                f"QLabel {{ background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid black; }}"
            )

            # Create cell ID label
            text_label = QtWidgets.QLabel(f"Cell {cell_id}")
            text_label.setFixedHeight(16)

            item_layout.addWidget(color_label)
            item_layout.addWidget(text_label)
            item_layout.addStretch()

            self.legend_layout.addWidget(item_widget, row, col)

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
