from __future__ import annotations

import logging
from pathlib import Path

import brian2.numpy_ as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.cell_overlay_widget import CellOverlayWidget
from neuroseq.data import SpikeData
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_cellpose import clone_cellpose_result
from neuroseq.importer_patchclamp import PeakDetectionParams
from neuroseq.importer_patchclamp import import_patchclamp_data
from neuroseq.mapping_validation import ensure_unique_ids
from neuroseq.utils import get_primary_available_geometry
from neuroseq.utils import is_vertical_screen


class PatchClampImportDialog(QtWidgets.QDialog):
    """Dialog for importing patch clamp data from .npy files."""

    def __init__(
        self,
        cellpose_import: CellposeImportResult | None = None,
        project: object | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        # Work on a copy so we don't mutate the caller's stored data
        self.cellpose_import = clone_cellpose_result(cellpose_import)
        self.project = project
        self.imported_spike_data: list = []
        self.imported_trace_data: list = []
        self.experiment_name = "PatchClamp Import"
        self.experiment_duration: float = 0.0
        self._auto_stimulus_ids: list[int] = []
        self._auto_from_cellpose = False
        self.append_data = False
        self.neurons_to_replace: list[int] = []
        self.file_path: Path | None = None
        self.npy_data: np.ndarray | None = None
        self.data_row_count: int = 0
        self.overlay_widget: CellOverlayWidget | None = None
        self.show_stimulus_checkbox: QtWidgets.QCheckBox | None = None
        self.preview_group: QtWidgets.QGroupBox | None = None
        self.options_group: QtWidgets.QGroupBox | None = None
        self.spike_group: QtWidgets.QGroupBox | None = None
        self.overlay_stack: QtWidgets.QStackedWidget | None = None
        self.overlay_placeholder: QtWidgets.QLabel | None = None
        self.file_group: QtWidgets.QGroupBox | None = None
        self.file_status_label: QtWidgets.QLabel | None = None
        self.button_row: QtWidgets.QWidget | None = None
        self.custom_mapping_preview: QtWidgets.QLabel | None = None
        self._has_maximised = False
        self._vertical_layout = False
        self._is_adjusting_position = False
        self._current_screen: QtGui.QScreen | None = None
        self._setup_ui()
        self._apply_existing_stim_mapping()
        self._populate_cellpose_preview()

    def _maximise_screen(self, screen: QtGui.QScreen | None = None) -> None:
        available = self._get_available_geometry(screen)
        if available is None:
            return
        adjusted = available.adjusted(70, 70, -70, -70)
        self.setGeometry(adjusted)
        self.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowMaximized)

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
        return is_vertical_screen(self._get_available_geometry(screen))  # updates

    @staticmethod
    def _double_spinbox(
        *,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int,
        suffix: str = "",
        tooltip: str | None = None,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(f" {suffix}")
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    @staticmethod
    def _int_spinbox(
        *,
        minimum: int,
        maximum: int,
        step: int,
        value: int,
        suffix: str = "",
        tooltip: str | None = None,
    ) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(f" {suffix}")
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    def _setup_ui(self) -> None:
        self.setWindowTitle("Import Patch Clamp Data")
        self.setModal(True)

        self._vertical_layout = self._should_use_vertical_layout()
        direction = (
            QtWidgets.QBoxLayout.Direction.TopToBottom
            if self._vertical_layout
            else QtWidgets.QBoxLayout.Direction.LeftToRight
        )
        self.main_layout = QtWidgets.QBoxLayout(direction, self)
        self.main_layout.setSpacing(12)

        self.left_container = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(8)

        self.overlay_stack = QtWidgets.QStackedWidget()
        self.overlay_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_widget = CellOverlayWidget("Cellpose Preview")
        self.overlay_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.overlay_placeholder = QtWidgets.QLabel("No cellpose data yet")
        self.overlay_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.overlay_placeholder.setStyleSheet("QLabel { color: gray; }")
        self.overlay_placeholder.setWordWrap(True)
        self.overlay_stack.addWidget(self.overlay_placeholder)
        self.overlay_stack.addWidget(self.overlay_widget)
        self.left_layout.addWidget(self.overlay_stack, stretch=1)
        self.left_layout.addStretch()

        self.file_group = QtWidgets.QGroupBox("Select .npy patchclamp file")
        file_layout = QtWidgets.QHBoxLayout(self.file_group)

        self.file_path_label = QtWidgets.QLabel("No file selected")
        self.file_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(browse_button)

        self.left_layout.addWidget(self.file_group)
        self.main_layout.addWidget(self.left_container, stretch=3)

        self.right_container = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(8)

        self.preview_group = QtWidgets.QGroupBox("Data Preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(140)
        preview_layout.addWidget(self.preview_text)
        self.right_layout.addWidget(self.preview_group)

        self.options_group = QtWidgets.QGroupBox("Import Options")
        options_layout = QtWidgets.QFormLayout(self.options_group)

        self.sampling_rate_spin = self._double_spinbox(
            minimum=0.1,
            maximum=10000.0,
            step=1.0,
            value=30.0,
            decimals=2,
            suffix="Hz",
        )
        options_layout.addRow("Sampling rate:", self.sampling_rate_spin)

        self.custom_mapping_line = QtWidgets.QLineEdit()
        self.custom_mapping_line.setPlaceholderText(
            "Comma-separated stim IDs (e.g., 2,3)"
        )
        self.custom_mapping_line.setVisible(True)  # not needed?

        custom_layout = QtWidgets.QVBoxLayout()
        custom_layout.addWidget(self.custom_mapping_line)
        self.custom_mapping_preview = QtWidgets.QLabel("")
        self.custom_mapping_preview.setWordWrap(True)
        self.custom_mapping_preview.setStyleSheet("QLabel { color: gray; }")
        self.custom_mapping_preview.setVisible(True)  # not needed?
        custom_layout.addWidget(self.custom_mapping_preview)
        self.custom_widget = QtWidgets.QWidget()
        self.custom_widget.setLayout(custom_layout)

        self.custom_mapping_label = QtWidgets.QLabel("Specify stim IDs to import:")
        self.custom_mapping_label.setVisible(True)  # not needed?
        self.custom_widget.setVisible(True)  # not needed?
        self.custom_mapping_hint = QtWidgets.QLabel("")

        options_layout.addRow(self.custom_mapping_label, self.custom_widget)
        options_layout.addRow("", self.custom_mapping_hint)

        self.show_stimulus_checkbox = QtWidgets.QCheckBox(
            "Show Stimulus IDs in preview"
        )
        self.show_stimulus_checkbox.setChecked(True)
        self.show_stimulus_checkbox.toggled.connect(self._on_show_stimulus_toggled)
        options_layout.addRow(self.show_stimulus_checkbox)

        self.spike_group = QtWidgets.QGroupBox("Peak Detection Parameters")
        spike_layout = QtWidgets.QFormLayout(self.spike_group)

        self.peak_height_spin = self._double_spinbox(
            minimum=0.1,
            maximum=10.0,
            step=0.1,
            value=3.0,
            decimals=2,
            suffix="std",
            tooltip="Minimum height above baseline in standard deviations",
        )
        spike_layout.addRow("Peak height threshold:", self.peak_height_spin)

        self.peak_prominence_spin = self._double_spinbox(
            minimum=0.1,
            maximum=5.0,
            step=0.1,
            value=0.5,
            decimals=2,
            suffix="std",
            tooltip="Minimum prominence for peak detection",
        )
        spike_layout.addRow("Peak prominence:", self.peak_prominence_spin)

        self.min_interval_spin = self._double_spinbox(
            minimum=0.0001,
            maximum=0.1,
            step=0.001,
            value=0.001,
            decimals=3,
            suffix="s",
            tooltip="Minimum interval between detected spikes",
        )
        spike_layout.addRow("Min spike interval:", self.min_interval_spin)

        self.peak_distance_spin = self._int_spinbox(
            minimum=1,
            maximum=1000,
            step=1,
            value=10,
            suffix="samples",
            tooltip="Minimum horizontal distance between peaks in samples",
        )
        spike_layout.addRow("Peak distance:", self.peak_distance_spin)

        self.peak_width_spin = self._int_spinbox(
            minimum=1,
            maximum=100,
            step=1,
            value=1,
            suffix="samples",
            tooltip="Minimum width of peaks in samples",
        )
        spike_layout.addRow("Peak width:", self.peak_width_spin)

        self.right_layout.addWidget(self.options_group)
        self.right_layout.addWidget(self.spike_group)

        self.button_row = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(self.button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.file_status_label = QtWidgets.QLabel("No file selected")
        self.file_status_label.setStyleSheet("QLabel { color: gray; }")
        self.file_status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        button_layout.addWidget(self.file_status_label, stretch=1)
        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        button_layout.addStretch()
        self.right_layout.addWidget(self.button_row)
        self.main_layout.addWidget(self.right_container, stretch=2)
        # Defer sizing until the dialog is shown so the window manager can apply it.

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_overlay_size()
        if self.npy_data is None:
            self._set_file_sections_visible(False)
            self._set_file_selector_visible(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_screen_layout(self._get_screen_for_window())

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
        Align the dialog layout and size to the target screen orientation
        and resync the preview when moving across monitors.
        """
        if screen is None:
            return
        self._current_screen = screen
        desired_vertical = self._should_use_vertical_layout(screen)
        if self.main_layout is not None and isinstance(
            self.main_layout, QtWidgets.QBoxLayout
        ):
            direction = (
                QtWidgets.QBoxLayout.Direction.TopToBottom
                if desired_vertical
                else QtWidgets.QBoxLayout.Direction.LeftToRight
            )
            self.main_layout.setDirection(direction)
        self._vertical_layout = desired_vertical
        self._is_adjusting_position = True
        self._maximise_screen(screen)
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

    def _sync_overlay_size(self) -> None:
        # wait until everyting is ready, activate and then proceed; this keeps consistent between
        # (A) Cellpose initial, (B) Cellpose reload(not initial import/flip), (C) Patchclamp open
        if self.left_container is None or self.right_container is None:
            return
        if self.overlay_stack is None or self.file_group is None:
            return
        if self.left_layout is None:
            return
        if self.main_layout is not None:
            self.main_layout.activate()
        self.left_layout.activate()
        main_rect = self.contentsRect()
        if main_rect.width() < 100 or main_rect.height() < 100:
            return  # Strange bug where QT restricts fastplotlib via dialog size. Always gets a second update, so this avoids it.
        # with more time I will investgiate this and remove it at source preferably to resolving here.
        if self._vertical_layout:  # this needs to adapt to moving across screen and when the application is across two screens
            # known bug that the background neuroseq application when behind cellpose preview causes awkward cross-screen
            # view. Hoped this would resolve it but it does nto get at the core issue (some kind of tying to the main window)
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
            left_rect.height() - self.file_group.height() - self.left_layout.spacing()
        )
        square_size = min(left_width, available_height)
        if square_size <= 0:
            return
        self.overlay_stack.setFixedSize(square_size, square_size)
        if self.overlay_widget is not None:
            self.overlay_widget.setFixedSize(square_size, square_size)
        self.overlay_stack.updateGeometry()
        if self.overlay_widget is not None:
            self.overlay_widget.updateGeometry()

    def _populate_cellpose_preview(self) -> None:
        """Fill the left-hand preview with Cellpose data if available."""
        if self.overlay_widget is None or self.overlay_stack is None:
            return

        if (
            self.cellpose_import is None
            or self.cellpose_import.overlay_image is None
            or self.cellpose_import.segmentation_array is None
        ):
            self._show_overlay_placeholder("No cellpose data yet")
            return

        self.overlay_stack.setCurrentWidget(self.overlay_widget)
        self.overlay_widget.set_overlay(
            self.cellpose_import.overlay_image,
            self.cellpose_import.segmentation_array,
            self.cellpose_import.original_cellpose_ids,
            self._get_cellpose_overlay_mapping(),
        )
        self._apply_overlay_display_options()

    def _show_overlay_placeholder(self, message: str) -> None:
        """Show a text placeholder instead of an empty overlay preview."""
        if self.overlay_stack is None or self.overlay_placeholder is None:
            return

        self.overlay_placeholder.setText(message)
        self.overlay_stack.setCurrentWidget(self.overlay_placeholder)

    def _on_mapping_changed(self) -> None:
        # Decide on if we need both the below.
        self.custom_mapping_label.setVisible(True)  # todo: refactor to remove this.
        self.custom_widget.setVisible(True)
        self._update_custom_mapping_hint()

    def _on_show_stimulus_toggled(self) -> None:
        self._apply_overlay_display_options()

    def _apply_overlay_display_options(self) -> None:
        if self.overlay_widget is None:
            return
        show_stimulus_ids = (
            self.show_stimulus_checkbox.isChecked()
            if self.show_stimulus_checkbox is not None
            else False
        )
        self.overlay_widget.set_display_options(
            show_stimulus_ids=show_stimulus_ids,
            show_cellpose_ids=False,
        )

    def _apply_existing_stim_mapping(self) -> None:
        """Prefill custom mapping when Cellpose data already has stimulus IDs."""
        if not self._has_custom_cellpose_mapping():
            return

        if self.cellpose_import is None:
            return
        mapping = self.cellpose_import.stimulus_to_cellpose_mapping or {}
        stim_ids = sorted(mapping.keys())
        self.custom_mapping_line.setText(",".join(str(stim) for stim in stim_ids))

        if len(stim_ids) > 1:
            self._auto_stimulus_ids = stim_ids
            self._auto_from_cellpose = True
        self._update_custom_mapping_hint()

    def _browse_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Patch Clamp Data File",
            "",
            "NumPy files (*.npy);;All files (*)",
        )
        if filename:
            self.load_file(filename)

    def load_file(self, filename: str) -> None:
        try:
            self._apply_screen_layout(self._get_screen_for_window())
            self.file_path = Path(filename)
            self.file_path_label.setText(self.file_path.name)
            self.file_path_label.setStyleSheet("QLabel { color: black; }")
            if self.file_status_label is not None:
                self.file_status_label.setText(self.file_path.name)
                self.file_status_label.setStyleSheet("QLabel { color: black; }")

            # Clear previous data
            self.npy_data = None
            self.data_row_count = 0
            self._auto_stimulus_ids = []
            self._auto_from_cellpose = False

            if self.file_path.suffix.lower() == ".npy":
                self._load_npy_file(filename)
            else:
                raise ValueError("Unsupported file format. Expected .npy file.")

            self._update_preview()
            self.import_button.setEnabled(True)
            self._set_file_sections_visible(True)
            self._set_file_selector_visible(False)
            self._apply_screen_layout(self._get_screen_for_window())
        except Exception as exc:
            self.logger.exception("Error loading patch clamp file")
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Could not load file {filename}:\n{exc}",
            )
            self.import_button.setEnabled(False)
            self._set_file_sections_visible(False)
            self._set_file_selector_visible(True)
            if self.file_status_label is not None:
                self.file_status_label.setText("No file selected")
                self.file_status_label.setStyleSheet("QLabel { color: gray; }")

    def _load_npy_file(self, filename: str) -> None:
        self.npy_data = np.load(filename)

        self.logger.debug(
            "Loaded patchclamp npy: path=%s shape=%s dtype=%s range=%.4f–%.4f",
            filename,
            getattr(self.npy_data, "shape", None),
            getattr(self.npy_data, "dtype", None),
            float(np.min(self.npy_data)),
            float(np.max(self.npy_data)),
        )

        if self.npy_data.ndim not in (1, 2):
            raise ValueError(
                f"NumPy patchclamp data must be 1D (single neuron) or 2D (multiple neurons), got {self.npy_data.ndim}D"
            )

        self.data_row_count = (
            1 if self.npy_data.ndim == 1 else int(self.npy_data.shape[0])
        )
        self._apply_cellpose_mapping_defaults(self.data_row_count)
        self._update_target_stimulus_ids()

        # Pathclamp has a higher hz in general
        self.sampling_rate_spin.setValue(10000.0)

    def _update_preview(
        self,
    ) -> None:  # important to have so the user can figure out mistakes
        lines = []

        if self.npy_data is not None:
            lines.append(f"NPY DATA: {self.npy_data.shape[0]} timepoints")
            lines.append(
                f"  mean {np.mean(self.npy_data):.3f} ± {np.std(self.npy_data):.3f}"
            )
            lines.append(
                f"  range {np.min(self.npy_data):.3f} to {np.max(self.npy_data):.3f}"
            )

            # Calculate duration at 10kHz
            duration_sec = self.npy_data.shape[0] / 10000.0
            lines.append(f"  duration: {duration_sec:.1f} seconds at 10kHz")
            lines.append("")

        else:
            self.preview_text.clear()
            return

        self.preview_text.setPlainText("\n".join(lines).strip())

    def _apply_cellpose_mapping_defaults(self, row_count: int) -> None:
        """Use Cellpose stimulus IDs in ascending order to prefill multi-row targets."""
        stim_ids = self._get_cellpose_stim_ids()
        if len(stim_ids) < 2 or row_count < 2:
            return

        derived_ids = stim_ids[:row_count]
        self.custom_mapping_line.setText(self._format_stimulus_display(derived_ids))
        self._auto_stimulus_ids = derived_ids
        self._auto_from_cellpose = True
        self._update_custom_mapping_hint()

    def _get_cellpose_stim_ids(self) -> list[int]:
        """Return sorted stimulus IDs from an existing Cellpose mapping."""
        if not self._has_custom_cellpose_mapping():
            return []  # Linear: Just get nothing here presently.
        if self.cellpose_import is None:
            return []
        mapping = self.cellpose_import.stimulus_to_cellpose_mapping or {}
        return sorted(mapping.keys())

    def _has_custom_cellpose_mapping(self) -> bool:
        if self.cellpose_import is None:
            return False
        strategy = getattr(self.cellpose_import, "mapping_strategy", "")
        return strategy == "Custom"

    # now generates Linear mapping identity itself for the preview widget here, rather than cellpose generating it
    # and passing it around like a custom mapping
    def _get_cellpose_overlay_mapping(self) -> dict[int, int] | None:
        if self.cellpose_import is None:
            return None
        mapping = self.cellpose_import.stimulus_to_cellpose_mapping
        strategy = getattr(self.cellpose_import, "mapping_strategy", "")
        if not strategy:  # these 4 lines below should never execute now, need to confirm later and remove -  todo.
            strategy = "Custom" if mapping else "Linear"
        if strategy == "Custom" and mapping:
            return mapping
        original_ids = self.cellpose_import.original_cellpose_ids
        if original_ids is None:
            return None
        return {int(cell_id): int(cell_id) for cell_id in original_ids}

    def _format_stimulus_display(self, stim_ids: list[int]) -> str:
        """Format stimulus IDs for display."""
        return ",".join(str(stim_id) for stim_id in stim_ids)

    def _set_file_sections_visible(self, visible: bool) -> None:
        for group in (self.preview_group, self.options_group, self.spike_group):
            if group is not None:
                group.setVisible(visible)
        if self.overlay_stack is not None:
            self.overlay_stack.setVisible(visible)
        if self.button_row is not None:
            self.button_row.setVisible(visible)
        if visible:
            self.resize(
                max(self.width(), getattr(self, "dialog_width", self.width())),
                max(self.height(), getattr(self, "dialog_height", self.height())),
            )

    def _set_file_selector_visible(self, visible: bool) -> None:
        if self.file_group is not None:
            self.file_group.setVisible(visible)

    def _parse_stim_neuron_ids(self, mapping_str: str) -> list[int]:
        """Parse comma-separated stim IDs into a list of ints."""
        if not mapping_str.strip():
            raise ValueError("Stimulus IDs cannot be empty")

        try:
            stim_ids: list[int] = []
            for raw_part in mapping_str.split(","):
                part = raw_part.strip()
                if not part:
                    continue
                if ":" in part:
                    part = part.split(":", 1)[0].strip()
                stim_ids.append(int(part))
            return stim_ids
        except ValueError as exc:
            raise ValueError(f"Invalid stimulus ID format: {exc}") from exc

    def _safe_parse_stimulus_ids(self) -> list[int]:
        """Parse stimulus IDs without raising, used for prefilling defaults."""
        text = self.custom_mapping_line.text()
        if not text.strip():
            return []
        try:
            return self._parse_stim_neuron_ids(text)
        except Exception:
            return []

    def _update_target_stimulus_ids(self) -> None:
        """Limit target stimulus IDs to the number of traces in the loaded data."""
        if self.npy_data is None:
            return

        auto_from_cellpose = False
        row_count = 1 if self.npy_data.ndim == 1 else int(self.npy_data.shape[0])
        self.data_row_count = row_count

        if row_count == 1:
            target_ids = self._safe_parse_stimulus_ids()
            target_ids = [0] if not target_ids else [target_ids[0]]
        else:
            target_ids = self._safe_parse_stimulus_ids()
            auto_from_cellpose = False
            if not target_ids:
                # Default to sorted Cellpose stim IDs when available
                cellpose_ids = self._get_cellpose_stim_ids()
                if cellpose_ids:
                    target_ids = cellpose_ids[:row_count]
                    auto_from_cellpose = True
            if target_ids:
                target_ids = target_ids[:row_count]
                # Pad with sequential IDs if we truncated a shorter list
                next_id = (target_ids[-1] + 1) if target_ids else 1
                while len(target_ids) < row_count:
                    target_ids.append(next_id)
                    next_id += 1
            else:
                target_ids = list(range(row_count))

        mapping_str = self._format_stimulus_display(target_ids)
        self.custom_mapping_line.setText(mapping_str)
        label_count = len(target_ids)
        label_suffix = "ID" if label_count == 1 else "IDs"
        self.custom_mapping_label.setText(
            f"Specify {label_count} {label_suffix} to import:"
        )
        self._auto_from_cellpose = auto_from_cellpose and len(target_ids) > 1
        self._auto_stimulus_ids = target_ids if self._auto_from_cellpose else []
        self._update_custom_mapping_hint()

    def _update_custom_mapping_hint(self) -> None:
        """Show a hint when custom mapping count mismatches neuron count."""
        if self.custom_mapping_hint is None:
            return

        row_count = self.data_row_count or 0
        stim_ids = self._safe_parse_stimulus_ids()
        if row_count <= 0 or not stim_ids:
            self.custom_mapping_hint.setVisible(False)
            return

        if len(stim_ids) > row_count:
            self.custom_mapping_hint.setText(
                f"Too many mappings for the {row_count} neurons being imported"
            )
            self.custom_mapping_hint.setVisible(True)
        elif len(stim_ids) < row_count:
            self.custom_mapping_hint.setText(
                f"Too few mappings for the {row_count} neurons being imported"
            )
            self.custom_mapping_hint.setVisible(True)
        else:
            self.custom_mapping_hint.setVisible(False)

    def import_data(self) -> None:
        if self.npy_data is None:
            return

        try:
            sampling_rate = self.sampling_rate_spin.value()
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be positive")
            dt = 1.0 / sampling_rate

            neuron_ids = ensure_unique_ids(
                self._safe_parse_stimulus_ids(),
                "Custom mapping would assign stimulus ID {value} to multiple patchclamp rows.",
            )
            self.neurons_to_replace = list(neuron_ids)

            self.logger.debug(
                "Importing patchclamp data: sampling_rate=%.2fHz dt=%.6f",
                sampling_rate,
                dt,
            )

            detection_params = PeakDetectionParams(
                peak_height=self.peak_height_spin.value(),
                peak_prominence=self.peak_prominence_spin.value(),
                min_interval=self.min_interval_spin.value(),
                peak_distance=self.peak_distance_spin.value(),
                peak_width=self.peak_width_spin.value(),
            )

            trace_data, spike_data_list = import_patchclamp_data(
                self.npy_data,
                neuron_ids,
                dt=dt,
                detection_params=detection_params,
                logger=self.logger,
            )

            self.imported_spike_data.extend(spike_data_list)
            self.imported_trace_data.append(trace_data)

            total_spikes = sum(len(sd.spike_times) for sd in spike_data_list)
            neuron_list = ", ".join(map(str, neuron_ids))
            duration_seconds = trace_data.traces.shape[1] / sampling_rate
            self.experiment_duration = duration_seconds

            if total_spikes > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Imported {total_spikes} spikes and trace data from neuron(s) {neuron_list}\n"
                    f"Duration: {duration_seconds:.1f} seconds at {sampling_rate:.0f} Hz.",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Spikes Detected",
                    f"No spikes were detected in the recording for neuron(s) {neuron_list}.\n"
                    f"You may need to adjust the detection parameters.",
                )
            self.accept()
        except Exception as exc:
            self.logger.exception("Error importing patch clamp data")
            QtWidgets.QMessageBox.critical(
                self,
                "Import Error",
                f"Error importing data:\n{exc}",
            )

    @property
    def imported_data(self) -> list:
        """Combined imported spike and trace data."""
        spike_data = None
        if self.imported_spike_data:
            if len(self.imported_spike_data) == 1:
                spike_data = self.imported_spike_data[0]
            else:
                spike_indices = np.concatenate(
                    [sd.spike_indices for sd in self.imported_spike_data]
                )
                spike_times = np.concatenate(
                    [sd.spike_times for sd in self.imported_spike_data]
                )
                if spike_times.size > 1:
                    order = np.argsort(spike_times)
                    spike_times = spike_times[order]
                    spike_indices = spike_indices[order]
                spike_data = SpikeData(
                    "Patch Clamp Spikes",
                    spike_indices.astype(np.uint32),
                    spike_times.astype(np.float32),
                )
        data = []
        if spike_data is not None:
            data.append(spike_data)
        return data + self.imported_trace_data

    def closeEvent(self, event):
        # explicitly close all fastplotlib figures to avoid renderer trying to access them after they have been destroyed
        self.overlay_widget.close()
        super().closeEvent(event)
