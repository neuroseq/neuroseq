"""Dialog for end-to-end import of 2-photon calcium imaging TIFFs.

Drives `importer_tif.run_suite2p_pipeline` in a background QThread and feeds the
resulting fluorescence / spike arrays into the existing Suite2p import path.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import numpy.typing as npt
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.data import Data
from neuroseq.importer_cellpose import CellPoseImporter
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_suite2p import _resolve_suite2p_indices
from neuroseq.importer_suite2p import prepare_suite2p_spikes
from neuroseq.importer_suite2p import prepare_suite2p_traces
from neuroseq.importer_tif import TifPipelineParams
from neuroseq.importer_tif import TifPipelineResult
from neuroseq.stim import StimArray


def _cellpose_result_to_masks(result: CellposeImportResult) -> npt.NDArray[Any]:
    """Convert stored display-oriented masks back to raw TIFF coordinates.

    The spatial-footprint importer flips Cellpose segmentation vertically for
    NeuroSeq/Fiji display coordinates. TIFF extraction operates on the original
    pixel coordinates, so undo that flip as well as restoring Cellpose's
    1-based labels and zero-valued background.
    """
    if result.segmentation_array is None:
        raise ValueError("Cellpose segmentation is missing; re-run Cellpose import.")
    seg = np.flipud(np.asarray(result.segmentation_array, dtype=np.int32))
    masks = seg.copy()
    # In CellposeImportResult: background = -1, cells = 0..N-1
    # Suite2p / Cellpose raw layout: background = 0, cells = 1..N
    masks[masks >= 0] += 1
    masks[masks < 0] = 0
    return masks


def _load_masks_from_seg_file(path: Path) -> npt.NDArray[Any]:
    """Load a Cellpose `_seg.npy` (or plain mask npy) into a Suite2p label image."""
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.ndim == 0:
        data = loaded.item()
        if isinstance(data, dict) and "masks" in data:
            return np.asarray(data["masks"], dtype=np.int32)
        raise ValueError(f"Could not find 'masks' key in {path}")
    return np.asarray(loaded, dtype=np.int32)


def _load_seg_file_background_image(path: Path) -> npt.NDArray[Any] | None:
    """Best-effort: recover the image a Cellpose `_seg.npy` was segmented on.

    Cellpose sometimes embeds it directly (an "img" key); other versions
    only store a "filename" pointing at the image file on disk (e.g. a Fiji
    Z-projection saved before segmenting) -- try the embedded image first,
    then fall back to loading from that referenced path. Returns None if
    neither is available/loadable, so the caller can fall back to its own
    computed background image instead.
    """
    try:
        loaded = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if not (
        isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.ndim == 0
    ):
        return None
    data = loaded.item()
    if not isinstance(data, dict):
        return None
    img = data.get("img")
    if img is not None:
        try:
            return np.asarray(img)
        except Exception:
            pass
    filename = data.get("filename")
    if filename:
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(filename)))
        except Exception:
            return None
    return None


def _prepare_background_rgb(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Scale a 2D grayscale or 3D RGB(A) image into a uint8 (H, W, 3) array."""
    image = np.asarray(image)
    if image.ndim == 2:
        gray = CellPoseImporter._scale_float_image_to_uint8(image)
        return np.stack([gray, gray, gray], axis=-1)
    if image.ndim == 3 and image.shape[-1] >= 3:
        return CellPoseImporter._scale_float_image_to_uint8(image[..., :3])
    raise ValueError(f"Unsupported background image shape {image.shape}")


def _build_cellpose_result_from_pipeline(
    result: TifPipelineResult,
    image_path: Path,
    *,
    segmentation_path: Path | None = None,
    background_image: npt.NDArray[Any] | None = None,
) -> CellposeImportResult:
    """Wrap pipeline masks and a background image as a Cellpose-style import
    result.

    Prefers `background_image` (e.g. recovered from a _seg.npy's embedded
    image or referenced file -- the actual image Cellpose was shown when
    segmenting) when its shape matches the masks; otherwise falls back to
    the pipeline's own computed mean projection, which may look visually
    different from what was actually segmented (e.g. a raw-stack mean vs.
    a Fiji Z-projection).
    """
    importer = CellPoseImporter()
    source = result.mean_image
    if (
        background_image is not None
        and background_image.shape[:2] == result.masks.shape
    ):
        source = background_image
    importer.image_data = np.flipud(_prepare_background_rgb(source))
    segmentation = result.masks.astype(np.int32) - 1
    importer.segmentation_data = np.flipud(segmentation)
    importer._segmentation_flipped_for_image = True
    unique_ids = np.unique(importer.segmentation_data)
    importer._original_cellpose_ids = unique_ids[unique_ids >= 0]
    importer.generate_overlay()
    return importer.create_result(
        image_path=image_path,
        segmentation_path=segmentation_path,
    )


class _QtLogHandler(logging.Handler):
    """Formats log records and hands them to a thread-safe sink."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self._sink(self.format(record))


class _PipelineWorker(QtCore.QObject):
    progress = QtCore.Signal(str, float)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(object)  # TifPipelineResult
    failed = QtCore.Signal(str)

    def __init__(
        self,
        tif_paths: list[Path],
        masks: npt.NDArray[Any] | None,
        params: TifPipelineParams,
        pipeline: str = "suite2p",
        stim_array: StimArray | None = None,
    ) -> None:
        super().__init__()
        self._tif_paths = tif_paths
        self._masks = masks
        self._params = params
        self._pipeline = pipeline
        self._stim_array = stim_array

    @QtCore.Slot()
    def run(self) -> None:
        handler = _QtLogHandler(self.log.emit)
        suite2p_logger = logging.getLogger("suite2p")
        neuroseq_logger = logging.getLogger("neuroseq.importer_tif")
        suite2p_level = suite2p_logger.level
        neuroseq_level = neuroseq_logger.level
        suite2p_logger.setLevel(logging.INFO)
        neuroseq_logger.setLevel(logging.INFO)
        suite2p_logger.addHandler(handler)
        neuroseq_logger.addHandler(handler)
        try:
            # Imported here so test environments without suite2p can still import
            # this module.
            def callback(msg: str, frac: float) -> None:
                self.progress.emit(msg, float(frac))

            if self._pipeline == "roi_average":
                from neuroseq.importer_tif import run_roi_average_pipeline

                if self._masks is None:
                    raise ValueError("ROI averaging requires Cellpose masks")
                result = run_roi_average_pipeline(
                    self._tif_paths,
                    self._masks,
                    self._params,
                    progress=callback,
                    stim_array=self._stim_array,
                )
            else:
                from neuroseq.importer_tif import run_suite2p_pipeline

                result = run_suite2p_pipeline(
                    self._tif_paths, self._masks, self._params, progress=callback
                )
            self.finished.emit(result)
        except Exception as exc:
            logging.getLogger(__name__).exception("Calcium extraction pipeline failed")
            self.failed.emit(str(exc))
        finally:
            suite2p_logger.removeHandler(handler)
            neuroseq_logger.removeHandler(handler)
            suite2p_logger.setLevel(suite2p_level)
            neuroseq_logger.setLevel(neuroseq_level)


class _PipelineProgressDialog(QtWidgets.QDialog):
    """QProgressDialog look-alike with an expandable log panel."""

    canceled = QtCore.Signal()

    def __init__(
        self,
        label_text: str,
        cancel_text: str,
        minimum: int,
        maximum: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        layout = QtWidgets.QVBoxLayout(self)
        self._label = QtWidgets.QLabel(label_text)
        layout.addWidget(self._label)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(minimum, maximum)
        layout.addWidget(self._bar)

        self._details_button = QtWidgets.QToolButton()
        self._details_button.setText("Details")
        self._details_button.setCheckable(True)
        self._details_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self._details_button.toggled.connect(self._toggle_details)
        layout.addWidget(self._details_button)

        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(200)
        self._log_view.setVisible(False)
        layout.addWidget(self._log_view)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QtWidgets.QPushButton(cancel_text)
        cancel_button.clicked.connect(self.canceled.emit)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)

    def _toggle_details(self, expanded: bool) -> None:
        self._details_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self._log_view.setVisible(expanded)
        self.adjustSize()

    def setLabelText(self, text: str) -> None:
        self._label.setText(text)

    def setValue(self, value: int) -> None:
        self._bar.setValue(value)

    def appendLog(self, message: str) -> None:
        self._log_view.appendPlainText(message)


class TifImportDialog(QtWidgets.QDialog):
    """End-to-end TIFF calcium-trace importer."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        cellpose_import: CellposeImportResult | None = None,
        stim_arrays: Mapping[str, StimArray] | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.experiment_name: str = "Calcium imaging (.tif)"
        self.experiment_duration: float = 0.0
        self.imported_data: list[Data] = []

        self._tif_paths: list[Path] = []
        self._seg_file_path: Path | None = None
        self._seg_file_background_image: npt.NDArray[Any] | None = None
        self._cellpose_import = cellpose_import
        self._stim_arrays = dict(stim_arrays or {})
        self._pipeline_result: TifPipelineResult | None = None
        self.new_cellpose_result: CellposeImportResult | None = None
        self._propagate_new_segmentation: bool = False
        self._new_segmentation_path: Path | None = None

        self._thread: QtCore.QThread | None = None
        self._worker: _PipelineWorker | None = None
        self._progress_dialog: _PipelineProgressDialog | None = None
        self._wheel_passthrough_widgets: set[QtCore.QObject] = set()

        self._setup_ui()

    # -- UI -----------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Import calcium imaging (.tif)")
        self.setModal(True)
        self.resize(560, 520)

        layout = QtWidgets.QVBoxLayout(self)
        self._scroll_area = QtWidgets.QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self._scroll_content)
        self._scroll_area.setWidget(self._scroll_content)
        layout.addWidget(self._scroll_area, stretch=1)

        instructions = QtWidgets.QLabel(
            "Extract calcium traces from TIFF stacks using Cellpose-derived ROIs. "
            "Choose Suite2p for neuropil correction and deconvolution, or simple "
            "ROI averaging for raw mean fluorescence."
        )
        instructions.setWordWrap(True)
        content_layout.addWidget(instructions)

        workflow_box = QtWidgets.QGroupBox("Recommended workflow for large TIFF stacks")
        workflow_layout = QtWidgets.QVBoxLayout(workflow_box)
        workflow_label = QtWidgets.QLabel(
            "Raw 2-photon stacks are often >1 GB and will crash Cellpose. "
            "Segment a projection of the stack instead:\n"
            "  1. Add your TIFF below and click <b>Open Fiji</b> to open it.\n"
            "  2. In Fiji: <b>Image → Stacks → Z Project…</b> "
            "(choose <i>Max Intensity</i> for sparse activity or "
            "<i>Mean Intensity</i> for dense activity).\n"
            "  3. <b>File → Save As → Tiff…</b> the projection (e.g. "
            "<code>MAX_&lt;name&gt;.tif</code>).\n"
            "  4. Click <b>Open Cellpose</b>, open the projection, segment, "
            "and save the <code>_seg.npy</code>.\n"
            "  5. Back here, pick <b>Use a Cellpose _seg.npy file</b> and "
            "load the file. Suite2p will then run on the full raw stack."
        )
        workflow_label.setWordWrap(True)
        workflow_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        workflow_layout.addWidget(workflow_label)
        content_layout.addWidget(workflow_box)

        # TIFF selection
        tif_group = QtWidgets.QGroupBox("TIFF file(s)")
        tif_layout = QtWidgets.QVBoxLayout(tif_group)
        self._tif_list = QtWidgets.QListWidget()
        self._tif_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        tif_layout.addWidget(self._tif_list)
        tif_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add TIFF…")
        add_btn.clicked.connect(self._browse_tifs)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_tifs)
        self._run_fiji_button = QtWidgets.QPushButton("Open Fiji")
        self._run_fiji_button.setToolTip(
            "Launch Fiji/ImageJ; if a TIFF is selected, open it for inspection."
        )
        self._run_fiji_button.clicked.connect(self._launch_fiji)
        tif_buttons.addWidget(add_btn)
        tif_buttons.addWidget(remove_btn)
        tif_buttons.addWidget(self._run_fiji_button)
        tif_buttons.addStretch(1)
        tif_layout.addLayout(tif_buttons)
        content_layout.addWidget(tif_group)

        # Cellpose mask source
        mask_group = QtWidgets.QGroupBox("Cellpose masks")
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        self._mask_source_buttons = QtWidgets.QButtonGroup(self)
        self._use_loaded_radio = QtWidgets.QRadioButton(
            "Use Cellpose masks from the current experiment with mapping applied"
        )
        self._use_seg_radio = QtWidgets.QRadioButton("Use a Cellpose _seg.npy file")
        self._auto_detect_radio = QtWidgets.QRadioButton(
            "Auto-detect ROIs with Suite2p (no Cellpose masks)"
        )
        self._mask_source_buttons.addButton(self._use_loaded_radio, 0)
        self._mask_source_buttons.addButton(self._use_seg_radio, 1)
        self._mask_source_buttons.addButton(self._auto_detect_radio, 2)
        mask_layout.addWidget(self._use_loaded_radio)
        mask_layout.addWidget(self._use_seg_radio)
        mask_layout.addWidget(self._auto_detect_radio)
        self._auto_detect_radio.toggled.connect(self._update_pipeline_fields)
        seg_row = QtWidgets.QHBoxLayout()
        self._seg_path_label = QtWidgets.QLabel("No _seg.npy selected")
        self._seg_path_label.setStyleSheet("QLabel { color: gray; }")
        browse_seg_btn = QtWidgets.QPushButton("Browse…")
        browse_seg_btn.clicked.connect(self._browse_seg_file)
        seg_row.addWidget(self._seg_path_label, stretch=1)
        seg_row.addWidget(browse_seg_btn)
        mask_layout.addLayout(seg_row)

        run_cellpose_row = QtWidgets.QHBoxLayout()
        run_cellpose_row.addStretch(1)
        self._run_cellpose_button = QtWidgets.QPushButton("Open Cellpose")
        self._run_cellpose_button.setToolTip(
            "Launch the Cellpose GUI to segment a mean projection, then load the "
            "resulting _seg.npy above."
        )
        self._run_cellpose_button.clicked.connect(self._launch_cellpose)
        run_cellpose_row.addWidget(self._run_cellpose_button)
        mask_layout.addLayout(run_cellpose_row)
        if self._cellpose_import is not None:
            self._use_loaded_radio.setChecked(True)
        else:
            self._use_loaded_radio.setEnabled(False)
            self._use_loaded_radio.setText(
                "Use Cellpose masks from the current experiment with mapping applied "
                "(none loaded)"
            )
            self._use_seg_radio.setChecked(True)
        content_layout.addWidget(mask_group)

        # Acquisition / pipeline parameters
        params_group = QtWidgets.QGroupBox("Acquisition & pipeline parameters")
        params_layout = QtWidgets.QFormLayout(params_group)

        self._pipeline_combo = QtWidgets.QComboBox()
        self._pipeline_combo.addItem("Suite2p", "suite2p")
        self._pipeline_combo.addItem("Simply average from ROI", "roi_average")
        self._pipeline_combo.setCurrentIndex(1)
        params_layout.addRow("Pipeline:", self._pipeline_combo)

        self._fs_spin = QtWidgets.QDoubleSpinBox()
        self._fs_spin.setRange(0.1, 1000.0)
        self._fs_spin.setDecimals(2)
        self._fs_spin.setValue(30.0)
        self._fs_spin.setSuffix(" Hz")
        params_layout.addRow("Frame rate (fs):", self._fs_spin)

        self._tau_spin = QtWidgets.QDoubleSpinBox()
        self._tau_spin.setRange(0.05, 10.0)
        self._tau_spin.setDecimals(3)
        self._tau_spin.setValue(0.5)
        self._tau_spin.setSuffix(" s")
        self._tau_spin.setToolTip(
            "Calcium indicator decay constant (e.g. jGCaMP8m ≈ 0.5 s, GCaMP6s ≈ 1.25 s)"
        )
        params_layout.addRow("Indicator tau:", self._tau_spin)

        self._nchannels_spin = QtWidgets.QSpinBox()
        self._nchannels_spin.setRange(1, 2)
        self._nchannels_spin.setValue(2)
        params_layout.addRow("# channels:", self._nchannels_spin)

        self._functional_chan_spin = QtWidgets.QSpinBox()
        self._functional_chan_spin.setRange(1, 2)
        self._functional_chan_spin.setValue(1)
        self._functional_chan_spin.setToolTip("1-indexed (Suite2p convention)")
        params_layout.addRow("Functional channel:", self._functional_chan_spin)
        self._nchannels_spin.valueChanged.connect(self._functional_chan_spin.setMaximum)

        self._neucoeff_spin = QtWidgets.QDoubleSpinBox()
        self._neucoeff_spin.setRange(0.0, 1.0)
        self._neucoeff_spin.setSingleStep(0.05)
        self._neucoeff_spin.setDecimals(2)
        self._neucoeff_spin.setValue(0.7)
        params_layout.addRow("Neuropil coefficient:", self._neucoeff_spin)

        self._diameter_spin = QtWidgets.QDoubleSpinBox()
        self._diameter_spin.setRange(2.0, 200.0)
        self._diameter_spin.setDecimals(1)
        self._diameter_spin.setValue(12.0)
        self._diameter_spin.setSuffix(" px")
        self._diameter_spin.setToolTip(
            "Approximate soma diameter in pixels, used by Suite2p's own "
            "ROI detector when no Cellpose masks are supplied."
        )
        params_layout.addRow("Cell diameter (auto-detect):", self._diameter_spin)

        self._spike_threshold_spin = QtWidgets.QDoubleSpinBox()
        self._spike_threshold_spin.setRange(0.0, 1.0)
        self._spike_threshold_spin.setSingleStep(0.01)
        self._spike_threshold_spin.setDecimals(3)
        self._spike_threshold_spin.setValue(0.02)
        params_layout.addRow("Spike threshold:", self._spike_threshold_spin)

        self._suppress_stimulation_artefact_checkbox = QtWidgets.QCheckBox(
            "Input artefact suppression"
        )
        self._suppress_stimulation_artefact_checkbox.setChecked(True)
        self._suppress_stimulation_artefact_checkbox.setToolTip(
            "Replace fluorescence in frames overlapping active input with "
            "the mean of the nearest clean values before and after the artefact."
        )
        params_layout.addRow(self._suppress_stimulation_artefact_checkbox)

        self._stim_array_combo = QtWidgets.QComboBox()
        for name, stim_array in self._stim_arrays.items():
            self._stim_array_combo.addItem(name, stim_array)
        params_layout.addRow("Input array:", self._stim_array_combo)
        has_stim_arrays = bool(self._stim_arrays)
        self._suppress_stimulation_artefact_checkbox.setEnabled(has_stim_arrays)
        if not has_stim_arrays:
            self._suppress_stimulation_artefact_checkbox.setToolTip(
                "No input arrays are available in the current project."
            )
        self._suppress_stimulation_artefact_checkbox.toggled.connect(
            self._update_pipeline_fields
        )

        self._params_layout = params_layout
        self._suite2p_parameter_widgets = [
            self._tau_spin,
            self._neucoeff_spin,
            self._spike_threshold_spin,
        ]
        self._auto_detect_parameter_widgets = [self._diameter_spin]
        self._pipeline_combo.currentIndexChanged.connect(self._on_pipeline_changed)

        content_layout.addWidget(params_group)

        # Run / Cancel
        button_layout = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run pipeline")
        self._run_button.clicked.connect(self._start_pipeline)
        button_layout.addWidget(self._run_button)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        button_layout.addWidget(cancel)
        layout.addLayout(button_layout)
        self._install_wheel_scroll_passthrough(
            [
                self._pipeline_combo,
                self._fs_spin,
                self._tau_spin,
                self._nchannels_spin,
                self._functional_chan_spin,
                self._neucoeff_spin,
                self._diameter_spin,
                self._spike_threshold_spin,
            ]
        )
        self._on_pipeline_changed()

    def _install_wheel_scroll_passthrough(
        self, widgets: list[QtWidgets.QWidget]
    ) -> None:
        for widget in widgets:
            self._wheel_passthrough_widgets.add(widget)
            widget.installEventFilter(self)
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                line_edit = widget.lineEdit()
                if line_edit is not None:
                    self._wheel_passthrough_widgets.add(line_edit)
                    line_edit.installEventFilter(self)

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            event is not None
            and watched is not None
            and event.type() == QtCore.QEvent.Type.Wheel
            and watched in self._wheel_passthrough_widgets
        ):
            wheel_event = cast(QtGui.QWheelEvent, event)
            self._scroll_from_wheel_event(wheel_event)
            return True
        return super().eventFilter(watched, event)

    def _scroll_from_wheel_event(self, event: QtGui.QWheelEvent) -> None:
        delta = event.pixelDelta().y()
        if delta == 0:
            delta = event.angleDelta().y()
        if delta == 0:
            return
        scroll_bar = self._scroll_area.verticalScrollBar()
        if scroll_bar is None:
            return
        scroll_bar.setValue(scroll_bar.value() - delta)

    def _update_pipeline_fields(self) -> None:
        """Hide parameters unrelated to the active pipeline/mask source."""
        auto_detect = self._auto_detect_radio.isChecked()
        if auto_detect:
            index = self._pipeline_combo.findData("suite2p")
            if index >= 0:
                self._pipeline_combo.setCurrentIndex(index)

        suite2p = self._pipeline_combo.currentData() == "suite2p"
        for widget in self._suite2p_parameter_widgets:
            widget.setVisible(suite2p)
            label = self._params_layout.labelForField(widget)
            if label is not None:
                label.setVisible(suite2p)

        for widget in self._auto_detect_parameter_widgets:
            widget.setVisible(auto_detect)
            label = self._params_layout.labelForField(widget)
            if label is not None:
                label.setVisible(auto_detect)

        self._pipeline_combo.setEnabled(not auto_detect)

        roi_average = not suite2p
        self._suppress_stimulation_artefact_checkbox.setVisible(roi_average)
        stim_label = self._params_layout.labelForField(self._stim_array_combo)
        show_stim_array = (
            roi_average
            and self._suppress_stimulation_artefact_checkbox.isChecked()
            and bool(self._stim_arrays)
        )
        self._stim_array_combo.setVisible(show_stim_array)
        if stim_label is not None:
            stim_label.setVisible(show_stim_array)

    def _on_pipeline_changed(self) -> None:
        """Apply pipeline-specific acquisition defaults and update the form."""
        if self._pipeline_combo.currentData() == "roi_average":
            self._nchannels_spin.setValue(1)
        self._update_pipeline_fields()

    # -- File pickers -------------------------------------------------------
    def _browse_tifs(self) -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TIFF stacks",
            "",
            "TIFF files (*.tif *.tiff);;All files (*)",
        )
        for filename in filenames:
            path = Path(filename)
            if path not in self._tif_paths:
                self._tif_paths.append(path)
                self._tif_list.addItem(str(path))

    def _remove_selected_tifs(self) -> None:
        for item in self._tif_list.selectedItems():
            row = self._tif_list.row(item)
            self._tif_list.takeItem(row)
            del self._tif_paths[row]

    def _launch_fiji(self) -> None:
        """Launch Fiji/ImageJ, opening the first selected TIFF if one is available."""
        from neuroseq.fiji import resolve_fiji_path

        fiji_path = resolve_fiji_path(self)
        if not fiji_path:
            return

        args: list[str] = []
        selected = self._tif_list.selectedItems()
        if selected:
            row = self._tif_list.row(selected[0])
            args.append(str(self._tif_paths[row]))
        elif self._tif_paths:
            args.append(str(self._tif_paths[0]))

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

    def _launch_cellpose(self) -> None:
        """Launch the Cellpose GUI as a detached subprocess (same pattern as the cellpose dialog)."""
        if importlib.util.find_spec("cellpose") is None:
            QtWidgets.QMessageBox.information(
                self,
                "Install Cellpose",
                "Cellpose is not installed in the current Python environment.",
            )
            return
        try:
            subprocess.Popen(
                [sys.executable, "-m", "cellpose"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Explain (once) if Cellpose's GPU option will be unavailable.
            from neuroseq.gpu import warn_if_gpu_unavailable

            warn_if_gpu_unavailable(self)
        except Exception as exc:
            self.logger.exception("Failed to launch Cellpose")
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to launch Cellpose",
                f"Could not launch Cellpose:\n{exc}",
            )

    def _browse_seg_file(self) -> None:
        filename, _ = getopenfilename(
            self,
            "Select Cellpose segmentation file",
            "",
            "Cellpose segmentation (*_seg.npy *.npy);;All files (*)",
        )
        if filename:
            self._seg_file_path = Path(filename)
            self._seg_path_label.setText(self._seg_file_path.name)
            self._seg_path_label.setStyleSheet("QLabel { color: black; }")
            self._use_seg_radio.setChecked(True)

    # -- Mask resolution ----------------------------------------------------
    def _resolve_masks(self) -> npt.NDArray[Any] | None:
        if self._auto_detect_radio.isChecked():
            return None
        if self._use_loaded_radio.isChecked():
            if self._cellpose_import is None:
                raise ValueError("No Cellpose data is loaded in the current experiment")
            return _cellpose_result_to_masks(self._cellpose_import)
        if self._seg_file_path is None:
            raise ValueError("No _seg.npy file selected")
        self._seg_file_background_image = _load_seg_file_background_image(
            self._seg_file_path
        )
        return _load_masks_from_seg_file(self._seg_file_path)

    def _selected_mapping(self) -> dict[int, int] | None:
        """Return the experiment mapping only for its stored Cellpose masks."""
        if not self._use_loaded_radio.isChecked() or self._cellpose_import is None:
            return None
        mapping = self._cellpose_import.stimulus_to_cellpose_mapping
        return mapping if isinstance(mapping, dict) else None

    # -- Pipeline launch ----------------------------------------------------
    def _start_pipeline(self) -> None:
        if not self._tif_paths:
            QtWidgets.QMessageBox.warning(
                self, "No TIFFs", "Add at least one TIFF stack before running."
            )
            return
        self.new_cellpose_result = None
        self._propagate_new_segmentation = False
        self._new_segmentation_path = None
        self._seg_file_background_image = None
        try:
            masks = self._resolve_masks()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cellpose masks required", str(exc))
            return

        if masks is None:
            proceed = QtWidgets.QMessageBox.question(
                self,
                "No cell segmentation",
                "No Cellpose segmentation was provided. Suite2p will "
                "auto-detect ROIs directly from the raw fluorescence movie "
                "instead, which is typically less reliable than a "
                "Cellpose-verified segmentation.\n\n"
                "Proceed without cell segmentation?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if proceed != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._propagate_new_segmentation = not self._use_loaded_radio.isChecked()
        if self._propagate_new_segmentation and self._use_seg_radio.isChecked():
            self._new_segmentation_path = self._seg_file_path

        params = TifPipelineParams(
            fs=self._fs_spin.value(),
            tau=self._tau_spin.value(),
            nchannels=self._nchannels_spin.value(),
            functional_chan=self._functional_chan_spin.value(),
            neuropil_coefficient=self._neucoeff_spin.value(),
            diameter=self._diameter_spin.value(),
        )

        self._run_button.setEnabled(False)
        self._progress_dialog = _PipelineProgressDialog(
            "Starting trace extraction…", "Cancel", 0, 100, self
        )
        self._progress_dialog.canceled.connect(self._cancel_pipeline)
        self._progress_dialog.show()

        self._thread = QtCore.QThread(self)
        pipeline = str(self._pipeline_combo.currentData())
        stim_array = None
        if (
            pipeline == "roi_average"
            and self._suppress_stimulation_artefact_checkbox.isChecked()
        ):
            stim_array = self._stim_array_combo.currentData()
        self._worker = _PipelineWorker(
            list(self._tif_paths), masks, params, pipeline, stim_array
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._progress_dialog.appendLog)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    @QtCore.Slot(str, float)
    def _on_progress(self, message: str, fraction: float) -> None:
        if self._progress_dialog is None:
            return
        self._progress_dialog.setLabelText(message)
        self._progress_dialog.setValue(int(max(0.0, min(1.0, fraction)) * 100))

    @QtCore.Slot(object)
    def _on_finished(self, result: TifPipelineResult) -> None:
        self._pipeline_result = result
        if self._progress_dialog is not None:
            self._progress_dialog.close()
        try:
            self._finalize_import(result)
        except Exception as exc:
            self.logger.exception("Error finalising TIFF import")
            QtWidgets.QMessageBox.critical(
                self, "Import error", f"Pipeline succeeded but import failed:\n{exc}"
            )
            self._run_button.setEnabled(True)

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
        QtWidgets.QMessageBox.critical(self, "Pipeline failed", message)
        self._run_button.setEnabled(True)

    def _cancel_pipeline(self) -> None:
        # Suite2p is not cooperatively cancellable from outside; we let it finish
        # but mark the dialog as rejected so results are dropped.
        if self._thread is not None and self._thread.isRunning():
            self.logger.info("Pipeline cancel requested — results will be discarded")
            self._worker = None  # results will be ignored on finished signal
        if self._progress_dialog is not None:
            self._progress_dialog.close()
        self.reject()

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._worker = None

    # -- Finalisation -------------------------------------------------------
    def _finalize_import(self, result: TifPipelineResult) -> None:
        fs = self._fs_spin.value()
        dt = 1.0 / fs
        n_neurons, n_timepoints = result.F.shape
        self.experiment_duration = n_timepoints * dt

        if self._propagate_new_segmentation:
            self.new_cellpose_result = _build_cellpose_result_from_pipeline(
                result,
                self._tif_paths[0],
                segmentation_path=self._new_segmentation_path,
                background_image=self._seg_file_background_image,
            )

        if result.extra.get("pipeline") == "roi_average":
            from neuroseq.data import TraceData

            indices = _resolve_suite2p_indices(n_neurons, self._selected_mapping())
            traces = TraceData(
                "Simply averaged ROI traces",
                result.F,
                indices,
                "fluorescence",
                dt,
            )
            self.imported_data = [traces]
            QtWidgets.QMessageBox.information(
                self,
                "Import successful",
                f"Imported {n_neurons} neurons × {n_timepoints} frames "
                f"({self.experiment_duration:.2f} s at {fs:.1f} Hz).",
            )
            self.accept()
            return

        traces = prepare_suite2p_traces(
            result.F,
            result.Fneu,
            max_neurons=n_neurons,
            sampling_rate=fs,
            stimulus_to_cellpose_mapping=self._selected_mapping(),
        )

        spike_data = prepare_suite2p_spikes(
            result.spks,
            dt=dt,
            threshold=self._spike_threshold_spin.value(),
            max_neurons=n_neurons,
            stimulus_to_cellpose_mapping=self._selected_mapping(),
        )

        self.imported_data = [traces, spike_data]
        QtWidgets.QMessageBox.information(
            self,
            "Import successful",
            f"Imported {n_neurons} neurons × {n_timepoints} frames "
            f"({self.experiment_duration:.2f} s at {fs:.1f} Hz).",
        )
        self.accept()
