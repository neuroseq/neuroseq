"""Dialog for end-to-end import of 2-photon calcium imaging TIFFs.

Drives `importer_tif.run_suite2p_pipeline` in a background QThread and feeds the
resulting fluorescence / spike arrays into the existing Suite2p import path.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.compat import getopenfilename

from neuroseq.data import Data
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.importer_suite2p import prepare_suite2p_spikes
from neuroseq.importer_suite2p import prepare_suite2p_traces
from neuroseq.importer_tif import TifPipelineParams
from neuroseq.importer_tif import TifPipelineResult


def _cellpose_result_to_masks(result: CellposeImportResult) -> npt.NDArray[Any]:
    """Convert a stored CellposeImportResult (0-based, bg=-1) back to Suite2p layout (bg=0)."""
    seg = np.asarray(result.segmentation_array, dtype=np.int32)
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


class _PipelineWorker(QtCore.QObject):
    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(object)  # TifPipelineResult
    failed = QtCore.Signal(str)

    def __init__(
        self,
        tif_paths: list[Path],
        masks: npt.NDArray[Any],
        params: TifPipelineParams,
        pipeline: str = "suite2p",
    ) -> None:
        super().__init__()
        self._tif_paths = tif_paths
        self._masks = masks
        self._params = params
        self._pipeline = pipeline

    @QtCore.Slot()
    def run(self) -> None:
        try:
            # Imported here so test environments without suite2p can still import
            # this module.
            def callback(msg: str, frac: float) -> None:
                self.progress.emit(msg, float(frac))

            if self._pipeline == "roi_average":
                from neuroseq.importer_tif import run_roi_average_pipeline

                result = run_roi_average_pipeline(
                    self._tif_paths, self._masks, self._params, progress=callback
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


class TifImportDialog(QtWidgets.QDialog):
    """End-to-end TIFF calcium-trace importer."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        cellpose_import: CellposeImportResult | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.experiment_name: str = "Calcium imaging (.tif)"
        self.experiment_duration: float = 0.0
        self.imported_data: list[Data] = []

        self._tif_paths: list[Path] = []
        self._seg_file_path: Path | None = None
        self._cellpose_import = cellpose_import
        self._pipeline_result: TifPipelineResult | None = None

        self._thread: QtCore.QThread | None = None
        self._worker: _PipelineWorker | None = None
        self._progress_dialog: QtWidgets.QProgressDialog | None = None

        self._setup_ui()

    # -- UI -----------------------------------------------------------------
    def _setup_ui(self) -> None:
        self.setWindowTitle("Import calcium imaging (.tif)")
        self.setModal(True)
        self.resize(560, 520)

        layout = QtWidgets.QVBoxLayout(self)

        instructions = QtWidgets.QLabel(
            "Extract calcium traces from TIFF stacks using Cellpose-derived ROIs. "
            "Choose Suite2p for neuropil correction and deconvolution, or simple "
            "ROI averaging for raw mean fluorescence."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        workflow_box = QtWidgets.QGroupBox("Recommended workflow for large TIFF stacks")
        workflow_layout = QtWidgets.QVBoxLayout(workflow_box)
        workflow_label = QtWidgets.QLabel(
            "Raw 2-photon stacks are often >1 GB and will crash Cellpose. "
            "Segment a projection of the stack instead:\n"
            "  1. Add your TIFF below and click <b>Run Fiji</b> to open it.\n"
            "  2. In Fiji: <b>Image → Stacks → Z Project…</b> "
            "(choose <i>Max Intensity</i> for sparse activity or "
            "<i>Mean Intensity</i> for dense activity).\n"
            "  3. <b>File → Save As → Tiff…</b> the projection (e.g. "
            "<code>MAX_&lt;name&gt;.tif</code>).\n"
            "  4. Click <b>Run Cellpose</b>, open the projection, segment, "
            "and save the <code>_seg.npy</code>.\n"
            "  5. Back here, pick <b>Use a Cellpose _seg.npy file</b> and "
            "load the file. Suite2p will then run on the full raw stack."
        )
        workflow_label.setWordWrap(True)
        workflow_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        workflow_layout.addWidget(workflow_label)
        layout.addWidget(workflow_box)

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
        self._run_fiji_button = QtWidgets.QPushButton("Run Fiji")
        self._run_fiji_button.setToolTip(
            "Launch Fiji/ImageJ; if a TIFF is selected, open it for inspection."
        )
        self._run_fiji_button.clicked.connect(self._launch_fiji)
        self._change_fiji_button = QtWidgets.QPushButton("Change Fiji path…")
        self._change_fiji_button.setToolTip(
            "Choose a different Fiji/ImageJ executable (e.g. if the saved path "
            "is wrong)."
        )
        self._change_fiji_button.clicked.connect(self._change_fiji_path)
        tif_buttons.addWidget(add_btn)
        tif_buttons.addWidget(remove_btn)
        tif_buttons.addWidget(self._run_fiji_button)
        tif_buttons.addWidget(self._change_fiji_button)
        tif_buttons.addStretch(1)
        tif_layout.addLayout(tif_buttons)
        layout.addWidget(tif_group)

        # Cellpose mask source
        mask_group = QtWidgets.QGroupBox("Cellpose masks")
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        self._mask_source_buttons = QtWidgets.QButtonGroup(self)
        self._use_loaded_radio = QtWidgets.QRadioButton(
            "Use Cellpose masks from the current experiment"
        )
        self._use_seg_radio = QtWidgets.QRadioButton("Use a Cellpose _seg.npy file")
        self._mask_source_buttons.addButton(self._use_loaded_radio, 0)
        self._mask_source_buttons.addButton(self._use_seg_radio, 1)
        mask_layout.addWidget(self._use_loaded_radio)
        mask_layout.addWidget(self._use_seg_radio)
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
        self._run_cellpose_button = QtWidgets.QPushButton("Run Cellpose")
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
                "Use Cellpose masks from the current experiment (none loaded)"
            )
            self._use_seg_radio.setChecked(True)
        layout.addWidget(mask_group)

        # Acquisition / pipeline parameters
        params_group = QtWidgets.QGroupBox("Acquisition & pipeline parameters")
        params_layout = QtWidgets.QFormLayout(params_group)

        self._pipeline_combo = QtWidgets.QComboBox()
        self._pipeline_combo.addItem("Suite2p", "suite2p")
        self._pipeline_combo.addItem("Simply average from ROI", "roi_average")
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

        self._spike_threshold_spin = QtWidgets.QDoubleSpinBox()
        self._spike_threshold_spin.setRange(0.0, 1.0)
        self._spike_threshold_spin.setSingleStep(0.01)
        self._spike_threshold_spin.setDecimals(3)
        self._spike_threshold_spin.setValue(0.02)
        params_layout.addRow("Spike threshold:", self._spike_threshold_spin)

        self._params_layout = params_layout
        self._suite2p_parameter_widgets = [
            self._tau_spin,
            self._neucoeff_spin,
            self._spike_threshold_spin,
        ]
        self._pipeline_combo.currentIndexChanged.connect(self._update_pipeline_fields)

        layout.addWidget(params_group)

        # Run / Cancel
        button_layout = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run pipeline")
        self._run_button.clicked.connect(self._start_pipeline)
        button_layout.addWidget(self._run_button)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        button_layout.addWidget(cancel)
        layout.addLayout(button_layout)

    def _update_pipeline_fields(self) -> None:
        """Hide parameters unrelated to simple ROI averaging."""
        suite2p = self._pipeline_combo.currentData() == "suite2p"
        for widget in self._suite2p_parameter_widgets:
            widget.setVisible(suite2p)
            label = self._params_layout.labelForField(widget)
            if label is not None:
                label.setVisible(suite2p)

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

    def _change_fiji_path(self) -> None:
        """Let the user pick a different Fiji executable, even if one is set."""
        from neuroseq.fiji import resolve_fiji_path

        resolve_fiji_path(self, force_prompt=True)

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
    def _resolve_masks(self) -> npt.NDArray[Any]:
        if self._use_loaded_radio.isChecked():
            if self._cellpose_import is None:
                raise ValueError("No Cellpose data is loaded in the current experiment")
            return _cellpose_result_to_masks(self._cellpose_import)
        if self._seg_file_path is None:
            raise ValueError("No _seg.npy file selected")
        return _load_masks_from_seg_file(self._seg_file_path)

    # -- Pipeline launch ----------------------------------------------------
    def _start_pipeline(self) -> None:
        if not self._tif_paths:
            QtWidgets.QMessageBox.warning(
                self, "No TIFFs", "Add at least one TIFF stack before running."
            )
            return
        try:
            masks = self._resolve_masks()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cellpose masks required", str(exc))
            return

        params = TifPipelineParams(
            fs=self._fs_spin.value(),
            tau=self._tau_spin.value(),
            nchannels=self._nchannels_spin.value(),
            functional_chan=self._functional_chan_spin.value(),
            neuropil_coefficient=self._neucoeff_spin.value(),
        )

        self._run_button.setEnabled(False)
        self._progress_dialog = QtWidgets.QProgressDialog(
            "Starting trace extraction…", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.canceled.connect(self._cancel_pipeline)
        self._progress_dialog.show()

        self._thread = QtCore.QThread(self)
        pipeline = str(self._pipeline_combo.currentData())
        self._worker = _PipelineWorker(list(self._tif_paths), masks, params, pipeline)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
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

        if result.extra.get("pipeline") == "roi_average":
            from neuroseq.data import TraceData

            traces = TraceData(
                "Simply averaged ROI traces",
                result.F,
                np.arange(n_neurons, dtype=np.uint32),
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
        )

        spike_data = prepare_suite2p_spikes(
            result.spks,
            dt=dt,
            threshold=self._spike_threshold_spin.value(),
            max_neurons=n_neurons,
        )

        self.imported_data = [traces, spike_data]
        QtWidgets.QMessageBox.information(
            self,
            "Import successful",
            f"Imported {n_neurons} neurons × {n_timepoints} frames "
            f"({self.experiment_duration:.2f} s at {fs:.1f} Hz).",
        )
        self.accept()
