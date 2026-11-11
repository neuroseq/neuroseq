from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_suite2p import compute_normalized_suite2p_traces
from neuroseq.importers.suite2p_import_dialog import Suite2pImportDialog
from tests.wait_timeout import WAIT_TIMEOUT_MS


class TestSuite2pImportDialog:
    """Test cases for the combined Suite2p import dialog"""

    def test_dialog_initialization(self, qtbot: QtBot):
        """Test that the dialog initializes correctly with all UI elements"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        assert dialog.f_file_path_label is not None
        assert dialog.import_button is not None

        assert not dialog.import_button.isEnabled()
        assert dialog.max_neurons_spin.value() == 2
        assert dialog.spike_threshold_spin.value() == 0.02
        assert dialog.sampling_rate_spin.value() == 30.0

    def test_file_loading_and_preview(self, qtbot: QtBot, suite2p_import_data_dir):
        """Test loading F.npy file and showing preview"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        test_f_path = test_data_dir / "F.npy"
        dialog.load_f_file(str(test_f_path))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        assert dialog.import_button.isEnabled()
        preview = dialog.preview_text.toPlainText()
        assert "Contrasting traces: 5 neurons × 900 timepoints" in preview

    def test_data_import_with_valid_files(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test importing data from F.npy and Fneu.npy files"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        # only get traces:
        assert len(dialog.imported_data) == 1
        assert dialog.experiment_duration == pytest.approx(30.0)  # 900 frames at 30 Hz
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        assert traces.shape == (5, 900)  # 5 neurons, 900 time steps
        assert np.all((traces >= 0.0) & (traces <= 1.0))
        assert np.all(np.isfinite(traces))  # no NaN or inf values
        assert dialog.f_data is not None
        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)
        np.testing.assert_allclose(traces, expected_traces[: traces.shape[0], :])
        assert trace_data.dt == pytest.approx(1.0 / 30.0)

    def test_contrasting_calculation(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test that F - 0.7 * Fneu calculation is performed correctly"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))

        assert dialog.f_data is not None
        assert dialog.fneu_data is not None
        f_data = dialog.f_data
        fneu_data = dialog.fneu_data
        expected_contrasted, _, _ = compute_normalized_suite2p_traces(f_data, fneu_data)

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        assert len(dialog.imported_data) == 1
        assert dialog.experiment_duration == pytest.approx(30.0)  # 900 frames at 30 Hz
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        np.testing.assert_array_almost_equal(traces, expected_contrasted, decimal=5)

    def test_import_without_fneu_file(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """F.npy-only imports should normalize fluorescence without neuropil subtraction."""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))

        assert dialog.import_button.isEnabled()

        dialog.import_data()
        assert len(qmessagebox_calls) == 1
        assert "successful" in qmessagebox_calls[-1]["title"].lower()

        assert len(dialog.imported_data) == 1
        assert dialog.experiment_duration == pytest.approx(30.0)  # 900 frames at 30 Hz
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        traces = trace_data.traces
        assert dialog.f_data is not None
        f_data = dialog.f_data
        expected_traces, _, _ = compute_normalized_suite2p_traces(f_data, None)
        np.testing.assert_allclose(traces, expected_traces[: traces.shape[0], :])

    def test_optional_spike_import_with_threshold(
        self, qtbot: QtBot, suite2p_import_data_dir, qmessagebox_calls: list
    ):
        """Test importing optional spks file with adjustable threshold"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        test_data_dir = suite2p_import_data_dir
        dialog.load_f_file(str(test_data_dir / "F.npy"))
        dialog.load_fneu_file(str(test_data_dir / "Fneu.npy"))
        dialog.load_spks_file(str(test_data_dir / "spks.npy"))

        dialog.spike_threshold_spin.setValue(
            0.02
        )  # realistic default, this will leave 36 spikes.

        dialog.import_data()
        assert dialog.experiment_duration == pytest.approx(30.0)  # 900 frames at 30 Hz
        assert len(dialog.imported_data) == 2
        trace_data = dialog.imported_data[0]
        assert isinstance(trace_data, TraceData)
        spike_data = dialog.imported_data[1]
        assert isinstance(spike_data, SpikeData)
        assert len(spike_data.spike_indices) > 0
        assert len(spike_data.spike_times) == len(spike_data.spike_indices)

    def test_toggle_individual_file_imports(self, qtbot: QtBot):
        """Test that the toggle button correctly shows/hides individual file imports"""
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)

        # Initially individual file imports should be hidden
        assert not dialog.individual_files_group.isVisible()
        assert (
            dialog.show_individual_files_button.text() == "Show individual file imports"
        )

        # Click to show individual file imports
        qtbot.mouseClick(dialog.show_individual_files_button, Qt.MouseButton.LeftButton)
        assert dialog.individual_files_group.isVisible()
        assert (
            dialog.show_individual_files_button.text() == "Hide individual file imports"
        )

        # Click again to hide individual file imports
        qtbot.mouseClick(dialog.show_individual_files_button, Qt.MouseButton.LeftButton)
        assert not dialog.individual_files_group.isVisible()
        assert (
            dialog.show_individual_files_button.text() == "Show individual file imports"
        )

    def test_load_data_directory_autofills_files(
        self,
        qtbot: QtBot,
        suite2p_import_data_dir,
        qmessagebox_calls: list,
    ) -> None:
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        detected = dialog.load_data_directory(suite2p_import_data_dir)
        assert detected == ["f", "fneu", "spks", "stat"]
        assert dialog.f_file_path is not None
        assert dialog.fneu_file_path is not None
        assert dialog.spks_file_path is not None
        assert dialog.stat_data is not None
        # one info message for autofill
        assert qmessagebox_calls[0]["name"] == "information"
        assert "Auto-filled" in qmessagebox_calls[0]["title"]

    def test_loading_from_csv_files(
        self,
        qtbot: QtBot,
        suite2p_import_data_dir,
        qmessagebox_calls: list,
    ) -> None:
        dialog = Suite2pImportDialog()
        qtbot.addWidget(dialog)

        dialog.load_f_file(str(suite2p_import_data_dir / "F.csv"))
        dialog.load_fneu_file(str(suite2p_import_data_dir / "Fneu.csv"))
        dialog.load_spks_file(str(suite2p_import_data_dir / "spks.csv"))

        assert dialog.import_button.isEnabled()
        dialog.import_data()
        # success message only (no folder auto fill)
        assert qmessagebox_calls[-1]["name"] == "information"
        assert "successful" in qmessagebox_calls[-1]["title"].lower()
        assert dialog.experiment_duration == pytest.approx(30.0)  # 900 frames at 30 Hz
        assert len(dialog.imported_data) == 2
        assert isinstance(dialog.imported_data[0], TraceData)
        assert dialog.imported_data[0].traces.shape == (5, 900)
        assert isinstance(dialog.imported_data[1], SpikeData)
        assert len(dialog.imported_data[1].spike_indices) > 0

    def test_import_maps_rows_through_the_project_mapping(
        self,
        qtbot: QtBot,
        suite2p_import_data_dir,
        qmessagebox_calls: list,
    ) -> None:
        """Indices are mapped at import, so a later remap keeps them unchanged."""
        mapping = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
        dialog = Suite2pImportDialog(stimulus_to_cellpose_mapping=mapping)
        qtbot.addWidget(dialog)

        dialog.load_f_file(str(suite2p_import_data_dir / "F.csv"))
        dialog.load_fneu_file(str(suite2p_import_data_dir / "Fneu.csv"))
        dialog.load_spks_file(str(suite2p_import_data_dir / "spks.csv"))
        dialog.import_data()

        traces, spikes = dialog.imported_data
        assert isinstance(traces, TraceData)
        assert isinstance(spikes, SpikeData)
        assert traces.raw_cellpose_ids is not None
        assert spikes.raw_cellpose_ids is not None
        assert list(traces.raw_cellpose_ids) == [0, 1, 2, 3, 4]
        assert list(traces.indices) == [4, 3, 2, 1, 0]
        assert np.array_equal(spikes.spike_indices, 4 - spikes.raw_cellpose_ids)


def test_workspace_passes_the_project_mapping_to_the_suite2p_dialog(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtpy import QtWidgets

    from neuroseq.project_workspace import ProjectWorkspace

    seen: dict[str, Any] = {}

    class _FakeDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr("neuroseq.project_workspace.Suite2pImportDialog", _FakeDialog)
    gui = ProjectWorkspace()
    qtbot.addWidget(gui)
    gui.project.cellpose_data = CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=np.array([[0, 1]], dtype=np.int32),
        overlay_image=np.zeros((1, 2, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=np.array([0, 1], dtype=np.int32),
        stimulus_to_cellpose_mapping={0: 1, 1: 0},
    )

    gui.btn_import_suite2p_data_clicked()

    assert seen["stimulus_to_cellpose_mapping"] == {0: 1, 1: 0}


class TestSuite2pDataValidation:
    """Test data validation for Suite2p imports"""

    def test_data_dimensions_validation(self, suite2p_import_data_dir):
        """Test that imported data has correct dimensions"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape == (5, 900)
        assert fneu_data.shape == (5, 900)
        assert f_data.shape == fneu_data.shape

    def test_data_content_validation(self, suite2p_import_data_dir):
        """Test that data contains expected non-zero values"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert np.count_nonzero(f_data) > 0
        assert np.count_nonzero(fneu_data) > 0

        assert np.all(np.isfinite(f_data))
        assert np.all(np.isfinite(fneu_data))

        contrasted_data = f_data - 0.7 * fneu_data
        assert np.count_nonzero(contrasted_data) > 0

    def test_timesteps_length(self, suite2p_import_data_dir):
        """Test that data has exactly 900 timesteps"""
        test_data_dir = suite2p_import_data_dir

        f_data = np.load(test_data_dir / "F.npy")
        fneu_data = np.load(test_data_dir / "Fneu.npy")

        assert f_data.shape[1] == 900
        assert fneu_data.shape[1] == 900
