from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from neuroseq.data import TraceData
from neuroseq.gui import Gui
from neuroseq.import_experiment_dialog import ImportExperimentDialog
from neuroseq.patchclamp_import_dialog import PatchClampImportDialog


def _suppress_message_boxes(monkeypatch) -> None:
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *_, **__: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_, **__: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_, **__: None)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_, **__: QtWidgets.QMessageBox.StandardButton.Yes,
    )


def _get_patchclamp_trace(experiment) -> TraceData:
    return next(
        data
        for data in experiment.data
        if isinstance(data, TraceData) and data.name == "Patch Clamp Traces"
    )


def _run_patchclamp_imports(gui, monkeypatch, stim_ids, steps) -> None:
    _suppress_message_boxes(monkeypatch)
    stim_iter = iter(stim_ids)
    step_iter = iter(steps)

    def _patchclamp_exec(self):
        self.load_file("tests/sampledata/3patchclampneurons.npy")
        self.custom_mapping_line.setText(next(stim_iter))
        self.sampling_rate_spin.setValue(1000.0)
        self.peak_height_spin.setValue(2.0)
        self.peak_prominence_spin.setValue(1.0)
        self.accept = lambda: None
        self.import_data()
        return QtWidgets.QDialog.DialogCode.Accepted

    def _experiment_exec(self):
        step = next(step_iter)
        self.append_to_experiment = step.get("append", False)
        if self.append_to_experiment:
            self.append_experiment_name = step.get(
                "name", self.existing_experiments[0].name
            )
            self.offset = step.get("offset", 0.0)
        else:
            name = step.get("name")
            if name:
                self.experiment.name = name
        return True

    monkeypatch.setattr(PatchClampImportDialog, "exec", _patchclamp_exec)
    monkeypatch.setattr(ImportExperimentDialog, "exec", _experiment_exec)
    for _ in stim_ids:
        gui.btn_import_patchclamp_data_clicked()


def test_gui_patchclamp_import_defaults_experiment_name(qtbot, monkeypatch):
    # Latest PR will change this data name anywya, but very key for checking where replaced/appended data is added for troubel shooting
    gui = Gui()
    qtbot.addWidget(gui)
    _run_patchclamp_imports(gui, monkeypatch, ["0, 1, 2"], [{"append": False}])
    trace = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import"])
    assert "PatchClamp Import" in gui.project.experiments
    assert trace.traces.shape[0] == 3
    assert np.array_equal(trace.indices, np.array([0, 1, 2], dtype=np.uint32))


def test_gui_patchclamp_append_same_ids_no_change(qtbot, monkeypatch):
    gui = Gui()
    qtbot.addWidget(gui)
    _run_patchclamp_imports(gui, monkeypatch, ["0, 1, 2"], [{"append": False}])
    trace = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import"])
    before_traces, before_indices = trace.traces.copy(), trace.indices.copy()
    _run_patchclamp_imports(
        gui,
        monkeypatch,
        ["0, 1, 2"],
        [{"append": True, "name": "PatchClamp Import", "offset": 0.0}],
    )
    trace_after = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import"])
    assert np.array_equal(trace_after.traces, before_traces)
    assert np.array_equal(trace_after.indices, before_indices)
    assert trace_after.dt == trace.dt


def test_gui_patchclamp_import_new_ids_keep_existing(qtbot, monkeypatch):
    gui = Gui()
    qtbot.addWidget(gui)
    _run_patchclamp_imports(gui, monkeypatch, ["0, 1, 2"], [{"append": False}])
    _run_patchclamp_imports(
        gui,
        monkeypatch,
        ["7, 8, 9"],
        [{"append": False, "name": "PatchClamp Import 2"}],
    )
    trace_a = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import"])
    trace_b = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import 2"])
    assert np.array_equal(trace_a.indices, np.array([0, 1, 2], dtype=np.uint32))
    assert np.array_equal(trace_b.indices, np.array([7, 8, 9], dtype=np.uint32))
    assert trace_a.traces.shape[0] == 3  # original traces are still there
    assert trace_b.traces.shape[0] == 3  # new traces are there


def test_gui_patchclamp_append_doubles_duration(qtbot, monkeypatch):
    gui = Gui()
    qtbot.addWidget(gui)
    raw = np.load("tests/sampledata/3patchclampneurons.npy")
    duration = raw.shape[1] / 1000.0
    _run_patchclamp_imports(gui, monkeypatch, ["0, 1, 2"], [{"append": False}])
    _run_patchclamp_imports(
        gui,
        monkeypatch,
        ["0, 1, 2"],
        [{"append": True, "name": "PatchClamp Import", "offset": duration}],
    )
    trace = _get_patchclamp_trace(gui.project.experiments["PatchClamp Import"])
    expected_timepoints = raw.shape[1] * 2
    expected_indices = np.array([0, 1, 2], dtype=np.uint32)
    trace_first_timepoint = trace.traces[:, 0]
    trace_last_timepoint = trace.traces[:, -1]
    raw_first_timepoint = raw[:, 0]
    raw_last_timepoint = raw[:, -1]

    assert trace.traces.shape[1] == expected_timepoints
    assert np.array_equal(trace.indices, expected_indices)
    assert np.array_equal(trace_first_timepoint, raw_first_timepoint)
    assert np.array_equal(trace_last_timepoint, raw_last_timepoint)


# next: need to assert errors occur when you try to extend with two specific neurons (only if Andrey doesn't want this).
