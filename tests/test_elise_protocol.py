"""Tests for the in-GUI ELiSe train/validation/replay protocol (Fig. 2)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.custom_analysis_widget import CustomAnalysisWindow
from neuroseq.example_projects import _build_fuer_elise_opsin_stims
from neuroseq.gui import Gui
from neuroseq.replay_metrics import save_learning_curves_csv
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

# Mirrors elise_learning_curves.CURVES_ATTR (the plugin loads via a file spec,
# not as an importable module, so we reference the attribute name directly).
_CURVES_ATTR = "_elise_protocol_curves"

_HAS_ELISE = importlib.util.find_spec("elise") is not None
_needs_elise = pytest.mark.skipif(
    not _HAS_ELISE, reason="requires the private 'elise' package"
)


# ---------- run_protocol (needs the elise package) ----------


@_needs_elise
def test_run_protocol_direct_mode_learns() -> None:
    from neuroseq.example_models import elise

    exc, inh = _build_fuer_elise_opsin_stims(tfinal_ms=1920)
    base = {k: v["default"] for k, v in elise.DEFAULT_PARAMS.items()}

    res = elise.run_protocol(
        exc.astype(float),
        inh.astype(float),
        stim_dt_s=0.001,
        base_params=base,
        teacher_mode="direct",
        training_epochs=8,
        training_cycles=15,
        validation_cycles=2,
        replay_epochs=3,
        replay_cycles=2,
    )

    assert res["num_vis"] == 13
    assert res["val_mse"].shape == (8,)
    assert res["val_corr"].shape == (8,)
    assert res["rep_mse"].shape == (3,)
    assert res["config"]["teacher_mode"] == "direct"
    assert res["target_rate"].shape[0] == 13
    assert res["final_replay_rate"].shape == (13, 2 * res["cycle_trace_samples"])

    # A standard replay Experiment is returned for import into the project so
    # the activity lines up with the opsin stim arrays.
    exp = res["experiment"]
    assert exp is not None
    trace_names = {d.name for d in exp.data}
    assert "firing_rate_r" in trace_names

    # The crisp direct teacher must actually drive learning.
    assert res["val_mse"][-1] < res["val_mse"][0]
    assert res["val_corr"][-1] > res["val_corr"][0]
    # Output rates stay in the sigmoid's [0, 1] range.
    assert np.all(res["final_replay_rate"] >= 0.0)
    assert np.all(res["final_replay_rate"] <= 1.0)


@_needs_elise
def test_run_protocol_opsin_mode_learns_delivered_teacher() -> None:
    from neuroseq.example_models import elise

    exc, inh = _build_fuer_elise_opsin_stims(tfinal_ms=1920)
    base = {k: v["default"] for k, v in elise.DEFAULT_PARAMS.items()}

    res = elise.run_protocol(
        exc.astype(float),
        inh.astype(float),
        stim_dt_s=0.001,
        base_params=base,
        teacher_mode="opsin",
        training_epochs=8,
        training_cycles=15,
        validation_cycles=2,
        replay_epochs=2,
        replay_cycles=2,
    )
    assert res["val_mse"].shape == (8,)
    assert res["config"]["teacher_mode"] == "opsin"
    assert np.all(np.isfinite(res["val_mse"]))
    # Scored against the delivered (smeared) opsin teacher, the network learns:
    # MSE falls and correlation rises, just like the direct teacher.
    assert res["val_mse"][-1] < res["val_mse"][0]
    assert res["val_corr"][-1] > res["val_corr"][0]


@_needs_elise
def test_run_protocol_rejects_bad_teacher_mode() -> None:
    from neuroseq.example_models import elise

    exc, inh = _build_fuer_elise_opsin_stims(tfinal_ms=480)
    with pytest.raises(ValueError, match="teacher_mode"):
        elise.run_protocol(
            exc.astype(float), inh.astype(float), 0.001, teacher_mode="x"
        )


def test_run_protocol_rejects_shape_mismatch() -> None:
    # Import guarded: build_target_rate is not reached before the shape check,
    # so this validation runs without the elise package.
    from neuroseq.example_models import elise

    exc = np.zeros((100, 13))
    inh = np.zeros((100, 7))
    with pytest.raises(ValueError, match="same shape"):
        elise.run_protocol(exc, inh, 0.001)


# ---------- CSV export ----------


def test_save_learning_curves_csv_roundtrip(tmp_path) -> None:
    curves = {
        "val_mse": np.array([0.08, 0.02, 0.006]),
        "val_corr": np.array([0.2, 0.7, 0.95]),
        "rep_mse": np.array([0.007, 0.008]),
        "rep_corr": np.array([0.94, 0.93]),
        "config": {"teacher_mode": "direct", "training_cycles": 25},
    }
    out = save_learning_curves_csv(tmp_path / "curves.csv", curves, source="proj.nseq")
    text = out.read_text()

    assert "# Analysis: ELiSe learning curves" in text
    assert "# teacher_mode: direct" in text
    assert "# Source: proj.nseq" in text
    # Header + 3 training rows + 2 replay rows.
    data_rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert data_rows[0] == "phase,epoch,mse,corr_coeff"
    assert sum(1 for r in data_rows if r.startswith("training,")) == 3
    assert sum(1 for r in data_rows if r.startswith("replay,")) == 2


# ---------- plugin discovery ----------


def test_learning_curves_plugin_is_discovered() -> None:
    ids = {p.id for p in discover_analysis_plugins()}
    assert "elise_learning_curves" in ids


# ---------- GUI action wiring ----------


def test_elise_protocol_action_enables_for_elise_project(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    try:
        assert hasattr(gui, "elise_protocol_action")
        # Default project is not an ELiSe project → disabled.
        assert not gui.elise_protocol_action.isEnabled()
        assert gui._elise_protocol_inputs() is None

        gui.project.external_simulate_path = "neuroseq:elise"
        exc, inh = _build_fuer_elise_opsin_stims(tfinal_ms=480)
        gui.project.add_stim_array(
            StimArray(
                name="opsin_exc",
                stim_type=StimType.Excitation,
                values=exc,
                units="mm**-2 / second",
                dt=0.001,
            )
        )
        gui.project.add_stim_array(
            StimArray(
                name="opsin_inh",
                stim_type=StimType.Inhibition,
                values=inh,
                units="mm**-2 / second",
                dt=0.001,
            )
        )
        gui._refresh_elise_replay_action()

        assert gui.elise_protocol_action.isEnabled()
        inputs = gui._elise_protocol_inputs()
        assert inputs is not None
        exc_vals, inh_vals, stim_dt = inputs
        assert exc_vals.shape == exc.shape
        assert stim_dt == pytest.approx(0.001)
    finally:
        gui.close()


def test_elise_protocol_dialog_values(qtbot: QtBot) -> None:
    from neuroseq.elise_protocol_dialog import EliseProtocolDialog

    dlg = EliseProtocolDialog()
    qtbot.addWidget(dlg)
    dlg.training_epochs.setValue(50)
    dlg.replay_learning.setChecked(False)

    values = dlg.get_values()
    assert set(values) == {
        "teacher_mode",
        "training_epochs",
        "training_cycles",
        "validation_cycles",
        "replay_epochs",
        "replay_cycles",
        "replay_learning",
    }
    assert values["teacher_mode"] == "direct"
    assert values["training_epochs"] == 50
    assert values["replay_learning"] is False
    # Estimate label reflects the updated epoch count.
    assert "training cycles" in dlg._estimate.text()


def _synthetic_result(experiment=None) -> dict:
    T = 40
    return {
        "val_mse": np.array([0.08, 0.02, 0.006]),
        "val_corr": np.array([0.2, 0.7, 0.95]),
        "rep_mse": np.array([0.007, 0.008]),
        "rep_corr": np.array([0.94, 0.93]),
        "target_rate": np.zeros((13, T)),
        "cycle_trace_samples": T,
        "trace_dt_s": 1e-4,
        "num_vis": 13,
        "final_replay_rate": np.zeros((13, 2 * T)),
        "experiment": experiment,
        "config": {"teacher_mode": "direct", "training_cycles": 25},
    }


def test_on_protocol_finished_opens_window_and_imports_experiment(
    qtbot: QtBot, monkeypatch, mocker
) -> None:
    monkeypatch.setattr("neuroseq.gui.getsavefilename", lambda *a, **k: ("", ""))
    gui = Gui()
    qtbot.addWidget(gui)
    try:
        append = mocker.patch.object(gui, "_append_experimental_data")
        sentinel_exp = object()
        gui._on_elise_protocol_finished(_synthetic_result(experiment=sentinel_exp))
        # Slot tore the worker state down and produced an analysis window.
        assert gui._elise_protocol_worker is None
        windows = gui.findChildren(CustomAnalysisWindow)
        assert windows
        widget = windows[-1].analysis_widget
        assert getattr(widget, _CURVES_ATTR, None) is not None
        # The replay Experiment is imported so traces line up with the stim.
        append.assert_called_once_with(sentinel_exp)
    finally:
        gui.close()


def test_on_protocol_failed_tears_down(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    try:
        gui._elise_protocol_worker = object()  # type: ignore[assignment]  # pretend a run is active
        gui._on_elise_protocol_failed("boom")
        assert gui._elise_protocol_worker is None
    finally:
        gui.close()


def test_on_protocol_cancelled_and_progress(qtbot: QtBot) -> None:
    gui = Gui()
    qtbot.addWidget(gui)
    try:
        gui._on_elise_protocol_progress(0.5)
        assert gui.progress_dialog.value() == 50
        gui._elise_protocol_worker = object()  # type: ignore[assignment]  # pretend a run is active
        gui._on_elise_protocol_cancelled()
        assert gui._elise_protocol_worker is None
    finally:
        gui.close()


@_needs_elise
def test_run_protocol_full_dispatch(qtbot: QtBot, monkeypatch) -> None:
    from qtpy import QtWidgets

    from neuroseq.example_projects import elise_project

    class _FakeDialog:
        def __init__(self, parent=None) -> None:
            pass

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def get_values(self) -> dict:
            return {
                "teacher_mode": "direct",
                "training_epochs": 2,
                "training_cycles": 2,
                "validation_cycles": 2,
                "replay_epochs": 2,
                "replay_cycles": 2,
                "replay_learning": True,
            }

    monkeypatch.setattr(
        "neuroseq.elise_protocol_dialog.EliseProtocolDialog", _FakeDialog
    )
    monkeypatch.setattr("neuroseq.gui.getsavefilename", lambda *a, **k: ("", ""))

    gui = Gui()
    qtbot.addWidget(gui)
    try:
        project = elise_project()
        gui.load_project(project)
        gui._refresh_elise_replay_action()
        assert gui.elise_protocol_action.isEnabled()

        gui._run_elise_protocol()
        # Worker dispatched on a thread; wait for the finished slot to tear down.
        qtbot.waitUntil(lambda: gui._elise_protocol_worker is None, timeout=60000)
        assert gui.findChildren(CustomAnalysisWindow)
    finally:
        gui.close()
