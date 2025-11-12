# This test file caused strange abort error due to graphics/WGPU initialization issues
# Tests have been moved to test_import_timeline_extending.py with a different approach

from __future__ import annotations

import uuid

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.gui import Gui
from neuroseq.import_dialog import Suite2pImportDialog

TRACE_KEY = "suite2p_contrasting_traces"
SAMPLING_RATE = 30.0
EXPECTED_DT = 1.0 / SAMPLING_RATE

# TODO: add the b2m pad arrays code (timed arrays) for efficiency/Liam's suggestion?


@pytest.fixture
def gui(qtbot: QtBot) -> Gui:
    pytest.skip(
        "Graphics initialization causes abort error - tests moved to test_import_timeline_extending.py"
    )
    gui = Gui()
    qtbot.addWidget(gui)
    return gui


def _write_temp_f_file(tmp_path, n_timepoints: int, n_neurons: int = 2) -> str:
    file_path = tmp_path / f"F_{n_neurons}n_{n_timepoints}t_{uuid.uuid4().hex}.npy"
    data = np.linspace(0.0, 1.0, n_neurons * n_timepoints, dtype=np.float64).reshape(
        n_neurons, n_timepoints
    )
    np.save(file_path, data)
    return str(file_path)


def _run_import_dialog(
    tmp_path, qtbot: QtBot, n_timepoints: int, *, replace_existing: bool
):
    dialog = Suite2pImportDialog()
    qtbot.addWidget(dialog)
    f_path = _write_temp_f_file(tmp_path, n_timepoints)
    dialog.sampling_rate_spin.setValue(SAMPLING_RATE)
    dialog.replace_checkbox.setChecked(replace_existing)
    dialog.load_f_file(f_path)
    dialog.import_data()
    imported_data = dialog.get_imported_data()
    assert imported_data is not None
    return imported_data


def _fetch_trace_dataset(gui: Gui):
    assert gui.external_simulation_data is not None
    assert TRACE_KEY in gui.external_simulation_data
    return gui.external_simulation_data[TRACE_KEY]


def _timeline_metrics(trace_dataset) -> tuple[int, float]:
    timestamps = trace_dataset["timestamps"]
    assert isinstance(timestamps, np.ndarray)
    count = len(timestamps)
    dt = float(timestamps[1] - timestamps[0]) if count > 1 else EXPECTED_DT
    duration_ms = count * dt * 1000.0
    return count, duration_ms


@pytest.mark.skip(
    reason="Graphics initialization causes abort error - tests moved to test_import_timeline_extending.py"
)
def test_import_extends_single_insert(gui: Gui, tmp_path, qtbot: QtBot):
    imported = _run_import_dialog(
        tmp_path, qtbot, n_timepoints=500, replace_existing=True
    )
    gui.handle_general_import(imported)

    trace_dataset = _fetch_trace_dataset(gui)
    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 500
    assert duration_ms == pytest.approx(500 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert gui._get_max_timepoints() == 500


@pytest.mark.skip(
    reason="Graphics initialization causes abort error - tests moved to test_import_timeline_extending.py"
)
def test_import_extends_append_mode(gui: Gui, tmp_path, qtbot: QtBot):
    first_payload = _run_import_dialog(
        tmp_path, qtbot, n_timepoints=500, replace_existing=True
    )
    gui.handle_general_import(first_payload)

    append_payload = _run_import_dialog(
        tmp_path, qtbot, n_timepoints=500, replace_existing=False
    )
    gui.handle_general_import(append_payload)

    trace_dataset = _fetch_trace_dataset(gui)
    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 1000
    assert duration_ms == pytest.approx(1000 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert gui._get_max_timepoints() == 1000


@pytest.mark.skip(
    reason="Graphics initialization causes abort error - tests moved to test_import_timeline_extending.py"
)
def test_import_extends_replace_mode(gui: Gui, tmp_path, qtbot: QtBot):
    first_payload = _run_import_dialog(
        tmp_path, qtbot, n_timepoints=500, replace_existing=True
    )
    gui.handle_general_import(first_payload)

    replace_payload = _run_import_dialog(
        tmp_path, qtbot, n_timepoints=500, replace_existing=True
    )
    gui.handle_general_import(replace_payload)

    trace_dataset = _fetch_trace_dataset(gui)
    count, duration_ms = _timeline_metrics(trace_dataset)

    assert count == 500
    assert duration_ms == pytest.approx(500 * (1000.0 / SAMPLING_RATE), rel=1e-3)
    assert gui._get_max_timepoints() == 500
