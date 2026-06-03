from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq.stimulation_generator_dialog as stimulation_generator_dialog_module
from neuroseq.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.exporter_syscon2 import GlobalSlmWindowViolation
from neuroseq.exporter_syscon2 import InitialSlmLoadViolation
from neuroseq.exporter_syscon2 import OverlappingPulseGroupViolation
from neuroseq.exporter_syscon2 import SimultaneousDurationViolation
from neuroseq.exporter_syscon2 import SlmGapViolation
from neuroseq.exporter_syscon2 import SlmOffOnOverlapViolation
from neuroseq.exporter_syscon2 import SysconExportReport
from neuroseq.exporter_syscon2 import SysconInputSanityReport
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.power_lut import PowerLut
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog

_REFERENCE_LUT_TEXT = """F2-EC-FD-4A-EB-0E-8C-29-3A-0D-22-7A-94-1D-D1-94
[_]
Description=1100nm 1 spot
Unit=mW
Enabled=False
TuplesCount=9
Tuple0=0;0
Tuple1=5;0
Tuple2=10;2.1
Tuple3=20;8.4
Tuple4=40;21.9
Tuple5=60;36
Tuple6=80;49
Tuple7=90;49
Tuple8=100;49
[/_]
"""


@pytest.fixture(autouse=True)
def _auto_accept_question_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )


def _cellpose_result(mapping: dict[int, int] | None = None) -> CellposeImportResult:
    segmentation = np.array(
        [
            [0, 0, -1],
            [1, 1, -1],
        ],
        dtype=np.int32,
    )
    overlay = np.zeros((2, 3, 3), dtype=np.uint8)
    original_ids = np.array([0, 1], dtype=np.int32)
    return CellposeImportResult(
        image_path=Path("dummy.png"),
        segmentation_array=segmentation,
        overlay_image=overlay,
        trace_data=None,
        spike_data=None,
        original_cellpose_ids=original_ids,
        stimulus_to_cellpose_mapping=mapping,
    )


def _build_project(values: np.ndarray, mapping: dict[int, int] | None) -> Project:
    project = Project()
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_SYSCON2_TTL_PULSE_QUANTUM_S,
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result(mapping=mapping)
    return project


def _patch_no_slm_violations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all pre-sanity SLM violation finders to return no violations.

    Use this in tests that mock check_syscon2_input_sanity directly and
    don't care about the pre-sanity steps.
    """
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_off_on_overlap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_simultaneous_duration_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_overlapping_pulse_group_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_global_slm_window_violations",
        lambda *a, **kw: [],
    )


def _build_export_report(
    *,
    total_neurons: int = 1,
    active_neurons: int = 1,
    mapped_active_neurons: int = 1,
    unmapped_active_neurons: list[int] | None = None,
    total_segments: int = 1,
    exported_segments: int = 1,
    active_laser_wavelength_nm: int = 1100,
) -> SysconExportReport:
    return SysconExportReport(
        total_neurons=total_neurons,
        active_neurons=active_neurons,
        mapped_active_neurons=mapped_active_neurons,
        unmapped_active_neurons=unmapped_active_neurons or [],
        total_segments=total_segments,
        exported_segments=exported_segments,
        skipped_segments_unmapped=max(total_segments - exported_segments, 0),
        entity_lanes=1,
        generated_objects=1,
        active_laser_wavelength_nm=active_laser_wavelength_nm,
        laser_timeline_index=1,
        ttl_out1_timeline_index=4,
        ttl_out2_timeline_index=5,
    )


def _build_sanity_report(
    *,
    is_valid: bool,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    export_report: SysconExportReport | None = None,
    sanitized_values: np.ndarray | None = None,
    fix_messages: list[str] | None = None,
    pulse_overrun_fix_values: np.ndarray | None = None,
) -> SysconInputSanityReport:
    return SysconInputSanityReport(
        is_valid=is_valid,
        errors=errors or [],
        warnings=warnings or [],
        export_report=export_report,
        sanitized_values=sanitized_values,
        fix_messages=fix_messages or [],
        pulse_overrun_fix_values=pulse_overrun_fix_values,
    )


def test_stimulation_generator_dialog_sanity_check_passes(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            warnings=[],
            export_report=_build_export_report(),
            sanitized_values=np.array(values, copy=True),
        ),
    )
    dialog = StimulationGeneratorDialog(
        project,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "sanity check passed" in qmessagebox_calls[0]["title"].lower()
    assert "exported segments" in qmessagebox_calls[0]["text"].lower()


def test_stimulation_generator_dialog_binary_mode_checkbox_defaults_to_enabled(
    qtbot: QtBot,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0}),
    )
    qtbot.addWidget(dialog)
    assert dialog.binary_mode_checkbox.isChecked()
    assert dialog.spiralling_checkbox.isChecked()
    assert dialog._selected_laser_wavelength_nm() == 1100


def test_stimulation_generator_dialog_sanity_check_uses_selected_laser_wavelength(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    _patch_no_slm_violations(monkeypatch)
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0}),
    )
    qtbot.addWidget(dialog)
    dialog.laser_combo.setCurrentIndex(dialog.laser_combo.findData(1030))

    captured: dict[str, int | bool] = {}

    def _fake_check_syscon2_input_sanity(*args, **kwargs) -> SysconInputSanityReport:
        del args
        captured["active_laser_wavelength_nm"] = kwargs["active_laser_wavelength_nm"]
        captured["spiralling"] = kwargs["spiralling"]
        return _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(
                active_laser_wavelength_nm=1030,
            ),
        )

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        _fake_check_syscon2_input_sanity,
    )
    dialog.check_sanity_button.click()

    assert captured["active_laser_wavelength_nm"] == 1030
    assert captured["spiralling"] is True
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"


def test_stimulation_generator_dialog_generate_passes_spiralling_checkbox(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.spiralling_checkbox.setChecked(False)

    captured: dict[str, bool] = {}

    def _fake_generate_syscon2_sequence_text(*args, **kwargs):
        del args
        captured["spiralling"] = kwargs["spiralling"]
        return "nohash\n", _build_export_report()

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "generate_syscon2_sequence_text",
        _fake_generate_syscon2_sequence_text,
    )

    dialog._generate_holo4d_script(
        project.stim_arrays["stim"],
        "stim",
        project.cellpose_data,
    )

    assert captured["spiralling"] is False


def test_stimulation_generator_dialog_load_lut_assigns_to_selected_wavelength(
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.laser_combo.setCurrentIndex(dialog.laser_combo.findData(1100))

    lut_path = tmp_path / "loaded_1100.lut"
    lut_path.write_text(_REFERENCE_LUT_TEXT, encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(lut_path), "SysCon LUT (*.lut)"),
    )

    dialog._on_load_lut_clicked()

    assert project.power_luts[1100].description == "1100nm 1 spot"
    assert "1100nm 1 spot" in dialog.lut_status_label.text()
    assert "(default)" not in dialog.lut_status_label.text()


def test_stimulation_generator_dialog_passes_target_power_and_lut_to_exporter(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    custom_lut = PowerLut(
        description="custom",
        tuples=((0.0, 0.0), (100.0, 50.0)),
    )
    project.power_luts[1100] = custom_lut
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.laser_combo.setCurrentIndex(dialog.laser_combo.findData(1100))
    dialog.target_power_spin.setValue(10.0)

    captured: dict[str, object] = {}

    def _fake_generate_syscon2_sequence_text(*args, **kwargs):
        del args
        captured["target_power_mw"] = kwargs["target_power_mw"]
        captured["power_lut"] = kwargs["power_lut"]
        return "nohash\n", _build_export_report()

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "generate_syscon2_sequence_text",
        _fake_generate_syscon2_sequence_text,
    )

    dialog._generate_holo4d_script(
        project.stim_arrays["stim"],
        "stim",
        project.cellpose_data,
    )

    assert captured["target_power_mw"] == 10.0
    assert captured["power_lut"] == custom_lut


def test_stimulation_generator_dialog_sanity_check_warns_when_report_contains_warnings(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 2), dtype=np.float32)
    values[20:22, 0] = 1.0
    values[20:22, 1] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            warnings=["Some active neurons are unmapped and will be skipped: 1"],
            export_report=_build_export_report(
                total_neurons=2,
                active_neurons=2,
                mapped_active_neurons=1,
                unmapped_active_neurons=[1],
                total_segments=2,
                exported_segments=1,
            ),
            sanitized_values=np.array(values, copy=True),
        ),
    )
    dialog = StimulationGeneratorDialog(
        project,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "warnings" in qmessagebox_calls[0]["title"].lower()
    assert "unmapped" in qmessagebox_calls[0]["text"].lower()


def test_stimulation_generator_dialog_sanity_check_fails_for_inactive_input(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    # dt must match the SysCon2 tick quantum so the dt check passes and we reach
    # the "no active stimulation" error.
    _DT_S = 0.0005208125
    values = np.zeros((64, 2), dtype=np.float32)
    project = Project()
    stim = StimArray(
        name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=_DT_S
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result(mapping={0: 0, 1: 1})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "sanity check failed" in qmessagebox_calls[0]["title"].lower()
    assert "no active stimulation samples" in qmessagebox_calls[0]["text"].lower()


def test_stimulation_generator_dialog_applies_sanitized_values_after_confirmation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)
    sanitized_values = np.array(values, copy=True)
    sanitized_values[20:22, 0] = 0.0
    sanitized_values[17:34, 0] = 1.0

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            warnings=["Expanded neuron 0 stimulus duration from 2ms to 17ms."],
            export_report=_build_export_report(),
            sanitized_values=sanitized_values,
            fix_messages=["Expanded neuron 0 stimulus duration from 2ms to 17ms."],
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    np.testing.assert_array_equal(project.stim_arrays["stim"].values, sanitized_values)
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "warning"
    assert "17ms" in qmessagebox_calls[0]["text"]


def test_stimulation_generator_dialog_declines_sanitized_values_when_confirmation_is_no(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)
    before = np.array(project.stim_arrays["stim"].values, copy=True)
    sanitized_values = np.array(values, copy=True)
    sanitized_values[20:22, 0] = 0.0
    sanitized_values[17:34, 0] = 1.0

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            warnings=[],
            export_report=_build_export_report(),
            sanitized_values=sanitized_values,
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.No,
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, before)
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"


def test_stimulation_generator_dialog_uses_stim_editor_undo_path_for_applied_sanitized_values(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    class _FakeStimEditor:
        def __init__(self, project: Project) -> None:
            self.project = project
            self.calls: list[tuple[str, np.ndarray]] = []

        def replace_stim_values_with_undo(
            self,
            stim_name: str,
            new_values: np.ndarray,
        ) -> None:
            self.calls.append((stim_name, np.array(new_values, copy=True)))
            self.project.stim_arrays[stim_name].values[:, :] = new_values

    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)
    sanitized_values = np.array(values, copy=True)
    sanitized_values[20:22, 0] = 0.0
    sanitized_values[17:34, 0] = 1.0

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            warnings=[],
            export_report=_build_export_report(),
            sanitized_values=sanitized_values,
        ),
    )

    parent = QtWidgets.QWidget()
    fake_editor = _FakeStimEditor(project)
    parent.stim_editor = fake_editor  # type: ignore[attr-defined]

    dialog = StimulationGeneratorDialog(project, parent=parent)
    qtbot.addWidget(parent)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(fake_editor.calls) == 1
    assert fake_editor.calls[0][0] == "stim"
    np.testing.assert_array_equal(fake_editor.calls[0][1], sanitized_values)
    np.testing.assert_array_equal(project.stim_arrays["stim"].values, sanitized_values)
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"


def test_stimulation_generator_dialog_generate_writes_seq_and_overview_png(
    qtbot: QtBot,
    tmp_path: Path,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    _DT_S = 0.0005208125  # SysCon2 tick quantum required for export
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1.0  # active pulse inside a valid SLM window
    project = Project()
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=_DT_S,
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result(mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.output_folder = tmp_path
    dialog.folder_edit.setText(str(tmp_path))
    dialog.name_edit.setText("preview_export")

    dialog._generate_scripts()

    seq_path = tmp_path / "preview_export.seq"
    png_path = tmp_path / "preview_export_overview.png"
    assert seq_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "scripts generated" in qmessagebox_calls[0]["title"].lower()


# ---------------------------------------------------------------------------
# dt pre-check (lines 415, 420, 426-451)
# ---------------------------------------------------------------------------


def test_stimulation_generator_dialog_shows_error_for_wrong_dt(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = Project()
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,  # wrong dt
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result()
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "sanity check failed" in qmessagebox_calls[0]["title"].lower()


def test_stimulation_generator_dialog_fixes_dt_when_fix_clicked(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = Project()
    stim = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=values,
        units="1",
        dt=0.001,  # wrong dt
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result()
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)

    # Override _show_scrollable_message to simulate clicking "Fix dt"
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_show_scrollable_message",
        lambda *a, **kw: True,
    )

    dialog.check_sanity_button.click()

    assert project.stim_arrays["stim"].dt == pytest.approx(_SYSCON2_TTL_PULSE_QUANTUM_S)
    assert np.all(project.stim_arrays["stim"].values == 0.0)
    # Confirmation info message shown after fixing
    assert any(c["name"] == "information" for c in qmessagebox_calls)


# ---------------------------------------------------------------------------
# Pre-sanity SLM load violation (lines 457, 460-468)
# ---------------------------------------------------------------------------


def test_stimulation_generator_dialog_cancels_on_slm_load_violation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: InitialSlmLoadViolation(
            first_active_idx=5, required_start_idx=32
        ),
    )
    # Patch remaining finders to avoid cascades
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [],
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    # patch_b2w_exec_show makes the prompt auto-cancel
    dialog.check_sanity_button.click()

    # Returned early — no sanity check ran, no message box
    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_applies_slm_load_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: InitialSlmLoadViolation(
            first_active_idx=5, required_start_idx=32
        ),
    )
    shifted_values = np.zeros_like(values)
    shifted_values[52:54, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "shift_stim_values_right_for_initial_slm_load",
        lambda *a, **kw: (shifted_values, None),
    )
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_initial_slm_start_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._InitialSlmFixChoice.SHIFT_RIGHT,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, shifted_values)


# ---------------------------------------------------------------------------
# Pre-sanity SLM gap violation (lines 473, 476-482, 785)
# ---------------------------------------------------------------------------


def _fake_slm_gap_violation() -> SlmGapViolation:
    return SlmGapViolation(
        previous_start_idx=0,
        previous_end_idx=32,
        current_start_idx=33,
        current_end_idx=64,
        previous_neuron_indices=(0,),
        current_neuron_indices=(1,),
        gap_samples=1,
        required_gap_samples=32,
    )


def test_stimulation_generator_dialog_cancels_on_slm_gap_violation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [_fake_slm_gap_violation()],
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_applies_slm_gap_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [_fake_slm_gap_violation()],
    )
    spaced_values = np.zeros_like(values)
    spaced_values[52:54, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "space_out_stim_values_for_slm",
        lambda *a, **kw: (spaced_values, None),
    )
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_slm_gap_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._SlmFixChoice.SPACE_OUT,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, spaced_values)


# ---------------------------------------------------------------------------
# Pre-sanity SLM off/on overlap violation (lines 489, 492-498)
# ---------------------------------------------------------------------------


def _fake_slm_offon_violation() -> SlmOffOnOverlapViolation:
    return SlmOffOnOverlapViolation(
        phase_start_idx=0,
        phase_start_tick=0,
        off_neuron_index=0,
        on_neuron_index=1,
        overlap_start_tick=10,
        overlap_end_tick=20,
        required_shift_samples=32,
        required_shift_ticks=32,
    )


def test_stimulation_generator_dialog_cancels_on_slm_offon_violation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_off_on_overlap_violations",
        lambda *a, **kw: [_fake_slm_offon_violation()],
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_applies_slm_offon_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_off_on_overlap_violations",
        lambda *a, **kw: [_fake_slm_offon_violation()],
    )
    spaced_values = np.zeros_like(values)
    spaced_values[52:54, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "space_out_stim_values_for_slm_off_on_overlap",
        lambda *a, **kw: (spaced_values, None),
    )
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_slm_off_on_overlap_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._SlmOffOnFixChoice.SPACE_RIGHT,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, spaced_values)


# ---------------------------------------------------------------------------
# Pre-sanity simultaneous duration violation (lines 505, 508-514)
# ---------------------------------------------------------------------------


def _fake_duration_violation() -> SimultaneousDurationViolation:
    return SimultaneousDurationViolation(
        start_idx=20,
        neuron_indices=(0, 1),
        durations_idx=(10, 20),
        min_duration_idx=10,
    )


def test_stimulation_generator_dialog_cancels_on_duration_violation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_off_on_overlap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_simultaneous_duration_violations",
        lambda *a, **kw: [_fake_duration_violation()],
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_applies_duration_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_simultaneous_duration_violations",
        lambda *a, **kw: [_fake_duration_violation()],
    )
    equalized_values = np.zeros_like(values)
    equalized_values[20:22, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "equalize_simultaneous_durations_to_min",
        lambda *a, **kw: (equalized_values, None),
    )
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_duration_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._DurationFixChoice.EQUALIZE_MIN,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, equalized_values)


# ---------------------------------------------------------------------------
# Overlap alignment auto-applied (lines 523-527)
# ---------------------------------------------------------------------------


def test_stimulation_generator_dialog_auto_applies_overlap_alignment(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_overlapping_pulse_group_violations",
        lambda *a, **kw: [
            OverlappingPulseGroupViolation(
                segment_neuron_indices=(0, 1),
                segment_starts_idx=(20, 21),
                segment_durations_idx=(5, 10),
                aligned_start_idx=20,
                aligned_duration_idx=5,
            )
        ],
    )
    aligned_values = np.zeros_like(values)
    aligned_values[20:25, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "align_overlapping_pulses_to_shortest",
        lambda *a, **kw: (aligned_values, None),
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, aligned_values)


# ---------------------------------------------------------------------------
# Global SLM window violation (lines 534-539, 542-544)
# ---------------------------------------------------------------------------


def _fake_global_window_violation() -> GlobalSlmWindowViolation:
    return GlobalSlmWindowViolation(
        neuron_index=0,
        start_idx=5,
        duration_idx=32,
        start_tick=5,
        snapped_start_tick=0,
        shift_ticks=-5,
        shift_samples=-5,
    )


def test_stimulation_generator_dialog_cancels_on_global_slm_window_violation(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_initial_slm_load_violation",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_gap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_slm_off_on_overlap_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_simultaneous_duration_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_overlapping_pulse_group_violations",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_global_slm_window_violations",
        lambda *a, **kw: [_fake_global_window_violation()],
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_snaps_to_global_slm_windows(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "find_global_slm_window_violations",
        lambda *a, **kw: [_fake_global_window_violation()],
    )
    snapped_values = np.zeros_like(values)
    snapped_values[16:18, 0] = 1.0
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "snap_stim_values_to_global_slm_windows",
        lambda *a, **kw: (snapped_values, None, None),
    )
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_global_slm_window_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._GlobalWindowFixChoice.SNAP,
    )
    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=True,
            export_report=_build_export_report(),
        ),
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, snapped_values)


# ---------------------------------------------------------------------------
# Pulse overrun fix button after sanity check failure (lines 604-605)
# ---------------------------------------------------------------------------


def test_stimulation_generator_dialog_applies_pulse_overrun_fix_when_clicked(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    _patch_no_slm_violations(monkeypatch)

    fix_values = np.zeros_like(values)
    fix_values[20:21, 0] = 1.0  # trimmed pulse

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        lambda *args, **kwargs: _build_sanity_report(
            is_valid=False,
            errors=["Laser pulse overrun detected."],
            pulse_overrun_fix_values=fix_values,
        ),
    )
    # Simulate "Fix pulse overrun" button click
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_show_scrollable_message",
        lambda *a, **kw: True,
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, fix_values)
