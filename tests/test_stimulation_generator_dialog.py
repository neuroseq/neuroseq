from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq.export.stimulation_generator_dialog as stimulation_generator_dialog_module
import neuroseq.export.syscon2_export_service as syscon2_export_service_module
from neuroseq.export.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.export.exporter_syscon2 import InitialSlmLoadViolation
from neuroseq.export.exporter_syscon2 import OverlappingPulseGroupViolation
from neuroseq.export.exporter_syscon2 import SimultaneousDurationViolation
from neuroseq.export.exporter_syscon2 import SlmGapViolation
from neuroseq.export.exporter_syscon2 import SlmOffOnOverlapViolation
from neuroseq.export.exporter_syscon2 import SysconExportReport
from neuroseq.export.exporter_syscon2 import SysconInputSanityReport
from neuroseq.export.power_lut import PowerLut
from neuroseq.export.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

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


def test_stimulation_generator_dialog_export_defaults(
    qtbot: QtBot,
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0}),
    )
    qtbot.addWidget(dialog)
    assert not dialog.binary_mode_checkbox.isChecked()
    assert not dialog.spiralling_checkbox.isChecked()
    assert dialog.ten_spots_per_roi_checkbox.isChecked()
    assert dialog.spot_algorithm_combo.isEnabled()
    assert dialog.spot_algorithm_combo.currentData() == "farthest_point"
    assert dialog._selected_laser_wavelength_nm() == 940
    assert dialog.target_power_spin.value() == 1.0


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
    assert captured["spiralling"] is False
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
        syscon2_export_service_module,
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
) -> None:
    values = np.zeros((40, 1), dtype=np.float32)
    values[20:22, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)

    lut_path = tmp_path / "loaded_1100.lut"
    lut_path.write_text(_REFERENCE_LUT_TEXT, encoding="utf-8")
    dialog._apply_lut_from_path(str(lut_path), 1100)
    dialog.laser_combo.setCurrentIndex(dialog.laser_combo.findData(1100))

    assert project.power_luts[1100].description == "1100nm 1 spot"
    # The label-status path is exercised by _format_lut_status; we avoid
    # asserting on dialog.lut_status_label.text() here because setting QLabel
    # text on a never-shown widget can leave a pending paint event that
    # interacts poorly with pytest-qt's _process_events on the macOS-arm64
    # runner. _format_lut_status is a pure function, easily covered separately.
    assert "(default)" not in dialog._format_lut_status()
    assert "1100nm 1 spot" in dialog._format_lut_status()


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
    dialog.ten_spots_per_roi_checkbox.setChecked(True)
    dialog.spot_algorithm_combo.setCurrentIndex(
        dialog.spot_algorithm_combo.findData("edge_balanced")
    )

    captured: dict[str, object] = {}

    def _fake_generate_syscon2_sequence_text(*args, **kwargs):
        del args
        captured["target_power_mw"] = kwargs["target_power_mw"]
        captured["power_lut"] = kwargs["power_lut"]
        captured["stimulation_spots_per_roi"] = kwargs["stimulation_spots_per_roi"]
        captured["stimulation_spot_algorithm"] = kwargs["stimulation_spot_algorithm"]
        return "nohash\n", _build_export_report()

    monkeypatch.setattr(
        syscon2_export_service_module,
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
    assert captured["stimulation_spots_per_roi"] == 10
    assert captured["stimulation_spot_algorithm"] == "edge_balanced"


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
    # the "no active input" error.
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
    assert "no active input samples" in qmessagebox_calls[0]["text"].lower()


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


def test_stimulation_generator_dialog_emits_sequence_generated_on_success(
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
    dialog.name_edit.setText("live_export")

    with qtbot.waitSignal(dialog.sequence_generated, timeout=1000) as blocker:
        dialog._generate_scripts()

    assert blocker.args == [
        str(tmp_path / "live_export.seq"),
        str(tmp_path / "live_export_overview.png"),
        frozenset({0}),
    ]
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "scripts generated" in qmessagebox_calls[0]["title"].lower()


def test_stimulation_generator_dialog_sequence_generated_excludes_unmapped_neurons(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    """Only neurons included in the generated sequence get target markers."""
    _DT_S = 0.0005208125
    values = np.zeros((64, 3), dtype=np.float32)
    values[33:36, 0] = 1.0
    values[33:36, 2] = 1.0
    project = Project()
    stim = StimArray(
        name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=_DT_S
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result(mapping={0: 0, 1: 1, 2: 2})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.output_folder = tmp_path
    dialog.folder_edit.setText(str(tmp_path))
    dialog.name_edit.setText("multi_target_export")

    with qtbot.waitSignal(dialog.sequence_generated, timeout=1000) as blocker:
        dialog._generate_scripts()

    assert blocker.args[2] == frozenset({0})


# ---------------------------------------------------------------------------
# dt pre-check (lines 415, 420, 426-451)
# ---------------------------------------------------------------------------


def test_stimulation_generator_dialog_cancels_on_wrong_dt(
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

    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_dt_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._DtFixChoice.CANCEL,
    )

    dialog.check_sanity_button.click()

    # Cancelling leaves the array untouched and shows no further message boxes.
    assert project.stim_arrays["stim"].dt == pytest.approx(0.001)
    assert not qmessagebox_calls


def test_stimulation_generator_dialog_clears_array_when_dt_fix_choice_is_clear(
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

    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_dt_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._DtFixChoice.CLEAR,
    )

    dialog.check_sanity_button.click()

    assert project.stim_arrays["stim"].dt == pytest.approx(_SYSCON2_TTL_PULSE_QUANTUM_S)
    assert np.all(project.stim_arrays["stim"].values == 0.0)
    # Confirmation info message shown after fixing
    assert any(c["name"] == "information" for c in qmessagebox_calls)


def test_stimulation_generator_dialog_resamples_and_continues_sanity_checks(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    # old_dt is exactly half the SysCon2 tick, so the resample mapping is unambiguous
    # (see test_resample_stim_values_to_dt_downsamples_via_nearest_neighbor).
    old_dt = _SYSCON2_TTL_PULSE_QUANTUM_S / 2
    values = np.zeros((10, 1), dtype=np.float32)
    values[2:6, 0] = 1.0
    project = _build_project(values.copy(), mapping={0: 0})
    project.stim_arrays["stim"].dt = old_dt  # force the mismatch _build_project avoids
    _patch_no_slm_violations(monkeypatch)

    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_dt_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._DtFixChoice.RESAMPLE,
    )
    sanity_calls: list[object] = []

    def _fake_check_sanity(*args, **kwargs):
        sanity_calls.append(args)
        return _build_sanity_report(is_valid=True, export_report=_build_export_report())

    monkeypatch.setattr(
        stimulation_generator_dialog_module,
        "check_syscon2_input_sanity",
        _fake_check_sanity,
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.check_sanity_button.click()

    stim = project.stim_arrays["stim"]
    assert stim.dt == pytest.approx(_SYSCON2_TTL_PULSE_QUANTUM_S)
    assert stim.values.shape == (5, 1)
    assert list(stim.values[:, 0]) == [0.0, 1.0, 1.0, 0.0, 0.0]
    # The pipeline must not stop after resampling -- it should reach the full sanity
    # check below, proving the other conditions are still evaluated on the new array.
    assert sanity_calls
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
    # Force the prompt to return CANCEL deterministically, without ever
    # showing a QMessageBox (relying on patch_b2w_exec_show's .show() default
    # leaves pending Qt events that hang subsequent tests on macOS-arm64-ssc).
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_initial_slm_start_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._InitialSlmFixChoice.CANCEL,
    )

    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
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
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_slm_gap_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._SlmFixChoice.CANCEL,
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
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_slm_off_on_overlap_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._SlmOffOnFixChoice.CANCEL,
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
    monkeypatch.setattr(
        StimulationGeneratorDialog,
        "_prompt_duration_fix_choice",
        lambda *a, **kw: StimulationGeneratorDialog._DurationFixChoice.CANCEL,
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


def _two_array_project() -> Project:
    project = _build_project(np.zeros((40, 1), dtype=np.float32), mapping={0: 0})
    project.add_stim_array(
        StimArray(
            name="second",
            stim_type=StimType.Excitation,
            values=np.zeros((40, 1), dtype=np.float32),
            units="1",
            dt=_SYSCON2_TTL_PULSE_QUANTUM_S,
        )
    )
    return project


def test_stimulation_generator_dialog_preselects_the_current_input_array(
    qtbot: QtBot,
) -> None:
    """#588: don't default to the first array created."""
    dialog = StimulationGeneratorDialog(
        _two_array_project(), current_stim_name="second"
    )
    qtbot.addWidget(dialog)

    assert dialog.stim_combo.currentText() == "second"


def test_stimulation_generator_dialog_defaults_to_first_without_a_current_array(
    qtbot: QtBot,
) -> None:
    for current in (None, "not-a-project-array"):
        dialog = StimulationGeneratorDialog(
            _two_array_project(), current_stim_name=current
        )
        qtbot.addWidget(dialog)
        assert dialog.stim_combo.currentText() == "stim"


def test_generate_rejects_invalid_name_without_emitting(
    qtbot, tmp_path, qmessagebox_calls
):
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.output_folder = tmp_path
    dialog.name_edit.setText("../escaped")
    emitted = []
    dialog.sequence_generated.connect(lambda *args: emitted.append(args))
    dialog._generate_scripts()
    assert not emitted
    assert not list(tmp_path.iterdir())
    assert qmessagebox_calls[-1]["name"] == "critical"


def test_generate_cancelled_overwrite_preserves_files(
    qtbot, tmp_path, qmessagebox_calls, mocker
):
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.output_folder = tmp_path
    dialog.name_edit.setText("existing")
    (tmp_path / "existing.seq").write_text("original")
    mocker.patch.object(dialog, "_confirm_export", return_value=False)
    emitted = []
    dialog.sequence_generated.connect(lambda *args: emitted.append(args))
    dialog._generate_scripts()
    assert not emitted
    assert (tmp_path / "existing.seq").read_text() == "original"


def test_standalone_sanity_fix_marks_project_dirty(qtbot):
    values = np.zeros((64, 1), dtype=np.float32)
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    project.dirty = False
    assert dialog._replace_stim_values_preserving_undo("stim", np.ones_like(values))
    assert project.dirty


def test_default_export_name_counts_todays_exports(tmp_path: Path) -> None:
    import datetime

    from neuroseq.export.stimulation_generator_dialog import next_default_export_name

    day = datetime.date(2026, 10, 8)
    assert next_default_export_name(None, day) == "export_seq_2026-10-08_1"
    (tmp_path / "export_seq_2026-10-08_1.seq").write_text("x")
    (tmp_path / "export_seq_2026-10-08_2.seq").write_text("x")
    (tmp_path / "export_seq_2026-10-07_9.seq").write_text("x")  # other day
    assert next_default_export_name(tmp_path, day) == "export_seq_2026-10-08_3"


def test_default_export_name_avoids_every_collision_source(tmp_path: Path) -> None:
    import datetime

    from neuroseq.export.stimulation_generator_dialog import next_default_export_name

    day = datetime.date(2026, 10, 8)
    (tmp_path / "export_seq_2026-10-08_1.seq").write_text("x")
    # Stray sibling files of an export count, even without the .seq.
    (tmp_path / "export_seq_2026-10-08_4_export_debug.csv").write_text("x")
    (tmp_path / "EXPORT_SEQ_2026-10-08_5_overview.PNG").write_text("x")
    assert next_default_export_name(tmp_path, day) == "export_seq_2026-10-08_6"
    # Names already in the stimulation setup count, with or without a folder part.
    reserved = ["Exp\\export_seq_2026-10-08_9.seq", "other/export_seq_2026-10-08_7.SEQ"]
    assert (
        next_default_export_name(tmp_path, day, reserved) == "export_seq_2026-10-08_10"
    )
    # A missing folder is not an error.
    assert (
        next_default_export_name(tmp_path / "missing", day) == "export_seq_2026-10-08_1"
    )


def test_dialog_rechecks_default_name_at_export_time(
    qtbot: QtBot, tmp_path: Path, qmessagebox_calls
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.output_folder = tmp_path
    dialog._refresh_default_name()
    stale = dialog.name_edit.text()
    # Someone else (e.g. the stimulation setup) takes that name after the dialog resolved it.
    dialog.reserved_names = lambda: [f"{stale}.seq"]
    dialog._generate_scripts()
    assert not (tmp_path / f"{stale}.seq").exists()
    assert any(tmp_path.glob("export_seq_*.seq"))


def test_dialog_default_name_advances_but_keeps_custom_names(
    qtbot: QtBot, tmp_path: Path, qmessagebox_calls
) -> None:
    values = np.zeros((64, 1), dtype=np.float32)
    values[33:36, 0] = 1.0
    project = _build_project(values, mapping={0: 0})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    assert dialog.name_edit.text().startswith("export_seq_")
    assert dialog.name_edit.text().endswith("_1")
    dialog.output_folder = tmp_path

    first = dialog.name_edit.text()
    dialog._generate_scripts()
    assert (tmp_path / f"{first}.seq").exists()
    assert dialog.name_edit.text() == first[:-1] + "2"

    dialog.name_edit.setText("my_name")
    dialog._generate_scripts()
    assert dialog.name_edit.text() == "my_name"


def test_loaded_940_lut_marks_dirty_and_survives_project_save(qtbot, tmp_path):
    from neuroseq.save_load_options import SaveLoadOptions

    project = Project()
    project.dirty = False
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    path = tmp_path / "940.lut"
    path.write_text(_REFERENCE_LUT_TEXT)
    dialog._apply_lut_from_path(str(path), 940)
    assert project.dirty
    saved = tmp_path / "calibrated.nseq"
    project.save(saved, SaveLoadOptions(True, True, True, True))
    loaded = Project(saved)
    assert loaded.power_luts[940] == project.power_luts[940]


def test_dialog_starts_with_the_given_output_folder(qtbot, tmp_path) -> None:
    """Issue #614: scripts can be generated without browsing for a folder."""
    from neuroseq.export.stimulation_generator_dialog import StimulationGeneratorDialog
    from neuroseq.project import Project

    dialog = StimulationGeneratorDialog(Project(), output_folder=tmp_path)
    qtbot.addWidget(dialog)
    assert dialog.output_folder == tmp_path
    assert dialog.folder_edit.text() == str(tmp_path)
    assert dialog.generate_button.isEnabled()
