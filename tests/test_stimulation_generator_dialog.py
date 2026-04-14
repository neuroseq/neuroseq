from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq.stimulation_generator_dialog as stimulation_generator_dialog_module
from neuroseq.exporter_syscon2 import _SYSCON2_TTL_PULSE_QUANTUM_S
from neuroseq.exporter_syscon2 import SysconExportReport
from neuroseq.exporter_syscon2 import SysconInputSanityReport
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog


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
    active_laser_wavelength_nm: int = 1140,
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
        laser_timeline_index=2,
        ttl_out1_timeline_index=3,
        ttl_out2_timeline_index=4,
    )


def _build_sanity_report(
    *,
    is_valid: bool,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    export_report: SysconExportReport | None = None,
    sanitized_values: np.ndarray | None = None,
    fix_messages: list[str] | None = None,
) -> SysconInputSanityReport:
    return SysconInputSanityReport(
        is_valid=is_valid,
        errors=errors or [],
        warnings=warnings or [],
        export_report=export_report,
        sanitized_values=sanitized_values,
        fix_messages=fix_messages or [],
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
    assert dialog._selected_laser_wavelength_nm() == 1140


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

    captured: dict[str, int] = {}

    def _fake_check_syscon2_input_sanity(*args, **kwargs) -> SysconInputSanityReport:
        del args
        captured["active_laser_wavelength_nm"] = kwargs["active_laser_wavelength_nm"]
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
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"


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
