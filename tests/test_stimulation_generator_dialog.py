from __future__ import annotations

from pathlib import Path

import brian2.numpy_ as np
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.exporter_syscon2 import find_slm_gap_violations
from neuroseq.importer_cellpose import CellposeImportResult
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
from neuroseq.stimulation_generator_dialog import StimulationGeneratorDialog


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
        dt=0.001,
    )
    project.add_stim_array(stim)
    project.cellpose_data = _cellpose_result(mapping=mapping)
    return project


def test_stimulation_generator_dialog_sanity_check_passes(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    values = np.zeros((6, 1), dtype=np.float32)
    values[1:3, 0] = 1.0
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0}),
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "sanity check passed" in qmessagebox_calls[0]["title"].lower()
    assert "exported segments" in qmessagebox_calls[0]["text"].lower()


def test_stimulation_generator_dialog_sanity_check_warns_for_unmapped(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    values = np.zeros((6, 2), dtype=np.float32)
    values[1:3, 0] = 1.0
    values[1:3, 1] = 1.0
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0}),
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
    values = np.zeros((6, 2), dtype=np.float32)
    dialog = StimulationGeneratorDialog(
        _build_project(values, mapping={0: 0, 1: 1}),
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    dialog.check_sanity_button.click()

    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "critical"
    assert "sanity check failed" in qmessagebox_calls[0]["title"].lower()
    assert "no active stimulation samples" in qmessagebox_calls[0]["text"].lower()


def test_stimulation_generator_dialog_sanity_check_space_out_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:10, 1] = 1.0
    project = _build_project(values, mapping={0: 0, 1: 1})
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert (
        len(find_slm_gap_violations(project.stim_arrays["stim"], active_threshold=0.0))
        > 0
    )

    monkeypatch.setattr(
        dialog,
        "_prompt_slm_gap_fix_choice",
        lambda violations, dt_seconds: dialog._SlmFixChoice.SPACE_OUT,
    )
    dialog.check_sanity_button.click()

    assert (
        len(find_slm_gap_violations(project.stim_arrays["stim"], active_threshold=0.0))
        == 0
    )
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
    assert "sanity check passed" in qmessagebox_calls[0]["title"].lower()


def test_stimulation_generator_dialog_sanity_check_cancel_slm_fix(
    qtbot: QtBot,
    qmessagebox_calls: list[dict[str, str]],
    monkeypatch,
) -> None:
    values = np.zeros((40, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:10, 1] = 1.0
    project = _build_project(values, mapping={0: 0, 1: 1})
    before = np.array(project.stim_arrays["stim"].values, copy=True)
    dialog = StimulationGeneratorDialog(project)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    monkeypatch.setattr(
        dialog,
        "_prompt_slm_gap_fix_choice",
        lambda violations, dt_seconds: dialog._SlmFixChoice.CANCEL,
    )
    dialog.check_sanity_button.click()

    assert np.array_equal(project.stim_arrays["stim"].values, before)
    assert len(qmessagebox_calls) == 0


def test_stimulation_generator_dialog_uses_stim_editor_undo_path_for_fixes(
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

    values = np.zeros((40, 2), dtype=np.float32)
    values[0:3, 0] = 1.0
    values[8:10, 1] = 1.0
    project = _build_project(values, mapping={0: 0, 1: 1})

    parent = QtWidgets.QWidget()
    fake_editor = _FakeStimEditor(project)
    parent.stim_editor = fake_editor  # type: ignore[attr-defined]

    dialog = StimulationGeneratorDialog(project, parent=parent)
    qtbot.addWidget(parent)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())

    monkeypatch.setattr(
        dialog,
        "_prompt_slm_gap_fix_choice",
        lambda violations, dt_seconds: dialog._SlmFixChoice.SPACE_OUT,
    )
    dialog.check_sanity_button.click()

    assert len(fake_editor.calls) == 1
    assert fake_editor.calls[0][0] == "stim"
    assert (
        len(find_slm_gap_violations(project.stim_arrays["stim"], active_threshold=0.0))
        == 0
    )
    assert len(qmessagebox_calls) == 1
    assert qmessagebox_calls[0]["name"] == "information"
