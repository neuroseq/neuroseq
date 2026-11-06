from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qtpy import QtWidgets

from neuroseq.stimulation.dynamic_input.dialog_helpers import confirm_start_routine


def _editor_with_stimuli() -> SimpleNamespace:
    stim = SimpleNamespace(name="Exc", values=np.ones((3, 2)))
    return SimpleNamespace(
        current_stim_modifier=SimpleNamespace(stim_array=stim),
        current_stim_array_has_any_data=lambda: True,
    )


def _answer(monkeypatch, label: str | None, seen: list[list[str]]) -> None:
    def exec_(box: QtWidgets.QMessageBox) -> int:
        seen.append([button.text() for button in box.buttons()])
        for button in box.buttons():
            if button.text() == label:
                button.click()
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Start in new arrays", "new"),
        ("Clear and start over", "clear"),
        ("Continue after last stimulus", "continue"),
    ],
)
def test_new_arrays_are_offered_next_to_clear_and_continue(
    qtbot, monkeypatch, label, expected
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    seen: list[list[str]] = []
    _answer(monkeypatch, label, seen)
    editor: Any = _editor_with_stimuli()
    assert confirm_start_routine(parent, editor, offer_new=True) == expected
    assert "Start in new arrays" in seen[0]


def test_new_arrays_are_not_offered_by_default(qtbot, monkeypatch) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    seen: list[list[str]] = []
    _answer(monkeypatch, "Start in new arrays", seen)
    editor: Any = _editor_with_stimuli()
    assert confirm_start_routine(parent, editor) == "cancel"
    assert "Start in new arrays" not in seen[0]
    assert {"Clear and start over", "Continue after last stimulus"} <= set(seen[0])
