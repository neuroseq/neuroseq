from __future__ import annotations

from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.visualization_support.neuron_subset_dialog import NeuronSubsetDialog


def _find_buttons(
    dialog: NeuronSubsetDialog,
) -> tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]:
    buttons = dialog.findChildren(QtWidgets.QPushButton)
    select_all = next(b for b in buttons if b.text() == "Select all")
    select_none = next(b for b in buttons if b.text() == "Select none")
    return select_all, select_none


def test_dialog_defaults_to_checking_every_neuron(qtbot: QtBot) -> None:
    dialog = NeuronSubsetDialog([1, 2, 3])
    qtbot.addWidget(dialog)

    assert dialog.selected_neuron_ids == [1, 2, 3]


def test_dialog_prechecks_only_the_given_ids(qtbot: QtBot) -> None:
    dialog = NeuronSubsetDialog([1, 2, 3, 4], checked_ids={2, 4})
    qtbot.addWidget(dialog)

    assert dialog.selected_neuron_ids == [2, 4]


def test_select_all_and_select_none_buttons(qtbot: QtBot) -> None:
    dialog = NeuronSubsetDialog([1, 2, 3], checked_ids={1})
    qtbot.addWidget(dialog)
    select_all_button, select_none_button = _find_buttons(dialog)

    select_all_button.click()
    assert dialog.selected_neuron_ids == [1, 2, 3]

    select_none_button.click()
    assert dialog.selected_neuron_ids == []


def test_unchecking_an_item_removes_it_from_selection(qtbot: QtBot) -> None:
    dialog = NeuronSubsetDialog([5, 7, 9])
    qtbot.addWidget(dialog)

    item = dialog.list_widget.item(1)
    assert item is not None
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    assert dialog.selected_neuron_ids == [5, 9]
