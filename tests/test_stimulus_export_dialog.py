from unittest.mock import Mock

import pytest
from qtpy import QtWidgets

from neuroseq.export import stimulus_export_dialog
from neuroseq.project import Project


@pytest.mark.parametrize(
    ("mode", "selection", "expected"),
    [
        ("patch-clamp", "Igor2 - single array (.ibw / .csv)", "single"),
        ("patch-clamp", "Igor2 - all arrays (.pxp)", "pxp"),
        ("light", "SysCon2", "syscon"),
        ("light", "Igor2 - all arrays (.pxp)", "pxp"),
        ("light", "Igor2 - single neuron (.ibw / .csv)", "single"),
    ],
)
def test_format_selection_routes_to_export(monkeypatch, mode, selection, expected):
    chooser = Mock(return_value=(selection, True))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", chooser)
    single = Mock()
    pxp = Mock()
    syscon = Mock()
    monkeypatch.setattr(stimulus_export_dialog, "_export_single_wave", single)
    monkeypatch.setattr(stimulus_export_dialog, "_export_pxp", pxp)
    monkeypatch.setattr(stimulus_export_dialog, "StimulationGeneratorDialog", syscon)
    project = Project()

    stimulus_export_dialog.show_stimulus_export_dialog(
        project, mode, current_stim_name="selected"
    )

    args = chooser.call_args.args
    assert args[1] == (
        "Export patch clamp input" if mode == "patch-clamp" else "Export light input"
    )
    assert args[3] == (
        ["Igor2 - single array (.ibw / .csv)", "Igor2 - all arrays (.pxp)"]
        if mode == "patch-clamp"
        else [
            "SysCon2",
            "Igor2 - all arrays (.pxp)",
            "Igor2 - single neuron (.ibw / .csv)",
        ]
    )
    assert [single.called, pxp.called, syscon.called] == [
        expected == "single",
        expected == "pxp",
        expected == "syscon",
    ]
    if expected == "single":
        single.assert_called_once_with(project, "selected", None)
    elif expected == "pxp":
        pxp.assert_called_once_with(project, None)
    else:
        syscon.assert_called_once_with(
            project, parent=None, current_stim_name="selected"
        )
        syscon.return_value.exec.assert_called_once_with()


@pytest.mark.parametrize("mode", ["patch-clamp", "light"])
def test_cancelled_format_selection_does_not_start_export(monkeypatch, mode):
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem", Mock(return_value=("", False))
    )
    exporters = [Mock(), Mock(), Mock()]
    for name, exporter in zip(
        ("_export_single_wave", "_export_pxp", "StimulationGeneratorDialog"),
        exporters,
        strict=True,
    ):
        monkeypatch.setattr(stimulus_export_dialog, name, exporter)

    stimulus_export_dialog.show_stimulus_export_dialog(Project(), mode)

    for exporter in exporters:
        exporter.assert_not_called()


def test_syscon_export_forwards_preview_and_refreshes_reserved_names(monkeypatch):
    dialog = Mock()
    dialog.export_settings.return_value = settings = object()
    dialog_class = Mock(return_value=dialog)
    monkeypatch.setattr(
        stimulus_export_dialog, "StimulationGeneratorDialog", dialog_class
    )
    names = {"existing"}

    def reserved_names():
        return names

    preview = Mock()
    project = Project()

    stimulus_export_dialog.show_stimulus_export_dialog(
        project,
        "syscon2",
        current_stim_name="selected",
        reserved_names=reserved_names,
        on_generated=preview,
    )

    dialog_class.assert_called_once_with(
        project, parent=None, current_stim_name="selected"
    )
    names.add("new")
    assert dialog.reserved_names() == {"existing", "new"}
    callback = dialog.sequence_generated.connect.call_args.args[0]
    callback("sequence.seq", "preview.png", frozenset({1}))
    preview.assert_called_once_with(
        "sequence.seq", "preview.png", frozenset({1}), settings
    )
    dialog.exec.assert_called_once_with()


def test_cancelled_pxp_dialog_does_not_choose_file_or_export(monkeypatch):
    project = Mock()
    project.stim_arrays = {"stim": object()}
    dialog = Mock()
    dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Rejected
    monkeypatch.setattr(
        stimulus_export_dialog, "Igor2PxpStimExportDialog", Mock(return_value=dialog)
    )
    choose_file = Mock()
    export = Mock()
    monkeypatch.setattr(stimulus_export_dialog, "getsavefilename", choose_file)
    monkeypatch.setattr(stimulus_export_dialog, "export_igor2_pxp", export)

    stimulus_export_dialog.show_stimulus_export_dialog(project, "igor-pxp")

    choose_file.assert_not_called()
    export.assert_not_called()
