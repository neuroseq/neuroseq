from unittest.mock import Mock

from qtpy import QtWidgets

from neuroseq.export import stimulus_export_dialog
from neuroseq.project import Project


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
