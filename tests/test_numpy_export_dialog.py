import numpy as np
from pytestqt.qtbot import QtBot

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData
from neuroseq.export.numpy_export_dialog import NPZ_FILTER
from neuroseq.export.numpy_export_dialog import NumpyExportDialog
from neuroseq.project import Project


def _project() -> Project:
    project = Project()
    trace = TraceData(
        "dF/F", np.arange(12.0).reshape(3, 4), np.array([0, 1, 2]), "dF/F", 0.05
    )
    spikes = SpikeData("Deconvolved spikes", np.array([1, 2]), np.array([0.0, 0.12]))
    project.add_experiment(Experiment("rec", 0.2, [], [trace, spikes]))
    return project


def test_traces_are_exported_as_neurons_by_samples(
    qtbot: QtBot, tmp_path, mocker
) -> None:
    project = _project()
    dialog = NumpyExportDialog(project, "traces", "rec :: dF/F")
    qtbot.addWidget(dialog)
    assert "3 neurons × 4 time points" in dialog.lbl_shape.text()
    target = tmp_path / "rec_dF_F"
    mocker.patch(
        "neuroseq.export.numpy_export_dialog.getsavefilename",
        return_value=(str(target), NPZ_FILTER),
    )
    dialog.btn_export.click()
    assert dialog.saved_path == f"{target}.npz"
    with np.load(dialog.saved_path) as archive:
        np.testing.assert_array_equal(archive["values"], np.arange(12.0).reshape(3, 4))


def test_spikes_are_binned_at_the_groups_frame_period(
    qtbot: QtBot, tmp_path, mocker
) -> None:
    project = _project()
    dialog = NumpyExportDialog(project, "spikes", "rec :: Deconvolved spikes")
    qtbot.addWidget(dialog)
    assert dialog.spn_bin_ms.value() == 50.0  # the trace's frame period
    matrix = dialog.matrix()
    assert matrix is not None
    assert matrix.values.shape[1] == 4  # 0.2 s at 50 ms
    assert matrix.values[1, 0] == 1 and matrix.values[2, 2] == 1
    target = tmp_path / "spikes.npy"
    mocker.patch(
        "neuroseq.export.numpy_export_dialog.getsavefilename",
        return_value=(str(target), ""),
    )
    dialog.btn_export.click()
    np.testing.assert_array_equal(np.load(target), matrix.values)


def test_nothing_to_export(qtbot: QtBot) -> None:
    dialog = NumpyExportDialog(Project(), "spikes")
    qtbot.addWidget(dialog)
    assert not dialog.btn_export.isEnabled()
    assert "No spikes" in dialog.lbl_shape.text()
