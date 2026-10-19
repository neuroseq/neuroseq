from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData
from neuroseq.project import Project
from neuroseq.project_history import ProjectCommand
from neuroseq.project_history import ProjectHistory
from neuroseq.save_load_options import SaveLoadOptions


def test_command_history_branches_and_releases_discarded_commands() -> None:
    history = ProjectHistory()
    values: list[int] = []
    events: list[str] = []
    unsubscribe = history.subscribe(events.append)

    def undo() -> None:
        values.pop()

    first = ProjectCommand("first", lambda: values.append(1), undo)
    history.execute(first)
    history.execute(ProjectCommand("second", lambda: values.append(2), undo))
    assert history.undo_label == "second"
    history.undo()
    assert values == [1]
    assert history.redo_label == "second"
    history.execute(ProjectCommand("third", lambda: values.append(3), undo))
    assert not history.can_redo()
    assert len(history) == 2
    history.undo()
    history.undo()
    history.undo()  # empty-history operations are harmless
    assert values == []
    assert history.undo_label == ""
    history.redo()
    history.redo()
    history.redo()
    assert values == [1, 3]
    assert history.redo_label == ""
    unsubscribe()
    count = len(events)
    history.clear()
    assert len(events) == count
    assert not history.can_undo()
    assert not history.can_redo()
    assert len(history) == 0


def test_failed_commands_preserve_history_position_and_redo_branch() -> None:
    history = ProjectHistory()

    def fail() -> None:
        raise ValueError("failed")

    history.execute(ProjectCommand("valid", lambda: None, fail))
    with pytest.raises(ValueError, match="failed"):
        history.undo()
    assert history.can_undo()
    assert not history.can_redo()
    history.clear()
    history.record_applied(ProjectCommand("redo fails", fail, lambda: None))
    history.undo()
    with pytest.raises(ValueError, match="failed"):
        history.redo()
    with pytest.raises(ValueError, match="failed"):
        history.execute(ProjectCommand("execute fails", fail, lambda: None))
    assert not history.can_undo()
    assert history.redo_label == "redo fails"


@pytest.mark.parametrize(
    "operation", ["undo", "redo", "clear", "execute", "record_applied"]
)
def test_commands_cannot_mutate_history_during_replay(operation: str) -> None:
    history = ProjectHistory()
    nested = ProjectCommand("nested", lambda: None, lambda: None)

    def undo() -> None:
        method = getattr(history, operation)
        if operation in ("execute", "record_applied"):
            method(nested)
        else:
            method()

    history.execute(ProjectCommand("outer", lambda: None, undo))
    with pytest.raises(RuntimeError):
        history.undo()
    assert history.current_index == 0
    assert len(history) == 1


def test_history_does_not_retain_closed_observers_and_listener_failure_is_isolated() -> (
    None
):
    history = ProjectHistory()

    class Observer:
        def changed(self, event: str) -> None:
            pass

    def failing_listener(event: str) -> None:
        raise RuntimeError("observer failed")

    observer = Observer()
    ref = weakref.ref(observer)
    history.subscribe(observer.changed)
    unsubscribe = history.subscribe(failing_listener)
    history.execute(ProjectCommand("edit", lambda: None, lambda: None))
    assert history.can_undo()
    unsubscribe()
    del observer
    gc.collect()
    assert ref() is None
    history.undo()
    assert not history.can_undo()


def _experiment(name: str = "recording") -> Experiment:
    return Experiment(
        name,
        4.0,
        [0.0, 4.0],
        [
            TraceData("voltage", np.array([[1, 2, 3, 4]]), np.array([0]), "mV", 1.0),
            SpikeData("spikes", np.array([0, 0]), np.array([0.5, 2.5])),
            SynapseData("weights", np.arange(4).reshape(1, 1, 4), np.arange(4)),
        ],
    )


def test_import_append_undo_restores_overwritten_data_and_metadata() -> None:
    project = Project()
    experiment = _experiment()
    project.add_experiment(experiment)
    original_buffers = [vars(datum).copy() for datum in experiment.data]
    incoming = _experiment()
    assert isinstance(incoming.data[0], TraceData)
    incoming.data[0].traces[:] = 9
    project.append_experimental_data("recording", 4.0, incoming.data, 2.0)
    assert experiment.duration == 6.0
    assert experiment.timepoints == [0, 2, 4, 6]
    assert isinstance(experiment.data[0], TraceData)
    assert experiment.data[0].traces.tolist() == [[1, 2, 9, 9, 9, 9]]
    after = [vars(datum).copy() for datum in experiment.data]
    project.history.undo()
    assert project.experiments["recording"] is experiment
    assert experiment.duration == 4.0
    assert experiment.timepoints == [0, 4]
    assert project.max_time_secs == 4.0
    for datum, saved in zip(experiment.data, original_buffers, strict=True):
        for key, value in saved.items():
            if isinstance(value, np.ndarray):
                assert getattr(datum, key) is value
    project.history.redo()
    assert experiment.duration == 6.0
    for datum, saved in zip(experiment.data, after, strict=True):
        for key, value in saved.items():
            if isinstance(value, np.ndarray):
                assert getattr(datum, key) is value


def test_failed_partial_append_rolls_back_data_dirty_state_and_history() -> None:
    project = Project()
    experiment = _experiment()
    project.add_experiment(experiment)
    project.history.clear()
    project.dirty = False
    incoming = _experiment()
    assert isinstance(incoming.data[0], TraceData)
    assert isinstance(incoming.data[-1], SynapseData)
    incoming.data[0].traces[:] = 8
    # Fail after the trace has already been appended.
    incoming.data[-1].values = np.zeros((2, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        project.append_experimental_data("recording", 4.0, incoming.data, 2.0)
    assert isinstance(experiment.data[0], TraceData)
    assert experiment.data[0].traces.tolist() == [[1, 2, 3, 4]]
    assert experiment.duration == 4.0
    assert experiment.timepoints == [0, 4]
    assert not project.dirty
    assert not project.history.can_undo()


def test_group_delete_replace_and_add_dataset_share_one_history() -> None:
    project = Project()
    first = _experiment("first")
    second = _experiment("second")
    project.add_experiment(first)
    project.add_experiment(second)
    project.remove_experiments(["first", "second", "missing"])
    assert not project.experiments
    project.history.undo()
    assert list(project.experiments) == ["first", "second"]
    assert project.experiments["first"] is first
    replacement = _experiment("first")
    replacement.duration = 9.0
    with project.edit_experiments(["first"], "Replace data group"):
        project.experiments["first"] = replacement
    project.history.undo()
    assert project.experiments["first"] is first
    project.history.redo()
    assert project.experiments["first"] is replacement
    extra = TraceData("calcium", np.ones((1, 12)), np.array([0]), "a.u.", 1.0)
    project.add_data_to_experiment("first", extra)
    assert replacement.duration == 12.0
    project.history.undo()
    assert replacement.duration == 9.0
    assert all(datum is not extra for datum in replacement.data)
    project.history.redo()
    assert replacement.data[-1] is extra
    project.history.undo()
    project.history.undo()
    project.history.undo()
    assert list(project.experiments) == ["first"]
    project.history.undo()
    assert not project.experiments


def test_noop_or_invalid_data_changes_do_not_create_history_entries() -> None:
    project = Project()
    project.remove_experiments(["missing"])
    with pytest.raises(KeyError):
        project.add_data_to_experiment("missing", _experiment().data[0])
    with pytest.raises(ValueError):
        project.append_experimental_data("missing", 1, [], 0)
    with pytest.raises(ValueError):
        project.add_experimental_data("missing", [])
    assert len(project.history) == 0
    project.add_experiment(_experiment())
    with pytest.raises(ValueError):
        project.add_experiment(_experiment())
    assert len(project.history) == 1


def test_discarding_redo_releases_retained_import_buffers() -> None:
    project = Project()
    original = _experiment("old")
    trace = original.data[0]
    assert isinstance(trace, TraceData)
    buffer_ref = weakref.ref(trace.traces)
    project.add_experiment(original)
    del original, trace
    project.history.undo()
    assert buffer_ref() is not None
    project.add_experiment(_experiment("new"))
    gc.collect()
    assert buffer_ref() is None


def test_history_is_per_project_and_not_serialized(tmp_path) -> None:
    project = Project()
    other = Project()
    project.add_experiment(_experiment())
    assert not other.history.can_undo()
    path = tmp_path / "history.nseq"
    project.save(path, SaveLoadOptions(True, True, False, True))
    assert not project.dirty
    project.history.undo()
    assert project.dirty
    assert not project.experiments
    loaded = Project(path)
    assert "recording" in loaded.experiments
    assert not loaded.history.can_undo()
    assert not loaded.dirty


@pytest.mark.parametrize("resize", ["pad", "resize", "replace"])
def test_nonundoable_neuron_changes_invalidate_old_stimulation_commands(
    resize: str,
) -> None:
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim import StimType

    project = Project()
    project.add_stim_array(
        StimArray("input", StimType.Excitation, np.zeros((10, 1)), "mV", 0.001)
    )
    project.add_experiment(_experiment())
    if resize == "pad":
        project.pad_stim_arrays_to_neuron_count(2)
    elif resize == "resize":
        project.resize_stim_arrays_to_neuron_count(2)
    else:
        project.add_stim_array(
            StimArray("input", StimType.Excitation, np.zeros((5, 1)), "mV", 0.001)
        )
    assert not project.history.can_undo()


def _stim_project(values: np.ndarray, dt: float = 0.001):
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim import StimType
    from neuroseq.stimulation.stim_array_modifier import StimArrayModifier

    project = Project()
    project.add_stim_array(StimArray("input", StimType.Excitation, values, "mV", dt))
    stim = project.stim_arrays["input"]
    modifier = StimArrayModifier(stim, None, project.history)  # type: ignore[arg-type]
    return project, stim, modifier


def test_history_drops_oldest_commands_beyond_max_commands() -> None:
    history = ProjectHistory(max_commands=3)
    values: list[int] = []

    def command(i: int) -> ProjectCommand:
        return ProjectCommand(
            str(i), lambda: values.append(i), lambda: values.remove(i)
        )

    for i in range(5):
        history.execute(command(i))
    assert len(history) == 3
    assert [c.label for c in history.action_stack] == ["2", "3", "4"]
    while history.can_undo():
        history.undo()
    assert values == [0, 1]
    history.redo()
    assert values == [0, 1, 2]
    assert Project().history.max_commands == 100
    with pytest.raises(ValueError):
        ProjectHistory(max_commands=0)


def test_replace_stim_values_is_undoable_and_keeps_earlier_draws_on_their_grid() -> (
    None
):
    from neuroseq.stimulation.stim_timing import remap_stim_values_to_dt

    project, stim, modifier = _stim_project(np.zeros((10, 2), dtype=np.float32))
    modifier.do(slice(0, 1), slice(6, 8), 1.0)
    original = stim.values
    resampled = remap_stim_values_to_dt(stim.values, 0.001, 0.0005)
    project.dirty = False
    project.replace_stim_values("input", resampled, 0.0005, label="Resample input")
    assert project.dirty
    assert stim.values is resampled and stim.dt == 0.0005
    assert project.max_time_secs == pytest.approx(0.01)
    assert project.history.undo_label == "Resample input"
    project.history.undo()
    assert stim.values is original and stim.dt == 0.001
    project.history.undo()  # the draw is undone at the indices it was made at
    assert not stim.values.any()
    project.history.redo()
    project.history.redo()
    assert stim.values is resampled
    assert stim.values[12:16, 0].tolist() == [1, 1, 1, 1]


def test_replace_stim_values_never_shares_the_retained_buffer() -> None:
    project, stim, _ = _stim_project(np.zeros((4, 1), dtype=np.float32))
    original = stim.values
    project.replace_stim_values("input", original)
    assert stim.values is not original
    stim.values[:] = 5  # in-place edit of the new buffer
    project.history.undo()
    assert stim.values is original and not original.any()


def test_slice_command_refuses_a_grid_changed_outside_history(caplog) -> None:
    project, stim, modifier = _stim_project(np.zeros((10, 1), dtype=np.float32))
    modifier.do(slice(0, 1), slice(2, 4), 1.0)
    # Out-of-band resample (e.g. a console script) the history knows nothing about.
    stim.values, stim.dt = np.full((20, 1), 7, dtype=np.float32), 0.0005
    with caplog.at_level("WARNING"):
        project.history.undo()
    assert "recorded on a (10, 1) grid" in caplog.text
    assert np.all(stim.values == 7)
    assert not project.history.can_undo()
    assert not project.history.can_redo()


def test_data_group_undo_refuses_to_pop_a_same_named_group_added_out_of_band(
    caplog,
) -> None:
    project = Project()
    old = _experiment("Rec")
    project.add_experiment(old)
    project.remove_experiments(["Rec"])
    newer = _experiment("Rec")
    project.experiments["Rec"] = newer  # bypasses the history
    with caplog.at_level("WARNING"):
        project.history.undo()
    assert project.experiments["Rec"] is newer
    assert "changed outside the edit history" in caplog.text
    assert not project.history.can_undo()


def test_edit_experiments_redo_also_checks_the_current_group() -> None:
    project = Project()
    project.add_experiment(_experiment("Rec"))
    project.history.undo()
    project.experiments["Rec"] = newer = _experiment("Rec")
    project.history.redo()
    assert project.experiments["Rec"] is newer
    assert not project.history.can_redo()
