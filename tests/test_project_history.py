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
    project.replace_experiment(replacement)
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
        project.replace_experiment(_experiment())
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
