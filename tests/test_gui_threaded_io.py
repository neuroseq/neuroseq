"""GUI-level tests for the threaded save / load / import flow."""

from __future__ import annotations

import threading

import pytest
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

import neuroseq
from neuroseq.example_projects import default_project
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.save_load_options import SaveLoadOptions


class _MockSaveLoadOptionsDialog:
    options = SaveLoadOptions(
        include_stim_arrays=True,
        include_model=True,
        include_simulation_data=False,
        include_experimental_data=False,
    )

    def __init__(self, *args, **kwargs):
        pass

    def exec(self):
        return True


@pytest.fixture
def gui(qtbot: QtBot):
    widget = ProjectWorkspace()
    qtbot.addWidget(widget)
    yield widget
    try:
        widget.close()
    except RuntimeError:
        return
    qtbot.waitUntil(lambda: widget._background_job.worker_thread is None, timeout=10000)


def _patch_save_dialog(monkeypatch: MonkeyPatch, filename: str) -> None:
    monkeypatch.setattr(
        neuroseq.project_workspace, "SaveLoadOptionsDialog", _MockSaveLoadOptionsDialog
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args: (filename, True),
    )


def _patch_open_dialog(monkeypatch: MonkeyPatch, filename: str) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args: (filename, True),
    )


@pytest.mark.parametrize("workflow", ["io", "brian2", "external", "elise"])
def test_running_job_keeps_progress_and_ownership_when_another_is_requested(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch: MonkeyPatch, workflow: str
) -> None:
    release = threading.Event()
    assert gui._dispatch_io_task(
        lambda report: release.wait(5), "Original operation", lambda _: None
    )
    original_worker = gui._background_job.worker
    original_label = gui.progress_dialog.labelText()
    try:
        if workflow == "io":
            assert not gui._dispatch_io_task(
                lambda report: pytest.fail("overlapping job started"),
                "Another operation",
                lambda _: None,
            )
        elif workflow == "brian2":
            gui._do_brian2_simulation_threaded(0.1)
        elif workflow == "external":
            gui.project.external_simulate_function = lambda *a, **kw: pytest.fail(
                "overlapping external simulation started"
            )
            gui._do_external_simulation(0.1)
        else:
            monkeypatch.setattr(
                gui, "_elise_protocol_inputs", lambda: (None, None, 0.001)
            )
            gui._run_elise_protocol()
        assert gui._background_job.worker is original_worker
        assert gui.progress_dialog.labelText() == original_label
        assert not gui.menu_file.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not gui._background_job.busy)


def test_close_keeps_window_open_when_worker_cannot_stop(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch: MonkeyPatch
) -> None:
    release = threading.Event()
    assert gui._dispatch_io_task(
        lambda report: release.wait(5), "Original operation", lambda _: None
    )
    gui._close_confirmed = True
    try:
        with monkeypatch.context() as patch:
            patch.setattr(gui._background_job, "stop", lambda timeout_ms: False)
            assert not gui.close()
        assert gui.isVisible()
        assert gui._background_job.busy
        assert not gui._close_confirmed
    finally:
        release.set()
        qtbot.waitUntil(lambda: not gui._background_job.busy)


def test_save_runs_on_worker_thread(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch: MonkeyPatch, tmp_path
):
    gui.load_project(default_project())
    save_thread_ids: list[int | None] = []

    original_save = Project.save

    def _spy_save(self, filepath, options):
        save_thread_ids.append(threading.current_thread().ident)
        return original_save(self, filepath, options)

    monkeypatch.setattr(Project, "save", _spy_save)

    out = tmp_path / "saved.nseq"
    _patch_save_dialog(monkeypatch, str(out))

    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=15000)

    assert out.exists()
    assert save_thread_ids
    assert save_thread_ids[0] != threading.main_thread().ident


def test_file_menu_disabled_during_save(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch: MonkeyPatch, tmp_path
):
    gui.load_project(default_project())

    block = threading.Event()
    original_save = Project.save

    def _blocking_save(self, filepath, options):
        block.wait(timeout=5.0)
        return original_save(self, filepath, options)

    monkeypatch.setattr(Project, "save", _blocking_save)
    out = tmp_path / "saved.nseq"
    _patch_save_dialog(monkeypatch, str(out))

    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is not None, timeout=2000)

    assert not gui.menu_file.isEnabled()
    assert not gui.toolbar_file.isEnabled()
    assert not gui.continue_simulation_action.isEnabled()

    block.set()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=15000)

    assert gui.menu_file.isEnabled()
    assert gui.toolbar_file.isEnabled()
    assert gui.continue_simulation_action.isEnabled()


def test_save_failure_surfaces_error_and_keeps_ui_unfrozen(
    qtbot: QtBot,
    gui: ProjectWorkspace,
    monkeypatch: MonkeyPatch,
    tmp_path,
    qmessagebox_calls,
):
    gui.load_project(default_project())

    def _failing_save(self, filepath, options):
        raise OSError("simulated permission denied")

    monkeypatch.setattr(Project, "save", _failing_save)
    out = tmp_path / "saved.nseq"
    _patch_save_dialog(monkeypatch, str(out))

    gui.btn_save_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=15000)

    assert not out.exists()
    # UI restored to enabled state after the failure.
    assert gui.menu_file.isEnabled()
    assert gui.toolbar_file.isEnabled()
    # The user got a critical error dialog.
    assert any(
        call["name"] == "critical" and "permission denied" in call["text"].lower()
        for call in qmessagebox_calls
    )


def test_load_runs_on_worker_thread(
    qtbot: QtBot, gui: ProjectWorkspace, monkeypatch: MonkeyPatch, tmp_path
):
    # First save a project synchronously via Project.save (worker test
    # already covers async save).
    src_project = default_project()
    src_path = tmp_path / "src.nseq"
    src_project.save(
        src_path,
        SaveLoadOptions(
            include_stim_arrays=True,
            include_model=True,
            include_simulation_data=False,
            include_experimental_data=False,
        ),
    )

    load_thread_ids: list[int | None] = []
    original_init = Project.__init__

    def _spy_init(self, *args, **kwargs):
        load_thread_ids.append(threading.current_thread().ident)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(Project, "__init__", _spy_init)

    _patch_open_dialog(monkeypatch, str(src_path))
    gui.btn_open_clicked()
    qtbot.waitUntil(lambda: gui._background_job.worker is None, timeout=15000)

    # First Project() call is from the file-path-driven load_project →
    # _load_task on the worker thread. (Any subsequent Project() calls
    # the load flow triggers — e.g. inside update_project — run on the
    # main thread and are not the one we care about.)
    assert load_thread_ids
    worker_init_seen = any(
        tid != threading.main_thread().ident for tid in load_thread_ids
    )
    assert worker_init_seen
    assert gui.project.filepath == src_path
