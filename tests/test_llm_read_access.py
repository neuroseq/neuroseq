from __future__ import annotations

import json
from pathlib import Path

import brian2 as b2
import numpy as np
import pytest

from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.chat_widget import _ChatWorker
from neuroseq.llm.read_access import PRIVATE_DATA_NOTICE
from neuroseq.llm.read_access import PRIVATE_PROJECT_SECTIONS
from neuroseq.llm.read_access import ReadAccess
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.wait_timeout import WAIT_TIMEOUT_MS


def test_project_reads_values_and_units_without_modifying_project():
    project = Project()
    project.b2_model.parameters["voltage"] = 5 * b2.mV
    project.b2_model.neurongroups["cells"] = {"equation": "dv/dt = -v / tau : volt"}
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    project.stim_arrays["input"] = StimArray(
        "input", StimType.Excitation, values, "mV", 0.001
    )
    reader = ReadAccess(project)

    def read(path, **kwargs):
        return json.loads(reader.execute({"kind": "project", "path": path, **kwargs}))

    assert "model" in read([])["keys"]
    assert (
        read(["model", "neurongroups", "cells", "equation"])["value"]
        == "dv/dt = -v / tau : volt"
    )
    voltage = read(["model", "parameters", "voltage"])
    assert voltage["values"] == [0.005]
    assert "units" in voltage
    page = read(["stim_arrays", "input", "values"], offset=4, limit=3)
    assert page["shape"] == [4, 3]
    assert page["values"] == [4, 5, 6]
    np.testing.assert_array_equal(project.stim_arrays["input"].values, values)
    assert "error" in read(["__dict__"])


def test_source_reads_pages_and_rejects_escape(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "outside.py").write_text("private", encoding="utf-8")
    reader = ReadAccess(None, source)
    listing = json.loads(reader.execute({"kind": "source"}))
    assert listing["files"] == ["example.py"]
    page = json.loads(
        reader.execute(
            {"kind": "source", "path": "example.py", "offset": 1, "limit": 1}
        )
    )
    assert page["lines"] == ["two\n"]
    for path in ("../outside.py", str(tmp_path / "outside.py"), "example.txt"):
        assert "error" in json.loads(reader.execute({"kind": "source", "path": path}))


@pytest.mark.parametrize(
    "read_request",
    [
        {"kind": "project"},
        {"kind": "source", "offset": -1},
        {"kind": "source", "limit": 501},
        {"kind": "source", "limit": True},
        {"kind": "execute", "code": "print('no')"},
    ],
)
def test_invalid_or_unavailable_reads_return_errors(read_request):
    assert "error" in json.loads(ReadAccess(None).execute(read_request))


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def stream_chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        yield from next(self.replies)


def test_worker_reads_then_answers_without_streaming_protocol(qtbot):
    backend = ScriptedBackend(
        [
            ['{"neuroseq_', 'read": {"kind": "source", "path": ""}}'],
            ["Here ", "is the answer."],
        ]
    )
    worker = _ChatWorker(backend, [Message(Role.USER, "Explain the code")], "fake")
    tokens = []
    worker.token.connect(tokens.append)
    worker.read_requested.connect(
        lambda request: worker.deliver_read('{"files": ["gui.py"]}')
    )
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    worker.wait()
    assert "".join(tokens) == "Here is the answer."
    assert "gui.py" in backend.calls[1][-1].content


def test_worker_stop_while_waiting_for_read(qtbot):
    worker = _ChatWorker(
        ScriptedBackend([['{"neuroseq_read": {"kind": "project"}}']]), [], "fake"
    )
    worker.read_requested.connect(lambda request: worker.request_stop())
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    assert worker.wait(WAIT_TIMEOUT_MS)


def test_worker_limits_reads(qtbot):
    request = '{"neuroseq_read": {"kind": "source"}}'
    backend = ScriptedBackend([[request]] * 13)
    worker = _ChatWorker(backend, [], "fake")
    tokens = []
    worker.token.connect(tokens.append)
    worker.read_requested.connect(lambda request: worker.deliver_read("{}"))
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    worker.wait()
    assert len(backend.calls) == 13
    assert "Read limit reached" in "".join(tokens)


PLUGIN_SOURCE = """PLUGIN_ID = "chat_test_plugin"
PLUGIN_NAME = "Chat test plugin"

def render_analysis(widget, data, neuron_slice, time_slice):
    widget.clear()
"""


class ConfirmStub:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls: list[tuple[Path, str, str | None]] = []

    def __call__(self, target: Path, content: str, previous: str | None) -> bool:
        self.calls.append((target, content, previous))
        return self.accept


def _write_request(path="custom.py", content=PLUGIN_SOURCE, **extra):
    return {"kind": "analysis_plugin", "path": path, "content": content, **extra}


def test_write_plugin_create_update_and_discover(isolate_user_plugin_dir):
    from neuroseq import analysis_plugins

    confirm = ConfirmStub(accept=True)
    reader = ReadAccess(None, confirm_write=confirm)
    write_request = _write_request()
    result = json.loads(reader.write_plugin(write_request))
    target = isolate_user_plugin_dir.resolve() / "custom.py"
    assert result["saved"] == str(target)
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE
    assert confirm.calls == [(target, PLUGIN_SOURCE, None)]
    ids = [plugin.id for plugin in analysis_plugins.discover_analysis_plugins()]
    assert "chat_test_plugin" in ids
    assert "mean_activity" in ids
    # Nothing is written into the package plugin folder.
    assert not (analysis_plugins._plugins_directory() / "custom.py").exists()

    updated = PLUGIN_SOURCE + "\n# Updated\n"
    write_request["content"] = updated
    assert "error" in json.loads(reader.write_plugin(write_request))
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE
    write_request["overwrite"] = True
    assert "saved" in json.loads(reader.write_plugin(write_request))
    assert target.read_text(encoding="utf-8") == updated
    # Overwrites hand the current file to the confirmation for a diff.
    assert confirm.calls[-1] == (target, updated, PLUGIN_SOURCE)
    assert not list(target.parent.glob("*.tmp"))


def test_declined_plugin_write_saves_nothing(isolate_user_plugin_dir):
    confirm = ConfirmStub(accept=False)
    result = json.loads(
        ReadAccess(None, confirm_write=confirm).write_plugin(_write_request())
    )
    assert "declined" in result
    assert "declined" in result["message"]
    assert len(confirm.calls) == 1
    assert not list(isolate_user_plugin_dir.glob("*"))


def test_declined_overwrite_keeps_existing_plugin(isolate_user_plugin_dir):
    isolate_user_plugin_dir.mkdir(parents=True, exist_ok=True)
    target = isolate_user_plugin_dir / "custom.py"
    target.write_text(PLUGIN_SOURCE, encoding="utf-8")
    confirm = ConfirmStub(accept=False)
    result = json.loads(
        ReadAccess(None, confirm_write=confirm).write_plugin(
            _write_request(content=PLUGIN_SOURCE + "# changed\n", overwrite=True)
        )
    )
    assert "declined" in result
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE


def test_default_confirmation_declines_without_explicit_accept(
    isolate_user_plugin_dir, qtbot
):
    # With exec() patched to show(), the dialog never reports Accepted.
    result = json.loads(ReadAccess(None).write_plugin(_write_request()))
    assert "declined" in result
    assert not (isolate_user_plugin_dir / "custom.py").exists()


@pytest.mark.parametrize(
    "filename",
    ["_helper.py", "__init__.py", "_replay_metrics.py"],
)
def test_plugin_writes_reject_underscore_names(isolate_user_plugin_dir, filename):
    confirm = ConfirmStub(accept=True)
    result = ReadAccess(None, confirm_write=confirm).write_plugin(
        _write_request(path=filename, overwrite=True)
    )
    assert "underscore" in json.loads(result)["error"]
    assert confirm.calls == []


@pytest.mark.parametrize("filename", ["mean_activity.py", "elise_target_mse.py"])
def test_plugin_writes_reject_builtin_names(isolate_user_plugin_dir, filename):
    confirm = ConfirmStub(accept=True)
    result = ReadAccess(None, confirm_write=confirm).write_plugin(
        _write_request(path=filename, overwrite=True)
    )
    assert "built-in" in json.loads(result)["error"]
    assert confirm.calls == []
    assert not (isolate_user_plugin_dir / filename).exists()


def test_plugin_write_falls_back_when_hard_links_unsupported(
    isolate_user_plugin_dir, monkeypatch
):
    from neuroseq.llm import read_access

    def no_link(*args, **kwargs):
        raise OSError("hard links not supported")

    monkeypatch.setattr(read_access.os, "link", no_link)
    reader = ReadAccess(None, confirm_write=ConfirmStub(accept=True))
    assert "saved" in json.loads(reader.write_plugin(_write_request()))
    target = isolate_user_plugin_dir / "custom.py"
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE
    assert not list(isolate_user_plugin_dir.glob("*.tmp"))


def test_plugin_write_rejects_change_during_confirmation(isolate_user_plugin_dir):
    isolate_user_plugin_dir.mkdir(parents=True, exist_ok=True)
    target = isolate_user_plugin_dir / "custom.py"
    target.write_text(PLUGIN_SOURCE, encoding="utf-8")

    def confirm(path, content, previous):
        path.write_text("# replaced meanwhile\n", encoding="utf-8")
        return True

    result = ReadAccess(None, confirm_write=confirm).write_plugin(
        _write_request(content=PLUGIN_SOURCE + "# new\n", overwrite=True)
    )
    assert "error" in json.loads(result)
    assert target.read_text(encoding="utf-8") == "# replaced meanwhile\n"


def test_plugin_write_preview_shows_code_or_diff(qtbot, tmp_path):
    from neuroseq.llm.plugin_write_dialog import PluginWriteConfirmDialog
    from neuroseq.llm.plugin_write_dialog import plugin_write_preview

    target = tmp_path / "custom.py"
    assert plugin_write_preview(target, PLUGIN_SOURCE, None) == PLUGIN_SOURCE
    diff = plugin_write_preview(target, PLUGIN_SOURCE + "x = 1\n", PLUGIN_SOURCE)
    assert diff.startswith(f"--- {target} (current)")
    assert "+x = 1" in diff

    dialog = PluginWriteConfirmDialog(target, PLUGIN_SOURCE + "x = 1\n", PLUGIN_SOURCE)
    qtbot.addWidget(dialog)
    assert dialog.code_view.isReadOnly()
    assert "+x = 1" in dialog.code_view.toPlainText()
    assert str(target) in dialog.message_label.text()
    assert dialog.cancel_button.isDefault()


def test_runtime_state_read_refused_while_background_job_runs():
    project = Project()
    busy = [True]
    reader = ReadAccess(project, is_busy=lambda: busy[0])
    result = json.loads(reader.execute({"kind": "project", "path": ["runtime_state"]}))
    assert "background job" in result["error"]
    busy[0] = False
    result = json.loads(reader.execute({"kind": "project", "path": ["runtime_state"]}))
    assert "error" not in result


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.py",
        "nested/plugin.py",
        "C:/outside.py",
        "gui.py:stream",
        "plugin.txt",
        "__init__.py",
    ],
)
def test_plugin_writes_reject_other_paths(tmp_path, isolate_user_plugin_dir, filename):
    result = ReadAccess(
        None, tmp_path, confirm_write=ConfirmStub(accept=True)
    ).write_plugin(_write_request(path=filename))
    assert "error" in json.loads(result)
    assert not list(tmp_path.rglob("*.py"))
    assert not list(isolate_user_plugin_dir.parent.rglob("*.py"))


def test_plugin_writes_reject_symlinked_target(tmp_path, isolate_user_plugin_dir):
    isolate_user_plugin_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    (isolate_user_plugin_dir / "custom.py").symlink_to(outside)
    result = ReadAccess(None, confirm_write=ConfirmStub(accept=True)).write_plugin(
        _write_request(overwrite=True)
    )
    assert "error" in json.loads(result)
    assert outside.read_text(encoding="utf-8") == "# outside\n"


@pytest.mark.parametrize(
    "content",
    ["def broken(", "print('must not run')", "x" * 200_001],
    ids=["invalid_syntax", "missing_render", "oversized"],
)
def test_plugin_writes_validate_before_saving(isolate_user_plugin_dir, content):
    confirm = ConfirmStub(accept=True)
    result = ReadAccess(None, confirm_write=confirm).write_plugin(
        _write_request(content=content)
    )
    assert "error" in json.loads(result)
    assert confirm.calls == []
    assert not list(isolate_user_plugin_dir.parent.rglob("*.py"))


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"neuroseq_read": {"kind": "source", "path": ""}}\n```',
        '```\n{"neuroseq_read": {"kind": "source", "path": ""}}\n```\n',
        '  {"neuroseq_read": {"kind": "source", "path": ""}}',
    ],
    ids=["json_fence", "bare_fence", "plain"],
)
def test_worker_accepts_code_fenced_requests(qtbot, reply):
    backend = ScriptedBackend([[reply[:5], reply[5:]], ["Done."]])
    worker = _ChatWorker(backend, [], "fake")
    tokens = []
    requests = []
    worker.token.connect(tokens.append)

    def receive(envelope):
        requests.append(envelope)
        worker.deliver_read("{}")

    worker.read_requested.connect(receive)
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    worker.wait()
    assert requests == [
        {"operation": "neuroseq_read", "request": {"kind": "source", "path": ""}}
    ]
    assert "".join(tokens) == "Done."


def test_worker_streams_backtick_answer_that_is_not_a_request(qtbot):
    backend = ScriptedBackend([["`x` is ", "a variable."]])
    worker = _ChatWorker(backend, [], "fake")
    tokens = []
    worker.token.connect(tokens.append)
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    worker.wait()
    assert "".join(tokens) == "`x` is a variable."


def test_worker_dispatches_plugin_writes(qtbot):
    backend = ScriptedBackend(
        [
            [
                json.dumps(
                    {
                        "neuroseq_write": {
                            "kind": "analysis_plugin",
                            "path": "custom.py",
                            "content": PLUGIN_SOURCE,
                        }
                    }
                )
            ],
            ["Saved custom.py."],
        ]
    )
    worker = _ChatWorker(backend, [], "fake")
    requests = []

    def receive(envelope):
        requests.append(envelope)
        worker.deliver_read('{"saved": "resources/analysis_plugins/custom.py"}')

    worker.read_requested.connect(receive)
    with qtbot.waitSignal(worker.completed, timeout=WAIT_TIMEOUT_MS):
        worker.start()
    worker.wait()
    assert requests[0]["operation"] == "neuroseq_write"
    assert requests[0]["request"]["content"] == PLUGIN_SOURCE
    assert "custom.py" in backend.calls[1][-1].content


def _plugin_chat_widget(qtbot, monkeypatch, accept):
    from neuroseq.llm import chat_widget

    monkeypatch.setattr(chat_widget.ChatWidget, "reload_backend", lambda self: None)
    confirm = ConfirmStub(accept=accept)
    monkeypatch.setattr(
        chat_widget.ChatWidget,
        "_confirm_plugin_write",
        lambda self, *args: confirm(*args),
    )
    widget = chat_widget.ChatWidget()
    qtbot.addWidget(widget)
    widget._backend = ScriptedBackend(
        [
            [
                "```json\n",
                json.dumps(
                    {
                        "neuroseq_write": {
                            "kind": "analysis_plugin",
                            "path": "from_chat.py",
                            "content": PLUGIN_SOURCE,
                        }
                    }
                ),
                "\n```",
            ],
            ["Done."],
        ]
    )
    return widget, confirm


def test_chat_widget_saves_confirmed_plugin(
    qtbot, monkeypatch, isolate_user_plugin_dir
):
    widget, confirm = _plugin_chat_widget(qtbot, monkeypatch, accept=True)
    widget.input_edit.setPlainText("Create an analysis plugin")
    widget._on_send_clicked()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)
    assert (isolate_user_plugin_dir / "from_chat.py").read_text(
        encoding="utf-8"
    ) == PLUGIN_SOURCE
    assert len(confirm.calls) == 1
    assert widget._history[-1].content == "Done."


def test_chat_widget_reports_declined_plugin(
    qtbot, monkeypatch, isolate_user_plugin_dir
):
    widget, confirm = _plugin_chat_widget(qtbot, monkeypatch, accept=False)
    widget.input_edit.setPlainText("Create an analysis plugin")
    widget._on_send_clicked()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)
    assert not (isolate_user_plugin_dir / "from_chat.py").exists()
    assert len(confirm.calls) == 1
    # The model is told the user declined.
    assert "declined" in widget._backend.calls[1][-1].content


def test_chat_widget_passes_busy_check_to_read_access(qtbot, monkeypatch):
    from neuroseq.llm import chat_widget

    monkeypatch.setattr(chat_widget.ChatWidget, "reload_backend", lambda self: None)
    widget = chat_widget.ChatWidget()
    qtbot.addWidget(widget)
    widget.set_project(Project())
    widget.set_busy_check(lambda: True)
    widget._backend = ScriptedBackend(
        [['{"neuroseq_read": {"kind": "project", "path": ["runtime_state"]}}'], ["Ok"]]
    )
    widget.input_edit.setPlainText("State?")
    widget._on_send_clicked()
    qtbot.waitUntil(lambda: widget._worker is None, timeout=WAIT_TIMEOUT_MS)
    assert "background job" in widget._backend.calls[1][-1].content


def test_private_data_hides_experimental_sections_but_keeps_model():
    project = Project()
    project.b2_model.parameters["voltage"] = 5 * b2.mV
    project.stim_arrays["input"] = StimArray(
        "input", StimType.Excitation, np.ones((4, 3), np.float32), "mV", 0.001
    )
    reader = ReadAccess(project, private_data=True)

    def read(path):
        return json.loads(reader.execute({"kind": "project", "path": path}))

    for section in PRIVATE_PROJECT_SECTIONS:
        assert read([section]) == {"error": PRIVATE_DATA_NOTICE}
    assert read(["stim_arrays", "input", "values"]) == {"error": PRIVATE_DATA_NOTICE}
    assert read(["model", "parameters", "voltage"])["values"] == [0.005]
    listed = read([])
    assert "experiments" in listed["keys"]

    shared = ReadAccess(project)
    shared_read = json.loads(
        shared.execute({"kind": "project", "path": ["stim_arrays", "input", "values"]})
    )
    assert shared_read["shape"] == [4, 3]
