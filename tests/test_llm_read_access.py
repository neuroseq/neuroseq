from __future__ import annotations

import json
from pathlib import Path

import brian2 as b2
import numpy as np
import pytest

from neuroseq.llm.backend import Message
from neuroseq.llm.backend import Role
from neuroseq.llm.chat_widget import _ChatWorker
from neuroseq.llm.read_access import ReadAccess
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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
    with qtbot.waitSignal(worker.completed):
        worker.start()
    worker.wait()
    assert "".join(tokens) == "Here is the answer."
    assert "gui.py" in backend.calls[1][-1].content


def test_worker_stop_while_waiting_for_read(qtbot):
    worker = _ChatWorker(
        ScriptedBackend([['{"neuroseq_read": {"kind": "project"}}']]), [], "fake"
    )
    worker.read_requested.connect(lambda request: worker.request_stop())
    with qtbot.waitSignal(worker.completed):
        worker.start()
    assert worker.wait(1000)


def test_worker_limits_reads(qtbot):
    request = '{"neuroseq_read": {"kind": "source"}}'
    backend = ScriptedBackend([[request]] * 13)
    worker = _ChatWorker(backend, [], "fake")
    tokens = []
    worker.token.connect(tokens.append)
    worker.read_requested.connect(lambda request: worker.deliver_read("{}"))
    with qtbot.waitSignal(worker.completed):
        worker.start()
    worker.wait()
    assert len(backend.calls) == 13
    assert "Read limit reached" in "".join(tokens)


PLUGIN_SOURCE = """PLUGIN_ID = "chat_test_plugin"
PLUGIN_NAME = "Chat test plugin"

def render_analysis(widget, data, neuron_slice, time_slice):
    widget.clear()
"""


def test_write_plugin_create_update_and_discover(tmp_path, monkeypatch):
    from neuroseq import analysis_plugins

    reader = ReadAccess(None, tmp_path)
    write_request = {
        "kind": "analysis_plugin",
        "path": "custom.py",
        "content": PLUGIN_SOURCE,
    }
    result = json.loads(reader.write_plugin(write_request))
    target = tmp_path / "resources" / "analysis_plugins" / "custom.py"
    assert result["saved"] == "resources/analysis_plugins/custom.py"
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE
    monkeypatch.setattr(analysis_plugins, "_plugins_directory", lambda: target.parent)
    assert [plugin.id for plugin in analysis_plugins.discover_analysis_plugins()] == [
        "chat_test_plugin"
    ]
    updated = PLUGIN_SOURCE + "\n# Updated\n"
    write_request["content"] = updated
    assert "error" in json.loads(reader.write_plugin(write_request))
    assert target.read_text(encoding="utf-8") == PLUGIN_SOURCE
    write_request["overwrite"] = True
    assert "saved" in json.loads(reader.write_plugin(write_request))
    assert target.read_text(encoding="utf-8") == updated
    assert not list(target.parent.glob("*.tmp"))


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
def test_plugin_writes_reject_other_paths(tmp_path, filename):
    result = ReadAccess(None, tmp_path).write_plugin(
        {"kind": "analysis_plugin", "path": filename, "content": PLUGIN_SOURCE}
    )
    assert "error" in json.loads(result)
    assert not list(tmp_path.rglob("*.py"))


@pytest.mark.parametrize(
    "content",
    ["def broken(", "print('must not run')", "x" * 200_001],
    ids=["invalid_syntax", "missing_render", "oversized"],
)
def test_plugin_writes_validate_before_saving(tmp_path, content):
    result = ReadAccess(None, tmp_path).write_plugin(
        {"kind": "analysis_plugin", "path": "custom.py", "content": content}
    )
    assert "error" in json.loads(result)
    assert not list(tmp_path.rglob("*.py"))


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
    with qtbot.waitSignal(worker.completed):
        worker.start()
    worker.wait()
    assert requests[0]["operation"] == "neuroseq_write"
    assert requests[0]["request"]["content"] == PLUGIN_SOURCE
    assert "custom.py" in backend.calls[1][-1].content


def test_chat_widget_saves_requested_plugin(qtbot, tmp_path, monkeypatch):
    from neuroseq.llm import chat_widget

    monkeypatch.setattr(chat_widget.ChatWidget, "reload_backend", lambda self: None)
    monkeypatch.setattr(
        chat_widget, "ReadAccess", lambda project: ReadAccess(project, tmp_path)
    )
    widget = chat_widget.ChatWidget()
    qtbot.addWidget(widget)
    widget._backend = ScriptedBackend(
        [
            [
                json.dumps(
                    {
                        "neuroseq_write": {
                            "kind": "analysis_plugin",
                            "path": "from_chat.py",
                            "content": PLUGIN_SOURCE,
                        }
                    }
                )
            ],
            ["Saved from_chat.py."],
        ]
    )
    widget.input_edit.setPlainText("Create an analysis plugin")
    widget._on_send_clicked()
    qtbot.waitUntil(lambda: widget._worker is None)
    assert (tmp_path / "resources" / "analysis_plugins" / "from_chat.py").read_text(
        encoding="utf-8"
    ) == PLUGIN_SOURCE
    assert widget._history[-1].content == "Saved from_chat.py."
