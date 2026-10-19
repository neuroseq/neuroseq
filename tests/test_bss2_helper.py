"""The in-container helper, imported by path with the BSS-2 stack stubbed."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from typing import Any

import numpy as np
import pytest

from neuroseq.device_control import bss2_protocol as bp


@pytest.fixture()
def helper():
    spec = importlib.util.spec_from_file_location(
        "bss2_helper_under_test", bp.HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_helper_imports_no_neuroseq() -> None:
    source = bp.HELPER_PATH.read_text()
    assert "import neuroseq" not in source and "from neuroseq" not in source


def test_helper_codec_is_compatible_with_protocol(helper) -> None:
    values = np.array([[1, 2], [3, 4]], np.uint16)
    assert np.array_equal(helper.decode_array(bp.encode_array(values, "<u2")), values)
    assert np.array_equal(bp.decode_array(helper.encode_array(values, "<u2")), values)
    with pytest.raises(ValueError):
        helper.encode_array(values, "<i8")
    with pytest.raises(ValueError):
        helper.decode_array({"dtype": "<i8", "shape": [1], "data": ""})
    with pytest.raises(ValueError):
        helper.decode_array({**bp.encode_array(values, "<u2"), "shape": [5]})


# ---- fake hardware stack ------------------------------------------------------


class _Quantity(np.ndarray):
    """Just enough of a ``quantities`` array: ``rescale`` and ``magnitude``."""

    def __new__(cls, values, annotations=None):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.annotations = annotations or {}
        return obj

    def __array_finalize__(self, obj):
        self.annotations = getattr(obj, "annotations", {})

    def rescale(self, unit):
        assert unit == "ms"
        return self

    @property
    def magnitude(self):
        return np.asarray(self)


class _Signal(_Quantity):
    times: Any


class FakePynn(types.ModuleType):
    def __init__(self, log: list) -> None:
        super().__init__("pynn_brainscales.brainscales2")
        self.log = log
        self.RunCommand = types.SimpleNamespace(APPEND="APPEND", EXECUTE="EXECUTE")
        fake = self

        class _View:
            def __init__(self, label):
                self.label = label

            def record(self, variables):
                fake.log.append(("record", self.label, list(variables)))

            def set(self, **kwargs):
                fake.log.append(("set", self.label, kwargs))

            def __getitem__(self, item):
                fake.log.append(("view", item.start, item.stop))
                return _View(f"{self.label}[{item.start}:{item.stop}]")

            def get_data(self, variable):
                fake.log.append(("get_data", self.label, variable))
                if variable == "spikes":
                    trains = [
                        _Quantity([0.25], {"source_index": 0}),
                        _Quantity([], {}),
                        _Quantity([0.5, 0.75], {"source_index": 2}),
                    ]
                    return types.SimpleNamespace(
                        segments=[types.SimpleNamespace(spiketrains=trains)]
                    )
                signal = _Signal([[1.0], [2.0], [3.0]])
                signal.times = _Quantity([0.0, 0.1, 0.2])
                segment = types.SimpleNamespace(
                    irregularlysampledsignals=[signal] if fake.irregular else [],
                    analogsignals=[] if fake.irregular else [signal],
                )
                return types.SimpleNamespace(segments=[segment])

        self.irregular = True
        self._View = _View
        self.cells = types.SimpleNamespace(
            CalibHXNeuronCoba=lambda **kw: ("CalibHXNeuronCoba", kw),
            SpikeSourceArray=lambda spike_times: ("SpikeSourceArray", spike_times),
        )
        self.synapses = types.SimpleNamespace(
            StaticSynapse=lambda weight: ("StaticSynapse", weight)
        )

    def setup(self):
        self.log.append(("setup",))

    def end(self):
        self.log.append(("end",))

    def run(self, duration, command=None):
        self.log.append(("run", duration, command))

    def Population(self, n, cell_type):
        self.log.append(("Population", n, cell_type[0]))
        return self._View(cell_type[0])

    def OneToOneConnector(self):
        return "one-to-one"

    def Projection(self, pre, post, connector, *, synapse_type, receptor_type):
        self.log.append(
            (
                "Projection",
                pre.label,
                post.label,
                connector,
                synapse_type,
                receptor_type,
            )
        )


@pytest.fixture()
def fake_stack(monkeypatch: pytest.MonkeyPatch):
    log: list = []
    pynn = FakePynn(log)
    package = types.ModuleType("pynn_brainscales")
    package.brainscales2 = pynn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynn_brainscales", package)
    monkeypatch.setitem(sys.modules, "pynn_brainscales.brainscales2", pynn)

    class Connection:
        def __enter__(self):
            log.append(("connect",))
            return self

        def __exit__(self, *exc):
            log.append(("disconnect",))

        def get_unique_identifier(self):
            return "hxcube7fpga3chip61_1"

    dlens = types.ModuleType("dlens_vx_v3")
    dlens.hxcomm = types.SimpleNamespace(ManagedConnection=Connection)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dlens_vx_v3", dlens)
    return pynn


def _serve(helper, requests: list[dict]) -> list[dict]:
    out = io.StringIO()
    helper.serve([json.dumps(r) + "\n" for r in requests], out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _cc_arguments(**overrides) -> dict:
    settings = bp.Bss2Settings(madc_neuron=1)
    values = np.array([[0.0, 1.0], [2.0, 1.0], [0.0, 0.0]], np.float32)
    arguments = bp.build_run_arguments(
        bp.plan_constant_current(values, 0.1, settings), settings
    )
    arguments.update(overrides)
    return arguments


def test_serve_ping_connect_shutdown(helper, fake_stack, monkeypatch) -> None:
    monkeypatch.setenv("QUIGGELDY_TOKEN", "tok")
    for key in ("QUIGGELDY_IP", "QUIGGELDY_PORT"):
        monkeypatch.delenv(key, raising=False)
    messages = _serve(
        helper,
        [
            {"id": 1, "command": "ping"},
            {
                "id": 2,
                "command": "connect",
                "arguments": {"host": "h.example", "port": 11717, "user": "alice"},
            },
            {
                "id": 3,
                "command": "connect",
                "arguments": {"host": "h.example", "identifier": "other"},
            },
            {"id": 4, "command": "shutdown"},
            {"id": 5, "command": "ping"},  # never answered
        ],
    )
    assert messages[0]["event"] == "ready" and messages[0]["id"] is None
    assert messages[1]["result"]["pong"] is True
    replies = {m["id"]: m for m in messages if "result" in m or "error" in m}
    assert replies[2]["result"] == {
        "identifier": "hxcube7fpga3chip61_1",
        "matches_expected": True,
    }
    assert replies[3]["result"]["matches_expected"] is False
    assert replies[4]["result"] == {"bye": True}
    assert 5 not in replies
    import os

    assert os.environ["QUIGGELDY_IP"] == "h.example"
    assert os.environ["QUIGGELDY_PORT"] == "11717"
    assert os.environ["QUIGGELDY_USER_NO_MUNGE"] == "alice"
    assert os.environ["QUIGGELDY_ENABLED"] == "1"
    assert "tok" not in json.dumps(messages)


def test_configure_quiggeldy_warns_without_token_or_host(helper) -> None:
    environ: dict[str, str] = {}
    warnings = helper.configure_quiggeldy({}, environ)
    assert environ == {"QUIGGELDY_ENABLED": "1"}
    assert len(warnings) == 2


def test_serve_reports_errors_and_skips_blank_lines(helper) -> None:
    out = io.StringIO()
    helper.serve(
        ["\n", "not json\n", "[1]\n", json.dumps({"id": 7, "command": "nope"}) + "\n"],
        out,
    )
    messages = [json.loads(line) for line in out.getvalue().splitlines()][1:]
    assert [m["id"] for m in messages] == [None, None, 7]
    assert all("error" in m for m in messages)
    assert "Unknown command" in messages[2]["error"]


def test_run_constant_current_appends_segments(helper, fake_stack) -> None:
    messages = _serve(
        helper, [{"id": 1, "command": "run", "arguments": _cc_arguments()}]
    )
    result = messages[-1]["result"]
    assert any(m.get("event") == "progress" for m in messages)
    log = fake_stack.log
    runs = [entry for entry in log if entry[0] == "run"]
    assert [r[2] for r in runs] == ["APPEND", "APPEND", "EXECUTE"]
    assert [r[1] for r in runs] == pytest.approx([0.1, 0.1, 0.1])
    sets = [entry[2] for entry in log if entry[0] == "set"]
    assert sets[1] == {
        "constant_current_enable": [True, True],
        "constant_current_i_offset": [500, 250],
    }
    assert ("record", "CalibHXNeuronCoba", ["spikes"]) in log
    assert ("record", "CalibHXNeuronCoba[1:2]", ["v"]) in log
    assert log[0] == ("setup",) and log[-1] == ("end",)
    assert bp.decode_array(result["spikes"]["indices"]).tolist() == [0, 2, 2]
    assert bp.decode_array(result["spikes"]["times_ms"]).tolist() == [0.25, 0.5, 0.75]
    assert result["v"]["neuron"] == 1
    assert bp.decode_array(result["v"]["values"]).tolist() == [1.0, 2.0, 3.0]
    readback = bp.decode_run_result(result, 1000.0)
    assert readback.trace is not None and readback.trace.traces.shape == (1, 3)


def test_run_spike_input_builds_projection(helper, fake_stack) -> None:
    fake_stack.irregular = False  # fall back to analog signals
    settings = bp.Bss2Settings(spike_weight=20)
    spikes = bp.SpikeData("s", np.array([1, 0, 1]), np.array([0.3, 0.1, 0.2]))
    arguments = bp.build_run_arguments(
        bp.plan_spikes_from_data(spikes, 1.0, settings), settings
    )
    messages = _serve(helper, [{"id": 1, "command": "run", "arguments": arguments}])
    assert "result" in messages[-1], messages[-1]
    log = fake_stack.log
    assert ("Population", 2, "SpikeSourceArray") in log
    projection = next(entry for entry in log if entry[0] == "Projection")
    assert projection[1:] == (
        "SpikeSourceArray",
        "CalibHXNeuronCoba",
        "one-to-one",
        ("StaticSynapse", 20),
        "excitatory",
    )
    assert ("run", pytest.approx(1.0), None) in log
    assert log[-1] == ("end",)


def test_spike_trains_without_pynn_sequence(helper, monkeypatch) -> None:
    trains = helper.spike_trains(np.array([1, 0, 1]), np.array([3.0, 1.0, 2.0]), 3)
    assert trains == [[1.0], [2.0, 3.0], []]


def test_run_ends_session_on_failure(helper, fake_stack, monkeypatch) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("chip unavailable")

    monkeypatch.setattr(fake_stack, "run", broken)
    messages = _serve(
        helper, [{"id": 1, "command": "run", "arguments": _cc_arguments()}]
    )
    assert "chip unavailable" in messages[-1]["error"]
    assert fake_stack.log[-1] == ("end",)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"protocol": 99}, "protocol"),
        ({"n_neurons": 0}, "n_neurons"),
        ({"duration_ms": 0}, "duration"),
        ({"record_v_neuron": 5}, "record_v_neuron"),
        ({"mode": "other"}, "Unknown input mode"),
    ],
)
def test_parse_run_arguments_validates(helper, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        helper.parse_run_arguments(_cc_arguments(**overrides))


def test_parse_run_arguments_validates_arrays(helper) -> None:
    arguments = _cc_arguments()
    segments = dict(arguments["segments"])
    segments["offset"] = bp.encode_array(np.full((3, 2), 2000), "<u2")
    with pytest.raises(ValueError, match="i_offset"):
        helper.parse_run_arguments({**arguments, "segments": segments})
    segments["offset"] = bp.encode_array(np.zeros((2, 2)), "<u2")
    with pytest.raises(ValueError, match="do not match"):
        helper.parse_run_arguments({**arguments, "segments": segments})
    segments["durations_ms"] = bp.encode_array(np.zeros(0), "<f4")
    with pytest.raises(ValueError, match="At least one"):
        helper.parse_run_arguments({**arguments, "segments": segments})
    segments = dict(arguments["segments"])
    segments["durations_ms"] = bp.encode_array(np.array([0.1, 0.0, 0.1]), "<f4")
    with pytest.raises(ValueError, match="durations"):
        helper.parse_run_arguments({**arguments, "segments": segments})

    spikes = {
        "indices": bp.encode_array(np.array([0, 9]), "<u4"),
        "times_ms": bp.encode_array(np.array([0.1, 0.2]), "<f4"),
        "weight": 1,
    }
    base = {**arguments, "mode": bp.INPUT_SPIKES}
    with pytest.raises(ValueError, match="outside"):
        helper.parse_run_arguments({**base, "spikes": spikes})
    spikes["indices"] = bp.encode_array(np.array([0]), "<u4")
    with pytest.raises(ValueError, match="differ"):
        helper.parse_run_arguments({**base, "spikes": spikes})
    spikes["indices"] = bp.encode_array(np.array([0, 1]), "<u4")
    spikes["weight"] = 64
    with pytest.raises(ValueError, match="weight"):
        helper.parse_run_arguments({**base, "spikes": spikes})


def test_collect_signal_rejects_mismatched_lengths(helper) -> None:
    signal = _Signal([1.0, 2.0])
    signal.times = _Quantity([0.0])
    with pytest.raises(ValueError, match="differ"):
        helper.collect_signal(signal)
    assert helper.collect_spikes([])[0].size == 0


def test_environment_report_masks_the_token(helper) -> None:
    report = helper.environment_report(
        {"QUIGGELDY_IP": "h", "QUIGGELDY_TOKEN": "abcdef", "SYSTEMNAME": "lab"}
    )
    assert report["QUIGGELDY_IP"] == "h" and report["SYSTEMNAME"] == "lab"
    assert report["QUIGGELDY_TOKEN_STATUS"] == "set (6 characters)"
    assert "abcdef" not in str(report) and "QUIGGELDY_TOKEN" not in report
    assert report["QUIGGELDY_PORT"] == "<unset>"
    assert helper.environment_report({})["QUIGGELDY_TOKEN_STATUS"] == "unset"
    assert (
        helper.environment_report({"QUIGGELDY_TOKEN": ""})["QUIGGELDY_TOKEN_STATUS"]
        == "empty"
    )


def test_package_versions_never_raise(helper, monkeypatch) -> None:
    def broken(name):
        if name == "numpy":
            raise RuntimeError("bad metadata")
        raise helper.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(helper.importlib.metadata, "version", broken)
    versions = helper.package_versions()
    assert versions["pynn-brainscales"] == "not installed"
    assert versions["numpy"] == "unknown (RuntimeError)"
