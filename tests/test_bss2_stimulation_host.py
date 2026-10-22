"""Running dynamic input blocks on BrainScaleS-2, against the fake chip of the
BrainScaleS-2 tab tests (its enabled neurons spike once in the middle of each
constant-current segment)."""

from __future__ import annotations

import numpy as np
import pytest

from neuroseq.data import SpikeData
from neuroseq.device_control import bss2_protocol as bp
from neuroseq.project import Project
from neuroseq.stimulation.dynamic_input.dynamic_input_measurement import (
    measure_calcium_spike_count,
)
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from tests.test_bss2_panel import _connect
from tests.test_bss2_panel import devices  # noqa: F401 - fixture
from tests.test_bss2_panel import panel  # noqa: F401 - fixture


def _probe_array() -> StimArray:
    """Two blocks on 3 neurons: a probe of neuron 1 at 0.5 of i_offset max from
    0.2-0.4 s of the second block (samples 10-20, dt 0.1 s)."""
    values = np.zeros((20, 3), np.float32)
    values[2:5, 0] = 0.9  # an earlier block, not sent
    values[12:14, 1] = 0.5
    return StimArray("probe", StimType.External, values, "a.u.", 0.1)


def test_block_runs_on_the_chip_and_its_spikes_reach_the_project(qtbot, panel):  # noqa: F811
    project = Project()
    host = panel.stimulation_host
    assert "Connect to BrainScaleS-2" in host.unavailable()
    _connect(qtbot, panel)
    assert host.unavailable() is None
    sent: list[dict] = []
    original_run = panel._session.run

    def capture(request_id, arguments):
        sent.append(arguments)
        return original_run(request_id, arguments)

    panel._session.run = capture  # type: ignore[method-assign]
    owner = object()
    with qtbot.waitSignal(host.block_completed, timeout=20000) as completed:
        assert host.send_block(
            project, _probe_array(), 10, 20, "rheobase", owner=owner, record_neuron=1
        )
        assert host.busy and panel.run_in_progress
        assert "still running" in host.unavailable()

    arguments = sent[0]
    assert arguments["n_neurons"] == 3 and arguments["record_v_neuron"] == 1
    # Only the block is sent, and 0.5 means half of i_offset max (500), not the
    # block's own maximum.
    assert arguments["duration_ms"] == pytest.approx(1.0)
    offsets = bp.decode_array(arguments["segments"]["offset"])
    assert offsets[:, 1].tolist() == [0, 250, 0]
    assert offsets[:, 0].tolist() == [0, 0, 0]

    payload_owner, payload = completed.args
    assert payload_owner is owner
    experiment = project.experiments[payload["experiment"]]
    assert payload["experiment"].startswith("BSS-2 rheobase ")
    spikes = project.get_data(payload["spikes"], SpikeData)
    assert spikes is not None and spikes in experiment.data
    assert "trace" in payload
    # The fake chip fires neuron 1 in the middle of the pulse (0.2-0.4 s).
    assert spikes.spike_indices.tolist() == [1]
    assert spikes.spike_times.tolist() == pytest.approx([0.3])
    assert measure_calcium_spike_count(spikes, 1, 0.2, 200.0) == 1
    assert measure_calcium_spike_count(spikes, 0, 0.2, 200.0) == 0
    assert not host.busy


def test_cancelled_block_result_is_ignored(qtbot, panel):  # noqa: F811
    project = Project()
    _connect(qtbot, panel)
    host = panel.stimulation_host
    owner = object()
    completed: list = []
    host.block_completed.connect(lambda *args: completed.append(args))
    host.send_block(project, _probe_array(), 10, 20, owner=owner)
    host.cancel(owner=object())  # someone else's: ignored
    assert host.busy
    host.cancel(owner=owner)
    assert not host.busy
    qtbot.waitUntil(lambda: not panel._session.busy, timeout=20000)
    assert not completed
    assert not project.experiments


def test_failed_run_and_invalid_blocks_fail_the_block(qtbot, panel):  # noqa: F811
    host = panel.stimulation_host
    owner = object()
    with qtbot.waitSignal(host.block_failed) as failed:
        assert not host.send_block(Project(), _probe_array(), 0, 5, owner=owner)
    assert failed.args[0] is owner and "Connect" in failed.args[1]
    _connect(qtbot, panel)
    with qtbot.waitSignal(host.block_failed) as failed:
        assert not host.send_block(Project(), _probe_array(), 5, 5, owner=owner)
    assert "Invalid sample range" in failed.args[1]
    with qtbot.waitSignal(host.block_failed) as failed:
        assert not host.send_block(
            Project(), _probe_array(), 0, 5, owner=owner, record_neuron=7
        )
    assert "recorded neuron 7" in failed.args[1]
    host.send_block(Project(), _probe_array(), 10, 20, owner=owner)
    request_id = host._run.request_id
    with qtbot.waitSignal(host.block_failed) as failed:
        panel._session.run_failed.emit("someone-else", "ignored")
        panel._session.run_failed.emit(request_id, "chip went away")
    assert failed.args == [owner, "chip went away"]
