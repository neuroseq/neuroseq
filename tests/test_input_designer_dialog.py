from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq.data import TraceData
from neuroseq.input_designer_dialog import EPSPMeasurement
from neuroseq.input_designer_dialog import InputDesignerDialog
from neuroseq.stim import StimArray
from neuroseq.stim import StimType


class _FakeProject:
    def __init__(self) -> None:
        self.trace = TraceData(
            "Simulation :: P / v",
            np.zeros((2, 2000), dtype=np.float32),
            np.array([0, 1]),
            "mV",
            0.001,
        )
        self.b2_model = SimpleNamespace(get_sim_time_secs=lambda: 0.0)

    def get_data_names(self, data_type: type[TraceData]) -> list[str]:
        assert data_type is TraceData
        return [self.trace.name]

    def get_data(self, name: str, data_type: type[TraceData]) -> TraceData:
        assert name == self.trace.name
        assert data_type is TraceData
        return self.trace


class _FakeEditor:
    def __init__(self, project: _FakeProject) -> None:
        self.project = project
        stim = StimArray(
            "Exc_stimulation",
            StimType.Excitation,
            np.zeros((10, 2)),
            "mwatt / um2",
            0.001,
        )
        self.current_stim_modifier = SimpleNamespace(stim_array=stim)
        self.calls: list[np.ndarray] = []
        self.focused_ranges: list[tuple[float, float]] = []
        self.controller = SimpleNamespace(
            focus_time_range=lambda start, stop: self.focused_ranges.append(
                (start, stop)
            )
        )

    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, values: np.ndarray
    ) -> None:
        del neuron_slice, time_slice
        self.calls.append(values.copy())
        stim = self.current_stim_modifier.stim_array
        stim.values = np.vstack((stim.values, values))


def test_input_designer_runs_probe_pairing_probe_and_reaches_target(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    amplitudes = iter([2.0, 2.5])
    measured_neurons: list[int] = []

    def measure(trace: TraceData) -> EPSPMeasurement:
        del trace
        measured_neurons.append(dialog.spn_postsynaptic.value())
        amplitude = next(amplitudes)
        return EPSPMeasurement(amplitude, (amplitude,) * 5)

    monkeypatch.setattr(dialog, "_measure_epsps", measure)
    dialog.spn_target.setValue(120.0)
    dialog.spn_tolerance.setValue(10.0)

    dialog.start()
    assert len(editor.calls) == 1
    assert np.count_nonzero(editor.calls[0][:, 0]) == 5
    assert np.count_nonzero(editor.calls[0][:, 1]) == 0

    dialog.simulation_finished()
    assert len(editor.calls) == 2
    pairing = editor.calls[1]
    pre_sample = int(np.flatnonzero(pairing[:, 0])[0])
    post_sample = int(np.flatnonzero(pairing[:, 1])[0])
    assert post_sample - pre_sample == 10

    dialog.simulation_finished()
    assert len(editor.calls) == 3
    dialog.simulation_finished()

    assert not dialog._running
    assert dialog._epsp_percentages == [100.0, 125.0]
    assert "target achieved" in dialog.log.toPlainText()
    assert "absolute 5.000%" in dialog.lbl_relative_difference.text()
    assert measured_neurons[0] == 1


def test_input_designer_uses_post_pre_pairing_for_depression(
    qtbot, monkeypatch
) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog,
        "_measure_epsps",
        lambda trace: EPSPMeasurement(2.0, (2.0,) * 5),
    )
    dialog.spn_target.setValue(80.0)
    dialog.spn_tolerance.setValue(5.0)

    dialog.start()
    dialog.simulation_finished()

    pairing = editor.calls[1]
    assert np.flatnonzero(pairing[:, 1])[0] < np.flatnonzero(pairing[:, 0])[0]
    assert "depression" in dialog.log.toPlainText()


def test_input_designer_camera_includes_three_prior_blocks(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)

    dialog.start()

    block_samples = editor.calls[0].shape[0]
    expected_start = max(0, 10 - 3 * block_samples) * 0.001
    assert editor.focused_ranges == [(expected_start, (10 + block_samples) * 0.001)]


def test_input_designer_reuses_epsp_plugin_detection(qtbot) -> None:
    project = _FakeProject()
    editor = _FakeEditor(project)
    dialog = InputDesignerDialog(editor)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    pulse_samples = [50, 150, 250, 350, 450]
    x = np.arange(project.trace.traces.shape[1])
    project.trace.traces[1] = -70.0
    for center in pulse_samples:
        project.trace.traces[1] += 5.0 * np.exp(-(((x - center) / 4.0) ** 2))
    dialog._probe_pulse_times = [sample * project.trace.dt for sample in pulse_samples]

    measurement = dialog._measure_epsps(project.trace)

    # The shared plugin's local-baseline window includes the early Gaussian
    # rise, so its measured prominence is slightly below the injected 5 mV.
    assert measurement.amplitude == pytest.approx(4.62, abs=0.05)
    assert len(measurement.pulse_amplitudes) == 5
