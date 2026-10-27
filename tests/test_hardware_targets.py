from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.stimulation.dynamic_input.hardware_targets import HostConnections
from neuroseq.stimulation.dynamic_input.hardware_targets import PerTargetValues


class _Host(QtCore.QObject):
    done = QtCore.Signal(object)


def test_host_connections_disconnect_exactly_what_was_connected() -> None:
    calls: list[object] = []
    host = _Host()
    connections = HostConnections()
    connections.connect(host, [(host.done, calls.append)])
    connections.connect(host, [(host.done, calls.append)])  # once per host
    host.done.emit(1)
    assert calls == [1]
    connections.disconnect_all()
    host.done.emit(2)
    assert calls == [1]
    connections.disconnect_all()  # nothing left: no error


def test_per_target_values_remember_each_target(qtbot) -> None:
    amplitude = QtWidgets.QDoubleSpinBox()
    amplitude.setRange(0, 1000)
    repeats = QtWidgets.QSpinBox()
    amplitude.setValue(100.0)
    repeats.setValue(1)
    values = PerTargetValues(
        (amplitude, repeats),
        lambda target: None if target == "sim" else (0.5, 3),
        "sim",
    )
    values.switch("chip")
    assert (amplitude.value(), repeats.value()) == (0.5, 3)
    amplitude.setValue(0.7)
    values.switch("sim")
    assert (amplitude.value(), repeats.value()) == (100.0, 1)
    values.switch("chip")
    assert amplitude.value() == 0.7
    values.switch("chip")  # same target: nothing changes
    assert amplitude.value() == 0.7
