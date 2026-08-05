import importlib

import numpy as np
from qtpy import QtWidgets

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.custom_analysis_widget import CustomAnalysisWidget
from neuroseq.data import TraceData

epsp = importlib.import_module("neuroseq.resources.analysis_plugins.epsp_amplitudes")


def _trace() -> np.ndarray:
    values = np.full(500, -70.0)
    x = np.arange(values.size)
    for center, amplitude, width in ((100, 5.0, 7.0), (250, 8.0, 9.0)):
        values += amplitude * np.exp(-(((x - center) / width) ** 2))
    values += 80.0 * np.exp(-(((x - 380) / 1.2) ** 2))  # action potential
    return values


def test_detect_epsps_excludes_action_potential_and_uses_local_baseline() -> None:
    result = epsp.detect_epsps(_trace(), dt_ms=0.1)
    np.testing.assert_allclose(result.times_ms, [10.0, 25.0], atol=0.2)
    np.testing.assert_allclose(result.amplitudes_mv, [5.0, 8.0], atol=0.2)


def test_detect_epsps_on_offset_removed_trace() -> None:
    # Same signal shifted so the baseline sits at ~0 mV (DC offset removed).
    # The absolute -20 mV action-potential guard must not apply here, but the
    # AP-sized deflection must still be rejected by the amplitude guard.
    result = epsp.detect_epsps(_trace() + 70.0, dt_ms=0.1)
    np.testing.assert_allclose(result.times_ms, [10.0, 25.0], atol=0.2)
    np.testing.assert_allclose(result.amplitudes_mv, [5.0, 8.0], atol=0.2)


def test_voltage_unit_aliases_are_converted_to_mv() -> None:
    values = np.array([-0.07, -0.065])
    np.testing.assert_allclose(epsp._to_mv(values, "volt"), [-70.0, -65.0])
    np.testing.assert_allclose(epsp._to_mv(values * 1000, "[millivolts]"), [-70, -65])
    np.testing.assert_allclose(epsp._to_mv(values * 1e6, "µvolt"), [-70, -65])


def test_unlabelled_voltage_units_are_inferred() -> None:
    np.testing.assert_allclose(epsp._to_mv(np.array([-0.070, -0.065]), ""), [-70, -65])
    np.testing.assert_allclose(epsp._to_mv(np.array([-70.0, -65.0]), ""), [-70, -65])


def test_epsp_plugin_controls_and_percent_plot(qtbot) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    plugin = next(p for p in discover_analysis_plugins() if p.id == "epsp_amplitudes")
    data = TraceData("Vm", _trace()[None, :], np.array([0]), "mV", 0.0001)
    widget.set_data(data, plugin, None, None)

    controls = widget._epsp_amplitude_controls
    checkbox = controls.findChild(QtWidgets.QCheckBox)
    spin = controls.findChild(QtWidgets.QSpinBox)
    units = controls.findChild(QtWidgets.QComboBox)
    assert [units.itemText(i) for i in range(units.count())] == [
        "Auto",
        "V",
        "mV",
        "µV",
    ]
    assert widget.subplot.get_ylabel() == "EPSP amplitude (mV)"
    checkbox.setChecked(True)
    spin.setValue(2)
    assert widget.subplot.get_ylabel() == "EPSP amplitude (%)"
    offsets = widget.subplot.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 1], [100 * 5 / 6.5, 100 * 8 / 6.5], atol=3)


def test_epsp_plugin_uses_different_markers_for_neurons(qtbot) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    plugin = next(p for p in discover_analysis_plugins() if p.id == "epsp_amplitudes")
    data = TraceData(
        "Vm", np.stack([_trace(), _trace()]), np.array([0, 1]), "mV", 0.0001
    )
    widget.set_data(data, plugin, None, None)

    first_path = widget.subplot.collections[0].get_paths()[0]
    second_path = widget.subplot.collections[1].get_paths()[0]
    assert not np.array_equal(first_path.vertices, second_path.vertices)
    assert [text.get_text() for text in widget.subplot.get_legend().get_texts()] == [
        "Neuron 0",
        "Neuron 1",
    ]


def test_epsp_percent_reference_is_calculated_per_neuron(qtbot) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    plugin = next(p for p in discover_analysis_plugins() if p.id == "epsp_amplitudes")
    data = TraceData(
        "Vm",
        np.stack([_trace(), -70.0 + 2.0 * (_trace() + 70.0)]),
        np.array([0, 1]),
        "mV",
        0.0001,
    )
    widget.set_data(data, plugin, None, None)

    controls = widget._epsp_amplitude_controls
    controls.findChild(QtWidgets.QCheckBox).setChecked(True)
    controls.findChild(QtWidgets.QSpinBox).setValue(2)

    expected = [100 * 5 / 6.5, 100 * 8 / 6.5]
    for collection in widget.subplot.collections:
        np.testing.assert_allclose(collection.get_offsets()[:, 1], expected, atol=3)


def test_explicit_input_units_override_unlabelled_trace(qtbot) -> None:
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    plugin = next(p for p in discover_analysis_plugins() if p.id == "epsp_amplitudes")
    data = TraceData("Vm", (_trace() / 1000)[None, :], np.array([0]), "", 0.0001)
    widget.set_data(data, plugin, None, None)

    units = widget._epsp_amplitude_controls.findChild(QtWidgets.QComboBox)
    units.setCurrentText("V")
    offsets = widget.subplot.collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 1], [5.0, 8.0], atol=0.2)
