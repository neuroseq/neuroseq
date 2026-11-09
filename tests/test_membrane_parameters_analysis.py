from __future__ import annotations

import brian2.numpy_ as np
import pytest

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _plugin_module():
    plugin = next(
        plugin
        for plugin in discover_analysis_plugins()
        if plugin.id == "membrane_parameters"
    )
    return plugin.render.__module__, plugin


def test_membrane_parameter_plugin_is_discovered() -> None:
    _, plugin = _plugin_module()
    assert plugin.label == "Estimate membrane parameters"
    assert plugin.open_dialog is not None


def test_fit_a_b_c_recovers_parameters() -> None:
    from neuroseq.resources.analysis_plugins.membrane_parameters import fit_a_b_c

    v = np.linspace(-72.0, -48.0, 100)
    current = np.sin(np.linspace(0.0, 8.0, 100)) * 20.0
    expected = (2.5, -65.0, 0.02)
    derivative = (-(v - expected[1]) + expected[0] * current) / expected[2]

    assert fit_a_b_c(derivative, v, current) == pytest.approx(expected)


def test_estimate_parameters_uses_selected_time_and_neuron() -> None:
    from neuroseq.resources.analysis_plugins.membrane_parameters import (
        estimate_parameters,
    )

    dt = 0.001
    r_m, v_rest, tau_m = 250e6, -0.065, 0.02
    current = np.tile(np.sin(np.linspace(0.0, 20.0, 300))[:, None], (1, 3)) * 40
    voltage = np.empty(300)
    voltage[0] = -0.060
    for sample in range(299):
        voltage[sample + 1] = (
            voltage[sample]
            + dt
            * (-(voltage[sample] - v_rest) + r_m * current[sample, 2] * 1e-12)
            / tau_m
        )
    stimulus = StimArray("current", StimType.External, current, "pA", dt)

    result = estimate_parameters(
        voltage[25:250][None, :],
        dt,
        stimulus,
        "mV",
        first_neuron=2,
        first_sample=25,
    )

    assert len(result) == 1
    assert result[0].neuron == 2
    assert (result[0].r_m, result[0].v_rest, result[0].tau_m) == pytest.approx(
        (r_m, v_rest, tau_m), rel=1e-5
    )
    assert result[0].c_m == pytest.approx(tau_m / r_m, rel=1e-5)


def test_format_capacitance_uses_cell_scale_units() -> None:
    from neuroseq.resources.analysis_plugins.membrane_parameters import (
        _format_capacitance,
    )

    assert _format_capacitance(80e-12) == "80 pF"
    assert _format_capacitance(1.5e-9) == "1.5 nF"


@pytest.mark.parametrize("units", ["A", "mA", "µA", "uA", "nA", "pA", "ampere"])
def test_current_unit_detection(units: str) -> None:
    from neuroseq.resources.analysis_plugins.membrane_parameters import is_current_unit

    assert is_current_unit(units)


def test_voltage_unit_is_not_supported_as_current() -> None:
    from neuroseq.resources.analysis_plugins.membrane_parameters import is_current_unit

    assert not is_current_unit("mV")


def test_membrane_dialog_fits_the_trace_row_of_the_selected_neuron_id(qtbot) -> None:
    from neuroseq.data import TraceData
    from neuroseq.resources.analysis_plugins.membrane_parameters import (
        MembraneParametersDialog,
    )

    dt = 0.001
    r_m, v_rest, tau_m = 250e6, -0.065, 0.02
    current = np.tile(np.sin(np.linspace(0.0, 20.0, 300))[:, None], (1, 2)) * 40
    current[:, 0] = 0.0  # neuron 0 gets no input
    voltage = np.empty(300)
    voltage[0] = -0.060
    for sample in range(299):
        voltage[sample + 1] = (
            voltage[sample]
            + dt
            * (-(voltage[sample] - v_rest) + r_m * current[sample, 1] * 1e-12)
            / tau_m
        )
    # Row 0 holds neuron 1 (the driven one), row 1 holds neuron 0.
    data = TraceData(
        "v", np.stack([voltage, np.full(300, v_rest)]), np.array([1, 0]), "mV", dt
    )
    stimulus = StimArray("current", StimType.External, current, "pA", dt)

    dialog = MembraneParametersDialog(
        data, [stimulus], slice(1, 2), slice(0, 250), None
    )
    qtbot.addWidget(dialog)

    text = dialog.results.toPlainText()
    assert text.startswith("Neuron 1\n")
    assert "250 MΩ" in text
