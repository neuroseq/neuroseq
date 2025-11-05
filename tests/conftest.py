from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import pytest
from qtpy import QtWidgets

from neuroseq import b2_model
from neuroseq.utils import example_models_path
from neuroseq.utils import init_brian2_logging

b2.prefs.codegen.target = "numpy"


def pytest_addoption(parser):
    parser.addoption(
        "--update-screenshot",
        action="store_true",
        default=False,
        help="Update screenshot.png",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--update-screenshot"):
        for item in items:
            if "screenshot" not in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Updating screenshot"))
    else:
        for item in items:
            if "screenshot" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping screenshot update"))


# replace all .exec() calls with .show() to avoid blocking the tests
@pytest.fixture(autouse=True)
def patch_b2w_exec_show(monkeypatch: pytest.MonkeyPatch) -> None:
    for widget in [
        QtWidgets.QDialog,
        QtWidgets.QMessageBox,
    ]:
        monkeypatch.setattr(widget, "exec", lambda self: self.show())


@pytest.fixture()
def qmessagebox_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    # create a list to store any QMessageBox calls that are made
    calls = []

    def _make_callback(name: str) -> Callable:
        # this function has the same calling interface as the qmessagebox static methods like information()
        def _mock(parent: Any, title: str, text: str) -> None:
            # but instead of displaying the message box it just appends the supplied arguments to the calls list
            calls.append({"name": name, "title": title, "text": text})

        return _mock

    for method_name in ["information", "warning", "critical"]:
        # mock QMessageBox static method `name` with our callback
        monkeypatch.setattr(
            QtWidgets.QMessageBox, method_name, _make_callback(method_name)
        )

    return calls


@pytest.fixture(autouse=True)
def reduce_brian2_logging_verbosity() -> None:
    init_brian2_logging()


# use temporary path for storing user settings
@pytest.fixture()
def tmp_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(
        "neuroseq.gui_settings.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    yield tmp_path


@pytest.fixture(scope="session")
def default_model_path() -> pathlib.Path:
    return example_models_path / "default_model.nseq"


@pytest.fixture(scope="session")
def reference_parameters() -> dict[str, Any]:
    data = np.array(
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype="int"
    )
    data_inv = np.array(~np.array(data, dtype="bool"), dtype="int")
    tfinal = 2000 * b2.ms

    stim = np.zeros((int(tfinal / b2.ms), 30), dtype="float")
    stim[:, :] = 0
    stim_constant = 300
    tfinal = 2000 * b2.ms
    stimulus = b2.TimedArray(stim * b2.nS, dt=1 * b2.ms)
    stim_inh = np.zeros(int(tfinal / b2.ms), dtype="float")
    stim_inh[0:500] = 0
    stim_inh[500:-1] = 0
    stimulus_inh = b2.TimedArray(stim_inh * b2.nS, dt=1 * b2.ms)

    for length in range(3):
        for k in range(20):
            stim[200 * length + 120 : 200 * length + 121, k + 4] = (
                stim_constant * data[k]
            )
            stim[200 * length + 118 : 200 * length + 119, k + 4] = (
                stim_constant * data_inv[k]
            )
        stim[200 * length + 119 : 200 * length + 120, 0] = stim_constant

    area = 20000 * b2.umetre**2

    taupre = 20 * b2.ms
    taupost = taupre
    dApre = 6 * b2.nS

    return {
        "n": 30,
        "tfinal": tfinal,
        "dt": 0.1 * b2.ms,
        "stimulus": stimulus,
        "stimulus_inh": stimulus_inh,
        "area": area,
        "Cm": (1 * b2.ufarad * b2.cm**-2) * area,
        "gl": (5e-5 * b2.siemens * b2.cm**-2) * area,
        "El": -60 * b2.mV,
        "EK": -90 * b2.mV,
        "ENa": 50 * b2.mV,
        "g_na": (100 * b2.msiemens * b2.cm**-2) * area,
        "g_kd": (30 * b2.msiemens * b2.cm**-2) * area,
        "VT": -63 * b2.mV,
        # Time constants
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        # Reversal potentials
        "Ee": 0 * b2.mV,
        "Ei": -80 * b2.mV,
        "Eiopto": -80 * b2.mV,  # −69.5*mV
        "Eeopto": 0 * b2.mV,
        "eqs": """
    dv/dt = (gl*(El-v)+ge*(Ee-v)+geopto*(Eeopto-v)+gi*(Ei-v)+giopto*(Eiopto-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK))/Cm : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dn/dt = alpha_n*(1-n)-beta_n*n : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    giopto = stimulus_inh(t): siemens
    geopto = stimulus(t, i): siemens
    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    """,
        "threshold": "v>-20*mV",
        "refractory": 3 * b2.ms,
        "method": "exponential_euler",
        "taupre": 20 * b2.ms,
        "taupost": taupre,
        "gmax": 60 * b2.nS,
        "dApre": 6 * b2.nS,
        "dApost": -dApre * taupre / taupost * 1.05,
        "ce_eq": """we : siemens
                    dApre/dt = -Apre / taupre : siemens (event-driven)
                    dApost/dt = -Apost / taupost : siemens (event-driven)""",
        "on_pre_ce": """ge += we
                        Apre += dApre
                        we = clip(we + Apost, -gmax, gmax)""",
        "on_post_ce": """Apost += dApost
                        we= clip(we + Apre, -gmax, gmax)""",
        "cei_eq": """we = 6*nS : siemens""",
        "cei_on_pre": "ge+=we",
        "cie_eq": """wi = 67*nS : siemens""",
        "cie_on_pre": "gi+=wi",
        "ce_connect": "i != j",
        "connect_p": 1.0,
        "seed": 4321,
        "v": "El + (randn() * 5 - 5)*mV",
        "ge": "(randn() * 1.5 + 4) * 10.*nS",
        "gi": "(randn() * 12 + 20) * 10.*nS",
    }


@pytest.fixture()
def mfx() -> b2_model.B2Model:
    return b2_model.B2Model()


@pytest.fixture
def mfx_ornstein() -> b2_model.B2Model:
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    model = b2_model.B2Model(seed=4321)
    model.define_neuronnumber("neurongroup", 5)
    model.define_dt("neurongroup", 0.5 * b2.ms)
    model.define_solver_method("neurongroup", "euler")
    model.define_equation("neurongroup", eqs)
    model._assemble_neurongroup(name="neurongroup")

    model.define_monitor_source("M", "neurongroup")
    model.define_monitor_type("M", "StateMonitor")
    model.define_monitor_variable("M", "y")
    model._assemble_monitor("M")

    # initial conditions
    model.define_init(
        "neurongroup", "tau", np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    )
    model.define_init("neurongroup", "y", 1)
    model.define_init("neurongroup", "sigma", np.array([0.1, 0.31622, 0.4, 0.2, 0.88]))

    for n in model.initials:
        model._assemble_initial_conditions(n)

    model.must_be_rebuilt = False

    return model


@pytest.fixture
def mfx_nmda() -> b2_model.B2Model:
    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = b2_model.B2Model(seed=4321)

    model.define_parameter("a", a)
    model.define_parameter("b", b)
    model.define_parameter("c", c)

    model.define_neuronnumber("neuron_input", 2)
    model.define_solver_method("neuron_input", "euler")
    model.define_equation("neuron_input", "dv/dt = 1/(10*ms) : 1")
    model.define_threshold("neuron_input", "v>1")
    model.define_reset("neuron_input", "v=0")

    model.define_neuronnumber("neurons", 1)
    model.define_equation(
        "neurons",
        """dv/dt = (g-v)/(10*ms) : 1
                            g : 1""",
    )
    model.define_solver_method("neurons", "exact")
    model._assemble_neurongroup("neurons")
    model._assemble_neurongroup("neuron_input")

    synapse_eq = """
        dg_syn/dt = -a*g_syn+b*x*(1-g_syn) : 1 (clock-driven)
        g_post = g_syn : 1 (summed)
        dx/dt=-c*x : 1 (clock-driven)
        w : 1 # synaptic weight
    """

    model.define_synapse_equation("S", synapse_eq)
    model.define_synapse_origin("S", "neuron_input")
    model.define_synapse_target("S", "neurons")
    model.define_synapse_pre("S", "x += w")
    model._assemble_synapse("S")

    model.define_neuronsubgroup_target("input_subgroup", "neuron_input")
    model.define_neuronsubgroup_idxs("input_subgroup", 1)
    model._assemble_neuronsubgroup("input_subgroup")

    model.define_monitor_source("M", "S")
    model.define_monitor_type("M", "StateMonitor")
    model.define_monitor_variable("M", "g")
    model._assemble_monitor("M")

    model.define_monitor_source("Mn", "neurons")
    model.define_monitor_type("Mn", "StateMonitor")
    model.define_monitor_variable("Mn", "g")
    model._assemble_monitor("Mn")

    model.define_init("S", "w", [1.0, 10.0])
    model.define_init("neuron_input", "v", [0.0, 0.5])

    for n in model.initials:
        model._assemble_initial_conditions(n)

    model.must_be_rebuilt = False
    return model


@pytest.fixture
def mfx_parameterized(
    mfx: b2_model.B2Model, reference_parameters: dict[str, Any]
) -> tuple[b2_model.B2Model, dict[str, Any]]:
    b2.seed(reference_parameters["seed"])
    n = 30
    threshold = "v>-20*mV"
    refractory = 3 * b2.ms
    method = "exponential_euler"
    dt = 0.1 * b2.ms
    mfx.define_equation("P", reference_parameters["eqs"])
    mfx.define_neuronnumber("P", n)
    mfx.define_threshold("P", threshold)
    mfx.define_refractory("P", refractory)
    mfx.define_dt("P", dt)
    mfx.define_solver_method("P", method)
    mfx._assemble_neurongroup("P")

    return mfx, reference_parameters


@pytest.fixture()
def mfx_with_synapse(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> tuple[b2_model.B2Model, dict[str, Any]]:
    taupre = 20 * b2.ms
    taupost = taupre
    gmax = 60 * b2.nS
    dApre = 6 * b2.nS
    dApost = -dApre * taupre / taupost * 1.05

    mfx, reference_parameters = mfx_parameterized

    mfx.define_neuronsubgroup_target("Pe", "P")
    mfx.define_neuronsubgroup_idxs("Pe", (0, 24))

    mfx.define_neuronsubgroup_target("Pi", "P")
    mfx.define_neuronsubgroup_idxs("Pi", (0, 24))

    mfx._assemble_neuronsubgroup("Pi")
    mfx._assemble_neuronsubgroup("Pe")

    mfx.define_parameter("taupre", taupre)

    mfx.define_parameter("taupost", taupost)

    mfx.define_parameter("gmax", gmax)

    mfx.define_parameter("dApre", dApre)

    mfx.define_parameter("dApost", dApost)

    mfx.define_synapse_connection("Ce", "i != j")
    mfx.define_synapse_connection_probability("Ce", 1.0)
    mfx.define_synapse_origin("Ce", "Pe")
    mfx.define_synapse_target("Ce", "Pi")
    mfx.define_synapse_equation(
        "Ce",
        """we : siemens
            dApre/dt = -Apre / taupre : siemens (event-driven)
            dApost/dt = -Apost / taupost : siemens (event-driven)""",
    )

    mfx.define_synapse_pre(
        "Ce",
        """ge += we
                    Apre += dApre
                    we = clip(we + Apost, -gmax, gmax)""",
    )

    mfx.define_synapse_post(
        "Ce",
        """Apost += dApost
                      we= clip(we + Apre, -gmax, gmax)""",
    )

    return mfx, reference_parameters
