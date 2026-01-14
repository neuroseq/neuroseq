from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import pytest
from qtpy import QtWidgets

from neuroseq.b2_model import B2Model
from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType
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


@pytest.fixture(autouse=True)
def suppress_message_boxes(monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
    )


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


def _generate_suite2p_trace_arrays(
    n_neurons: int = 5,
    n_timepoints: int = 900,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple Suite2p-style arrays with fixed means/stds."""
    rng = np.random.default_rng(seed)

    f_data = rng.normal(
        loc=146.0,
        scale=14.401437,
        size=(n_neurons, n_timepoints),
    ).astype(np.float32)

    fneu_data = rng.normal(
        loc=133.0,
        scale=12.630169,
        size=(n_neurons, n_timepoints),
    ).astype(np.float32)

    spike_data = np.zeros((n_neurons, n_timepoints), dtype=np.float32)
    total_points = n_neurons * n_timepoints
    n_active = int(total_points * 0.04)  # make 4% of points have a spike
    active_indices = rng.choice(total_points, size=n_active, replace=False)
    spike_values = rng.uniform(0.0, 0.05, size=n_active).astype(np.float32)
    spike_data.ravel()[active_indices] = spike_values

    return f_data, fneu_data, spike_data


@pytest.fixture(scope="session")
def suite2p_import_data_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> pathlib.Path:
    """Provide a temp directory with generated Suite2p-style .npy files for tests."""
    data_dir = pathlib.Path(tmp_path_factory.mktemp("suite2p_import"))
    f_data, fneu_data, spike_data = _generate_suite2p_trace_arrays()

    np.save(data_dir / "F.npy", f_data)
    np.save(data_dir / "Fneu.npy", fneu_data)
    np.save(data_dir / "spks.npy", spike_data)
    # optional metadata file
    stat_data = np.zeros(5, dtype=object)
    np.save(data_dir / "stat.npy", stat_data)

    # CSV fallbacks for less advanced users
    np.savetxt(data_dir / "F.csv", f_data, delimiter=",")
    np.savetxt(data_dir / "Fneu.csv", fneu_data, delimiter=",")
    np.savetxt(data_dir / "spks.csv", spike_data, delimiter=",")

    return data_dir


@pytest.fixture(scope="session")
def test_stim_arrays() -> dict[str, StimArray]:
    tfinal = 2000 * b2.ms
    data = np.array(
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype="int"
    )
    data_inv = np.array(~np.array(data, dtype="bool"), dtype="int")
    stim = np.zeros((int(tfinal / b2.ms), 30), dtype="float")
    stim[:, :] = 0
    stim_constant = 300
    stimulus = StimArray("stimulus", StimType.Excitation, stim, "siemens", 0.001)
    stim_inh = np.zeros(int(tfinal / b2.ms), dtype="float")
    stim_inh[0:500] = 0
    stim_inh[500:-1] = 0
    stimulus_inh = StimArray(
        "stimulus_inh", StimType.Inhibition, stim_inh, "siemens", 0.001
    )

    for length in range(3):
        for k in range(20):
            stim[200 * length + 120 : 200 * length + 121, k + 4] = (
                stim_constant * data[k]
            )
            stim[200 * length + 118 : 200 * length + 119, k + 4] = (
                stim_constant * data_inv[k]
            )
        stim[200 * length + 119 : 200 * length + 120, 0] = stim_constant

    return {
        "stimulus": stimulus,
        "stimulus_inh": stimulus_inh,
    }


@pytest.fixture(scope="session")
def reference_parameters() -> dict[str, Any]:
    tfinal = 2000 * b2.ms

    area = 20000 * b2.umetre**2

    taupre = 20 * b2.ms
    taupost = taupre
    dApre = 6 * b2.nS

    return {
        "n": 30,
        "tfinal": tfinal,
        "dt": 0.1 * b2.ms,
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
def b2model() -> B2Model:
    return B2Model()


@pytest.fixture
def project_ornstein() -> Project:
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """
    b2_model = B2Model(seed=4321)
    b2_model.define_neuronnumber("neurongroup", 5)
    b2_model.define_dt("neurongroup", 0.5 * b2.ms)
    b2_model.define_solver_method("neurongroup", "euler")
    b2_model.define_equation("neurongroup", eqs)
    b2_model._assemble_neurongroup(name="neurongroup")

    b2_model.define_monitor_source("M", "neurongroup")
    b2_model.define_monitor_type("M", "StateMonitor")
    b2_model.define_monitor_variable("M", "y")
    b2_model._assemble_monitor("M")

    # initial conditions
    b2_model.define_init(
        "neurongroup", "tau", np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    )
    b2_model.define_init("neurongroup", "y", 1)
    b2_model.define_init(
        "neurongroup", "sigma", np.array([0.1, 0.31622, 0.4, 0.2, 0.88])
    )

    for n in b2_model.initials:
        b2_model._assemble_initial_conditions(n)

    b2_model.must_be_rebuilt = False

    project = Project()
    project.b2_model = b2_model
    return project


@pytest.fixture
def b2_model_nmda() -> B2Model:
    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = B2Model(seed=4321)

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
def project_nmda(b2_model_nmda: B2Model) -> Project:
    project = Project()
    project.b2_model = b2_model_nmda
    return project


@pytest.fixture
def mfx_parameterized(
    b2model: B2Model, reference_parameters: dict[str, Any]
) -> tuple[B2Model, dict[str, Any]]:
    b2.seed(reference_parameters["seed"])
    n = 30
    threshold = "v>-20*mV"
    refractory = 3 * b2.ms
    method = "exponential_euler"
    dt = 0.1 * b2.ms
    b2model.define_equation("P", reference_parameters["eqs"])
    b2model.define_neuronnumber("P", n)
    b2model.define_threshold("P", threshold)
    b2model.define_refractory("P", refractory)
    b2model.define_dt("P", dt)
    b2model.define_solver_method("P", method)
    b2model._assemble_neurongroup("P")

    return b2model, reference_parameters


@pytest.fixture()
def mfx_with_synapse(
    mfx_parameterized: tuple[B2Model, dict[str, Any]],
) -> tuple[B2Model, dict[str, Any]]:
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


@pytest.fixture
def test_project() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 30
    nneurons_exc = 24

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("gl", 10.0 * b2.nsiemens)
    b2_model.define_parameter("gmax", 60.0 * b2.nsiemens)

    b2_model.define_parameter("El", -60 * b2.mV)
    b2_model.define_parameter("EK", -90 * b2.mV)
    b2_model.define_parameter("ENa", 50 * b2.mV)
    b2_model.define_parameter("g_na", 20.0 * b2.usiemens)
    b2_model.define_parameter("g_kd", 6.0 * b2.usiemens)
    b2_model.define_parameter("VT", -63 * b2.mV)
    # Time constants
    b2_model.define_parameter("taue", 5 * b2.ms)
    b2_model.define_parameter("taui", 10 * b2.ms)
    b2_model.define_parameter("taupost", 20 * b2.ms)
    b2_model.define_parameter("taupre", 20 * b2.ms)
    ## Reversal potentials
    b2_model.define_parameter("Ee", 0 * b2.mV)
    b2_model.define_parameter("Ei", -80 * b2.mV)

    ### optogenetics parameters
    b2_model.define_parameter("Eeopto", 0 * b2.mV)
    b2_model.define_parameter("Eiopto", -80 * b2.mV)

    b2_model.define_parameter("we", 6 * b2.nS)
    b2_model.define_parameter("wi", 67 * b2.nS)

    b2_model.define_parameter("dApost", -6.3 * b2.nsiemens)
    b2_model.define_parameter("dApre", 6.0 * b2.nsiemens)

    eqs = """
dv/dt = (gl*(El-v)+ge*(Ee-v)+geopto*(Eeopto-v)+gi*(Ei-v)+giopto*(Eiopto-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
giopto = stimulus_inh(t, i): siemens
geopto = stimulus(t, i): siemens
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    """

    # neuron group
    b2_model.define_neuronnumber("P", nneurons)
    b2_model.define_solver_method("P", "exponential_euler")
    b2_model.define_equation("P", eqs)
    b2_model.define_threshold("P", "v>-20*mV")
    b2_model.define_refractory("P", 3 * b2.ms)

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    ce_eq = """
we : siemens
dApre/dt = -Apre / taupre : siemens (event-driven)
dApost/dt = -Apost / taupost : siemens (event-driven)
    """
    b2_model.define_synapse_equation("Ce", ce_eq)
    b2_model.define_synapse_connection("Ce", connection_condition="i != j")
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "Pe")

    b2_model.define_synapse_equation("Cie", "wi = 67*nS : siemens")
    b2_model.define_synapse_connection("Cie", None)
    b2_model.define_synapse_origin("Cie", "Pi")
    b2_model.define_synapse_target("Cie", "Pe")

    b2_model.define_synapse_equation("Cei", "we = 6*nS : siemens")
    b2_model.define_synapse_connection("Cei", None)
    b2_model.define_synapse_origin("Cei", "Pe")
    b2_model.define_synapse_target("Cei", "Pi")

    ce_on_pre = """
ge += we
Apre += dApre
we = clip(we + Apost, -gmax, gmax)
"""
    b2_model.define_synapse_pre("Ce", ce_on_pre)
    b2_model.define_synapse_pre("Cei", "ge+=we")
    b2_model.define_synapse_pre("Cie", "gi+=wi")

    ce_on_post = """
Apost += dApost
we= clip(we + Apre, -gmax, gmax)
"""
    b2_model.define_synapse_post("Ce", ce_on_post)

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("spike", "P")
    b2_model.define_monitor_type("spike", "SpikeMonitor")
    b2_model.define_monitor_variable("spike", "v")

    b2_model.define_monitor_source("synapse", "Ce")
    b2_model.define_monitor_type("synapse", "StateMonitor")
    b2_model.define_monitor_variable("synapse", "we")

    # define stim arrays

    tfinal = 1000 * b2.ms

    stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    stim[120, :nneurons_exc:2] = 1
    stim[121, 1:nneurons_exc:2] = 1
    stim[320, :nneurons_exc:2] = 1
    stim[321, 1:nneurons_exc:2] = 1
    stim[520, :nneurons_exc:2] = 1
    stim[521, 1:nneurons_exc:2] = 1
    stim_array = StimArray(
        name="stimulus",
        stim_type=StimType.Excitation,
        values=stim,
        units="siemens",
        dt=0.001,
    )
    project.add_stim_array(stim_array)

    stim_inh = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    stim_inh[800:802, :nneurons_exc] = 1
    stim_array_inh = StimArray(
        name="stimulus_inh",
        stim_type=StimType.Inhibition,
        values=stim_inh,
        units="siemens",
        dt=0.001,
    )
    project.add_stim_array(stim_array_inh)

    return project
