from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import brian2 as b2
import h5py
import numpy as np
import pytest
from formozov_project import b2_model
from pytest_mock import MockerFixture

b2.prefs.codegen.target = "cython"


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
    mfx.define_dt(dt)
    mfx.define_solver_method("P", method)
    mfx.assemble_neurongroup("P")

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
    mfx.define_neuron_subgroup("P", "Pe", slice(0, 24))
    mfx.define_neuron_subgroup("P", "Pi", slice(24, 30))

    mfx.define_parameter("taupre", taupre)

    mfx.define_parameter("taupost", taupost)

    mfx.define_parameter("gmax", gmax)

    mfx.define_parameter("dApre", dApre)

    mfx.define_parameter("dApost", dApost)

    mfx.define_synapse_connection("Ce", "i != j")
    mfx.define_synapse_connection_probablitiy("Ce", 1.0)
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


def ornstein() -> b2_model.B2Model:
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    model = b2_model.B2Model(seed=4321)
    model.define_neuronnumber("neurongroup", 2)
    model.define_dt(0.2 * b2.ms)
    model.define_solver_method("neurongroup", "euler")
    model.define_equation("neurongroup", eqs)
    model.assemble_neurongroup(name="neurongroup")
    model.define_monitor("M", b2.StateMonitor, "neurongroup", "y")

    # init
    model.network["neurongroup"].tau = np.array([1, 0.1]) * b2.second
    model.network["neurongroup"].sigma = np.array([0.1, 0.31622])
    model.network["neurongroup"].y = 1

    return model


def test_construction(tmp_path: Path) -> None:
    model = b2_model.B2Model()
    assert model.parameters is None
    assert model.network is not None
    assert model.network.name == "b2Network"
    assert model.synapses == {}
    assert model.monitors == {}


def test_define_equation(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}

    mfx.define_equation("P", "dge/dt = -ge*(1./taue) : siemens")

    assert mfx.neurongroups["P"]["model"] == "dge/dt = -ge*(1./taue) : siemens\n"

    with pytest.raises(
        ValueError, match="Equation must have a colon that separates equation and unit"
    ):
        mfx.define_equation("P", "dge/dt = -ge*(1./taue)")

    with pytest.raises(
        ValueError,
        match="Equation must have an equal sign that separates variable and expression",
    ):
        mfx.define_equation("P", "-ge*(1./taue): siemens")


def test_define_equation_incremental(mfx: b2_model.B2Model) -> None:
    mfx.define_equation("P", "dApre/dt = -Apre / taupre : siemens")
    mfx.define_equation("P", "dApost/dt = -Apost / taupost : siemens")

    e = b2.Equations(mfx.neurongroups["P"]["model"])

    assert set(e.names) == set(["Apre", "Apost"])
    assert e.parameter_names == set()


def test_define_dt(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}

    mfx.define_dt(0.1 * b2.ms)

    assert mfx.dt is not None
    assert np.isclose(mfx.dt.base, 0.0001)
    assert mfx.dt.has_same_dimensions(0.1 * b2.ms)


def test_define_parameter(mfx: b2_model.B2Model) -> None:
    assert mfx.parameters is None

    mfx.define_parameter("taue", 5 * b2.ms)
    mfx.define_parameter("taui", 10 * b2.ms)
    mfx.define_parameter("area", 20000 * b2.umetre**2)

    assert mfx.parameters == {
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        "area": 20000 * b2.umetre**2,
    }


def test_define_threshold(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}
    mfx.define_threshold("P", "v > 10 * mV")
    assert mfx.neurongroups["P"]["th"] == "v > 10 * mV"


def test_define_solver_method(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}
    mfx.define_solver_method("P", "euler")
    assert mfx.neurongroups["P"]["method"] == "euler"


def test_define_refractory(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}
    mfx.define_refractory("P", 5 * b2.ms)
    assert mfx.neurongroups["P"]["refractory"] == 5 * b2.ms


def test_define_neuron_number(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}
    mfx.define_neuronnumber("P", 100)
    assert mfx.neurongroups["P"]["n"] == 100


def test_assemble_neurongroup(mfx: b2_model.B2Model) -> None:
    eq = "dge/dt = -ge*(1./taue): siemens"
    n = 30
    threshold = "v > 10 * mV"
    refractory = 5 * b2.ms
    method = "euler"
    dt = 0.1 * b2.ms
    mfx.define_equation("P", eq)
    mfx.define_neuronnumber("P", n)
    mfx.define_threshold("P", threshold)
    mfx.define_refractory("P", refractory)
    mfx.define_dt(dt)
    mfx.define_solver_method("P", method)
    mfx.define_parameter("ge", "(randn() * 1.5 + 4) * 10.*nS")

    ng = b2.NeuronGroup(
        n,
        eq,
        threshold=threshold,
        refractory=refractory,
        method=method,
        dt=dt,
        name="P",
    )

    assert mfx.neurongroups["P"] == {
        "n": n,
        "th": threshold,
        "refractory": refractory,
        "method": method,
        "model": eq + "\n",
    }
    mfx.assemble_neurongroup("P")
    assert len(mfx.network.objects) == 1

    assert mfx.network["P"].name == ng.name
    assert mfx.network["P"]._N == ng._N


def test_assemble_neurongroup_exception(mfx: b2_model.B2Model) -> None:
    with pytest.raises(RuntimeError, match="Error in assembling neuron group N"):
        mfx.assemble_neurongroup("N")


def test_neuron_subgroup(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_neuron_subgroup("P", "excitatory", slice(0, 20))
    mfx.define_neuron_subgroup("P", "inhibitory", slice(20, 30))

    assert list(
        itertools.islice(
            itertools.count(),
            mfx.neurongroups["excitatory"]["idx"].start,
            mfx.neurongroups["excitatory"]["idx"].stop,
        )
    ) == list(range(0, 20))

    assert list(
        itertools.islice(
            itertools.count(),
            mfx.neurongroups["inhibitory"]["idx"].start,
            mfx.neurongroups["inhibitory"]["idx"].stop,
        )
    ) == list(range(20, 30))

    print(list(mfx.neurongroups.keys()))

    assert mfx.network[mfx.neurongroups["excitatory"]["name"]] is not None
    assert mfx.network[mfx.neurongroups["inhibitory"]["name"]] is not None

    mfx.network = b2.Network()
    with pytest.raises(KeyError, match="'No object with name \"ng\" found'"):
        mfx.define_neuron_subgroup("ng", "excitatory", slice(0, 20))


def test_define_synapse_eq_exceptions(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    with pytest.raises(
        ValueError,
        match="Equation must have an equal sign that separates variable and expression",
    ):
        mfx.define_synapse_equation(
            "S", "dApost/dt -Apost / taupost : siemens (event-driven)"
        )

    with pytest.raises(
        ValueError, match="Equation must have a colon that separates equation and unit"
    ):
        mfx.define_synapse_equation("S", "dApost/dt = -Apost / taupost (event-driven)")


def test_synapse_connection_prob_numerical(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_synapse_connection_probablitiy("Ce", 0.5)

    assert np.isclose(mfx.synapses["Ce"]["p"], 0.5)

    boundary = np.nextafter(1.0, 2.0)
    with pytest.raises(
        ValueError,
        match="Error, supplied value 1.0000000000000002 is not a probability.",
    ):
        mfx.define_synapse_connection_probablitiy("Ce", boundary)

    boundary = np.nextafter(0.0, -1.0)
    with pytest.raises(
        ValueError, match="Error, supplied value -5e-324 is not a probability."
    ):
        mfx.define_synapse_connection_probablitiy("Ce", boundary)


def test_define_synapse(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    taupre = 20 * b2.ms
    taupost = taupre
    gmax = 60 * b2.nS
    dApre = 6 * b2.nS
    dApost = -dApre * taupre / taupost * 1.05

    mfx, _ = mfx_parameterized
    mfx.define_neuron_subgroup("P", "Pe", slice(0, 24))
    mfx.define_neuron_subgroup("P", "Pi", slice(24, 30))

    mfx.define_parameter("taupre", taupre)

    mfx.define_parameter("taupost", taupost)

    mfx.define_parameter("gmax", gmax)

    mfx.define_parameter("dApre", dApre)

    mfx.define_parameter("dApost", dApost)

    mfx.define_synapse_connection("Ce", "i != j")
    mfx.define_synapse_connection_probablitiy("Ce", 1.0)
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

    mfx.assemble_synapse("Ce")

    assert len(mfx.synapses) == 1
    assert mfx.synapses["Ce"]["origin"] == "Pe"
    assert mfx.synapses["Ce"]["target"] == "Pi"


def test_assemble_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "formozov_project.b2_model.b2.Synapses", side_effect=Exception("fake exception")
    )
    mfx, _ = mfx_with_synapse

    with pytest.raises(RuntimeError, match="Error in defining synapse Ce"):
        mfx.assemble_synapse("Ce")


def test_connect_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "formozov_project.b2_model.b2.Synapses.connect",
        side_effect=Exception("fake exception"),
    )

    mfx, _ = mfx_with_synapse
    with pytest.raises(RuntimeError, match="Error in connecting synapse Ce"):
        mfx.assemble_synapse("Ce")


def test_define_monitor(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    assert mfx.monitors == {}

    mfx.define_monitor("Traces_ge", b2.StateMonitor, "P", "ge")

    assert "Traces_ge" in mfx.monitors
    assert mfx.network["Traces_ge"].source == mfx.network["P"]
    assert mfx.network["Traces_ge"].record_variables == [
        "ge",
    ]


def test_define_monitor_exception(
    mocker: MockerFixture, mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "formozov_project.b2_model.b2.StateMonitor",
        side_effect=Exception("fake exception"),
    )

    mfx, _ = mfx_parameterized

    with pytest.raises(RuntimeError, match="Error in defining monitor M"):
        mfx.define_monitor("M", b2.StateMonitor, "G", "v")


def test_ornstein_uhlenbeck(tmp_path: Path) -> None:
    model = ornstein()
    model.run_simulation(1 * b2.second)

    b2.seed(4321)
    N = b2.NeuronGroup(
        2,
        """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
        """,
        method="euler",
    )

    N.tau = np.array([1, 0.1]) * b2.second
    N.sigma = np.array([0.1, 0.31622])
    N.y = 1

    M = b2.StateMonitor(N, "y", record=True)

    network = b2.Network([N, M])

    network.run(1 * b2.second)

    assert np.allclose(M.t / b2.second, model.network["M"].t / b2.second)
    assert np.allclose(M.y[1], model.network["M"].y[1])
    assert np.allclose(M.y[0], model.network["M"].y[0])


def test_NMDA_model() -> None:
    # https://brian2.readthedocs.io/en/stable/examples/synapses.nonlinear.html

    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = b2_model.B2Model(seed=4321)
    # model.define_dt(0.2 * b2.ms)
    model.define_solver_method("neuron_input", "euler")

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
    model.assemble_neurongroup("neurons")
    model.assemble_neurongroup("neuron_input")

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
    model.assemble_synapse("S")

    model.define_monitor("M", b2.StateMonitor, "S", "g")
    model.define_monitor("Mn", b2.StateMonitor, "neurons", "g")

    model.network["S"].w = [1.0, 10.0]
    model.network["neuron_input"].v = [0.0, 0.5]

    model.run_simulation(100 * b2.ms)

    # test model
    neuron_input = b2.NeuronGroup(
        2,
        "dv/dt = 1/(10*ms) : 1",
        threshold="v>1",
        reset="v = 0",
        method="euler",
        namespace={"a": a, "b": b, "c": c},
    )

    neurons = b2.NeuronGroup(
        1,
        """dv/dt = (g-v)/(10*ms) : 1
                                g : 1""",
        method="exact",
        namespace={"a": a, "b": b, "c": c},
    )

    S = b2.Synapses(
        neuron_input,
        neurons,
        """
                    dg_syn/dt = -a*g_syn+b*x*(1-g_syn) : 1 (clock-driven)
                    g_post = g_syn : 1 (summed)
                    dx/dt=-c*x : 1 (clock-driven)
                    w : 1 # synaptic weight
                """,
        on_pre="x += w",
        name="synapses_test",
        namespace={"a": a, "b": b, "c": c},
    )  # NMDA synapses

    S.connect()

    # initilize
    S.w = [1.0, 10.0]

    neuron_input.v = [0.0, 0.5]

    M = b2.StateMonitor(
        S,
        "g",
        name="Mtest",
        # If not using standalone mode, this could also simply be
        # record=np.arange(len(neuron_input) * len(neurons)),
        record=True,
    )

    Mn = b2.StateMonitor(neurons, "g", name="Mntest", record=0)

    test_network = b2.Network(name="testnetwork")
    test_network.add(neuron_input)
    test_network.add(neurons)
    test_network.add(S)
    test_network.add(M)
    test_network.add(Mn)
    test_network.run(100 * b2.ms)

    # assert result equality
    np.testing.assert_almost_equal(M.g, model.network["M"].g)
    np.testing.assert_almost_equal(Mn.g, model.network["Mn"].g)


def test_hdf5(tmp_path: Path) -> None:
    model = ornstein()
    model.run_simulation(10 * b2.second)

    model.save_to_h5(tmp_path / "data.h5")

    data = None

    with h5py.File(Path(tmp_path, "data.h5"), "r") as h5file:
        data = h5file["M"]["y"][:]

    assert np.allclose(data, model.network["M"].y)

    with pytest.raises(ValueError, match="Path fakepath/data.h5 does not exist"):
        model.save_to_h5("fakepath/data.h5")

    model.monitors = {}

    with pytest.raises(
        RuntimeError, match="Error, cannot save data when no monitors are defined"
    ):
        model.save_to_h5(tmp_path / "data.h5")


def test_logger(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="formozov_project.b2_model")
    model = ornstein()
    model.run_simulation(1 * b2.second, savepath=tmp_path / "data.h5 ")

    # bunch of asserts checking that logfile content works as expected
    logs = [record.message for record in caplog.records]
    assert "Model creation complete" in logs[0]
    assert "Neuron number defined: 2" in logs[1]
    assert "dt defined: 200. us" in logs[2]
    assert "Solver method defined: euler" in logs[3]
    assert "Equation defined" in logs[4]
    assert "assembling neurongroup 'neurongroup'" in logs[5]
    assert "adding monitor 'M'" in logs[6]
    assert f"run the system for 1. s with savepath = {tmp_path}" in logs[7]

    pathname = tmp_path / "data.h5"
    assert f"save data to hdf5 at {pathname}" in logs[8]


def test_store_restore() -> None:
    model = ornstein()
    sim_time = 10 * b2.ms
    assert len(model.checkpoints) == 0
    model.run_simulation(sim_time)
    # default initial state checkpoint is made on first simulation if no other checkpoints are made
    assert len(model.checkpoints) == 1
    assert model.network["M"].t[-1] == sim_time - model.dt
    # restore initial state
    model.restore(model.checkpoints[0])
    assert len(model.network["M"].t) == 0
    # store two new checkpoints
    model.run_simulation(sim_time)
    assert model.network["M"].t[-1] == sim_time - model.dt
    model.store("t1")
    assert len(model.checkpoints) == 2
    model.run_simulation(sim_time)
    assert model.network["M"].t[-1] == 2 * sim_time - model.dt
    model.store("t2")
    assert len(model.checkpoints) == 3
    # restore second checkpoint
    model.restore("t1")
    assert model.network["M"].t[-1] == sim_time - model.dt
    # overwrite existing checkpoint
    assert len(model.checkpoints) == 3
    model.store("t2")
    assert len(model.checkpoints) == 3
    # restore initial state
    model.restore(model.checkpoints[0])
    assert len(model.network["M"].t) == 0
    # restore non-existing checkpoint
    with pytest.raises(ValueError, match=".*'idontexist'.*"):
        model.restore("idontexist")
