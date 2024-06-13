import itertools
import logging
import typing
from pathlib import Path

import brian2 as b2
import numpy as np
import pytest
from formozov_project.b2_model import B2Model

b2.prefs.codegen.target = "cython"


@pytest.fixture()
def mfx(tmp_path) -> typing.Generator[B2Model, None, None]:
    yield B2Model(tmp_path, loglevel="DEBUG")


@pytest.fixture
def mfx_parameterized(
    mfx, reference_parameters
) -> typing.Generator[B2Model, None, None]:
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

    yield mfx, reference_parameters


def ornstein(path):
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    model = B2Model(path, seed=4321)
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


def test_construction(tmp_path) -> None:
    model = B2Model(tmp_path, loglevel="DEBUG")

    assert model.path == tmp_path
    assert model.logger.name == "Model"
    assert model.logger.level == logging.DEBUG
    assert model.logger.handlers[0].baseFilename == str(tmp_path / "logfile.log")

    assert model.parameters is None
    assert model.network is not None
    assert model.network.name == "b2Network"
    assert model.neuron_subgroups == {}
    assert model.synapses == {}
    assert model.monitors == {}


def test_define_equation(mfx) -> None:
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


def test_define_dt(mfx) -> None:
    assert mfx.neurongroups == {}

    mfx.define_dt(0.1 * b2.ms)

    with pytest.raises(ValueError, match="dt must be a Quantity"):
        mfx.define_dt(0.1)


def test_define_parameter(mfx) -> None:
    assert mfx.parameters is None

    mfx.define_parameter("taue", 5 * b2.ms)
    mfx.define_parameter("taui", 10 * b2.ms)
    mfx.define_parameter("area", 20000 * b2.umetre**2)

    assert mfx.parameters == {
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        "area": 20000 * b2.umetre**2,
    }


def test_define_threshold(mfx) -> None:
    assert mfx.neurongroups == {}
    mfx.define_threshold("P", "v > 10 * mV")
    assert mfx.neurongroups["P"]["th"] == "v > 10 * mV"


def test_define_solver_method(mfx) -> None:
    assert mfx.neurongroups == {}
    mfx.define_solver_method("P", "euler")
    assert mfx.neurongroups["P"]["method"] == "euler"


def test_define_refractory(mfx) -> None:
    assert mfx.neurongroups == {}
    mfx.define_refractory("P", 5 * b2.ms)
    assert mfx.neurongroups["P"]["refractory"] == 5 * b2.ms

    with pytest.raises(ValueError, match="Refractory period must be a Quantity"):
        mfx.define_refractory("P", 5)


def test_define_neuron_number(mfx) -> None:
    assert mfx.neurongroups == {}
    mfx.define_neuronnumber("P", 100)
    assert mfx.neurongroups["P"]["n"] == 100

    with pytest.raises(ValueError, match="Neuron number must be an integer"):
        mfx.define_neuronnumber("P", 100.0)


def test_assemble_neurongroup(mfx) -> None:
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
        name="test_ng",
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

    ng = list(mfx.network.objects)[0]

    assert mfx.network["P"].name == ng.name
    assert mfx.network["P"]._N == ng._N


def test_neuron_subgroup(mfx_parameterized) -> None:
    mfx, _ = mfx_parameterized
    assert mfx.neuron_subgroups == {}
    mfx.define_neuron_subgroup("P", "excitatory", slice(0, 20))
    mfx.define_neuron_subgroup("P", "inhibitory", slice(20, 30))

    assert list(
        itertools.islice(
            itertools.count(),
            mfx.neuron_subgroups["excitatory"]["idx"].start,
            mfx.neuron_subgroups["excitatory"]["idx"].stop,
        )
    ) == list(range(0, 20))
    assert list(
        itertools.islice(
            itertools.count(),
            mfx.neuron_subgroups["inhibitory"]["idx"].start,
            mfx.neuron_subgroups["inhibitory"]["idx"].stop,
        )
    ) == list(range(20, 30))

    assert (
        mfx.network[mfx.neuron_subgroups["excitatory"]["name_in_network"]] is not None
    )
    assert (
        mfx.network[mfx.neuron_subgroups["inhibitory"]["name_in_network"]] is not None
    )

    mfx.network = b2.Network()
    with pytest.raises(KeyError, match="'No object with name \"ng\" found'"):
        mfx.define_neuron_subgroup("ng", "excitatory", slice(0, 20))


def test_define_synapse(mfx_parameterized) -> None:
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

    mfx.define_synapse(
        "Ce",
        "Pe",
        "Pi",
        """we : siemens
            dApre/dt = -Apre / taupre : siemens (event-driven)
            dApost/dt = -Apost / taupost : siemens (event-driven)""",
        on_pre="""ge += we
                    Apre += dApre
                    we = clip(we + Apost, -gmax, gmax)""",
        on_post="""Apost += dApost
                      we= clip(we + Apre, -gmax, gmax)""",
        connection_condition="i != j",
        connection_probability=1.0,
    )

    assert len(mfx.synapses) == 1
    assert mfx.synapses["Ce"]["origin"] == "Pe"
    assert mfx.synapses["Ce"]["target"] == "Pi"


def test_define_monitor(mfx_parameterized) -> None:
    mfx, _ = mfx_parameterized

    assert mfx.monitors == {}

    mfx.define_monitor("Traces_ge", b2.StateMonitor, "P", "ge", record=True)

    assert "Traces_ge" in mfx.monitors
    assert mfx.network["Traces_ge"].source == mfx.network["P"]
    assert mfx.network["Traces_ge"].record_variables == [
        "ge",
    ]


def test_ornstein_uhlenbeck(tmp_path):
    model = ornstein(tmp_path)
    model.run_simulation(10 * b2.second)

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

    network.run(10 * b2.second)

    assert np.allclose(M.t / b2.second, model.network["M"].t / b2.second)
    assert np.allclose(M.y[1], model.network["M"].y[1])
    assert np.allclose(M.y[0], model.network["M"].y[0])


def test_NMDA_model(tmp_path):
    # https://brian2.readthedocs.io/en/stable/examples/synapses.nonlinear.html
    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = B2Model(tmp_path, seed=4321)

    model.define_reset("v = 0")
    model.define_neuronnumber(2)
    model.define_dt()
    model.define_synapse()
    model.define_solver_method()
    model.define_equation()
    model.define_threshold()

    neuron_input = b2.NeuronGroup(
        2, "dv/dt = 1/(10*ms) : 1", threshold="v>1", reset="v = 0", method="euler"
    )
    neurons = b2.NeuronGroup(
        1,
        """dv/dt = (g-v)/(10*ms) : 1
                                g : 1""",
        method="exact",
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
    )  # NMDA synapses

    S.connect()
    S.w = [1.0, 10.0]
    neuron_input.v = [0.0, 0.5]

    M = b2.StateMonitor(
        S,
        "g",
        # If not using standalone mode, this could also simply be
        record=True,
    )
    # record=np.arange(len(neuron_input)*len(neurons)))

    Mn = b2.StateMonitor(neurons, "g", record=0)

    test_network = b2.Network([neuron_input, neuron_input, S, M, Mn])

    test_network.run(1000 * b2.ms)


def test_hdf5(tmp_path):
    from pathlib import Path

    import h5py

    model = ornstein(tmp_path)

    model.run_simulation(10 * b2.second, save=False)

    model.save_to_h5()

    data = None
    with h5py.File(Path(model.path, "data.h5"), "r") as h5file:
        data = h5file["y"][:]

    assert np.allclose(data, model.network["M"].y)


def test_logger(tmp_path):
    model = ornstein(tmp_path)

    model.run_simulation(10 * b2.second, save=True)

    with open(Path(model.path, "logfile.log")) as file:
        logs = [line.strip() for line in file.readlines()]

    # bunch of asserts checking that logfile content works as expected
    assert "Model creation complete" in logs[0]
    assert "Neuron number defined: 2" in logs[1]
    assert "dt defined: 200. us" in logs[2]
    assert "Solver method defined: euler" in logs[3]
    assert "Equation defined" in logs[4]
    assert "assembling neurongroup 'neurongroup'" in logs[5]
    assert "adding monitor 'M'" in logs[6]
    assert "run the system for 10. s with save = True" in logs[7]
    assert f"save data to hdf5 at {tmp_path / "data.h5"}" in logs[8]
