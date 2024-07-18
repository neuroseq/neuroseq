from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import h5py
import pytest
from brian2.tests.utils import assert_allclose
from formozov_project import b2_model
from formozov_project import b2json
from pytest_mock import MockerFixture


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
        ValueError,
        match="Equation must have a colon that separates equation and unit or an equal sign",
    ):
        mfx.define_equation("P", "dge/dt -ge*(1./taue)")

    with pytest.raises(
        ValueError,
        match="Equation must have a colon that separates equation and unit or an equal sign",
    ):
        mfx.define_equation("P", "-ge*(1./taue) siemens")


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
    timed_array = b2.TimedArray(np.random.rand(10, 100) * b2.mS, dt=1 * b2.ms)
    mfx.define_parameter("stim", timed_array)

    assert mfx.parameters == {
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        "area": 20000 * b2.umetre**2,
        "stim": timed_array,
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

    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    mfx.define_neuronsubgroup_target("inhibitory", "P")
    mfx.define_neuronsubgroup_idxs("inhibitory", (20, 30))

    assert (
        mfx.neurongroups["excitatory"]["start"],
        mfx.neurongroups["excitatory"]["stop"],
    ) == (0, 20)

    assert (
        mfx.neurongroups["inhibitory"]["start"],
        mfx.neurongroups["inhibitory"]["stop"],
    ) == (20, 30)

    mfx.assemble_neuronsubgroup("excitatory")
    mfx.assemble_neuronsubgroup("inhibitory")

    assert mfx.network[mfx.neurongroups["excitatory"]["name"]] is not None
    assert mfx.network[mfx.neurongroups["inhibitory"]["name"]] is not None

    mfx.network = b2.Network()
    mfx.define_neuronsubgroup_idxs("ng", (20, 30))

    with pytest.raises(
        KeyError, match="Error, target Px could not be found in neurongroups"
    ):
        mfx.define_neuronsubgroup_target("ng", "Px")

    with pytest.raises(
        RuntimeError, match="Error when trying to create a neuronsubgroup"
    ):
        mfx.assemble_neuronsubgroup("ng")


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

    mfx.define_neuronsubgroup_target("Pe", "P")
    mfx.define_neuronsubgroup_idxs("Pe", (0, 24))

    mfx.define_neuronsubgroup_target("Pi", "P")
    mfx.define_neuronsubgroup_idxs("Pi", (0, 24))

    mfx.assemble_neuronsubgroup("Pi")
    mfx.assemble_neuronsubgroup("Pe")

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

    mfx.define_monitor_source("Traces_ge", "P")
    mfx.define_monitor_type("Traces_ge", "StateMonitor")
    mfx.define_monitor_variable("Traces_ge", "ge")

    mfx.assemble_monitor("Traces_ge")

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
        mfx.define_monitor_source("M", "G")
        mfx.define_monitor_type("M", "StateMonitor")
        mfx.define_monitor_variable("M", "v")
        mfx.assemble_monitor("M")


def test_define_init(mfx_ornstein: b2_model.B2Model) -> None:
    mfx = mfx_ornstein

    b2.seed(4321)
    N = b2.NeuronGroup(
        5,
        """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
        """,
        method="euler",
        name="neurongroup",
    )

    N.tau = np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    N.sigma = np.array([0.1, 0.31622, 0.4, 0.2, 0.88])
    N.y = 1

    M = b2.StateMonitor(N, "y", record=True)

    network = b2.Network([N, M])

    assert all(network["neurongroup"].tau == mfx.network["neurongroup"].tau)
    assert all(network["neurongroup"].sigma == mfx.network["neurongroup"].sigma)
    assert all(network["neurongroup"].y == mfx.network["neurongroup"].y)


def test_ornstein_uhlenbeck(mfx_ornstein: b2_model.B2Model) -> None:
    b2.seed(4321)
    mfx_ornstein.run_simulation(1 * b2.second)

    b2.seed(4321)
    N = b2.NeuronGroup(
        5,
        """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
        """,
        method="euler",
    )

    N.tau = np.array([1, 0.8, 0.6, 0.3, 0.1]) * b2.second
    N.sigma = np.array([0.1, 0.31622, 0.4, 0.2, 0.88])
    N.y = 1

    M = b2.StateMonitor(N, "y", record=True)

    network = b2.Network([N, M])

    network.run(1 * b2.second)

    assert np.allclose(M.t / b2.second, mfx_ornstein.network["M"].t / b2.second)
    assert np.allclose(M.y[1], mfx_ornstein.network["M"].y[1])
    assert np.allclose(M.y[0], mfx_ornstein.network["M"].y[0])


def test_NMDA_model(mfx_nmda: b2_model.B2Model) -> None:
    # https://brian2.readthedocs.io/en/stable/examples/synapses.nonlinear.html

    mfx_nmda.run_simulation(100 * b2.ms)

    # test model
    neuron_input = b2.NeuronGroup(
        2,
        "dv/dt = 1/(10*ms) : 1",
        threshold="v>1",
        reset="v = 0",
        method="euler",
        namespace=mfx_nmda.parameters,
    )

    neurons = b2.NeuronGroup(
        1,
        """dv/dt = (g-v)/(10*ms) : 1
                                g : 1""",
        method="exact",
        namespace=mfx_nmda.parameters,
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
        namespace=mfx_nmda.parameters,
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
    assert_allclose(M.g, mfx_nmda.network["M"].g)
    assert_allclose(Mn.g, mfx_nmda.network["Mn"].g)


def test_hdf5(tmp_path: Path, mfx_ornstein: b2_model.B2Model) -> None:
    mfx_ornstein.run_simulation(10 * b2.second)

    mfx_ornstein.save_to_h5(tmp_path / "data.h5")

    data = None

    with h5py.File(Path(tmp_path, "data.h5"), "r") as h5file:
        data = h5file["M"]["y"][:]

    assert np.allclose(data, mfx_ornstein.network["M"].y)

    with pytest.raises(ValueError, match="Path fakepath/data.h5 does not exist"):
        mfx_ornstein.save_to_h5("fakepath/data.h5")

    mfx_ornstein.monitors = {}

    with pytest.raises(
        RuntimeError, match="Error, cannot save data when no monitors are defined"
    ):
        mfx_ornstein.save_to_h5(tmp_path / "data.h5")


def test_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, mfx_ornstein: b2_model.B2Model
) -> None:
    caplog.set_level(logging.INFO, logger="formozov_project.b2_model")
    mfx_ornstein.run_simulation(1 * b2.second, savepath=tmp_path / "data.h5 ")

    logs = [record.message for record in caplog.records]
    assert f"run the system for 1. s with savepath = {tmp_path}" in logs[0]

    pathname = tmp_path / "data.h5"
    assert f"save data to hdf5 at {pathname}" in logs[1]


def test_store_restore(mfx_ornstein: b2_model.B2Model) -> None:
    sim_time = 10 * b2.ms
    assert len(mfx_ornstein.checkpoints) == 0
    mfx_ornstein.run_simulation(sim_time)
    # default initial state checkpoint is made on first simulation if no other checkpoints are made
    assert len(mfx_ornstein.checkpoints) == 1
    assert mfx_ornstein.network["M"].t[-1] == sim_time - mfx_ornstein.dt
    # restore initial state
    mfx_ornstein.restore(mfx_ornstein.checkpoints[0])
    assert len(mfx_ornstein.network["M"].t) == 0
    # store two new checkpoints
    mfx_ornstein.run_simulation(sim_time)
    assert mfx_ornstein.network["M"].t[-1] == sim_time - mfx_ornstein.dt
    mfx_ornstein.store("t1")
    assert len(mfx_ornstein.checkpoints) == 2
    mfx_ornstein.run_simulation(sim_time)
    assert mfx_ornstein.network["M"].t[-1] == 2 * sim_time - mfx_ornstein.dt
    mfx_ornstein.store("t2")
    assert len(mfx_ornstein.checkpoints) == 3
    # restore second checkpoint
    mfx_ornstein.restore("t1")
    assert mfx_ornstein.network["M"].t[-1] == sim_time - mfx_ornstein.dt
    # overwrite existing checkpoint
    assert len(mfx_ornstein.checkpoints) == 3
    mfx_ornstein.store("t2")
    assert len(mfx_ornstein.checkpoints) == 3
    # restore initial state
    mfx_ornstein.restore(mfx_ornstein.checkpoints[0])
    assert len(mfx_ornstein.network["M"].t) == 0
    # restore non-existing checkpoint
    with pytest.raises(ValueError, match=".*'idontexist'.*"):
        mfx_ornstein.restore("idontexist")


def test_save_load(mfx_nmda: b2_model.B2Model, tmp_path: Path) -> None:
    mfx_nmda.save_model(tmp_path / "model.json")

    with open(tmp_path / "model.json") as file:
        loaded_data = b2json.load(file)

    for k in [
        "neurons",
        "neuron_input",
        "S",
        "S_pre",
        "M",
        "Mn",
        "seed",
        "input_subgroup",
    ]:
        assert k in loaded_data

    reconstructed_model = b2_model.B2Model.load_model(tmp_path / "model.json")

    assert reconstructed_model.seed == mfx_nmda.seed
    assert reconstructed_model.dt == mfx_nmda.dt

    # synapses
    for neurongroupname in ["neuron_input", "neurons"]:
        for var in mfx_nmda.network.get_states()[neurongroupname]:
            assert_allclose(
                getattr(reconstructed_model.network[neurongroupname], var),
                getattr(mfx_nmda.network[neurongroupname], var),
                verbose=True,
            )

    assert_allclose(
        reconstructed_model.network["neurons"].g,
        mfx_nmda.network["neurons"].g,
        verbose=True,
    )

    # test synapses
    for var in mfx_nmda.network.get_states()["S"]:
        assert_allclose(
            getattr(reconstructed_model.network["S"], var),
            getattr(mfx_nmda.network["S"], var),
            verbose=True,
        )

    for name in ["M", "Mn"]:
        for var in mfx_nmda.network.get_states()[name]:
            assert_allclose(
                getattr(reconstructed_model.network[name], var),
                getattr(mfx_nmda.network[name], var),
                verbose=True,
            )

    assert reconstructed_model.parameters == mfx_nmda.parameters

    # resetting the seed is necessary, since brian2 has only one global rng (and buffer of cached random numbers),
    # and if the seed is not reset the second simulation will run with different random numbers and produce different results
    b2.seed(loaded_data["seed"])
    mfx_nmda.run_simulation(1 * b2.second)

    b2.seed(loaded_data["seed"])
    reconstructed_model.run_simulation(1 * b2.second)

    assert_allclose(mfx_nmda.network["M"].g, reconstructed_model.network["M"].g)
    assert_allclose(mfx_nmda.network["Mn"].g, reconstructed_model.network["Mn"].g)
