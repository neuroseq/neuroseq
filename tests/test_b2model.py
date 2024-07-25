from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import brian2 as b2
import brian2.numpy_ as np
import h5py
import pytest
from brian2.tests.utils import assert_allclose
from formozov_project import b2_model
from formozov_project import b2json
from pytest import MonkeyPatch
from pytest_mock import MockerFixture


def test_construction(tmp_path: Path) -> None:
    model = b2_model.B2Model()
    assert model.parameters == {}
    assert model.network is not None
    assert model.network.name == "b2Network"
    assert model.synapses == {}
    assert model.monitors == {}


def test_define_equation(mfx: b2_model.B2Model) -> None:
    assert mfx.neurongroups == {}

    mfx.define_equation("P", "dge/dt = -ge*(1./taue) : siemens")

    assert mfx.neurongroups["P"]["model"] == "dge/dt = -ge*(1./taue) : siemens\n"
    assert all(
        f in mfx.functions["neurongroups"]["P"]
        for f in ["rename", "delete", "element_added", "element_removed"]
    )

    assert all(
        f in mfx.functions["neurongroups"]["P"]["model"]
        for f in ["add", "delete", "modify", "rename"]
    )

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
    assert mfx.parameters == {}
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
    assert mfx.neurongroups["P"]["neuronnumber"] == 100


def test_assemble_neurongroup_exception(mfx: b2_model.B2Model) -> None:
    with pytest.raises(RuntimeError, match="Error in assembling neuron group N"):
        mfx._assemble_neurongroup("N")


def test_rename_neurongroup(mfx_nmda: b2_model.B2Model) -> None:
    assert "neurons" in mfx_nmda.neurongroups
    test_data = deepcopy(mfx_nmda.neurongroups["neurons"])

    mfx_nmda.functions["neurongroups"]["neurons"]["rename"].rename("renamed")
    assert "neurons" not in mfx_nmda.neurongroups
    assert "neurons" not in mfx_nmda.functions["neurongroups"]

    assert "renamed" in mfx_nmda.neurongroups
    assert "renamed" in mfx_nmda.functions["neurongroups"]

    assert mfx_nmda.neurongroups["renamed"] == test_data

    assert all(
        f in mfx_nmda.functions["neurongroups"]["renamed"]
        for f in ["rename", "delete", "element_added", "element_removed"]
    )


def test_rename_synapse(mfx_nmda: b2_model.B2Model) -> None:
    assert "S" in mfx_nmda.synapses
    test_data = deepcopy(mfx_nmda.synapses["S"])

    mfx_nmda.functions["synapses"]["S"]["rename"].rename("renamed")

    assert "S" not in mfx_nmda.synapses
    assert "S" not in mfx_nmda.functions["synapses"]

    assert "renamed" in mfx_nmda.synapses
    assert "renamed" in mfx_nmda.functions["synapses"]

    assert mfx_nmda.synapses["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["synapses"]["renamed"]
        for f in ["rename", "delete", "element_added", "element_removed"]
    )


def test_rename_parameter(mfx_nmda: b2_model.B2Model) -> None:
    assert "a" in mfx_nmda.parameters
    test_data = deepcopy(mfx_nmda.parameters["a"])

    mfx_nmda.functions["parameters"]["a"]["rename"]("renamed")
    assert "a" not in mfx_nmda.parameters
    assert "a" not in mfx_nmda.functions["parameters"]

    assert "renamed" in mfx_nmda.parameters
    assert "renamed" in mfx_nmda.functions["parameters"]

    assert mfx_nmda.parameters["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["parameters"]["renamed"]
        for f in ["add", "delete", "modify", "rename"]
    )


def test_rename_monitor(mfx_nmda: b2_model.B2Model) -> None:
    assert "M" in mfx_nmda.monitors
    test_data = deepcopy(mfx_nmda.monitors["M"])
    test_data["name"] = "renamed"

    mfx_nmda.functions["monitors"]["M"]["rename"].rename("renamed")
    assert "M" not in mfx_nmda.monitors
    assert "M" not in mfx_nmda.functions["monitors"]

    assert "renamed" in mfx_nmda.monitors
    assert "renamed" in mfx_nmda.functions["monitors"]

    assert mfx_nmda.monitors["renamed"] == test_data

    assert all(
        f in mfx_nmda.functions["monitors"]["renamed"]
        for f in ["rename", "delete", "element_added", "element_removed"]
    )


def test_rename_init(mfx_nmda: b2_model.B2Model) -> None:
    assert "S" in mfx_nmda.initials
    test_data = deepcopy(mfx_nmda.initials["S"])

    mfx_nmda.functions["initials"]["S"]["rename"].rename("renamed")

    assert "S" not in mfx_nmda.initials
    assert "S" not in mfx_nmda.functions["initials"]

    assert "renamed" in mfx_nmda.initials
    assert "renamed" in mfx_nmda.functions["initials"]

    assert mfx_nmda.initials["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["initials"]["renamed"]
        for f in ["rename", "delete", "element_added", "element_removed"]
    )


def test_rename_init_element(mfx_nmda: b2_model.B2Model) -> None:
    assert "w" in mfx_nmda.initials["S"]
    test_data = deepcopy(mfx_nmda.initials["S"]["w"])
    mfx_nmda.functions["initials"]["S"]["w"]["rename"]("renamed")

    assert "w" not in mfx_nmda.initials["S"]
    assert "w" not in mfx_nmda.functions["initials"]["S"]

    assert "renamed" in mfx_nmda.initials["S"]
    assert "renamed" in mfx_nmda.functions["initials"]["S"]

    assert mfx_nmda.initials["S"]["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["initials"]["S"]["renamed"]
        for f in ["add", "delete", "modify", "rename"]
    )


def test_rename_neurongroup_element_with_modification(
    mfx_nmda: b2_model.B2Model,
) -> None:
    assert "model" in mfx_nmda.neurongroups["neurons"]
    test_data = deepcopy(mfx_nmda.neurongroups["neurons"]["model"])

    mfx_nmda.functions["neurongroups"]["neurons"]["model"]["rename"]("renamed")

    assert "model" not in mfx_nmda.neurongroups["neurons"]
    assert "model" not in mfx_nmda.functions["neurongroups"]["neurons"]

    assert "renamed" in mfx_nmda.neurongroups["neurons"]
    assert "renamed" in mfx_nmda.functions["neurongroups"]["neurons"]

    assert mfx_nmda.neurongroups["neurons"]["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["neurongroups"]["neurons"]["renamed"]
        for f in ["add", "delete", "modify", "rename"]
    )

    mfx_nmda.functions["neurongroups"]["neurons"]["renamed"]["modify"]("dv/dt=-v:1")

    assert mfx_nmda.neurongroups["neurons"]["renamed"] == "dv/dt=-v:1"


def test_rename_synapse_element(mfx_nmda: b2_model.B2Model) -> None:
    assert "model" in mfx_nmda.synapses["S"]
    test_data = deepcopy(mfx_nmda.synapses["S"]["model"])

    mfx_nmda.functions["synapses"]["S"]["model"]["rename"]("renamed")

    assert "model" not in mfx_nmda.synapses["S"]
    assert "model" not in mfx_nmda.functions["synapses"]["S"]

    assert "renamed" in mfx_nmda.synapses["S"]
    assert "renamed" in mfx_nmda.functions["synapses"]["S"]

    assert mfx_nmda.synapses["S"]["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["synapses"]["S"]["renamed"]
        for f in ["add", "delete", "modify", "rename"]
    )

    mfx_nmda.functions["synapses"]["S"]["renamed"]["rename"]("another_name")

    assert "renamed" not in mfx_nmda.synapses["S"]
    assert "renamed" not in mfx_nmda.functions["synapses"]["S"]

    assert "another_name" in mfx_nmda.synapses["S"]
    assert "another_name" in mfx_nmda.functions["synapses"]["S"]

    assert mfx_nmda.synapses["S"]["another_name"] == test_data
    assert all(
        f in mfx_nmda.functions["synapses"]["S"]["another_name"]
        for f in ["add", "delete", "modify", "rename"]
    )


def test_rename_monitor_element(mfx_nmda: b2_model.B2Model) -> None:
    assert "type" in mfx_nmda.monitors["M"]
    test_data = deepcopy(mfx_nmda.monitors["M"]["type"])

    mfx_nmda.functions["monitors"]["M"]["type"]["rename"]("renamed")

    assert "type" not in mfx_nmda.monitors["M"]
    assert "type" not in mfx_nmda.functions["monitors"]["M"]

    assert "renamed" in mfx_nmda.monitors["M"]
    assert "renamed" in mfx_nmda.functions["monitors"]["M"]

    assert mfx_nmda.monitors["M"]["renamed"] == test_data
    assert all(
        f in mfx_nmda.functions["monitors"]["M"]["renamed"]
        for f in ["add", "delete", "modify", "rename"]
    )


def test_neuron_subgroup(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    mfx.define_neuronsubgroup_target("inhibitory", "P")
    mfx.define_neuronsubgroup_idxs("inhibitory", (20, 30))

    assert mfx.neurongroups["excitatory"]["idxs"] == (0, 20)
    assert mfx.neurongroups["inhibitory"]["idxs"] == (20, 30)

    mfx._assemble_neuronsubgroup("excitatory")
    mfx._assemble_neuronsubgroup("inhibitory")

    assert mfx.network["excitatory"] is not None
    assert mfx.network["inhibitory"] is not None


def test_neuron_subgroup_remove_exception(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))
    mfx._assemble_neuronsubgroup("excitatory")

    def do_raise(name: Any) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2.Network, "remove", do_raise)

    with pytest.raises(
        RuntimeError, match="Error when removing existing subgroup excitatory:"
    ):
        mfx._assemble_neuronsubgroup("excitatory")


def test_neuron_subgroup_add_exception(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    def do_raise(name: Any) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2.Network, "add", do_raise)

    with pytest.raises(RuntimeError, match="Error when adding subgroup"):
        mfx._assemble_neuronsubgroup("excitatory")


def test_neuron_subgroup_creation_exception(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]], monkeypatch: MonkeyPatch
) -> None:
    mfx, _ = mfx_parameterized
    mfx.define_neuronsubgroup_target("excitatory", "P")
    mfx.define_neuronsubgroup_idxs("excitatory", (0, 20))

    def do_raise(*_, **__) -> None:
        raise RuntimeError("dummy")

    monkeypatch.setattr(b2, "Subgroup", do_raise)

    with pytest.raises(
        RuntimeError, match="Error when trying to create a neuronsubgroup"
    ):
        mfx._assemble_neuronsubgroup("excitatory")


def test_neuron_subgroup_wrong_group(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.network = b2.Network()
    with pytest.raises(
        KeyError, match="Error, target Px could not be found in neurongroups"
    ):
        mfx.define_neuronsubgroup_target("ng", "Px")


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


def test_define_synapse_delay(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    mfx.define_synapse_delay("Ce", 1e-3 * b2.second)

    assert mfx.synapses["Ce"]["delay"] == 1e-3 * b2.second


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

    mfx._assemble_neuronsubgroup("Pi")
    mfx._assemble_neuronsubgroup("Pe")

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
    mfx.define_synapse_solver_method("Ce", "exponential_euler")

    mfx._assemble_synapse("Ce")

    assert len(mfx.synapses) == 1
    assert mfx.synapses["Ce"]["origin"] == "Pe"
    assert mfx.synapses["Ce"]["target"] == "Pi"
    assert mfx.synapses["Ce"]["method"] == "exponential_euler"
    ce = mfx.network["Ce"]

    # test removal and reassembly
    mfx.define_synapse_connection_probablitiy("Ce", 0.5)
    mfx._assemble_synapse("Ce")
    assert ce is not mfx.network["Ce"]
    assert mfx.synapses["Ce"]["p"] == 0.5


def test_assemble_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "formozov_project.b2_model.b2.Synapses", side_effect=Exception("fake exception")
    )
    mfx, _ = mfx_with_synapse

    with pytest.raises(RuntimeError, match="Error in defining synapse Ce"):
        mfx._assemble_synapse("Ce")


def test_connect_synapse_exception(
    mocker: MockerFixture, mfx_with_synapse: tuple[b2_model.B2Model, dict[str, Any]]
) -> None:
    mocker.patch(
        "formozov_project.b2_model.b2.Synapses.connect",
        side_effect=Exception("fake exception"),
    )

    mfx, _ = mfx_with_synapse
    with pytest.raises(RuntimeError, match="Error in connecting synapse Ce"):
        mfx._assemble_synapse("Ce")


def test_define_monitor(
    mfx_parameterized: tuple[b2_model.B2Model, dict[str, Any]],
) -> None:
    mfx, _ = mfx_parameterized

    assert mfx.monitors == {}

    mfx.define_monitor_source("Traces_ge", "P")
    mfx.define_monitor_type("Traces_ge", "StateMonitor")
    mfx.define_monitor_variable("Traces_ge", "ge")

    mfx._assemble_monitor("Traces_ge")

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
        mfx._assemble_monitor("M")


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
    b2.seed(4321)
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

    # initialize
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

    assert_allclose(S.w, mfx_nmda.network["S"].w)
    assert_allclose(neuron_input.v, mfx_nmda.network["neuron_input"].v)

    test_network.run(100 * b2.ms)

    # assert result equality
    assert_allclose(M.g, mfx_nmda.network["M"].g)
    assert_allclose(Mn.g, mfx_nmda.network["Mn"].g)

    mfx_nmda.must_be_rebuilt = True
    mfx_nmda.run_simulation(100 * b2.ms)
    assert (
        mfx_nmda.must_be_rebuilt is False
    )  # false indicate that the model has been rebuild


def test_NMDA_model_delete(mfx_nmda: b2_model.B2Model) -> None:
    assert "neuronnumber" in mfx_nmda.neurongroups["neuron_input"]
    mfx_nmda.delete_neurongroup_element("neuron_input", "neuronnumber")
    assert "neuronnumber" not in mfx_nmda.neurongroups["neuron_input"]
    assert "neuronnumber" not in mfx_nmda.functions["neurongroups"]["neuron_input"]

    assert "on_pre" in mfx_nmda.synapses["S"]
    mfx_nmda.delete_synapse_element("S", "on_pre")
    assert "on_pre" not in mfx_nmda.synapses["S"]
    assert "on_pre" not in mfx_nmda.functions["synapses"]["S"]

    assert "source" in mfx_nmda.monitors["Mn"]
    mfx_nmda.delete_monitor_element("Mn", "source")
    assert "source" not in mfx_nmda.monitors["Mn"]
    assert "source" not in mfx_nmda.functions["monitors"]["Mn"]

    assert "w" in mfx_nmda.initials["S"]
    mfx_nmda.delete_init_element("S", "w")
    assert "w" not in mfx_nmda.initials["S"]
    assert "w" not in mfx_nmda.functions["initials"]["S"]

    assert "Mn" in mfx_nmda.monitors
    mfx_nmda.delete_monitor("Mn")
    assert "Mn" not in mfx_nmda.monitors
    assert "Mn" not in mfx_nmda.functions["monitors"]

    with pytest.raises(
        KeyError, match="Error when deleting initial conditions w of group S"
    ):
        mfx_nmda.delete_init_element("S", "w")

    assert "S" in mfx_nmda.initials
    mfx_nmda.delete_init("S")
    assert "S" not in mfx_nmda.initials

    assert "neuron_input" in mfx_nmda.neurongroups
    mfx_nmda.delete_neurongroup("neuron_input")
    assert "neuron_input" not in mfx_nmda.neurongroups

    assert "S" in mfx_nmda.synapses
    mfx_nmda.delete_synapse("S")
    assert "S" not in mfx_nmda.synapses

    with pytest.raises(RuntimeError, match="Error when deleting X in neurons"):
        mfx_nmda.delete_neurongroup_element("neurons", "X")

    assert "a" in mfx_nmda.parameters  # type: ignore
    mfx_nmda.delete_parameter("a")
    assert "a" not in mfx_nmda.parameters  # type: ignore

    with pytest.raises(KeyError, match="Error when deleting x"):
        mfx_nmda.delete_parameter("x")

    with pytest.raises(
        KeyError, match="Error when deleting initial conditions for group X"
    ):
        mfx_nmda.delete_init("X")


def test_assemble_network(mfx_nmda, monkeypatch: MonkeyPatch) -> None:
    a = 1 / (10 * b2.ms)
    b = 1 / (10 * b2.ms)
    c = 1 / (10 * b2.ms)

    model = b2_model.B2Model(seed=4321)
    # model.define_dt(0.2 * b2.ms)

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

    model.define_neuronsubgroup_target("input_subgroup", "neuron_input")
    model.define_neuronsubgroup_idxs("input_subgroup", 1)

    model.define_monitor_source("M", "S")
    model.define_monitor_type("M", "StateMonitor")
    model.define_monitor_variable("M", "g")

    model.define_monitor_source("Mn", "neurons")
    model.define_monitor_type("Mn", "StateMonitor")
    model.define_monitor_variable("Mn", "g")

    model.define_init("S", "w", [1.0, 10.0])
    model.define_init("neuron_input", "v", [0.0, 0.5])

    model.assemble_network()

    assert model.neurongroups == mfx_nmda.neurongroups
    assert model.synapses == mfx_nmda.synapses
    assert model.monitors == mfx_nmda.monitors
    assert model.parameters == mfx_nmda.parameters

    assert sorted(list(model.network.get_states().keys())) == sorted(
        list(mfx_nmda.network.get_states().keys())
    )

    assert "M" in [o.name for o in model.get_network_objects()]
    assert "neuron_input" in [o.name for o in model.get_network_objects()]
    assert "S" in [o.name for o in model.get_network_objects()]

    nrg = model.get_neurongroup("neurons")

    assert nrg["definition"] == model.neurongroups["neurons"]
    assert nrg["object"] is model.network["neurons"]

    assert model.get_variables_of("neurons") == list(model.network["neurons"].variables)
    assert model.get_equation_names_of("neurons") == list(
        model.network["neurons"].equations.names
    )

    with pytest.raises(
        KeyError,
        match="Error, name X is neither synapse nor neurongroup and hence does not have equations to get variables from",
    ):
        model.get_equation_names_of("X")

    def raise_error(x: Any) -> None:
        raise RuntimeError("Dummy")

    monkeypatch.setattr(b2.Network, "__getattribute__", raise_error)

    with pytest.raises(
        RuntimeError,
        match="Error, something went wrong when getting equation variables for neurons",
    ):
        model.get_equation_names_of("neurons")


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
    assert_allclose(mfx_nmda.network["S"].g, reconstructed_model.network["S"].g)

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
