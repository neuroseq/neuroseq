import logging
import typing

import brian2 as b2
import pytest
from formozov_project.b2_model import B2Model


@pytest.fixture()
def mfx(tmp_path) -> typing.Generator[B2Model, None, None]:
    yield B2Model(tmp_path, loglevel="DEBUG")


@pytest.fixture
def mfx_parameterized(mfx) -> typing.Generator[B2Model, None, None]:
    eq = "dge/dt = -ge*(1./taue): siemens\n"
    n = 30
    threshold = "v > 10 * mV"
    refractory = 5 * b2.ms
    method = "euler"
    dt = 0.1 * b2.ms
    mfx.define_equation(eq)
    mfx.define_neuronnumber(n)
    mfx.define_threshold(threshold)
    mfx.define_refractory(refractory)
    mfx.define_dt(dt)
    mfx.define_solver_method(method)
    mfx.define_parameter("ge", "(randn() * 1.5 + 4) * 10.*nS")
    mfx.assemble_neurongroup()
    yield mfx


@pytest.fixture()
def mfx_parameterized_with_synapse(
    mfx_parameterized,
) -> typing.Generator[B2Model, None, None]:
    mfx = mfx_parameterized

    taupre = 20 * b2.ms
    taupost = taupre
    gmax = 60 * b2.nS
    dApre = 6 * b2.nS
    dApost = -dApre * taupre / taupost * 1.05

    mfx = mfx_parameterized
    mfx.define_neuron_subgroup("Pe", slice(0, 24))
    mfx.define_neuron_subgroup("Pi", slice(24, 30))

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

    yield mfx


def test_construction(tmp_path) -> None:
    model = B2Model(tmp_path, loglevel="DEBUG")

    assert model.path == tmp_path
    assert model.logger.name == "Model"
    assert model.logger.level == logging.DEBUG
    assert model.logger.handlers[0].baseFilename == str(tmp_path / "logfile.log")

    assert model.equation == ""
    assert model.parameters == {}
    assert model.neurongroup is None
    assert model.neuron_subgroups == {}
    assert model.synapses == {}
    assert model.monitors == {}


def test_define_equation(mfx) -> None:
    mfx.define_equation("dge/dt = -ge*(1./taue) : siemens")

    assert mfx.equation == "dge/dt = -ge*(1./taue) : siemens\n"


def test_equation_no_unit(mfx) -> None:
    with pytest.raises(
        ValueError, match="Equation must have a colon that separates equation and unit"
    ):
        mfx.define_equation("dge/dt = -ge*(1./taue)")


def test_equation_no_variable(mfx) -> None:
    with pytest.raises(
        ValueError,
        match="Equation must have an equal sign that separates variable and expression",
    ):
        mfx.define_equation("-ge*(1./taue): siemens")


def test_define_dt(mfx) -> None:
    assert mfx.dt is None

    mfx.define_dt(0.1 * b2.ms)

    with pytest.raises(ValueError, match="dt must be a Quantity"):
        mfx.define_dt(0.1)


def test_define_parameter(mfx) -> None:
    assert mfx.parameters == {}

    mfx.define_parameter("taue", 5 * b2.ms)
    mfx.define_parameter("taui", 10 * b2.ms)
    mfx.define_parameter("area", 20000 * b2.umetre**2)

    assert mfx.parameters == {
        "taue": 5 * b2.ms,
        "taui": 10 * b2.ms,
        "area": 20000 * b2.umetre**2,
    }

    with pytest.raises(
        ValueError, match="Parameter value must be a Quantity or a string"
    ):
        mfx.define_parameter("taue", 5)


def test_define_threshold(mfx) -> None:
    assert mfx.th is None
    mfx.define_threshold("v > 10 * mV")
    assert mfx.th == "v > 10 * mV"


def test_define_solver_method(mfx) -> None:
    assert mfx.solver is None
    mfx.define_solver_method("euler")
    assert mfx.solver == "euler"

    # TODO: add solver method verification


def test_define_refractory(mfx) -> None:
    assert mfx.refractory is None
    mfx.define_refractory(5 * b2.ms)
    assert mfx.refractory == 5 * b2.ms

    with pytest.raises(ValueError, match="Refractory period must be a Quantity"):
        mfx.define_refractory(5)


def test_define_neuron_number(mfx) -> None:
    assert mfx.n is None
    mfx.define_neuronnumber(100)
    assert mfx.n == 100

    with pytest.raises(ValueError, match="Neuron number must be an integer"):
        mfx.define_neuronnumber(100.0)


def test_assemble_neurongroup(mfx) -> None:
    eq = "dge/dt = -ge*(1./taue): siemens\n"
    n = 30
    threshold = "v > 10 * mV"
    refractory = 5 * b2.ms
    method = "euler"
    dt = 0.1 * b2.ms
    mfx.define_equation(eq)
    mfx.define_neuronnumber(n)
    mfx.define_threshold(threshold)
    mfx.define_refractory(refractory)
    mfx.define_dt(dt)
    mfx.define_solver_method(method)
    mfx.define_parameter("ge", "(randn() * 1.5 + 4) * 10.*nS")

    ng = b2.NeuronGroup(
        n,
        eq,
        threshold=threshold,
        refractory=refractory,
        method=method,
        dt=dt,
    )

    assert mfx.neurongroup is None
    mfx.assemble_neurongroup()
    assert mfx.neurongroup.equations == ng.equations
    assert mfx.neurongroup._refractory == ng._refractory


def test_neuron_subgroup(mfx_parameterized) -> None:
    mfx = mfx_parameterized
    assert mfx.neuron_subgroups == {}
    mfx.define_neuron_subgroup("excitatory", slice(0, 20))
    mfx.define_neuron_subgroup("inhibitory", slice(20, 30))

    assert list(mfx.neuron_subgroups["excitatory"].indices) == list(range(0, 20))
    assert list(mfx.neuron_subgroups["inhibitory"].indices) == list(range(20, 30))

    mfx.neurongroup = None
    with pytest.raises(ValueError, match="Neurongroup not defined yet"):
        mfx.define_neuron_subgroup("excitatory", slice(0, 20))


def test_define_synapse(mfx_parameterized) -> None:
    taupre = 20 * b2.ms
    taupost = taupre
    gmax = 60 * b2.nS
    dApre = 6 * b2.nS
    dApost = -dApre * taupre / taupost * 1.05

    mfx = mfx_parameterized
    mfx.define_neuron_subgroup("Pe", slice(0, 24))
    mfx.define_neuron_subgroup("Pi", slice(24, 30))

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
    assert mfx.synapses["Ce"]["synapse"].source == mfx.neuron_subgroups["Pe"]
    assert mfx.synapses["Ce"]["synapse"].target == mfx.neuron_subgroups["Pi"]


def test_define_monitor(mfx_parameterized) -> None:
    mfx = mfx_parameterized

    assert mfx.monitors == {}

    mfx.define_monitor("Traces_ge", b2.StateMonitor, mfx.neurongroup, "ge", record=True)

    assert "Traces_ge" in mfx.monitors
    assert mfx.monitors["Traces_ge"].source == mfx.neurongroup
    assert mfx.monitors["Traces_ge"].record_variables == [
        "ge",
    ]
