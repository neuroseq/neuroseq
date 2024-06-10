import logging

import brian2 as b2
import pytest
from formozov_project.b2_model import B2Model


@pytest.fixture()
def mfx(tmp_path) -> B2Model:
    return B2Model(tmp_path, loglevel="DEBUG")


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
    assert model.synapses is None
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

    with pytest.raises(ValueError, match="Parameter value must be a Quantity"):
        mfx.define_parameter("taue", 5)


def test_define_threshold() -> None:
    pass


def test_define_solver_method() -> None:
    pass


def test_define_refractory() -> None:
    pass


def test_define_neuron_number() -> None:
    pass


def test_define_synapse() -> None:
    pass


def test_define_synapse_exception_in_def() -> None:
    pass


def test_define_synapse_exception_in_connect() -> None:
    pass


def test_define_monitor() -> None:
    pass


def test_assemble_neurongroup() -> None:
    pass


def test_save_to_yaml() -> None:
    pass


def test_load_from_yaml() -> None:
    pass


def test_run_simulation() -> None:
    pass
