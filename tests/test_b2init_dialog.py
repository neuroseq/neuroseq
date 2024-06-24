import formozov_project.b2_model_widget as B2W
import numpy as np
import pytest
from brian2 import ms
from brian2 import second
from formozov_project.b2_model import B2Model
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtCore


@pytest.fixture
def initdlg(qtbot: QtBot) -> B2W.InitialConditionWidget:
    model = B2Model()  # Assuming B2Model is a stub or mock if needed
    dialog = B2W.InitialConditionWidget(model)
    qtbot.addWidget(dialog)
    return dialog


def test_initialization(initdlg: B2W.InitialConditionWidget) -> None:
    assert initdlg.source is not None
    assert initdlg.var is not None
    assert initdlg.value is not None
    assert initdlg.params == {}


def test_accept_source(initdlg: B2W.InitialConditionWidget, qtbot: QtBot) -> None:
    qtbot.keyClicks(initdlg.source, "test_source")
    qtbot.mouseClick(initdlg.source_button, QtCore.Qt.LeftButton)
    assert initdlg.params["elementname"] == "test_source"
    assert initdlg.source_button.isDown()
    assert initdlg.source_button.text() == "saved"


def test_accept_variable(initdlg: B2W.InitialConditionWidget, qtbot: QtBot) -> None:
    qtbot.keyClicks(initdlg.var, "test_variable")
    qtbot.mouseClick(initdlg.var_button, QtCore.Qt.LeftButton)
    assert initdlg.var_button.isDown()
    assert initdlg.var_button.text() == "saved"
    assert initdlg.params["varname"] == "test_variable"


def test_accept_value(
    initdlg: B2W.InitialConditionWidget, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    qtbot.keyClicks(initdlg.value, "np.zeros(3)")
    qtbot.mouseClick(initdlg.value_button, QtCore.Qt.LeftButton)
    assert initdlg.params["value"] == eval("np.zeros(3)")
    assert initdlg.value_button.text() == "saved"
    assert initdlg.value_button.isDown()

    qtbot.keyClicks(initdlg.value, "np.zeros(3)")

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
    )

    def mock_eval_raise_error(_: str) -> None:
        raise ValueError("Mocked error")

    monkeypatch.setattr("builtins.eval", mock_eval_raise_error)

    qtbot.mouseClick(initdlg.value_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(initdlg.error_dialog.isVisible)

    assert (
        initdlg.error_dialog.text()
        == "Error in evaluating value expression np.zeros(3): Mocked error"
    )


def test_accept(
    initdlg: B2W.InitialConditionWidget, qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    # define neurongroup
    eqs = """
        tau : second
        sigma : 1
        dy/dt = -y/tau + sqrt(2*sigma**2/tau)*xi : 1
    """

    initdlg.model.define_neuronnumber("neurongroup", 2)
    initdlg.model.define_dt(0.2 * ms)
    initdlg.model.define_solver_method("neurongroup", "euler")
    initdlg.model.define_equation("neurongroup", eqs)
    initdlg.model.assemble_neurongroup(name="neurongroup")

    # define params

    initdlg.params["elementname"] = "neurongroup"
    initdlg.params["varname"] = "tau"
    initdlg.params["value"] = np.array([1, 0.8, 0.6, 0.3, 0.1]) * second

    # Trigger the accept method
    initdlg.accept()

    # Verify the expected outcome, possibly by mocking model.define_init
    assert initdlg.model.network["tau"] is not None

    monkeypatch.setattr(
        B2W.QtWidgets.QMessageBox, "exec", B2W.QtWidgets.QMessageBox.show
    )

    del initdlg.params["elementname"]

    initdlg.accept()

    qtbot.waitUntil(initdlg.error_dialog.isVisible)

    assert (
        initdlg.error_dialog.text()
        == "Cannot define incompletely specified initial condition"
    )

    initdlg.error_dialog.close()
