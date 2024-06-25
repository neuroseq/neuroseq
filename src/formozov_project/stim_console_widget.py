from __future__ import annotations

from typing import Any

import numpy as np
from pyqtgraph.console.repl_widget import ReplWidget
from qtpy import QtCore
from qtpy import QtWidgets


class StimConsoleWidget(ReplWidget):
    stims_updated = QtCore.Signal(np.ndarray, bool)

    def __init__(
        self,
        stims: np.ndarray | None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(globals=self.namespace, locals=self.namespace, parent=parent)
        self.namespace_dict = {"np": np, "stims": stims}
        self._print_intro()

    def _print_intro(self) -> None:
        self.output.insertPlainText(
            "This is a Python console where you can edit the stimuli by modifying the 2-d numpy array 'stims'.\n"
        )
        self.output.insertPlainText(
            "The first index is the neuron index, the second index is the timestamp.\n"
        )
        self.output.insertPlainText(
            "Here are some examples you can try entering below:\n\n"
        )
        self.write(">>> stims = np.ones((2,100))", style="command")
        self.output.insertPlainText("\n")
        self.write(">>> stims = np.random.randint(2,size=(24,500))", style="command")
        self.output.insertPlainText("\n")
        self.write(">>> stims[::2,::10] = 1", style="command")
        self.output.insertPlainText(
            "\n\nNumpy is available as np, and indentation can be done with a single space.\n\n\n"
        )

    def namespace(self) -> dict[str, Any]:
        return self.namespace_dict

    def validate_stims(self) -> tuple[bool, str]:
        stims = self.namespace()["stims"]
        if stims is None:
            return False, "None"
        if not isinstance(stims, np.ndarray):
            return False, f"{type(stims)}"
        if stims.ndim != 2:
            return False, f"{stims.ndim}d numpy array"
        return True, "2d numpy array"

    def runCmd(self, cmd: str) -> None:
        super().runCmd(cmd)
        stims_is_valid, stims_type = self.validate_stims()
        if stims_is_valid:
            self.stims_updated.emit(self.namespace()["stims"], stims_is_valid)
        else:
            self.output.insertPlainText(
                f"    Invalid stim array: expected a 2d numpy array, found {stims_type}\n"
            )
            self.stims_updated.emit(np.ndarray(shape=(0,)), stims_is_valid)
