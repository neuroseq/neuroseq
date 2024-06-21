from __future__ import annotations

from typing import Any

import numpy as np
from pyqtgraph.console.repl_widget import ReplWidget
from qtpy import QtCore
from qtpy import QtWidgets


class StimConsoleWidget(ReplWidget):
    stim_updated = QtCore.Signal(np.ndarray, bool)

    def __init__(
        self,
        stim: np.ndarray | None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(globals=self.namespace, locals=self.namespace, parent=parent)
        self.namespace_dict = {"np": np, "stim": stim}
        self._print_intro()

    def _print_intro(self) -> None:
        self.output.insertPlainText(
            "This is a Python console where you can edit the stimuli by modifying the 2-d numpy array 'stim'.\n"
        )
        self.output.insertPlainText(
            "The first index is the neuron index, the second index is the timestamp.\n"
        )
        self.output.insertPlainText(
            "Here are some examples you can try entering below:\n\n"
        )
        self.write(">>> stim = np.ones((2,100))", style="command")
        self.output.insertPlainText("\n")
        self.write(">>> stim = np.random.randint(2,size=(24,500))", style="command")
        self.output.insertPlainText("\n")
        self.write(">>> stim[::2,::10] = 1", style="command")
        self.output.insertPlainText(
            "\n\nNumpy is available as np, and indentation can be done with a single space.\n\n\n"
        )

    def namespace(self) -> dict[str, Any]:
        return self.namespace_dict

    def validate_stim(self) -> tuple[bool, str]:
        stim = self.namespace()["stim"]
        if stim is None:
            return False, "None"
        if not isinstance(stim, np.ndarray):
            return False, f"{type(stim)}"
        if stim.ndim != 2:
            return False, f"{stim.ndim}d numpy array"
        return True, "2d numpy array"

    def runCmd(self, cmd: str) -> None:
        super().runCmd(cmd)
        stim_is_valid, stim_type = self.validate_stim()
        if not stim_is_valid:
            self.output.insertPlainText(
                f"    Invalid stim array: expected a 2d numpy array, found {stim_type}\n"
            )
        self.stim_updated.emit(self.namespace()["stim"], stim_is_valid)
