from __future__ import annotations

from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from pyqtgraph.console.repl_widget import ReplWidget
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.project import Project


class ConsoleWidget(ReplWidget):
    console_command_entered = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(globals=self.namespace, locals=self.namespace, parent=parent)
        self.namespace_dict: dict[str, Any] = {
            "np": np,
            "b2": b2,
            "project": None,
        }
        self._print_intro()

    def _print_intro(self) -> None:
        self.output.insertPlainText(
            "This is a Python console where you can interact directly with the Model using the 'project' variable.\n"
        )
        self.output.insertPlainText(
            "Here are some examples you can try entering below:\n\n"
        )
        self.write(
            ">>> project.stim_arrays['stimulus'].values[::10,::2] = 1", style="command"
        )
        self.output.insertPlainText("\n")
        self.write(
            ">>> project.stim_arrays['stimulus'].values = np.random.randint(3,size=(1000, 80))",
            style="command",
        )
        self.output.insertPlainText(
            "\n\nNumpy is available as np, brian2 as b2, and indentation can be done with a single space.\n\n\n"
        )

    def set_project(self, project: Project) -> None:
        self.namespace_dict["project"] = project

    def namespace(self) -> dict[str, Any]:
        return self.namespace_dict

    def runCmd(self, cmd: str) -> None:
        super().runCmd(cmd)
        self.namespace()["project"].update_limits()
        self.console_command_entered.emit()
