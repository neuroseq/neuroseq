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
        sample_btn = QtWidgets.QPushButton("Sample python code")
        sample_btn.clicked.connect(self._show_sample_code)
        self.layout.addWidget(sample_btn)

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

    def _show_sample_code(self) -> None:
        samples = """\
# ── Project ──────────────────────────────────────────────────
project                              # the active Project object
project.stim_arrays                  # dict of StimArray objects
project.experiments                  # dict of Experiment objects
project.dt                           # canonical sample interval (s)

# ── Stimulus arrays ───────────────────────────────────────────
list(project.stim_arrays.keys())     # names of all stim arrays
sa = project.stim_arrays['MyStim']   # get one stim array
sa.values                            # numpy array, shape (time, neuron)
sa.dt                                # sample interval (s)
sa.units                             # e.g. 'pA'

# ── Edit stimulus values (no undo) ───────────────────────────
sa.values[:] = 0                     # zero out entire array
sa.values[100:200, 0] = 500          # set samples 100-199, neuron 0 to 500 pA
sa.values[:, 1] = np.sin(           \\
    np.linspace(0, 2*np.pi, sa.values.shape[0])\\
) * 200                              # sinusoidal pattern on neuron 1

# ── Experiment / trace data ───────────────────────────────────
exp = project.experiments['MyExp']
exp.data                             # list of TraceData / SpikeData objects

# ── Utilities ─────────────────────────────────────────────────
np                                   # numpy (already imported)
b2                                   # brian2 (already imported)
"""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Console — sample python code")
        dlg.resize(620, 480)
        layout = QtWidgets.QVBoxLayout(dlg)
        text_edit = QtWidgets.QPlainTextEdit(samples)
        text_edit.setReadOnly(True)
        text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        text_edit.setMaximumHeight(400)
        font = text_edit.font()
        font.setFamily("Monospace")
        font.setPointSize(9)
        text_edit.setFont(font)
        layout.addWidget(text_edit)
        close_btn = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close_btn.rejected.connect(dlg.reject)
        layout.addWidget(close_btn)
        dlg.exec()

    def set_project(self, project: Project) -> None:
        self.namespace_dict["project"] = project

    def namespace(self) -> dict[str, Any]:
        return self.namespace_dict

    def runCmd(self, cmd: str) -> None:
        super().runCmd(cmd)
        self.namespace()["project"].update_limits()
        self.console_command_entered.emit()
