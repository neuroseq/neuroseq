from __future__ import annotations

from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from pyqtgraph.console.repl_widget import ReplWidget
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import console_art
from neuroseq.project import Project

SAMPLE_CODE = """\
# ── Project ──────────────────────────────────────────────────
project                              # the active Project object
project.stim_arrays                  # dict of StimArray objects
project.experiments                  # dict of Experiment objects
project.dt                           # canonical sample interval (s)

# ── Input arrays ───────────────────────────────────────────────
list(project.stim_arrays.keys())     # names of all input arrays
sa = project.stim_arrays['MyStim']   # get one input array
sa.values                            # numpy array, shape (time, neuron)
sa.dt                                # sample interval (s)
sa.units                             # e.g. 'pA'

# ── Edit input values (no undo) ───────────────────────────────
sa.values[:] = 0                     # zero out entire array
sa.values[100:200, 0] = 500          # set samples 100-199, neuron 0 to 500 pA
sa.values[:, 1] = np.sin(           \\
    np.linspace(0, 2*np.pi, sa.values.shape[0])\\
) * 200                              # sinusoidal pattern on neuron 1

# ── Ramp input patterns ────────────────────────────────────────
# Single ramp: linearly increase from 0 to 500 pA over 200 samples on neuron 0
sa.values[100:300, 0] = np.linspace(0, 500, 200)

# Ramp up then back down (triangle) on neuron 0
n = 200
sa.values[100:100+n, 0] = np.linspace(0, 500, n)
sa.values[100+n:100+2*n, 0] = np.linspace(500, 0, n)

# Staircase: 5 steps, each 100 samples wide, amplitude 0→500 pA on neuron 0
n_steps, step_len, amp = 5, 100, 500
for i in range(n_steps):
 sa.values[i*step_len:(i+1)*step_len, 0] = i * amp / (n_steps - 1)

# Sequential ramp across all neurons (each neuron gets a later ramp)
n_neurons = sa.values.shape[1]
for i in range(n_neurons):
 start = i * 50
 sa.values[start:start+100, i] = np.linspace(0, 300, 100)

# Sequential 100 ms pulses on all neurons with 18 ms inter-pulse gap
pulse_n = int(round(0.100 / sa.dt))
gap_n = int(round(0.018 / sa.dt))
n_neurons = sa.values.shape[1]
total_n = n_neurons * (pulse_n + gap_n)
amp = 1.0
new_len = max(sa.values.shape[0], total_n)
sa.values = np.zeros((new_len, n_neurons), dtype=sa.values.dtype)
for i in range(n_neurons):
 t0 = i * (pulse_n + gap_n)
 sa.values[t0:t0+pulse_n, i] = amp

# ── Experiment / trace data ───────────────────────────────────
exp = project.experiments['MyExp']
exp.data                             # list of TraceData / SpikeData objects

# ── Utilities ─────────────────────────────────────────────────
np                                   # numpy (already imported)
b2                                   # brian2 (already imported)
"""


class ConsoleWidget(ReplWidget):
    console_command_entered = QtCore.Signal()
    close_requested = QtCore.Signal()

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

        # Create button layout
        btn_layout = QtWidgets.QHBoxLayout()

        sample_btn = QtWidgets.QPushButton("Sample Python Code")
        sample_btn.clicked.connect(self._show_sample_code)
        sample_btn.setMaximumWidth(150)
        sample_btn.setFixedHeight(20)
        btn_layout.addWidget(sample_btn)

        history_btn = QtWidgets.QPushButton("Console History")
        history_btn.clicked.connect(self._show_console_history)
        history_btn.setMaximumWidth(150)
        history_btn.setFixedHeight(20)
        btn_layout.addWidget(history_btn)

        close_console_btn = QtWidgets.QPushButton("Close Console")
        close_console_btn.clicked.connect(self.close_requested)
        close_console_btn.setMaximumWidth(150)
        close_console_btn.setFixedHeight(20)
        btn_layout.addWidget(close_console_btn)

        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

    def _print_intro(self) -> None:
        # A fresh brain is grown for every launch, so the splash is never
        # quite the same twice.
        self._insert_centered(console_art.console_splash())
        self.output.insertPlainText(
            "This is a Python console where you can interact directly with the Model using the 'project' object.\n"
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
        self.output.insertPlainText(
            "\n\nClick 'Sample Python Code' for more information and examples.\n\n\n"
        )
        # Inserting all this text auto-scrolls the view to the bottom, hiding
        # the "Welcome to NeuroSeq" splash at the top on startup - scroll
        # back up once Qt has finished laying out the freshly inserted text.
        QtCore.QTimer.singleShot(0, lambda: self.output.verticalScrollBar().setValue(0))

    def _insert_centered(self, text: str) -> None:
        """Insert ``text`` as horizontally centered blocks, then go back to left.

        Qt centers each line on its own, so this only preserves ASCII art when
        every line is the same length - which ``console_splash`` guarantees.
        Centering is done per block rather than by padding with spaces so it
        stays correct when the console is resized.
        """
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        centered = QtGui.QTextBlockFormat()
        centered.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        cursor.setBlockFormat(centered)
        cursor.insertText(text)
        left = QtGui.QTextBlockFormat()
        left.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        cursor.setBlockFormat(left)
        self.output.setTextCursor(cursor)

    def _show_sample_code(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Console: sample python code")
        dlg.resize(620, 480)
        layout = QtWidgets.QVBoxLayout(dlg)
        text_edit = QtWidgets.QPlainTextEdit(SAMPLE_CODE)
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

    def _show_console_history(self) -> None:
        """Show a dialog with the console command history."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Console: command history")
        dlg.resize(620, 480)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Get history from CmdInput widget (self.input.history is a list)
        history_list = [cmd for cmd in self.input.history if cmd.strip()]
        history_text = "\n".join(history_list) if history_list else "(no history)"

        text_edit = QtWidgets.QPlainTextEdit(history_text)
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
        # A console command may mutate project state directly (e.g.
        # ``project.stim_arrays['x'].values = ...``) rather than through one
        # of Project's own mutating methods, which this can't tell apart
        # from a read-only command - mark dirty unconditionally rather than
        # risk silently missing a real edit.
        self.namespace()["project"].update_limits()
        self.namespace()["project"].mark_dirty()
        self.console_command_entered.emit()
