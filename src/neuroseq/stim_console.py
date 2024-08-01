from __future__ import annotations

from typing import Any

import brian2 as b2
import brian2.numpy_ as np
from pyqtgraph.console.repl_widget import ReplWidget
from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.b2_model import B2Model
from neuroseq.timed_array_modifer import TimedArrayModifier


class StimConsole(ReplWidget):
    sig_stims_values_dict_updated = QtCore.Signal(dict)  # dict[str, np.ndarray]

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(globals=self.namespace, locals=self.namespace, parent=parent)
        self.namespace_dict: dict[str, Any] = {
            "np": np,
            "b2": b2,
            "stims": None,
            "model": None,
        }
        self._print_intro()

    def _print_intro(self) -> None:
        self.output.insertPlainText(
            "This is a Python console where you can edit the stimuli by modifying the dict of 2-d numpy array 'stims'.\n"
        )
        self.output.insertPlainText(
            "The first index is the timestamp index, the second index is the neuron index.\n"
        )
        self.output.insertPlainText(
            "Here are some examples you can try entering below:\n\n"
        )
        self.write(">>> stims['stimulus'] = np.ones((100, 2))", style="command")
        self.output.insertPlainText("\n")
        self.write(
            ">>> stims['stimulus'] = np.random.randint(2,size=(500, 24))",
            style="command",
        )
        self.output.insertPlainText("\n")
        self.write(">>> stims['stimulus'][::10,::2] = 1", style="command")
        self.output.insertPlainText(
            "\n\nNumpy is available as np, and indentation can be done with a single space.\n\n\n"
        )

    def set_stims(self, stims: dict[str, TimedArrayModifier]) -> None:
        self.namespace_dict["stims"] = {
            k: v.timed_array.values.copy() for k, v in stims.items()
        }

    def set_model(self, model: B2Model) -> None:
        self.namespace_dict["model"] = model

    def namespace(self) -> dict[str, Any]:
        return self.namespace_dict

    def validate_stims(self) -> tuple[bool, str]:
        stims = self.namespace()["stims"]
        if stims is None:
            return False, "None"
        if not isinstance(stims, dict):
            return False, f"{type(stims)}"
        for name, stim in stims.items():
            if stim is None:
                return False, f"dict with None for key '{name}'"
            if not isinstance(stim, np.ndarray):
                return False, f"dict with {type(stim)} for key '{name}'"
            if not 1 <= stim.ndim <= 2:
                return False, f"dict with {stim.ndim}d numpy array for key '{name}'"
        return True, "dict of 1d or 2d numpy arrays"

    def runCmd(self, cmd: str) -> None:
        super().runCmd(cmd)
        valid_stims_dict, stims_type = self.validate_stims()
        if valid_stims_dict:
            self.sig_stims_values_dict_updated.emit(self.namespace()["stims"])
        else:
            self.output.insertPlainText(
                f"    Invalid stims: expected dict[str, 1d or 2d numpy array], found {stims_type}\n"
            )
