import logging
from typing import Any

import brian2 as b2
from qtpy import QtWidgets

from formozov_project.stim_dialog import StimDialog


class ModelLoadingDialog(QtWidgets.QDialog):
    """
    Dialog for loading a model from File and making the gui wait until it has been fully constructed

    Parameters
    ----------
    QtWidgets : QWidget
        Basic QT wiget type.
    """

    def __init__(self, parent=None):
        """
        Create a new ModelLoadingDialog instance

        Parameters
        ----------
        parent : QtWidget, optional
            parent QtWidget, by default None
        """
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Loading model data from file")
        self.layout = QtWidgets.QGridLayout()
        self.logger = logging.getLogger(__name__)
        qtbtn = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        self.button_box = QtWidgets.QDialogButtonBox(qtbtn)

        self.button_box.accepted.connect(self.accept)

        self.button_box.rejected.connect(self.reject)

        self.trace_selector_name = QtWidgets.QLabel("Select trace to display", self)
        self.layout.addWidget(
            self.trace_selector_name,
            1,
            1,
        )

        self.trace_selector = QtWidgets.QComboBox(self)
        self.layout.addWidget(
            self.trace_selector,
            self.layout.rowCount(),
            1,
        )
        self.trace_time_selector = QtWidgets.QComboBox(self)
        self.layout.addWidget(
            self.trace_time_selector,
            self.layout.rowCount(),
            1,
        )

        self.heatmap_selector_name = QtWidgets.QLabel("Select heatmap to display", self)

        self.layout.addWidget(
            self.heatmap_selector_name, self.layout.rowCount(), 1, 1, 1
        )

        self.heatmap_selector = QtWidgets.QComboBox(self)
        self.layout.addWidget(self.heatmap_selector, self.layout.rowCount(), 1, 1, 1)

        self.heatmap_time_selector = QtWidgets.QComboBox(self)
        self.layout.addWidget(
            self.heatmap_time_selector, self.layout.rowCount(), 1, 1, 1
        )

        self.stim_selector_name = QtWidgets.QLabel(
            "Select stimulus array to display", self
        )
        self.stim_selector = QtWidgets.QComboBox(self)

        self.layout.addWidget(self.stim_selector_name, self.layout.rowCount(), 1, 1, 1)

        self.layout.addWidget(self.stim_selector, self.layout.rowCount(), 1, 1, 1)

        self.layout.addWidget(self.button_box, self.layout.rowCount(), 1, 1, 1)

        self.setLayout(self.layout)

        self.flattend_states = {}
        states = parent.model.network.get_states()
        states["parameters"] = parent.model.parameters
        self._flatten_dict_recursive(states, "", "", self.flattend_states)

        states["None"] = None  # dummy value where we want nothing

        self.heatmap_selector.addItems(sorted(list(self.flattend_states.keys())))

        self.trace_selector.addItems(sorted(list(self.flattend_states.keys())))
        self.trace_time_selector.addItems(
            sorted([k for k in self.flattend_states if ".t" in k] + ["None"])
        )

        self.heatmap_selector.currentTextChanged.connect(self.select_stim)
        self.heatmap_time_selector.addItems(
            sorted([k for k in self.flattend_states if ".t" in k] + ["None"])
        )

        self.stim_selector.addItems(
            [k for k in self.flattend_states if "parameters" in k]
        )

        self.trace_selector.currentTextChanged.connect(self.select_trace)
        self.trace_time_selector.currentTextChanged.connect(self.select_trace)

        self.heatmap_selector.currentTextChanged.connect(self.select_trace)
        self.heatmap_time_selector.currentTextChanged.connect(self.select_trace)

        self.stim_selector.currentTextChanged.connect(self.select_stim)

        self.heatmap_data: Any = None
        self.heatmap_time: Any = None

        self.trace_data: Any = None
        self.trace_time: Any = None

        self.stim_data: Any = None

        self.show()

    def _flatten_dict_recursive(self, d, k, key, res: dict):
        if isinstance(d, dict):
            key += str(k) + "." if k != "" else str(k)
            for k in d:
                self._flatten_dict_recursive(d[k], k, key, res)
        else:
            key += str(k)
            res[key] = d

    def _set_element(self, target: str, value: Any) -> None:
        self.logger.debug(f"Element {target} set to {value}")
        setattr(self, target, value)

    def select_trace(self) -> None:
        self._set_element(
            "trace_data", self.flattend_states[self.trace_selector.currentText()]
        )

    def select_trace_time(self) -> None:
        self._set_element(
            "trace_time", self.flattend_states[self.trace_selector.currentText()]
        )

    def select_heatmap(self) -> None:
        self._set_element(
            "heatmap_time", self.flattend_states[self.heatmap_selector.currentText()]
        )

    def select_heatmap_time(self) -> None:
        self._set_element(
            "heatmap_data",
            self.flattend_states[self.heatmap_time_selector.currentText()],
        )

    def select_stim(self) -> None:
        if not isinstance(
            self.flattend_states[self.stim_selector.currentText()], b2.TimedArray
        ):
            message = (
                f"Stimulus data must be a TimedArray, currently selected element {self.stim_selector.currentText()} is of type {type(self.flattend_states[self.stim_selector.currentText()])}",
            )

            self.logger.critical(message)

            QtWidgets.QMessageBox.critical(self, "Model data loading error", message)

        else:
            self._set_element(
                "stim_data", self.flattend_states[self.stim_selector.currentText()]
            )

    def accept(self) -> None:
        if not isinstance(self.stim_data, b2.TimedArray):
            message = (
                f"Stimulus data must be a TimedArray, currently selected element {self.stim_selector.currentText()} is of type {type(self.flattend_states[self.stim_selector.currentText()])}",
            )

            self.logger.critical(message)

            QtWidgets.QMessageBox.critical(self, "Model data loading error", message)
        else:
            self.logger.info("setting loaded data")
            self.parent.trace_widget.set_data(
                self.trace_data,
                self.trace_time,
            )

            self.parent.heatmap_widget.set_data(
                self.heatmap_data,
                self.heatmap_time,
            )

            self.parent.stim_widget = StimDialog(
                self.stim_data.values, self.stim_data.dt, self.parent
            )

            super().accept()
