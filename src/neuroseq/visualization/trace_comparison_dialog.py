"""Dialog choosing which neurons and which trace arrays to compare."""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Sequence

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.visualization_support.trace_comparison import parse_neuron_ids

_CHECKED = QtCore.Qt.CheckState.Checked
_UNCHECKED = QtCore.Qt.CheckState.Unchecked
_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _check_state(checked: bool) -> QtCore.Qt.CheckState:
    return _CHECKED if checked else _UNCHECKED


class TraceComparisonDialog(QtWidgets.QDialog):
    """Pick any number of neurons and the trace arrays to compare them across.

    The neuron list can be filtered by id; **Select all**/**Select none** act
    on the neurons the filter shows, and the range field ticks ids such as
    ``0-5, 9``. The arrays list holds every trace array of the project. OK
    stays disabled until at least one neuron and one array are ticked.
    """

    def __init__(
        self,
        neuron_ids: Sequence[int],
        array_names: Sequence[str],
        checked_neuron_ids: Collection[int] = (),
        checked_array_names: Collection[str] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare traces")
        self.resize(560, 480)
        layout = QtWidgets.QHBoxLayout(self)

        # Neurons
        neuron_box = QtWidgets.QGroupBox("Neurons", self)
        neuron_layout = QtWidgets.QVBoxLayout(neuron_box)
        self.filter_edit = QtWidgets.QLineEdit(neuron_box)
        self.filter_edit.setPlaceholderText("Filter, e.g. 1 or 10-20")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        neuron_layout.addWidget(self.filter_edit)

        neuron_buttons = QtWidgets.QHBoxLayout()
        self.select_all_button = QtWidgets.QPushButton("Select all", neuron_box)
        self.select_none_button = QtWidgets.QPushButton("Select none", neuron_box)
        self.select_all_button.setToolTip("Tick every neuron the filter shows")
        self.select_none_button.setToolTip("Untick every neuron the filter shows")
        self.select_all_button.clicked.connect(lambda: self._set_shown_checked(True))
        self.select_none_button.clicked.connect(lambda: self._set_shown_checked(False))
        neuron_buttons.addWidget(self.select_all_button)
        neuron_buttons.addWidget(self.select_none_button)
        neuron_buttons.addStretch()
        neuron_layout.addLayout(neuron_buttons)

        checked_ids = {int(n) for n in checked_neuron_ids}
        self.neuron_list = QtWidgets.QListWidget(neuron_box)
        for neuron_id in neuron_ids:
            item = QtWidgets.QListWidgetItem(f"Neuron {int(neuron_id)}")
            item.setData(_ID_ROLE, int(neuron_id))
            item.setCheckState(_check_state(int(neuron_id) in checked_ids))
            self.neuron_list.addItem(item)
        neuron_layout.addWidget(self.neuron_list)

        range_row = QtWidgets.QHBoxLayout()
        self.range_edit = QtWidgets.QLineEdit(neuron_box)
        self.range_edit.setPlaceholderText("Tick ids, e.g. 0-5, 9")
        self.range_edit.returnPressed.connect(self._tick_range)
        self.range_button = QtWidgets.QPushButton("Tick", neuron_box)
        self.range_button.clicked.connect(self._tick_range)
        range_row.addWidget(self.range_edit)
        range_row.addWidget(self.range_button)
        neuron_layout.addLayout(range_row)
        self.range_error = QtWidgets.QLabel("", neuron_box)
        self.range_error.setWordWrap(True)
        self.range_error.hide()
        neuron_layout.addWidget(self.range_error)
        layout.addWidget(neuron_box, stretch=1)

        # Trace arrays
        right = QtWidgets.QVBoxLayout()
        array_box = QtWidgets.QGroupBox("Trace arrays", self)
        array_layout = QtWidgets.QVBoxLayout(array_box)
        self.array_list = QtWidgets.QListWidget(array_box)
        checked_arrays = set(checked_array_names)
        for name in array_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(_ID_ROLE, name)
            item.setCheckState(_check_state(name in checked_arrays))
            self.array_list.addItem(item)
        array_layout.addWidget(self.array_list)
        array_buttons = QtWidgets.QHBoxLayout()
        self.select_all_arrays_button = QtWidgets.QPushButton("Select all", array_box)
        self.select_no_arrays_button = QtWidgets.QPushButton("Select none", array_box)
        self.select_all_arrays_button.clicked.connect(
            lambda: self._set_arrays_checked(True)
        )
        self.select_no_arrays_button.clicked.connect(
            lambda: self._set_arrays_checked(False)
        )
        array_buttons.addWidget(self.select_all_arrays_button)
        array_buttons.addWidget(self.select_no_arrays_button)
        array_buttons.addStretch()
        array_layout.addLayout(array_buttons)
        right.addWidget(array_box, stretch=1)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        right.addWidget(self.button_box)
        layout.addLayout(right, stretch=1)

        self.neuron_list.itemChanged.connect(self._update_ok_enabled)
        self.array_list.itemChanged.connect(self._update_ok_enabled)
        self._update_ok_enabled()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _items(list_widget: QtWidgets.QListWidget) -> list[QtWidgets.QListWidgetItem]:
        items = (list_widget.item(row) for row in range(list_widget.count()))
        return [item for item in items if item is not None]

    @property
    def ok_button(self) -> QtWidgets.QPushButton:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        return button

    def _update_ok_enabled(self, *_args) -> None:
        self.ok_button.setEnabled(
            bool(self.selected_neuron_ids) and bool(self.selected_array_names)
        )

    def _apply_filter(self, text: str) -> None:
        """Show neurons whose id matches the filter (an id/range or a substring)."""
        text = text.strip()
        wanted: set[int] | None
        try:
            wanted = set(parse_neuron_ids(text)) if text else None
        except ValueError:
            wanted = None
        for item in self._items(self.neuron_list):
            neuron_id = int(item.data(_ID_ROLE))
            if not text:
                hidden = False
            elif wanted is not None:
                hidden = neuron_id not in wanted
            else:
                hidden = text.lower() not in item.text().lower()
            item.setHidden(hidden)

    def _set_shown_checked(self, checked: bool) -> None:
        for item in self._items(self.neuron_list):
            if not item.isHidden():
                item.setCheckState(_check_state(checked))

    def _set_arrays_checked(self, checked: bool) -> None:
        for item in self._items(self.array_list):
            item.setCheckState(_check_state(checked))

    def _tick_range(self) -> None:
        try:
            ids = set(parse_neuron_ids(self.range_edit.text()))
        except ValueError as error:
            self.range_error.setText(str(error))
            self.range_error.show()
            return
        self.range_error.hide()
        found = set()
        for item in self._items(self.neuron_list):
            neuron_id = int(item.data(_ID_ROLE))
            if neuron_id in ids:
                item.setCheckState(_CHECKED)
                found.add(neuron_id)
        if ids - found:
            self.range_error.setText(
                f"{len(ids - found)} id(s) not in this array were ignored."
            )
            self.range_error.show()

    # -- results ---------------------------------------------------------

    @property
    def selected_neuron_ids(self) -> list[int]:
        return [
            int(item.data(_ID_ROLE))
            for item in self._items(self.neuron_list)
            if item.checkState() == _CHECKED
        ]

    @property
    def selected_array_names(self) -> list[str]:
        return [
            str(item.data(_ID_ROLE))
            for item in self._items(self.array_list)
            if item.checkState() == _CHECKED
        ]
