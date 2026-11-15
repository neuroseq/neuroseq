"""Dialog choosing which neurons and which trace arrays to compare."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.data_names import group_name_problem
from neuroseq.visualization_support.trace_comparison import ComparisonView
from neuroseq.visualization_support.trace_comparison import array_labels
from neuroseq.visualization_support.trace_comparison import concatenation_group_name
from neuroseq.visualization_support.trace_comparison import in_ranges
from neuroseq.visualization_support.trace_comparison import parse_neuron_ranges
from neuroseq.visualization_support.trace_comparison import range_size
from neuroseq.visualization_support.trace_comparison import shared_dt

_CHECKED = QtCore.Qt.CheckState.Checked
_UNCHECKED = QtCore.Qt.CheckState.Unchecked
_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _check_state(checked: bool) -> QtCore.Qt.CheckState:
    return _CHECKED if checked else _UNCHECKED


class TraceComparisonDialog(QtWidgets.QDialog):
    """Pick any number of neurons and the trace arrays to compare them across.

    The neuron list can be filtered by id; **Select all**/**Select none** act
    on the neurons the filter shows, and the range field ticks ids such as
    ``0-5, 9``. The arrays list holds every trace array of the project and
    can be reordered (drag, or Up/Down) for joining. **View** picks how the
    traces are shown; each view's options are enabled only while it is
    chosen. "Also add as new trace array" needs every ticked array to share
    one ``dt`` (``array_dt`` reports it per array name) and a free group name
    (``existing_group_names``). OK stays disabled until at least one neuron
    and one array are ticked and, when adding, the name is usable.
    """

    def __init__(
        self,
        neuron_ids: Sequence[int],
        array_names: Sequence[str],
        checked_neuron_ids: Collection[int] = (),
        checked_array_names: Collection[str] = (),
        parent: QtWidgets.QWidget | None = None,
        *,
        array_dt: Callable[[str], float | None] | None = None,
        existing_group_names: Collection[str] = (),
        view: ComparisonView = ComparisonView.OVERLAY,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare traces")
        self.resize(640, 600)
        self._array_dt = array_dt
        self._dt_cache: dict[str, float | None] = {}
        self._existing_group_names = set(existing_group_names)
        self._name_edited = False
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
        self.array_list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.array_list.setToolTip(
            "Drag, or use Up/Down, to set the order in which arrays are joined"
        )
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
        self.move_up_button = QtWidgets.QPushButton("Up", array_box)
        self.move_down_button = QtWidgets.QPushButton("Down", array_box)
        self.move_up_button.setToolTip("Move the highlighted array up")
        self.move_down_button.setToolTip("Move the highlighted array down")
        self.move_up_button.clicked.connect(lambda: self._move_current_array(-1))
        self.move_down_button.clicked.connect(lambda: self._move_current_array(1))
        array_buttons.addWidget(self.select_all_arrays_button)
        array_buttons.addWidget(self.select_no_arrays_button)
        array_buttons.addStretch()
        array_buttons.addWidget(self.move_up_button)
        array_buttons.addWidget(self.move_down_button)
        array_layout.addLayout(array_buttons)
        right.addWidget(array_box, stretch=1)

        right.addWidget(self._build_view_box(view))

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
        self.array_list.itemChanged.connect(self._arrays_changed)
        model = self.array_list.model()
        assert model is not None
        model.rowsMoved.connect(self._arrays_changed)
        self._arrays_changed()

    def _build_view_box(self, view: ComparisonView) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("View", self)
        box_layout = QtWidgets.QVBoxLayout(box)
        self.view_group = QtWidgets.QButtonGroup(box)
        self.view_buttons: dict[ComparisonView, QtWidgets.QRadioButton] = {}
        texts = {
            ComparisonView.OVERLAY: (
                "Overlay per neuron",
                "One row per neuron, every array's trace drawn on top of each other",
            ),
            ComparisonView.SIDE_BY_SIDE: (
                "Side by side per neuron",
                "One row per neuron, one column per array",
            ),
            ComparisonView.SEPARATE_WINDOWS: (
                "Separate window per array",
                "One window per array with the neurons stacked; axes stay linked while locked",
            ),
            ComparisonView.CONCATENATED: (
                "Concatenated",
                "The arrays joined end to end, in the list's order, into one timeline per neuron",
            ),
        }
        indent = QtWidgets.QStyle.PixelMetric.PM_IndicatorWidth
        indent_px = (self.style().pixelMetric(indent) if self.style() else 16) + 8
        for mode, (text, tip) in texts.items():
            button = QtWidgets.QRadioButton(text, box)
            button.setToolTip(tip)
            self.view_group.addButton(button)
            self.view_buttons[mode] = button
            box_layout.addWidget(button)
            if mode is ComparisonView.SEPARATE_WINDOWS:
                row = QtWidgets.QHBoxLayout()
                row.setContentsMargins(indent_px, 0, 0, 0)
                self.lock_time_checkbox = QtWidgets.QCheckBox("Lock time axis", box)
                self.lock_time_checkbox.setChecked(True)
                self.lock_value_checkbox = QtWidgets.QCheckBox("Lock value axis", box)
                self.lock_time_checkbox.setToolTip(
                    "Zooming or panning time in one window moves the others"
                )
                self.lock_value_checkbox.setToolTip(
                    "Each neuron's value range follows across the windows"
                )
                row.addWidget(self.lock_time_checkbox)
                row.addWidget(self.lock_value_checkbox)
                row.addStretch()
                box_layout.addLayout(row)
            elif mode is ComparisonView.CONCATENATED:
                form = QtWidgets.QFormLayout()
                form.setContentsMargins(indent_px, 0, 0, 0)
                self.gap_spin = QtWidgets.QDoubleSpinBox(box)
                self.gap_spin.setRange(0.0, 1e7)
                self.gap_spin.setDecimals(1)
                self.gap_spin.setSuffix(" ms")
                self.gap_spin.setToolTip("Empty time left between joined arrays")
                form.addRow("Gap:", self.gap_spin)
                self.add_to_project_checkbox = QtWidgets.QCheckBox(
                    "Also add as new trace array to the project", box
                )
                form.addRow(self.add_to_project_checkbox)
                self.name_edit = QtWidgets.QLineEdit(box)
                self.name_edit.textEdited.connect(self._name_edited_by_user)
                self.name_edit.textChanged.connect(self._update_ok_enabled)
                form.addRow("Name:", self.name_edit)
                self.name_error = QtWidgets.QLabel("", box)
                self.name_error.setWordWrap(True)
                self.name_error.hide()
                form.addRow(self.name_error)
                box_layout.addLayout(form)
        self.view_buttons[view].setChecked(True)
        self.view_group.buttonToggled.connect(self._update_options)
        self.add_to_project_checkbox.toggled.connect(self._update_options)
        return box

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
        problem = ""
        if self.add_to_project:
            problem = group_name_problem(
                self.name_edit.text(), self._existing_group_names
            )
        self.name_error.setText(problem)
        self.name_error.setVisible(bool(problem))
        self.ok_button.setEnabled(
            bool(self.selected_neuron_ids)
            and bool(self.selected_array_names)
            and not problem
        )

    def _dt_of(self, name: str) -> float | None:
        if name not in self._dt_cache:
            self._dt_cache[name] = (
                self._array_dt(name) if self._array_dt is not None else None
            )
        return self._dt_cache[name]

    def _arrays_changed(self, *_args) -> None:
        if not self._name_edited:
            self.name_edit.setText(
                concatenation_group_name(
                    self.selected_array_names, self._existing_group_names
                )
            )
        self._update_options()

    def _name_edited_by_user(self, _text: str) -> None:
        self._name_edited = True

    def _update_options(self, *_args) -> None:
        """Enable each view's options only while that view is chosen."""
        view = self.view
        separate = view is ComparisonView.SEPARATE_WINDOWS
        self.lock_time_checkbox.setEnabled(separate)
        self.lock_value_checkbox.setEnabled(separate)
        concatenated = view is ComparisonView.CONCATENATED
        self.gap_spin.setEnabled(concatenated)
        reason = self._add_to_project_problem()
        self.add_to_project_checkbox.setEnabled(concatenated and not reason)
        self.add_to_project_checkbox.setToolTip(
            reason
            or "Store the joined traces as a new trace array (undoable with Edit → Undo)"
        )
        self.name_edit.setEnabled(self.add_to_project)
        self._update_ok_enabled()

    def _add_to_project_problem(self) -> str:
        """Why the joined arrays cannot be stored in the project ('' if they can)."""
        names = self.selected_array_names
        if not names:
            return "Tick the trace arrays to join first."
        dts = [self._dt_of(name) for name in names]
        if any(dt is None for dt in dts):
            return "The sampling interval of a ticked array is unknown."
        if shared_dt(dt for dt in dts if dt is not None) is None:
            listed = ", ".join(
                f"{label}: {float(dt) * 1e3:.4g} ms"
                for label, dt in zip(array_labels(names), dts, strict=True)
                if dt is not None
            )
            return (
                "Only arrays with the same sampling interval (dt) can be joined "
                f"into one project trace array. Ticked arrays: {listed}."
            )
        return ""

    def _move_current_array(self, step: int) -> None:
        row = self.array_list.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.array_list.count():
            return
        item = self.array_list.takeItem(row)
        self.array_list.insertItem(target, item)
        self.array_list.setCurrentRow(target)
        self._arrays_changed()

    def _apply_filter(self, text: str) -> None:
        """Show neurons whose id matches the filter (an id/range or a substring)."""
        text = text.strip()
        wanted: list[tuple[int, int]] | None
        try:
            wanted = parse_neuron_ranges(text) if text else None
        except ValueError:
            wanted = None
        for item in self._items(self.neuron_list):
            neuron_id = int(item.data(_ID_ROLE))
            if not text:
                hidden = False
            elif wanted is not None:
                hidden = not in_ranges(neuron_id, wanted)
            else:
                hidden = text.lower() not in item.text().lower()
            item.setHidden(hidden)

    @contextmanager
    def _bulk_check(self, list_widget: QtWidgets.QListWidget) -> Iterator[None]:
        """Tick/untick many items with one update afterwards: without it
        every item's ``itemChanged`` rescans the whole list (O(n²))."""
        blocked = list_widget.blockSignals(True)
        try:
            yield
        finally:
            list_widget.blockSignals(blocked)
        if list_widget is self.array_list:
            self._arrays_changed()
        else:
            self._update_ok_enabled()

    def _set_shown_checked(self, checked: bool) -> None:
        with self._bulk_check(self.neuron_list):
            for item in self._items(self.neuron_list):
                if not item.isHidden():
                    item.setCheckState(_check_state(checked))

    def _set_arrays_checked(self, checked: bool) -> None:
        with self._bulk_check(self.array_list):
            for item in self._items(self.array_list):
                item.setCheckState(_check_state(checked))

    def _tick_range(self) -> None:
        try:
            ranges = parse_neuron_ranges(self.range_edit.text())
        except ValueError as error:
            self.range_error.setText(str(error))
            self.range_error.show()
            return
        self.range_error.hide()
        found = set()
        with self._bulk_check(self.neuron_list):
            for item in self._items(self.neuron_list):
                neuron_id = int(item.data(_ID_ROLE))
                if in_ranges(neuron_id, ranges):
                    item.setCheckState(_CHECKED)
                    found.add(neuron_id)
        ignored = range_size(ranges) - len(found)
        if ignored:
            self.range_error.setText(f"{ignored} id(s) not in this array were ignored.")
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
        """Ticked arrays, in the list's (possibly reordered) order."""
        return [
            str(item.data(_ID_ROLE))
            for item in self._items(self.array_list)
            if item.checkState() == _CHECKED
        ]

    @property
    def view(self) -> ComparisonView:
        for mode, button in self.view_buttons.items():
            if button.isChecked():
                return mode
        return ComparisonView.OVERLAY

    @property
    def lock_time(self) -> bool:
        return self.lock_time_checkbox.isChecked()

    @property
    def lock_value(self) -> bool:
        return self.lock_value_checkbox.isChecked()

    @property
    def gap_ms(self) -> float:
        return float(self.gap_spin.value())

    @property
    def add_to_project(self) -> bool:
        """Whether a concatenation is also stored as a new project trace array."""
        return (
            self.view is ComparisonView.CONCATENATED
            and self.add_to_project_checkbox.isEnabled()
            and self.add_to_project_checkbox.isChecked()
        )

    @property
    def new_array_name(self) -> str:
        return self.name_edit.text().strip()
