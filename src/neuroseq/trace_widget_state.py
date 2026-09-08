from __future__ import annotations

from qtpy import QtCore

from neuroseq.data import TraceData


class TraceWidgetState(QtCore.QObject):
    """Canonical, shareable state for a trace panel.

    Multiple `TraceWidget` views can be bound to the same `TraceWidgetState`
    instance (see `TraceWidget.__init__`'s `state` parameter) and stay in
    sync: a mutation via any one view (or via `gui.py`) updates this object
    and emits a signal, which every connected view uses to refresh its own
    rendering. No widget/rendering logic lives here - only data and the
    operations that change it.
    """

    data_changed = QtCore.Signal()
    greyed_changed = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.data: TraceData | None = None
        self.greyed_neuron_ids: set[int] = set()

    def set_data(self, data: TraceData | None) -> None:
        self.data = data
        self.greyed_neuron_ids = set()
        self.data_changed.emit()

    def set_neuron_greyed(self, neuron_index: int, greyed: bool) -> None:
        """Hide (or restore) one neuron. The underlying data is never touched."""
        if greyed:
            if neuron_index in self.greyed_neuron_ids:
                return
            self.greyed_neuron_ids.add(neuron_index)
        else:
            if neuron_index not in self.greyed_neuron_ids:
                return
            self.greyed_neuron_ids.discard(neuron_index)
        self.greyed_changed.emit()

    def clear_greyed_neurons(self) -> None:
        if not self.greyed_neuron_ids:
            return
        self.greyed_neuron_ids = set()
        self.greyed_changed.emit()
