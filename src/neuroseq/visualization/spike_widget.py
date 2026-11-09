from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.data import SpikeData
from neuroseq.utils import neuron_indices_to_colors
from neuroseq.visualization.timeseries_widget import TimeseriesWidget
from neuroseq.visualization_support.constrained_pan_zoom_controller import (
    ConstrainedPanZoomController,
)


class SpikeWidget(TimeseriesWidget):
    """A widget that displays neuron spikes.

    spikes is a 1d array of spiking neuron indices, and timestamps is a 1d array of the corresponding timestamps of these spikes
    """

    def __init__(
        self,
        controller: pygfx.Controller,
        parent: QtWidgets.QWidget | None = None,
        default_controller: ConstrainedPanZoomController | None = None,
        show_axis_link_controls: bool = False,
    ) -> None:
        super().__init__(
            controller,
            parent,
            default_controller=default_controller,
            show_axis_link_controls=show_axis_link_controls,
        )

    def _rebuild_custom_analysis_menu(self) -> None:
        super()._rebuild_custom_analysis_menu()
        self.context_menu.insertSeparator(self.context_menu.actions()[0])
        preview_action = QtGui.QAction("Preview vector graphic…", self)
        preview_action.setEnabled(
            isinstance(self.data, SpikeData) and self.graphic is not None
        )
        preview_action.triggered.connect(self.preview_spikes)
        self.context_menu.insertAction(self.context_menu.actions()[0], preview_action)

    def preview_spikes(self) -> QtWidgets.QDialog | None:
        """Preview the selected region, or the current view when nothing is selected."""
        controller = self.controller
        if (
            not isinstance(self.data, SpikeData)
            or self.graphic is None
            or controller is None
        ):
            return None

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        from matplotlib.figure import Figure

        from neuroseq.visualization_support.preview_ticks import set_neuron_id_ticks

        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"Spike preview: {self.data.name}")
        dialog.resize(1000, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        figure = Figure(layout="tight", facecolor="none")
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(NavigationToolbar2QT(canvas, dialog))
        layout.addWidget(canvas)
        ax = figure.add_subplot(111, facecolor="none")
        points, spike_ids = self._display_spike_points()
        spikes = ax.scatter(
            points[:, 0] + self.graphic.offset[0],
            points[:, 1] + self.graphic.offset[1],
            c=neuron_indices_to_colors(spike_ids),
            marker="|",
            s=(7 * 72 / figure.dpi) ** 2,
            linewidths=72 / figure.dpi,
        )
        spikes.set_gid("spikes")
        state = controller._get_camera_state()
        center_x, center_y = state["position"][:2]
        ax.set_xlim(center_x - state["width"] / 2, center_x + state["width"] / 2)
        ax.set_ylim(center_y - state["height"] / 2, center_y + state["height"] / 2)
        if self.selected_time_slice is not None:
            start, stop, _ = self.selected_time_slice.indices(self.n_timestamps)
            if stop > start:
                ax.set_xlim(start * self.dt_ms, stop * self.dt_ms)
        if self.selected_neuron_slice is not None:
            start, stop, _ = self.selected_neuron_slice.indices(self.n_neurons)
            if stop > start:
                ax.set_ylim(start - 0.5, stop - 0.5)
        set_neuron_id_ticks(ax.yaxis, self._visible_neuron_ids)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_title(self.data.name)
        dialog.show()
        return dialog

    def set_data(self, data: SpikeData | None) -> None:
        self.clear_data()
        if data is None or data.spike_times.shape[0] == 0:
            return
        self.data = data
        n_neurons_with_spikes = int(max(data.spike_indices)) + 1
        controller_neuron_count = (
            int(self.controller.max_height) if self.controller is not None else 0
        )
        # SpikeData only records neurons that actually fired, so the highest
        # spiking index alone would undercount any silent neurons above it.
        # Falling back to the shared controller's neuron limit (kept in sync
        # with the rest of the project via Gui.update_limits()) keeps this
        # panel's row count - and therefore its y-axis tick labels and its
        # alignment with the Traces/Stimuli panels - correct even when some
        # neurons never spike.
        total_neurons = max(n_neurons_with_spikes, controller_neuron_count)
        self.n_neurons = total_neurons
        self._all_neuron_ids = list(range(total_neurons))
        self._rebuild_visible_neuron_mapping()
        self.logger.info(
            f"{data.spike_indices.shape} spikes with {self.n_neurons} neurons"
        )
        # spikes don't have a dt, but we need one when selecting time slices,
        # so we choose dt=1ms so time slices have a resolution of 1ms
        self.dt_ms = 1.0
        self.n_timestamps = int(1000.0 * max(data.spike_times) / self.dt_ms) + 1
        self._add_spike_graphic()
        self.post_set_data()

    def _display_spike_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Spikes of the visible neurons as (time ms, display row) points,
        plus each point's real neuron id. Hidden neurons' spikes are dropped
        and the rest move to their compacted rows."""
        assert isinstance(self.data, SpikeData)
        ids = self.data.spike_indices.astype(np.int64)
        row_of_id = np.full(int(ids.max()) + 1 if ids.size else 0, -1, np.int64)
        for neuron_id, row in self._neuron_id_to_display_row.items():
            if neuron_id < row_of_id.shape[0]:
                row_of_id[neuron_id] = row
        rows = row_of_id[ids] if ids.size else ids
        keep = rows >= 0
        points = np.stack(
            [
                self.data.spike_times[keep].astype(np.float32) * np.float32(1000),
                rows[keep].astype(np.float32),
            ],
            axis=1,
        )
        return points, ids[keep]

    def _add_spike_graphic(self) -> None:
        points, spike_ids = self._display_spike_points()
        visible = points.shape[0] > 0
        if not visible:
            # A scatter needs at least one point; keep a hidden placeholder.
            points = np.zeros((1, 2), dtype=np.float32)
            spike_ids = np.zeros(1, dtype=np.int64)
        self.graphic = self.subplot.add_scatter(
            points,
            colors=neuron_indices_to_colors(spike_ids),
            sizes=7,
            size_space="screen",
        )
        self.graphic.visible = visible

    def _apply_visible_neuron_layout(self) -> None:
        # Scatter buffers are fixed-size, so rebuild with the visible spikes.
        if self.graphic is None or not isinstance(self.data, SpikeData):
            return
        self.subplot.remove_graphic(self.graphic)
        self._add_spike_graphic()
