from __future__ import annotations

import brian2.numpy_ as np
import pygfx
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.constrained_pan_zoom_controller import ConstrainedPanZoomController
from neuroseq.data import SpikeData
from neuroseq.timeseries_widget import TimeseriesWidget
from neuroseq.utils import neuron_indices_to_colors


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
        from matplotlib.ticker import MaxNLocator

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
        points = np.asarray(self.graphic.data[:]).copy()
        spikes = ax.scatter(
            points[:, 0] + self.graphic.offset[0],
            points[:, 1] + self.graphic.offset[1],
            c=neuron_indices_to_colors(self.data.spike_indices),
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
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
        spike_data = np.stack(
            [
                data.spike_times.astype(np.float32) * np.float32(1000),
                data.spike_indices.astype(np.float32),
            ],
            axis=1,
        )
        self.graphic = self.subplot.add_scatter(
            spike_data,
            colors=neuron_indices_to_colors(data.spike_indices),
            sizes=7,
            size_space="screen",
        )
        self.post_set_data()
