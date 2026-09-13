from __future__ import annotations

import brian2.numpy_ as np
import pygfx
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
            uniform_size=True,
        )
        self.post_set_data()
