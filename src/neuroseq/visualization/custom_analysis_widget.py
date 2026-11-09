from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from types import MethodType

import brian2.numpy_ as np
import numpy.typing as npt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from neuroseq import theme as theme_module
from neuroseq.analysis_plugins import AnalysisPlugin
from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData


class CustomAnalysisType(Enum):
    Histogram = "mdi6.chart-histogram"
    Histogram2d = "mdi6.data-matrix"
    SpikeCountPerNeuron = "mdi6.chart-bar"
    PearsonCorrelation = "mdi6.chart-bell-curve-cumulative"

    @property
    def label(self) -> str:
        return {
            CustomAnalysisType.Histogram: "Histogram",
            CustomAnalysisType.Histogram2d: "Histogram 2D",
            CustomAnalysisType.SpikeCountPerNeuron: "Spike count per neuron",
            CustomAnalysisType.PearsonCorrelation: "Pearson correlation",
        }[self]

    def __str__(self) -> str:
        return self.label


class CustomAnalysisWindow(QtWidgets.QWidget):
    """A window that contains a CustomAnalysisWidget."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.setWindowTitle("Custom Analysis")
        self.resize(800, 600)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.analysis_widget = CustomAnalysisWidget(self)
        self.layout.addWidget(self.analysis_widget)

    def closeEvent(self, event) -> None:
        self.analysis_widget.clear_data()
        super().closeEvent(event)


class CustomAnalysisWidget(MatplotlibWidget):
    """A custom analysis widget that analyses a slice of data and displays the results using matplotlib."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        # Lets the figure re-flow its margins for a wrapped (multi-line)
        # title or a colorbar instead of letting them overlap the axes (#546).
        self.getFigure().set_layout_engine("constrained")
        self.subplot = self.getFigure().add_subplot(111)
        original_draw_idle = self.canvas._draw_idle

        def _safe_draw_idle(canvas) -> None:
            try:
                original_draw_idle()
            except RuntimeError as exc:
                if "deleted" not in str(exc).lower():
                    raise
            except AttributeError as exc:
                if "screenchanged" not in str(exc).lower():
                    raise

        self.canvas._draw_idle = MethodType(_safe_draw_idle, self.canvas)
        self.traces: npt.NDArray | None = None
        self.spike_times_ms: npt.NDArray | None = None
        self.spike_indices: npt.NDArray | None = None
        self.spike_neuron_indices: npt.NDArray | None = None
        self.neuron_ids: list[int] | None = None
        self.dt_ms: float = 1.0

    def _redraw(self) -> None:
        if self.isVisible() and self.windowHandle() is not None:
            try:
                self.draw()
                return
            except AttributeError as exc:
                if "screenchanged" not in str(exc).lower():
                    raise

        self.canvas.draw_idle()

    def clear_data(self) -> None:
        self.traces = None
        self.spike_times_ms = None
        self.spike_indices = None
        self.spike_neuron_indices = None
        self.neuron_ids = None
        self.subplot.clear()
        self._redraw()

    def apply_theme(self, theme: str) -> None:
        """Recolor the matplotlib figure to match the application theme."""
        face, text = theme_module.mpl_colors(theme)
        figure = self.getFigure()
        figure.set_facecolor(face)
        self.subplot.set_facecolor(face)
        self.subplot.tick_params(colors=text)
        for spine in self.subplot.spines.values():
            spine.set_color(text)
        for axis_label in (self.subplot.xaxis.label, self.subplot.yaxis.label):
            axis_label.set_color(text)
        self.subplot.title.set_color(text)
        self._redraw()

    def _preprocess_data(
        self,
        data: TraceData | SpikeData | SynapseData,
        analysis_label: str,
        neuron_slice: slice | Sequence[int] | None,
        time_slice: slice | None,
    ) -> None:
        self.traces = None
        self.spike_times_ms = None
        self.spike_indices = None
        self.spike_neuron_indices = None
        self.neuron_ids = None
        if isinstance(data, TraceData):
            self.dt_ms = data.dt * 1000.0
        else:
            self.dt_ms = 1.0
        if neuron_slice is None:
            neuron_slice = slice(None, None)
            neuron_str = "all neurons"
        elif isinstance(neuron_slice, slice):
            if neuron_slice.stop == neuron_slice.start + 1:
                neuron_str = f"neuron {neuron_slice.start}"
            else:
                neuron_str = f"neurons {neuron_slice.start}-{neuron_slice.stop - 1}"
        else:
            neuron_ids = list(neuron_slice)
            neuron_str = (
                f"neuron {neuron_ids[0]}"
                if len(neuron_ids) == 1
                else f"{len(neuron_ids)} selected neurons"
            )
        self.neuron_slice = neuron_slice
        if time_slice is None:
            time_slice = slice(None, None)
            time_str = "all time"
        else:
            time_str = f"time {time_slice.start * self.dt_ms:.0f}ms-{time_slice.stop * self.dt_ms:.0f}ms"
        self.time_slice = time_slice
        self.subplot.set_title(
            f"{data.name} {analysis_label}: {neuron_str}, {time_str}"
        )
        if isinstance(data, TraceData):
            # The selection holds neuron ids; trace rows follow data.indices.
            rows = data.rows_for_neurons(neuron_slice)
            self.traces = np.asarray(data.traces)[rows][:, time_slice]
            self.neuron_ids = [int(data.indices[row]) for row in rows]
        elif isinstance(data, SpikeData):
            assert isinstance(neuron_slice, slice)
            all_spike_indices = np.asarray(data.spike_indices)
            if neuron_slice.start is not None and neuron_slice.stop is not None:
                self.spike_neuron_indices = np.arange(
                    neuron_slice.start, neuron_slice.stop
                )
            elif all_spike_indices.size:
                self.spike_neuron_indices = np.arange(int(all_spike_indices.max()) + 1)
            else:
                self.spike_neuron_indices = np.asarray([], dtype=np.int64)
            spike_times_ms = (
                np.asarray(data.spike_times) * 1000.0
            )  # convert to milliseconds
            spike_indices = all_spike_indices
            if time_slice.start is not None and time_slice.stop is not None:
                times_mask = (spike_times_ms >= time_slice.start * self.dt_ms) & (
                    spike_times_ms < time_slice.stop * self.dt_ms
                )
                spike_indices = spike_indices[times_mask]
                spike_times_ms = spike_times_ms[times_mask]
            if neuron_slice.start is not None and neuron_slice.stop is not None:
                neurons_mask = (spike_indices >= neuron_slice.start) & (
                    spike_indices < neuron_slice.stop
                )
                spike_indices = spike_indices[neurons_mask]
                spike_times_ms = spike_times_ms[neurons_mask]
            self.spike_indices = spike_indices
            self.spike_times_ms = spike_times_ms

    def set_data(
        self,
        data: TraceData | SpikeData | SynapseData | None,
        analysis_type: CustomAnalysisType | AnalysisPlugin | None,
        neuron_slice: slice | Sequence[int] | None,
        time_slice: slice | None,
    ) -> None:
        self.subplot.clear()
        if data is None or analysis_type is None:
            return
        self.logger.info(
            f"Plotting {analysis_type} of {type(data)} data for neuron slice {neuron_slice} and time slice {time_slice}"
        )
        analysis_label = (
            analysis_type.label
            if isinstance(analysis_type, CustomAnalysisType)
            else analysis_type.label
        )
        self._preprocess_data(data, analysis_label, neuron_slice, time_slice)

        if isinstance(analysis_type, AnalysisPlugin):
            # Only CustomAnalysisType.PearsonCorrelation ever receives an
            # arbitrary neuron subset (via the right-click "Pearson
            # correlation" dialog) - plugins always get a contiguous slice.
            assert isinstance(neuron_slice, slice) or neuron_slice is None
            try:
                analysis_type.render(self, data, neuron_slice, time_slice)
            except Exception as exc:
                self.logger.exception(
                    f"Analysis plugin {analysis_type.id} failed while plotting: {exc}"
                )
                self.subplot.text(
                    0.5,
                    0.5,
                    "Plugin execution failed. See logs for details.",
                    ha="center",
                    va="center",
                    transform=self.subplot.transAxes,
                )
            self._redraw()
            return

        if analysis_type == CustomAnalysisType.Histogram:
            if isinstance(data, TraceData):
                x = np.asarray(self.traces).flatten()
            elif isinstance(data, SpikeData):
                x = self.spike_indices
            self.subplot.hist(x, bins=30, density=True)
            self.subplot.set_xlabel("Value")
            self.subplot.set_ylabel("Density")
        elif analysis_type == CustomAnalysisType.Histogram2d:
            if isinstance(data, TraceData):
                assert self.traces is not None
                x = np.tile(
                    self.dt_ms * np.arange(self.traces.shape[1]), self.traces.shape[0]
                )
                if time_slice is not None and time_slice.start is not None:
                    x += time_slice.start * self.dt_ms
                y = np.asarray(self.traces).flatten()
                self.subplot.set_ylabel("Values")
            elif isinstance(data, SpikeData):
                x = self.spike_times_ms
                y = self.spike_indices
                self.subplot.set_ylabel("Neurons")
            self.subplot.hist2d(x, y, bins=30, density=True)
            self.subplot.set_xlabel("Time")
        elif analysis_type == CustomAnalysisType.SpikeCountPerNeuron:
            if not isinstance(data, SpikeData):
                return
            assert self.spike_neuron_indices is not None
            self.subplot.set_xlabel("Neuron Index")
            self.subplot.set_ylabel("Spike Count")
            if self.spike_indices is None or self.spike_indices.size == 0:
                self.subplot.text(
                    0.5,
                    0.5,
                    "No spikes in selected region",
                    ha="center",
                    va="center",
                    transform=self.subplot.transAxes,
                )
            else:
                max_neuron_index = (
                    int(np.max(self.spike_neuron_indices))
                    if self.spike_neuron_indices.size
                    else int(np.max(self.spike_indices))
                )
                spike_counts = np.bincount(
                    self.spike_indices.astype(np.int64),
                    minlength=max_neuron_index + 1,
                )
                self.subplot.bar(
                    self.spike_neuron_indices,
                    spike_counts[self.spike_neuron_indices],
                )
        elif analysis_type == CustomAnalysisType.PearsonCorrelation:
            if not isinstance(data, TraceData):
                return
            assert self.traces is not None
            traces = np.asarray(self.traces)
            self.subplot.set_xlabel("Neuron index")
            self.subplot.set_ylabel("Neuron index")
            # The title set by _preprocess_data is one long line that runs
            # past the window edge once the neuron/time description gets
            # non-trivial (#546) - break it onto a second line right after
            # the analysis label, which is always present and never part of
            # the neuron/time description itself.
            self.subplot.set_title(self.subplot.get_title().replace(": ", ":\n", 1))
            if traces.shape[0] == 0 or traces.shape[1] < 2:
                self.subplot.text(
                    0.5,
                    0.5,
                    "Need at least one neuron and two time samples",
                    ha="center",
                    va="center",
                    transform=self.subplot.transAxes,
                )
            else:
                corr = np.asarray(np.corrcoef(traces), dtype=np.float64)
                corr = np.atleast_2d(corr)
                image = self.subplot.imshow(
                    corr,
                    vmin=-1.0,
                    vmax=1.0,
                    cmap="coolwarm",
                    origin="lower",
                )
                self.subplot.figure.colorbar(
                    image,
                    ax=self.subplot,
                    label="Pearson r",
                )
                # Ticks otherwise read as 0, 1, 2, ... regardless of which
                # actual neurons were selected (#546) - relabel the matrix's
                # default 0-based positions with the real neuron ids they
                # correspond to, keeping matplotlib's own "nice" tick
                # spacing rather than labelling every single row/column.
                neuron_ids = self.neuron_ids or list(range(corr.shape[0]))

                def _neuron_id_at(value: float, _pos, _ids=neuron_ids) -> str:
                    index = round(value)
                    return str(_ids[index]) if 0 <= index < len(_ids) else ""

                for axis in (self.subplot.xaxis, self.subplot.yaxis):
                    axis.set_major_locator(MaxNLocator(integer=True))
                    axis.set_major_formatter(FuncFormatter(_neuron_id_at))
        else:
            return
        self._redraw()
