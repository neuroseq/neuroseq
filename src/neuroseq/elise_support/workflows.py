"""ELiSe replay analysis and protocol result workflows.

The main window owns worker threads and action wiring; this module handles
model-specific inputs, metrics, CSV output, and analysis presentation.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import brian2.numpy_ as np
from qtpy import QtWidgets
from qtpy.compat import getsavefilename

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.data import TraceData
from neuroseq.elise_support.plugin_keys import CURVES_ATTR
from neuroseq.elise_support.plugin_keys import LEARNING_CURVES_PLUGIN_ID
from neuroseq.elise_support.plugin_keys import TARGET_ATTR
from neuroseq.elise_support.plugin_keys import TARGET_DT_ATTR
from neuroseq.elise_support.plugin_keys import TARGET_MSE_PLUGIN_ID
from neuroseq.project import Project
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWindow

if TYPE_CHECKING:
    from neuroseq.project_workspace import ProjectWorkspace


def replay_inputs(
    project: Project,
) -> tuple[TraceData, np.ndarray, float] | None:
    """Return (firing_rate_r, opsin_exc_values, opsin_dt_s) if available."""
    if project.external_simulate_path != "neuroseq:elise":
        return None
    trace = None
    for experiment in project.experiments.values():
        for datum in experiment.data:
            if isinstance(datum, TraceData) and datum.name == "firing_rate_r":
                trace = datum
    if trace is None or np.asarray(trace.traces).size == 0:
        return None
    stim = project.stim_arrays.get("opsin_exc")
    if stim is None:
        return None
    return trace, np.asarray(stim.values), float(stim.dt)


def analyse_replay(window: ProjectWorkspace) -> None:
    from neuroseq.elise_support.replay_metrics import compute_performance_against_target
    from neuroseq.elise_support.replay_metrics import save_replay_metrics_csv
    from neuroseq.example_models.elise import build_target_rate_pattern

    inputs = window._elise_replay_inputs()
    if inputs is None:
        QtWidgets.QMessageBox.information(
            window,
            "ELiSe replay analysis",
            "This action requires an ELiSe project with a completed "
            "simulation (needs the 'firing_rate_r' trace and an "
            "'opsin_exc' input).",
        )
        return
    trace, opsin_exc, stim_dt_s = inputs
    params_dict = window.project.external_simulate_params or {}

    def _param(key: str, default: float) -> float:
        entry = params_dict.get(key)
        if isinstance(entry, dict) and "value" in entry:
            try:
                return float(entry["value"])
            except (TypeError, ValueError):
                return default
        return default

    target_rate, cycle_stim_samples = build_target_rate_pattern(
        opsin_exc,
        E_l=_param("E_l", -70.0),
        a=_param("a", 0.3),
        b=_param("b", -58.0),
    )

    trace_dt_s = float(trace.dt)
    if trace_dt_s <= 0:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            "Firing-rate trace has non-positive dt; cannot align target.",
        )
        return
    upsample = int(round(stim_dt_s / trace_dt_s))
    if upsample < 1 or abs(stim_dt_s - upsample * trace_dt_s) > 1e-9:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            f"Input dt ({stim_dt_s * 1000:.4g} ms) is not an integer "
            f"multiple of trace dt ({trace_dt_s * 1000:.4g} ms); "
            "cannot align target.",
        )
        return
    target_rate_resampled = np.repeat(target_rate, upsample, axis=1)
    num_vis = target_rate_resampled.shape[0]

    n_total_neurons = int(np.asarray(trace.traces).shape[0])
    if num_vis > n_total_neurons:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            f"Opsin input has {num_vis} pitches but the trace only "
            f"covers {n_total_neurons} neurons.",
        )
        return

    # Ask the user where to save the per-cycle CSV before running.
    default_stem = "elise_replay_mse"
    if window.project.filepath is not None:
        default_stem = pathlib.Path(window.project.filepath).stem + "_replay_mse"
    csv_filename, _ = getsavefilename(
        window,
        "Save replay MSE analysis as CSV",
        default_stem + ".csv",
        "CSV files (*.csv)",
    )
    if not csv_filename:
        return
    csv_path = pathlib.Path(csv_filename)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")

    output_rates = np.asarray(trace.traces)[:num_vis, :].astype(np.float64)
    try:
        mses, corrs, shifts = compute_performance_against_target(
            output_rates, target_rate_resampled
        )
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            f"Cannot compute replay metrics: {exc}",
        )
        return

    try:
        save_replay_metrics_csv(
            csv_path,
            mses,
            corrs,
            shifts,
            cycle_dt_ms=trace_dt_s * 1000.0,
            cycle_length_samples=int(target_rate_resampled.shape[1]),
            n_output_neurons=num_vis,
            source=str(window.project.filepath) if window.project.filepath else None,
        )
    except OSError as exc:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            f"Could not write CSV to {csv_path}: {exc}",
        )
        return

    plugin = next(
        (p for p in discover_analysis_plugins() if p.id == TARGET_MSE_PLUGIN_ID),
        None,
    )
    if plugin is None:
        QtWidgets.QMessageBox.warning(
            window,
            "ELiSe replay analysis",
            "Could not find the ELiSe target MSE analysis plugin.",
        )
        return
    analysis_window = CustomAnalysisWindow(parent=window)
    analysis_widget = analysis_window.analysis_widget
    setattr(analysis_widget, TARGET_ATTR, target_rate_resampled)
    setattr(analysis_widget, TARGET_DT_ATTR, trace_dt_s)

    n_samples = int(np.asarray(trace.traces).shape[1])
    analysis_widget.set_data(
        trace,
        plugin,
        slice(0, num_vis),
        slice(0, n_samples),
    )
    analysis_window.setWindowTitle(
        window._popup_title(f"ELiSe replay MSE — saved to {csv_path.name}")
    )
    analysis_window.show()


def protocol_inputs(
    project: Project,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (opsin_exc, opsin_inh, stim_dt_s) if this is an ELiSe project."""
    if project.external_simulate_path != "neuroseq:elise":
        return None
    exc = project.stim_arrays.get("opsin_exc")
    inh = project.stim_arrays.get("opsin_inh")
    if exc is None or inh is None:
        return None
    exc_values = np.asarray(exc.values)
    inh_values = np.asarray(inh.values)
    if exc_values.size == 0 or exc_values.shape != inh_values.shape:
        return None
    return exc_values, inh_values, float(exc.dt)


def update_stim_arrays(window: ProjectWorkspace, result: dict) -> None:
    """Replace opsin_exc/opsin_inh with the protocol's full-timeline stims.

    The teaching pattern is present during training cycles and blank during
    teacher-off validation/replay, so the stim array reflects the true
    timeline of what was simulated and lines up with the imported traces.
    """
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.stimulation.stim import StimType

    exc_full = result.get("stim_exc_full")
    inh_full = result.get("stim_inh_full")
    if exc_full is None or inh_full is None:
        return
    dt = float(result.get("stim_dt_s") or 0.0)
    if dt <= 0:
        return
    defaults = {
        "opsin_exc": StimType.Excitation,
        "opsin_inh": StimType.Inhibition,
    }
    for name, values in (("opsin_exc", exc_full), ("opsin_inh", inh_full)):
        existing = window.project.stim_arrays.get(name)
        stim_type = existing.stim_type if existing is not None else defaults[name]
        units = existing.units if existing is not None else "mm**-2 / second"
        window.project.add_stim_array(
            StimArray(
                name=name,
                stim_type=stim_type,
                values=np.asarray(values),
                units=units,
                dt=dt,
            )
        )
    window.stim_editor.set_project(window.project)


def finish_protocol(window: ProjectWorkspace, result: dict) -> None:
    from neuroseq.elise_support.replay_metrics import save_learning_curves_csv

    # Rebuild the opsin stim arrays to the full protocol timeline (teaching
    # pattern during training, blank during teacher-off validation/replay)
    # so the stim array shows the true timeline of what was simulated, then
    # import the aligned activity Experiment.
    window._update_elise_stim_arrays(result)
    experiment = result.get("experiment")
    if experiment is not None:
        window._append_experimental_data(experiment)

    default_stem = "elise_learning_curves"
    if window.project.filepath is not None:
        default_stem = pathlib.Path(window.project.filepath).stem + "_learning_curves"
    csv_filename, _ = getsavefilename(
        window,
        "Save ELiSe learning curves as CSV",
        default_stem + ".csv",
        "CSV files (*.csv)",
    )
    csv_path: pathlib.Path | None = None
    if csv_filename:
        csv_path = pathlib.Path(csv_filename)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        try:
            save_learning_curves_csv(
                csv_path,
                result,
                source=(
                    str(window.project.filepath) if window.project.filepath else None
                ),
            )
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                window,
                "Run ELiSe protocol",
                f"Could not write CSV to {csv_path}: {exc}",
            )
            csv_path = None

    num_vis = int(result["num_vis"])
    final_rate = result.get("final_replay_rate")
    if final_rate is None:
        # Nothing to plot the curves against (e.g. the protocol stopped
        # before its final replay); the data and CSV are already saved.
        return
    plugin = next(
        (p for p in discover_analysis_plugins() if p.id == LEARNING_CURVES_PLUGIN_ID),
        None,
    )
    if plugin is None:
        QtWidgets.QMessageBox.warning(
            window,
            "Run ELiSe protocol",
            "Could not find the ELiSe learning-curves analysis plugin.",
        )
        return
    analysis_window = CustomAnalysisWindow(parent=window)
    analysis_widget = analysis_window.analysis_widget
    setattr(analysis_widget, CURVES_ATTR, result)
    # The plugin reads the curves off the widget; set_data still needs a
    # data object, so hand it the final replay trace.
    final_arr = np.asarray(final_rate, dtype=np.float64)
    data = TraceData(
        name="firing_rate_r (final replay)",
        traces=final_arr,
        indices=np.arange(final_arr.shape[0], dtype=np.uint32),
        units="1",
        dt=float(result["trace_dt_s"]),
    )
    analysis_widget.set_data(
        data, plugin, slice(0, num_vis), slice(0, final_arr.shape[1])
    )
    title = "ELiSe learning curves"
    if csv_path is not None:
        title += f" — saved to {csv_path.name}"
    analysis_window.setWindowTitle(window._popup_title(title))
    analysis_window.show()
