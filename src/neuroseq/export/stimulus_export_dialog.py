"""UI entry point for stimulus export; keep format-specific workflows out of Gui."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from collections.abc import Iterable
from typing import Literal

from qtpy import QtWidgets
from qtpy.compat import getsavefilename

from neuroseq.export.exporter_igor2 import export_igor2_csv
from neuroseq.export.exporter_igor2 import export_igor2_ibw
from neuroseq.export.exporter_igor2 import export_igor2_pxp
from neuroseq.export.igor2_pxp_stim_export_dialog import Igor2PxpStimExportDialog
from neuroseq.export.igor2_single_wave_export_dialog import Igor2SingleWaveExportDialog
from neuroseq.export.stimulation_generator_dialog import StimulationGeneratorDialog
from neuroseq.export.syscon2_export_service import SysconExportSettings
from neuroseq.project import Project

logger = logging.getLogger(__name__)


def show_stimulus_export_dialog(
    project: Project,
    export_format: Literal["igor-single", "igor-pxp", "syscon2"],
    *,
    parent: QtWidgets.QWidget | None = None,
    current_stim_name: str | None = None,
    reserved_names: Callable[[], Iterable[str]] | None = None,
    on_generated: Callable[[str, str, object, SysconExportSettings], None]
    | None = None,
) -> None:
    """Run the selected stimulus export workflow, including file writing and feedback."""
    if export_format == "igor-single":
        _export_single_wave(project, current_stim_name, parent)
    elif export_format == "igor-pxp":
        _export_pxp(project, parent)
    elif export_format == "syscon2":
        dialog = StimulationGeneratorDialog(
            project, parent=parent, current_stim_name=current_stim_name
        )
        if reserved_names is not None:
            dialog.reserved_names = reserved_names
        if on_generated is not None:
            dialog.sequence_generated.connect(
                lambda seq, png, targets: on_generated(
                    seq, png, targets, dialog.export_settings()
                )
            )
        dialog.exec()
    else:
        raise ValueError(f"Unknown stimulus export format: {export_format}")


def _export_single_wave(
    project: Project, current_stim_name: str | None, parent: QtWidgets.QWidget | None
) -> None:
    if not project.stim_arrays:
        QtWidgets.QMessageBox.information(
            parent,
            "No input entities",
            "No input data entities available. Create one before exporting.",
        )
        return

    stim_name = current_stim_name
    if not stim_name or stim_name not in project.stim_arrays:
        QtWidgets.QMessageBox.critical(
            parent,
            "Export error",
            "No input array is currently selected for export.",
        )
        return

    stim_array = project.stim_arrays[stim_name]
    if int(stim_array.values.shape[1]) <= 0:
        QtWidgets.QMessageBox.critical(
            parent,
            "Export error",
            "The selected input array has no neurons to export.",
        )
        return

    dialog = Igor2SingleWaveExportDialog(stim_name, stim_array, parent=parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    suggested_filename = f"{dialog.experiment_name}.ibw"
    filename, _ = getsavefilename(
        parent,
        "Export single wave to Igor2",
        suggested_filename,
        "Igor Binary Wave (*.ibw);;CSV file (*.csv)",
    )
    if filename == "":
        return

    output_path = pathlib.Path(filename)
    suffix = output_path.suffix.lower()
    if suffix not in {".ibw", ".csv"}:
        output_path = output_path.with_suffix(".ibw")
        suffix = ".ibw"

    try:
        if suffix == ".ibw":
            export_report = export_igor2_ibw(
                output_path,
                stim_array,
                wave_name=dialog.experiment_name,
                neuron_index=dialog.selected_neuron_index,
                unit_override=dialog.unit_override,
                scaling_factor=dialog.scaling_factor,
                as_sweeps=dialog.as_sweeps,
            )
            if dialog.as_sweeps and export_report.n_sweeps == 1:
                logger.info(
                    "Sweep export requested for '%s' but no period was detected; "
                    "exported as single sweep.",
                    stim_array.name,
                )
            fmt_label = f"Igor Binary Wave '{export_report.experiment_name}'"
        else:
            if dialog.as_sweeps:
                logger.info(
                    "Sweep export requested for '%s' but CSV was selected; "
                    "exported as single sweep.",
                    stim_array.name,
                )
            export_report = export_igor2_csv(
                output_path,
                stim_array,
                experiment_name=dialog.experiment_name,
                neuron_index=dialog.selected_neuron_index,
                scaling_factor=dialog.scaling_factor,
            )
            fmt_label = f"CSV experiment '{export_report.experiment_name}'"
    except Exception as e:
        logger.warning(
            "Failed to export Igor2 single wave %s: %s",
            output_path,
            e,
        )
        QtWidgets.QMessageBox.critical(
            parent,
            "Export error",
            f"Could not export Igor2 single wave to '{output_path}': {e}",
        )
        return

    if export_report.n_sweeps > 1:
        sweep_note = (
            f" Split into {export_report.n_sweeps} sweeps × "
            f"{export_report.sample_count // export_report.n_sweeps} samples."
        )
    else:
        sweep_note = ""
    QtWidgets.QMessageBox.information(
        parent,
        "Export Successful",
        f"Exported {fmt_label} for "
        f"input '{stim_name}', neuron {export_report.neuron_index}, "
        f"{export_report.sample_count} samples.{sweep_note}",
    )


def _export_pxp(project: Project, parent: QtWidgets.QWidget | None) -> None:
    if not project.stim_arrays:
        QtWidgets.QMessageBox.information(
            parent,
            "No input entities",
            "No input data entities available. Create one before exporting.",
        )
        return

    dialog = Igor2PxpStimExportDialog(project, parent=parent)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    selected = dialog.selected_array_names
    if not selected:
        QtWidgets.QMessageBox.warning(
            parent,
            "Nothing selected",
            "No input arrays were selected for export.",
        )
        return

    filename, _ = getsavefilename(
        parent,
        "Export Igor2 Packed Experiment",
        "stim_arrays.pxp",
        "Igor Packed Experiment (*.pxp)",
    )
    if filename == "":
        return

    output_path = pathlib.Path(filename)
    if output_path.suffix.lower() != ".pxp":
        output_path = output_path.with_suffix(".pxp")

    try:
        export_report = export_igor2_pxp(
            output_path,
            project.stim_arrays,
            array_names=selected,
            unit_override=dialog.unit_override,
            scaling_factor=dialog.scaling_factor,
        )
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            parent,
            "Export error",
            f"PXP export failed:\n{e}",
        )
        return

    QtWidgets.QMessageBox.information(
        parent,
        "Export complete",
        f"Exported {len(export_report.wave_names)} wave(s) to:\n{export_report.path}",
    )
