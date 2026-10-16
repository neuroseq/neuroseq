"""Qt-free SysCon2 ``.seq`` export used by the generator dialog, the panel's one-click
export, and :mod:`neuroseq.syscon2_stimulation_host`.

Keeping a single code path here means every way of producing a sequence applies the same
export settings and the same sanity gate.
"""

from __future__ import annotations

import datetime
import pathlib
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

import brian2.numpy_ as np

from neuroseq.export.exporter_syscon2 import SysconExportReport
from neuroseq.export.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.export.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.export.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.export.power_lut import resolve_power_lut
from neuroseq.importers.importer_syscon2 import original_stim_sidecar_path
from neuroseq.importers.importer_syscon2 import write_original_stim_sidecar
from neuroseq.stim import StimArray

if TYPE_CHECKING:
    from neuroseq.export.power_lut import PowerLut
    from neuroseq.importers.importer_cellpose import CellposeImportResult
    from neuroseq.project import Project

TICK_RESOLUTION_US = 500


def default_export_directory() -> pathlib.Path:
    """Where exported scripts go until the user picks a folder (issue #614)."""
    return pathlib.Path.home() / "Documents" / "NeuroSeq" / "SysCon exports"


def default_target_directory(day: datetime.date | None = None) -> str:
    """Folder on the SysCon PC for imported scripts: one per experiment day,
    so scripts are not all collected in "neuroseq" (issue #614)."""
    day = day or datetime.date.today()
    return f"neuroseq\\{day.isoformat()}"


class SysconExportError(RuntimeError):
    """The array cannot be exported as-is (failed sanity check, nothing to export...)."""


@dataclass(frozen=True)
class SysconExportSettings:
    """The user-facing knobs of a HOLO4D ``.seq`` export."""

    config_text: str
    force_binary_intensity: bool = False
    spiralling: bool = False
    ten_spots_per_roi: bool = True
    spot_algorithm: str = "farthest_point"
    laser_wavelength_nm: int = 940
    target_power_mw: float = 1.0
    exact_slm_timing: bool = True


@dataclass(frozen=True)
class SysconExportResult:
    seq_path: pathlib.Path
    overview_png_path: pathlib.Path
    targeted_stim_ids: frozenset[int]
    report: SysconExportReport


def _power_lut(project: Project, wavelength_nm: int) -> PowerLut | None:
    lut, _ = resolve_power_lut(project.power_luts, wavelength_nm)
    return lut


def _export_kwargs(
    project: Project, settings: SysconExportSettings
) -> dict[str, object]:
    return {
        "template_text": settings.config_text,
        "tick_resolution_us": TICK_RESOLUTION_US,
        "active_threshold": 0.0,
        "binary_intensity": 1.0,
        "force_binary_intensity": settings.force_binary_intensity,
        "active_laser_wavelength_nm": settings.laser_wavelength_nm,
        "spiralling": settings.spiralling,
        "target_power_mw": settings.target_power_mw,
        "power_lut": _power_lut(project, settings.laser_wavelength_nm),
        "stimulation_spots_per_roi": 10 if settings.ten_spots_per_roi else 1,
        "stimulation_spot_algorithm": settings.spot_algorithm,
        "exact_slm_timing": settings.exact_slm_timing,
    }


def generate_sequence_text(
    project: Project,
    stim_array: StimArray,
    settings: SysconExportSettings,
    cellpose_data: CellposeImportResult | None = None,
) -> tuple[str, SysconExportReport]:
    """Generate the ``.seq`` text for ``stim_array`` (no sanity gate, no files)."""
    return generate_syscon2_sequence_text(
        stim_array,
        cellpose_data if cellpose_data is not None else project.cellpose_data,
        **_export_kwargs(project, settings),  # type: ignore[arg-type]
    )


def slice_stim_array(
    stim_array: StimArray, start_sample: int, stop_sample: int, name: str | None = None
) -> StimArray:
    """Copy of ``stim_array`` restricted to ``[start_sample, stop_sample)``."""
    if not 0 <= start_sample < stop_sample <= stim_array.values.shape[0]:
        raise SysconExportError(
            f"Invalid sample range [{start_sample}, {stop_sample}) for an array of "
            f"{stim_array.values.shape[0]} samples."
        )
    return replace(
        stim_array,
        name=name or stim_array.name,
        values=np.array(stim_array.values[start_sample:stop_sample], copy=True),
    )


def validate_export_name(name: str) -> None:
    """Require a single filename stem usable on the SysCon Windows machine."""
    if (
        not name
        or name in {".", ".."}
        or name[-1:] in {".", " "}
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
    ):
        raise SysconExportError(
            "Script name must be a valid Windows filename, without a folder or extension."
        )


def export_stim_array(
    project: Project,
    stim_array: StimArray,
    settings: SysconExportSettings,
    output_dir: pathlib.Path,
    name: str,
    *,
    start_sample: int | None = None,
    stop_sample: int | None = None,
    confirm_warnings: Callable[[list[str]], bool] | None = None,
) -> SysconExportResult:
    """Sanity-check and export ``stim_array`` (optionally a sample range) to ``output_dir``.

    Raises :class:`SysconExportError` if the sanity check fails; fixes that need user
    judgement (resampling dt, spacing out pulses, ...) stay in the generator dialog.
    """
    validate_export_name(name)
    if (
        not np.isfinite(float(stim_array.dt))
        or not np.isfinite(settings.target_power_mw)
        or settings.target_power_mw < 0
    ):
        raise SysconExportError(
            "Sample interval and target power must be finite; power cannot be negative."
        )
    if start_sample is not None or stop_sample is not None:
        stim_array = slice_stim_array(
            stim_array,
            0 if start_sample is None else start_sample,
            stim_array.values.shape[0] if stop_sample is None else stop_sample,
        )
    if not np.any(stim_array.values != 0):
        raise SysconExportError("The input array has no active samples to export.")

    sanity = check_syscon2_input_sanity(
        stim_array,
        project.cellpose_data,
        **_export_kwargs(project, settings),  # type: ignore[arg-type]
    )
    if not sanity.is_valid:
        raise SysconExportError(
            "Input sanity check failed - open the export dialog to fix it:\n- "
            + "\n- ".join(sanity.errors)
        )

    if sanity.fix_messages:
        raise SysconExportError(
            "Timing fixes require review. Open the export dialog and run Check Input Sanity:\n- "
            + "\n- ".join(sanity.fix_messages)
        )
    warnings = list(sanity.warnings)
    lut = _power_lut(project, settings.laser_wavelength_nm)
    if settings.target_power_mw > 0 and lut is None:
        warnings.append(
            f"No power LUT is available for {settings.laser_wavelength_nm} nm. "
            f"The requested {settings.target_power_mw:g} mW will NOT be applied; "
            "the sequence will omit calibrated power fields. Load a suitable LUT "
            "or explicitly accept export without calibrated power."
        )
    if (
        lut is not None
        and settings.target_power_mw > 0
        and (
            not lut.tuples
            or any(
                not np.isfinite(percent)
                or not np.isfinite(mw)
                or not 0 <= percent <= 100
                or mw < 0
                for percent, mw in lut.tuples
            )
        )
    ):
        raise SysconExportError(
            "Power LUT must contain finite percentages in 0–100 and nonnegative mW values."
        )
    if warnings and (confirm_warnings is None or not confirm_warnings(warnings)):
        raise SysconExportError(
            "Export warnings require review:\n- " + "\n- ".join(warnings)
        )

    script, report = generate_sequence_text(project, stim_array, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_path = output_dir / f"{name}.seq"
    overview_path = output_dir / f"{name}_overview.png"
    original_path = original_stim_sidecar_path(seq_path)
    # Render everything before replacing any existing output. Publish the sequence last.
    with tempfile.TemporaryDirectory(
        prefix=".syscon-export-", dir=output_dir
    ) as staging:
        staged = pathlib.Path(staging)
        (staged / seq_path.name).write_text(script, encoding="utf-8")
        write_original_stim_sidecar(
            staged / original_path.name, stim_array.values, stim_array.dt, script
        )
        generate_syscon2_timeline_overview_png(script, staged / overview_path.name)
        for destination in (overview_path, original_path, seq_path):
            (staged / destination.name).replace(destination)
    targeted = report.exported_neuron_ids
    return SysconExportResult(seq_path, overview_path, targeted, report)
