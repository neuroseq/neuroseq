"""Qt-free SysCon2 ``.seq`` export used by the generator dialog, the panel's one-click
export, and :mod:`neuroseq.syscon2_stimulation_host`.

Keeping a single code path here means every way of producing a sequence applies the same
export settings and the same sanity gate.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

import brian2.numpy_ as np

from neuroseq.exporter_syscon2 import SysconExportReport
from neuroseq.exporter_syscon2 import check_syscon2_input_sanity
from neuroseq.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.exporter_syscon2 import generate_syscon2_timeline_overview_png
from neuroseq.power_lut import resolve_power_lut
from neuroseq.stim import StimArray

if TYPE_CHECKING:
    from neuroseq.importer_cellpose import CellposeImportResult
    from neuroseq.power_lut import PowerLut
    from neuroseq.project import Project

TICK_RESOLUTION_US = 500


class SysconExportError(RuntimeError):
    """The array cannot be exported as-is (failed sanity check, nothing to export...)."""


@dataclass(frozen=True)
class SysconExportSettings:
    """The user-facing knobs of a HOLO4D ``.seq`` export."""

    config_text: str
    force_binary_intensity: bool = True
    spiralling: bool = True
    ten_spots_per_roi: bool = False
    spot_algorithm: str = "farthest_point"
    laser_wavelength_nm: int = 940
    target_power_mw: float = 92.0


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


def export_stim_array(
    project: Project,
    stim_array: StimArray,
    settings: SysconExportSettings,
    output_dir: pathlib.Path,
    name: str,
    *,
    start_sample: int | None = None,
    stop_sample: int | None = None,
) -> SysconExportResult:
    """Sanity-check and export ``stim_array`` (optionally a sample range) to ``output_dir``.

    Raises :class:`SysconExportError` if the sanity check fails; fixes that need user
    judgement (resampling dt, spacing out pulses, ...) stay in the generator dialog.
    """
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

    script, report = generate_sequence_text(project, stim_array, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_path = output_dir / f"{name}.seq"
    overview_path = output_dir / f"{name}_overview.png"
    debug_csv_path = output_dir / f"{name}_export_debug.csv"
    seq_path.write_text(script, encoding="utf-8")
    debug_csv_path.write_text(report.debug_csv_text, encoding="utf-8")
    generate_syscon2_timeline_overview_png(script, overview_path)
    targeted = frozenset(
        int(i) for i in np.flatnonzero(np.any(stim_array.values != 0, axis=0))
    )
    return SysconExportResult(seq_path, overview_path, targeted, report)
