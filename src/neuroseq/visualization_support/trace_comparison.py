"""Qt-free data gathering for comparing traces across trace arrays.

A project often holds several trace arrays recorded under the same
stimulation (repeats 1..N). ``build_trace_comparison`` collects, for each
chosen neuron, that neuron's trace from every chosen array together with the
array's own time axis (arrays may differ in ``dt`` and length) and a colour
per array, and notes which neurons an array does not contain. The comparison
window in ``neuroseq.visualization.trace_comparison_window`` only draws what
this module returns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

import numpy as np
from numpy.typing import NDArray

from neuroseq.data import TraceData

# Categorical colours (matplotlib's "tab10"), one per compared array, cycled
# when more arrays are compared than there are colours.
ARRAY_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_RANGE_PART = re.compile(r"^\s*(\d+)\s*(?:(?:-|\.\.|:)\s*(\d+)\s*)?$")


def parse_neuron_ids(text: str) -> list[int]:
    """Parse ``"0-3, 7, 10..12"`` into sorted unique neuron ids.

    Parts are separated by commas, semicolons or whitespace; a range ``a-b``
    (or ``a..b``, ``a:b``) includes both ends and may be given backwards.
    Raises ``ValueError`` naming the first part that is not an id or range.
    """
    ids: set[int] = set()
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        match = _RANGE_PART.match(part)
        if match is None:
            raise ValueError(f"'{part}' is not a neuron id or range")
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) is not None else start
        if stop < start:
            start, stop = stop, start
        ids.update(range(start, stop + 1))
    return sorted(ids)


def array_color(position: int) -> str:
    """Colour of the ``position``-th compared array."""
    return ARRAY_COLORS[position % len(ARRAY_COLORS)]


def array_labels(names: Sequence[str]) -> list[str]:
    """Distinct, short legend labels for the project data ``names``.

    When every name shares the same data name after ``" :: "`` (typical for
    repeats: ``"Repeat 1 :: dF/F"``, ``"Repeat 2 :: dF/F"``) only the
    experiment part differs, so that part alone labels each array. Otherwise
    the full names are used.
    """
    parts = [name.partition(" :: ") for name in names]
    if (
        len(names) > 1
        and all(sep for _, sep, _ in parts)
        and len({data for _, _, data in parts}) == 1
        and len({group for group, _, _ in parts}) == len(names)
        and not any(group == "Simulation" for group, _, _ in parts)
    ):
        return [group for group, _, _ in parts]
    return list(names)


@dataclass(frozen=True)
class ComparisonSeries:
    """One neuron's trace from one array, ready to plot."""

    array_name: str
    label: str
    color: str
    time_ms: NDArray[np.float64]
    values: NDArray[np.float32]
    units: str


@dataclass(frozen=True)
class ComparisonRow:
    """Every compared array's trace of one neuron (arrays lacking it skipped)."""

    neuron_id: int
    series: list[ComparisonSeries]


@dataclass
class TraceComparison:
    """Rows to plot plus the arrays' legend entries and missing neurons."""

    rows: list[ComparisonRow]
    # (label, colour) per compared array, in the order the arrays were given.
    legend: list[tuple[str, str]]
    # Array label -> neuron ids that array has no trace for.
    missing: dict[str, list[int]] = field(default_factory=dict)

    @property
    def units(self) -> str:
        """The traces' unit when all arrays agree, else an empty string."""
        units = {series.units for row in self.rows for series in row.series}
        return units.pop() if len(units) == 1 else ""

    @property
    def max_time_ms(self) -> float:
        """End of the longest compared trace (0 when there is nothing to plot)."""
        ends = [
            float(series.time_ms[-1])
            for row in self.rows
            for series in row.series
            if series.time_ms.size
        ]
        return max(ends, default=0.0)

    def missing_note(self) -> str:
        """Human-readable note about neurons skipped per array ('' if none)."""
        parts = [
            f"{label}: {format_neuron_ids(ids)}"
            for label, ids in self.missing.items()
            if ids
        ]
        if not parts:
            return ""
        return "Not in every array (skipped there): " + "; ".join(parts)


def format_neuron_ids(ids: Iterable[int]) -> str:
    """Compact ``"0-3, 7"`` form of neuron ids (inverse of parse_neuron_ids)."""
    ordered = sorted(set(int(i) for i in ids))
    chunks: list[str] = []
    start = previous = None
    for neuron_id in ordered:
        if previous is not None and neuron_id == previous + 1:
            previous = neuron_id
            continue
        if start is not None:
            chunks.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = neuron_id
    if start is not None:
        chunks.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(chunks)


def time_axis_ms(data: TraceData) -> NDArray[np.float64]:
    """Sample times of ``data`` in milliseconds, on its own ``dt``."""
    return np.arange(data.traces.shape[1], dtype=np.float64) * float(data.dt) * 1e3


def build_trace_comparison(
    arrays: Sequence[tuple[str, TraceData]], neuron_ids: Iterable[int]
) -> TraceComparison:
    """Gather each neuron's trace from every array in ``arrays``.

    ``arrays`` pairs a project data name with its ``TraceData``. Rows follow
    the order of ``neuron_ids`` (duplicates dropped); a row is kept when at
    least one array holds that neuron. Each array is plotted against its own
    time axis, so arrays with different ``dt`` or length line up in time.
    """
    names = [name for name, _ in arrays]
    labels = array_labels(names)
    colors = [array_color(position) for position in range(len(arrays))]
    row_by_id_per_array = [
        {int(neuron_id): row for row, neuron_id in enumerate(data.indices)}
        for _, data in arrays
    ]
    times = [time_axis_ms(data) for _, data in arrays]
    missing: dict[str, list[int]] = {label: [] for label in labels}
    rows: list[ComparisonRow] = []
    for neuron_id in dict.fromkeys(int(n) for n in neuron_ids):
        series: list[ComparisonSeries] = []
        for position, (name, data) in enumerate(arrays):
            row = row_by_id_per_array[position].get(neuron_id)
            if row is None:
                missing[labels[position]].append(neuron_id)
                continue
            series.append(
                ComparisonSeries(
                    array_name=name,
                    label=labels[position],
                    color=colors[position],
                    time_ms=times[position],
                    values=np.asarray(data.traces[row]),
                    units=data.units,
                )
            )
        if series:
            rows.append(ComparisonRow(neuron_id, series))
    return TraceComparison(
        rows=rows,
        legend=list(zip(labels, colors, strict=True)),
        missing={label: ids for label, ids in missing.items() if ids},
    )
