"""Qt-free data gathering for comparing traces across trace arrays.

A project often holds several trace arrays recorded under the same
stimulation (repeats 1..N). ``build_trace_comparison`` collects, for each
chosen neuron, that neuron's trace from every chosen array together with the
array's own time axis (arrays may differ in ``dt`` and length) and a colour
per array, and notes which neurons an array does not contain.
``build_trace_concatenation`` instead joins the arrays end to end into one
timeline per neuron (and can turn that into a new ``TraceData``), and
``AxisLinkGroup`` keeps the time/value ranges of several per-array windows in
step. The windows in ``neuroseq.visualization`` only draw what this module
returns.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from collections.abc import Hashable
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

# Separator between a data group (experiment) and a data set in project data
# names ("Repeat 1 :: dF/F"); a group name must not contain it.
NAME_SEPARATOR = " :: "


class ComparisonView(enum.Enum):
    """How the "Compare traces" dialog shows the chosen arrays."""

    OVERLAY = "overlay"
    SIDE_BY_SIDE = "side_by_side"
    SEPARATE_WINDOWS = "separate_windows"
    CONCATENATED = "concatenated"


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
        return missing_note(self.missing, "skipped there")


def missing_note(missing: dict[str, list[int]], what_happens: str) -> str:
    """``"Not in every array (<what_happens>): A: 1-3; B: 7"`` ('' if none)."""
    parts = [
        f"{label}: {format_neuron_ids(ids)}" for label, ids in missing.items() if ids
    ]
    if not parts:
        return ""
    return f"Not in every array ({what_happens}): " + "; ".join(parts)


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


# -- Concatenation ---------------------------------------------------------


@dataclass(frozen=True)
class ConcatSegment:
    """One array's stretch of the concatenated timeline."""

    array_name: str
    label: str
    color: str
    start_ms: float
    dt: float
    n_samples: int

    @property
    def duration_ms(self) -> float:
        return self.n_samples * self.dt * 1e3

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms

    @property
    def time_ms(self) -> NDArray[np.float64]:
        """Sample times of this segment on the joint timeline."""
        return self.start_ms + np.arange(self.n_samples, dtype=np.float64) * (
            self.dt * 1e3
        )


@dataclass(frozen=True)
class ConcatRow:
    """One neuron's values per segment (all NaN where an array lacks it)."""

    neuron_id: int
    values: list[NDArray[np.float32]]


@dataclass
class TraceConcatenation:
    """Arrays joined end to end: segments, per-neuron rows and missing ids."""

    segments: list[ConcatSegment]
    rows: list[ConcatRow]
    gap_ms: float
    units: str
    missing: dict[str, list[int]] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.segments[-1].end_ms if self.segments else 0.0

    @property
    def boundaries_ms(self) -> list[float]:
        """Where one segment hands over to the next (middle of any gap)."""
        return [
            (previous.end_ms + following.start_ms) / 2
            for previous, following in zip(
                self.segments, self.segments[1:], strict=False
            )
        ]

    @property
    def same_dt(self) -> bool:
        return shared_dt([segment.dt for segment in self.segments]) is not None

    def missing_note(self) -> str:
        return missing_note(self.missing, "left empty there")

    def to_trace_data(self, name: str) -> TraceData:
        """The joined timeline as one ``TraceData`` (needs one shared ``dt``).

        Rows are the concatenated neurons sorted by id; a gap becomes NaN
        samples, as does the segment of an array lacking a neuron.
        """
        dt = shared_dt([segment.dt for segment in self.segments])
        if dt is None:
            raise ValueError(
                "Only arrays with the same sampling interval can be joined into one array"
            )
        gap_samples = int(round(self.gap_ms / (dt * 1e3)))
        rows = sorted(self.rows, key=lambda row: row.neuron_id)
        blocks: list[NDArray[np.float32]] = []
        for position, segment in enumerate(self.segments):
            if position and gap_samples:
                blocks.append(np.full((len(rows), gap_samples), np.nan, np.float32))
            if rows:
                blocks.append(np.stack([row.values[position] for row in rows]))
            else:
                blocks.append(np.zeros((0, segment.n_samples), np.float32))
        traces = np.concatenate(blocks, axis=1)
        indices = np.array([row.neuron_id for row in rows], dtype=np.uint32)
        return TraceData(name, traces, indices, self.units, dt)


def shared_dt(dts: Iterable[float]) -> float | None:
    """The common sampling interval of ``dts``, or None if they differ (or none)."""
    values = [float(dt) for dt in dts]
    if not values:
        return None
    first = values[0]
    if all(np.isclose(dt, first, rtol=1e-9, atol=0.0) for dt in values):
        return first
    return None


def build_trace_concatenation(
    arrays: Sequence[tuple[str, TraceData]],
    neuron_ids: Iterable[int],
    gap_ms: float = 0.0,
) -> TraceConcatenation:
    """Join the chosen neurons' traces of ``arrays`` end to end, in order.

    Each array keeps its own ``dt``: a segment starts where the previous one
    ends (``n_samples * dt``) plus ``gap_ms``. When every array shares one
    ``dt`` the gap is rounded to whole samples, so the plotted timeline
    matches :meth:`TraceConcatenation.to_trace_data`. A neuron missing from
    an array is NaN-filled in that segment; a neuron in no array gets no row.
    """
    names = [name for name, _ in arrays]
    labels = array_labels(names)
    gap_ms = max(0.0, float(gap_ms))
    dt = shared_dt([data.dt for _, data in arrays])
    if dt is not None and gap_ms:
        gap_ms = round(gap_ms / (dt * 1e3)) * dt * 1e3
    segments: list[ConcatSegment] = []
    start = 0.0
    for position, (name, data) in enumerate(arrays):
        if position:
            start += gap_ms
        segment = ConcatSegment(
            array_name=name,
            label=labels[position],
            color=array_color(position),
            start_ms=start,
            dt=float(data.dt),
            n_samples=int(data.traces.shape[1]),
        )
        segments.append(segment)
        start = segment.end_ms
    row_by_id_per_array = [
        {int(neuron_id): row for row, neuron_id in enumerate(data.indices)}
        for _, data in arrays
    ]
    missing: dict[str, list[int]] = {label: [] for label in labels}
    rows: list[ConcatRow] = []
    for neuron_id in dict.fromkeys(int(n) for n in neuron_ids):
        found = [lookup.get(neuron_id) for lookup in row_by_id_per_array]
        values: list[NDArray[np.float32]] = []
        for position, ((_, data), row) in enumerate(zip(arrays, found, strict=True)):
            if row is None:
                missing[labels[position]].append(neuron_id)
                values.append(np.full(segments[position].n_samples, np.nan, np.float32))
            else:
                values.append(np.asarray(data.traces[row], dtype=np.float32))
        if any(row is not None for row in found):
            rows.append(ConcatRow(neuron_id, values))
    units = {data.units for _, data in arrays}
    return TraceConcatenation(
        segments=segments,
        rows=rows,
        gap_ms=gap_ms,
        units=units.pop() if len(units) == 1 else "",
        missing={label: ids for label, ids in missing.items() if ids},
    )


def concatenation_group_name(
    array_names: Sequence[str], existing: Iterable[str] = ()
) -> str:
    """Default data-group name for joined arrays, unique among ``existing``.

    ``"Concatenated: Rep 1 + Rep 2 + Rep 3"``; more than four arrays are
    shortened to ``"Rep 1 + … + Rep 9 (9 arrays)"``.
    """
    labels = [
        label.replace(NAME_SEPARATOR, " / ") for label in array_labels(array_names)
    ]
    if len(labels) > 4:
        joined = f"{labels[0]} + … + {labels[-1]} ({len(labels)} arrays)"
    else:
        joined = " + ".join(labels)
    base = f"Concatenated: {joined}" if joined else "Concatenated"
    return unique_name(base, existing)


def concatenation_data_name(array_names: Sequence[str]) -> str:
    """Data-set name for joined arrays: their shared data name, else "traces"."""
    data_names = {name.partition(NAME_SEPARATOR)[2] for name in array_names}
    if len(data_names) == 1:
        name = data_names.pop()
        if name:
            return name
    return "traces"


def unique_name(base: str, existing: Iterable[str]) -> str:
    """``base``, or ``"base (2)"``, ``"base (3)"`` … when already taken."""
    taken = set(existing)
    if base not in taken:
        return base
    number = 2
    while f"{base} ({number})" in taken:
        number += 1
    return f"{base} ({number})"


def group_name_problem(name: str, existing: Iterable[str]) -> str:
    """Why ``name`` cannot be a new data-group name ('' when it can)."""
    name = name.strip()
    if not name:
        return "Enter a name for the new trace array."
    if NAME_SEPARATOR.strip() in name:
        return f"The name must not contain '{NAME_SEPARATOR.strip()}'."
    if name == "Simulation" or name in set(existing):
        return f"A data group named '{name}' already exists."
    return ""


# -- Linked axes across windows ---------------------------------------------

TIME_AXIS = "time"
VALUE_AXIS = "value"


class AxisLinkGroup[M: Hashable]:
    """Keeps the time and/or value ranges of several views in step.

    Each member locks its time and value axis independently. A range change
    on a member is passed on (through ``apply``) only when that member has
    the axis locked, and only to other members that have it locked too; a
    change made while passing one on is not passed on again, so views that
    report their own range changes cannot loop.
    """

    def __init__(self) -> None:
        self._locks: dict[M, dict[str, bool]] = {}
        self._propagating = False

    @property
    def members(self) -> list[M]:
        return list(self._locks)

    def add(self, member: M, *, lock_time: bool, lock_value: bool) -> None:
        self._locks[member] = {TIME_AXIS: lock_time, VALUE_AXIS: lock_value}

    def remove(self, member: M) -> None:
        self._locks.pop(member, None)

    def is_locked(self, member: M, axis: str) -> bool:
        return self._locks.get(member, {}).get(axis, False)

    def set_locked(self, member: M, axis: str, locked: bool) -> None:
        if member in self._locks:
            self._locks[member][axis] = locked

    def targets(self, source: M, axis: str) -> list[M]:
        """Members a range change of ``source`` on ``axis`` should reach."""
        if not self.is_locked(source, axis):
            return []
        return [
            member
            for member in self._locks
            if member is not source and self.is_locked(member, axis)
        ]

    def reference(self, member: M, axis: str) -> M | None:
        """Another member locked on ``axis``, whose range ``member`` adopts when
        it locks again (None when no other member is locked)."""
        for other in self._locks:
            if other is not member and self.is_locked(other, axis):
                return other
        return None

    def propagate(self, source: M, axis: str, apply: Callable[[M], None]) -> list[M]:
        """Call ``apply(target)`` for every target of ``source`` on ``axis``.

        Returns the targets reached. A change reported while a propagation is
        running (the targets echoing it back) is not passed on again.
        """
        if self._propagating:
            return []
        targets = self.targets(source, axis)
        self._propagating = True
        try:
            for target in targets:
                apply(target)
        finally:
            self._propagating = False
        return targets


def cascade_offsets(count: int, step: int = 32) -> list[tuple[int, int]]:
    """Top-left offsets that fan ``count`` windows out diagonally."""
    return [(index * step, index * step) for index in range(count)]
