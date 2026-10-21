"""Qt-free BrainScaleS-2 (BSS-2) delivery logic.

NeuroSeq never imports the BSS-2 software stack (``pynn_brainscales``,
``dlens_vx_v3``): it only exists inside the EBRAINS container image. Delivery
therefore runs ``resources/bss2/neuroseq_bss2_helper.py`` *inside* the
container and exchanges JSON lines with it over stdin/stdout. This module holds
everything around that exchange that does not need Qt or hardware:

* turning a :class:`~neuroseq.stimulation.stim.StimArray` into constant-current
  segments or spike trains,
* building the ``run`` request and decoding its result into NeuroSeq data,
* parsing the quiggeldy setup list and picking the default setup,
* constructing the container launch command (argv + environment).

Time base
---------
BSS-2 is an *accelerated* analog system: its neurons evolve roughly 1000x
faster than biological neurons, and ``pynn.run`` durations as well as recorded
spike/sample times are in *hardware* milliseconds. NeuroSeq stim arrays and
experiment data are in *biological* seconds. The conversion uses an explicit,
user-configurable ``speedup`` (default :data:`DEFAULT_SPEEDUP` = 1000)::

    hardware_ms = biological_s * 1000 / speedup
    biological_s = hardware_ms / 1000 * speedup

so 1 s of stimulus becomes a 1 ms hardware run, and a spike recorded at
0.5 ms hardware time is stored at 0.5 s. This factor is an assumption about
the calibrated neuron time constants; it has to be checked against the real
container/chip (see the BrainScaleS-2 user guide page).

Wire protocol (one JSON object per line)
----------------------------------------
Requests (NeuroSeq -> helper)::

    {"id": <int>, "command": "ping" | "connect" | "run" | "shutdown",
     "arguments": {...}}

Messages (helper -> NeuroSeq)::

    {"id": null, "event": "ready", "versions": {...}}      # once at start-up
    {"id": <int>, "event": "progress", "message": <str>}   # zero or more
    {"id": <int>, "result": {...}}                          # final reply, or
    {"id": <int>, "error": <str>}

Arrays travel as ``{"dtype": "<f4", "shape": [...], "data": <base64>}`` of
little-endian bytes (see :func:`encode_array`). The EBRAINS token is never part
of a message: it only reaches the helper through its environment.
"""

from __future__ import annotations

import base64
import csv
import getpass
import hashlib
import io
import json
import os
import re
import shlex
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from neuroseq.data import Data
from neuroseq.data import Experiment
from neuroseq.data import SpikeData
from neuroseq.data import TraceData

PROTOCOL_VERSION = 1
#: Biological time / hardware time. Assumed, user-configurable; see module docs.
DEFAULT_SPEEDUP = 1000.0
#: Neurons on one BSS-2 (HICANN-X) chip.
MAX_HARDWARE_NEURONS = 512
#: ``constant_current_i_offset`` is a 10-bit CapMem value.
I_OFFSET_LIMIT = 1022
#: Synapse weights are 6-bit integers.
WEIGHT_LIMIT = 63
#: Each segment is one ``pynn.run(..., APPEND)`` call; warn above this many.
SEGMENT_WARNING_LIMIT = 1000

INPUT_CONSTANT_CURRENT = "constant_current"
INPUT_SPIKES = "spike_input"
SPIKE_SOURCE_STIM = "stim"
SPIKE_SOURCE_EXPERIMENT = "experiment"

QUIGGELDY_SETUPS_URL = "https://quiggeldy.bioai.eu/nmpi/quiggeldy_setups.csv"
DEFAULT_CONTAINER_IMAGE = "/containers/stable/latest"
#: Activates the BSS-2 software stack inside the EBRAINS image. Its output is
#: sent to stderr so that stdout carries protocol messages only.
CONTAINER_ENV_SETUP = (
    'export SYSTEMNAME="ebrainslab"; '
    "source /esd/vendor/spack/share/spack/setup-env.sh; "
    "spack env activate esd"
)
HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "bss2"
    / "neuroseq_bss2_helper.py"
)

# pynn-brainscales CalibHXNeuronCoba targets (hardware units: dimensionless
# voltages, microsecond time constants). Calix solves the per-neuron CapMem
# codes on-chip. Used by the PyNN example model only: stimulus delivery uses
# HXNeuron with the nightly calibration, which supports constant current.
BSS2_NEURON_PARAMETERS: dict[str, float | int] = dict(
    v_rest=110.0,
    v_reset=100.0,
    v_thresh=160.0,
    tau_m=10.0,
    tau_syn_E=10.0,
    tau_syn_I=10.0,
    tau_refrac=2.0,
    cm=63,
    e_rev_E=300.0,
    e_rev_I=30.0,
)

_ARRAY_DTYPES = {"<f4", "<u4", "<u2", "|u1"}


# ---- array transport -------------------------------------------------------


def encode_array(array: np.ndarray, dtype: str) -> dict[str, Any]:
    """Encode ``array`` as little-endian ``dtype`` bytes in base64."""
    if dtype not in _ARRAY_DTYPES:
        raise ValueError(f"Unsupported array dtype {dtype!r}")
    data = np.ascontiguousarray(np.asarray(array).astype(np.dtype(dtype)))
    return {
        "dtype": dtype,
        "shape": list(data.shape),
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


def decode_array(payload: Mapping[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array`, validating dtype and size."""
    dtype = str(payload["dtype"])
    if dtype not in _ARRAY_DTYPES:
        raise ValueError(f"Unsupported array dtype {dtype!r}")
    shape = tuple(int(n) for n in payload["shape"])
    if any(n < 0 for n in shape):
        raise ValueError("Negative array dimension")
    raw = base64.b64decode(payload["data"], validate=True)
    item = np.dtype(dtype)
    if len(raw) != int(np.prod(shape, dtype=np.int64)) * item.itemsize:
        raise ValueError("Array payload size does not match its shape")
    return np.frombuffer(raw, dtype=item).reshape(shape).copy()


# ---- time base -------------------------------------------------------------


def bio_s_to_hw_ms(seconds: np.ndarray | float, speedup: float) -> Any:
    """Biological seconds -> hardware milliseconds."""
    _check_speedup(speedup)
    return np.asarray(seconds, dtype=np.float64) * 1000.0 / float(speedup)


def hw_ms_to_bio_s(milliseconds: np.ndarray | float, speedup: float) -> Any:
    """Hardware milliseconds -> biological seconds."""
    _check_speedup(speedup)
    return np.asarray(milliseconds, dtype=np.float64) / 1000.0 * float(speedup)


def _check_speedup(speedup: float) -> None:
    if not float(speedup) > 0:
        raise ValueError("The speed-up factor must be > 0")


# ---- settings --------------------------------------------------------------


@dataclass
class Bss2Settings:
    """Persisted tab settings. The EBRAINS token is deliberately not a field."""

    container_image: str = DEFAULT_CONTAINER_IMAGE
    container_app: str = "dls"
    setup_host: str = ""
    setup_port: int = 11717
    setup_identifier: str = ""
    input_mode: str = INPUT_CONSTANT_CURRENT
    #: 0 means "as many neurons as stim channels".
    n_neurons: int = 0
    max_neurons: int = MAX_HARDWARE_NEURONS
    i_offset_max: int = 500
    #: Stim value mapped to ``i_offset_max``; 0 means "the array maximum".
    value_at_max: float = 0.0
    spike_source: str = SPIKE_SOURCE_STIM
    spike_threshold: float = 0.0
    spike_event_ms: float = 20.0
    spike_weight: int = WEIGHT_LIMIT
    madc_neuron: int = 0
    speedup: float = DEFAULT_SPEEDUP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> Bss2Settings:
        """Build settings from stored values, keeping defaults for anything
        unknown, missing or of the wrong type."""
        settings = cls()
        if not isinstance(values, Mapping):
            return settings
        for f in fields(cls):
            if f.name not in values:
                continue
            default = getattr(settings, f.name)
            try:
                value = type(default)(values[f.name])
            except (TypeError, ValueError):
                continue
            setattr(settings, f.name, value)
        if settings.input_mode not in (INPUT_CONSTANT_CURRENT, INPUT_SPIKES):
            settings.input_mode = INPUT_CONSTANT_CURRENT
        if settings.spike_source not in (SPIKE_SOURCE_STIM, SPIKE_SOURCE_EXPERIMENT):
            settings.spike_source = SPIKE_SOURCE_STIM
        settings.max_neurons = int(np.clip(settings.max_neurons, 1, 4096))
        settings.i_offset_max = int(np.clip(settings.i_offset_max, 0, I_OFFSET_LIMIT))
        settings.spike_weight = int(np.clip(settings.spike_weight, 0, WEIGHT_LIMIT))
        if not settings.speedup > 0:
            settings.speedup = DEFAULT_SPEEDUP
        return settings


# ---- stim conversion -------------------------------------------------------


def channels_to_neurons(sampled: np.ndarray, n_neurons: int) -> np.ndarray:
    """Map a (time, n_channels) stim block onto (time, n_neurons).

    Pass channels through when they already match, broadcast a single
    channel, truncate extra channels, and zero-pad when there are too few.
    """
    if sampled.shape[1] == n_neurons:
        return sampled
    if sampled.shape[1] == 1:
        return np.repeat(sampled, n_neurons, axis=1)
    if sampled.shape[1] > n_neurons:
        return sampled[:, :n_neurons]
    padded = np.zeros((sampled.shape[0], n_neurons), dtype=sampled.dtype)
    padded[:, : sampled.shape[1]] = sampled
    return padded


def stim_to_spike_times(
    arrays: Iterable[tuple[np.ndarray, float]],
    simulation_time: float,
    n_neurons: int,
    threshold: float,
    event_ms: float,
) -> list[np.ndarray]:
    """Convert stim arrays into one evoked-spike train per neuron (ms).

    ``arrays`` yields ``(values, dt_seconds)`` pairs. They are summed and
    sampled on a regular ``event_ms`` grid (slot centres); wherever a neuron's
    summed stim exceeds ``threshold`` a spike is emitted at that grid time.
    This models optogenetic drive: a non-zero stimulus evokes a spike.
    """
    if event_ms <= 0:
        raise ValueError("The spike event interval must be > 0 ms")
    total_ms = simulation_time * 1000.0
    n_slots = max(1, int(total_ms // event_ms))
    slot_times_ms = np.arange(n_slots) * event_ms + 0.5 * event_ms
    slot_times_s = slot_times_ms / 1000.0

    summed = np.zeros((n_slots, n_neurons), dtype=np.float64)
    for values, dt in arrays:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        stim_dt_s = float(dt)
        if stim_dt_s <= 0:
            raise ValueError("Stim dt must be > 0")
        bins = np.minimum((slot_times_s / stim_dt_s).astype(np.int64), arr.shape[0] - 1)
        summed += channels_to_neurons(arr[bins], n_neurons)

    active = summed > threshold
    return [slot_times_ms[active[:, i]].astype(np.float64) for i in range(n_neurons)]


def resolve_neuron_count(
    n_channels: int, settings: Bss2Settings
) -> tuple[int, list[str]]:
    """The number of driven hardware neurons, plus warnings about the mapping."""
    if n_channels < 1:
        raise ValueError("The stim array has no channels")
    warnings: list[str] = []
    requested = int(settings.n_neurons) if settings.n_neurons > 0 else n_channels
    n_neurons = min(requested, int(settings.max_neurons))
    if requested > n_neurons:
        warnings.append(
            f"{requested} neurons requested, but at most {settings.max_neurons} are "
            f"available; using neurons 0-{n_neurons - 1}."
        )
    if n_channels > n_neurons:
        warnings.append(
            f"The stim array has {n_channels} channels; only the first {n_neurons} "
            "are delivered."
        )
    elif n_channels < n_neurons and n_channels > 1:
        warnings.append(
            f"The stim array has {n_channels} channels; neurons {n_channels}-"
            f"{n_neurons - 1} receive no input."
        )
    return n_neurons, warnings


def change_segments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (time, channel) array where any channel's value changes.

    Returns ``(starts, lengths)`` in samples; consecutive equal samples form
    one segment, so each segment has a single constant value per channel.
    """
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected a non-empty (time, channel) array")
    changed = np.any(values[1:] != values[:-1], axis=1)
    starts = np.concatenate(([0], np.flatnonzero(changed) + 1)).astype(np.int64)
    lengths = np.diff(np.append(starts, values.shape[0])).astype(np.int64)
    return starts, lengths


@dataclass
class ConstantCurrentPlan:
    """Per-segment ``constant_current_enable`` / ``constant_current_i_offset``."""

    durations_s: np.ndarray  # (segments,), biological seconds
    enable: np.ndarray  # (segments, neurons) bool
    offset: np.ndarray  # (segments, neurons) uint16
    n_channels: int
    value_at_max: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n_neurons(self) -> int:
        return int(self.enable.shape[1])

    @property
    def n_segments(self) -> int:
        return int(self.durations_s.size)

    @property
    def duration_s(self) -> float:
        return float(self.durations_s.sum())


def plan_constant_current(
    values: np.ndarray, dt: float, settings: Bss2Settings
) -> ConstantCurrentPlan:
    """Segment a stim array into constant-current steps.

    Per segment and neuron: ``enable = value > 0`` and
    ``offset = round(clip(value / value_at_max, 0, 1) * i_offset_max)``.
    Adjacent segments that end up identical after quantisation are merged.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("The stim array is empty")
    if not float(dt) > 0:
        raise ValueError("Stim dt must be > 0")
    n_neurons, warnings = resolve_neuron_count(values.shape[1], settings)
    mapped = channels_to_neurons(values, n_neurons)
    value_at_max = float(settings.value_at_max)
    if value_at_max <= 0:
        value_at_max = float(mapped.max())
    i_max = int(np.clip(settings.i_offset_max, 0, I_OFFSET_LIMIT))
    starts, lengths = change_segments(mapped)
    levels = mapped[starts]
    enable = levels > 0
    if value_at_max > 0:
        scaled = np.clip(levels / value_at_max, 0.0, 1.0) * i_max
        offset = np.rint(scaled).astype(np.uint16)
    else:
        offset = np.zeros(levels.shape, dtype=np.uint16)
    offset[~enable] = 0
    # Re-merge runs that quantisation made identical.
    keys = np.concatenate([enable, offset], axis=1)
    keep = np.concatenate(([True], np.any(keys[1:] != keys[:-1], axis=1)))
    group = np.cumsum(keep) - 1
    merged_lengths = np.bincount(group, weights=lengths).astype(np.int64)
    plan = ConstantCurrentPlan(
        durations_s=merged_lengths * float(dt),
        enable=enable[keep],
        offset=offset[keep],
        n_channels=int(values.shape[1]),
        value_at_max=value_at_max,
        warnings=warnings,
    )
    if plan.n_segments > SEGMENT_WARNING_LIMIT:
        plan.warnings.append(
            f"{plan.n_segments} segments: each is a separate hardware run step, "
            "so preparing the run may take long."
        )
    return plan


@dataclass
class SpikeInputPlan:
    """External spike trains fed one-to-one into the hardware neurons."""

    indices: np.ndarray  # uint32 neuron index per spike
    times_s: np.ndarray  # biological seconds per spike
    n_neurons: int
    duration_s: float
    weight: int
    warnings: list[str] = field(default_factory=list)

    @property
    def n_spikes(self) -> int:
        return int(self.indices.size)


def plan_spikes_from_stim(
    values: np.ndarray, dt: float, settings: Bss2Settings
) -> SpikeInputPlan:
    """Threshold/event-rate conversion of a stim array (as the PyNN example)."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("The stim array is empty")
    if not float(dt) > 0:
        raise ValueError("Stim dt must be > 0")
    n_neurons, warnings = resolve_neuron_count(values.shape[1], settings)
    duration_s = values.shape[0] * float(dt)
    trains_ms = stim_to_spike_times(
        [(values, float(dt))],
        duration_s,
        n_neurons,
        threshold=float(settings.spike_threshold),
        event_ms=float(settings.spike_event_ms),
    )
    indices = np.concatenate(
        [np.full(t.size, i, dtype=np.uint32) for i, t in enumerate(trains_ms)]
    )
    times_s = np.concatenate(trains_ms) / 1000.0
    return _spike_plan(indices, times_s, n_neurons, duration_s, settings, warnings)


def plan_spikes_from_data(
    spikes: SpikeData, duration_s: float, settings: Bss2Settings
) -> SpikeInputPlan:
    """Replay recorded/simulated spikes: neuron ``i`` drives hardware neuron ``i``."""
    indices = np.asarray(spikes.spike_indices, dtype=np.int64)
    times_s = np.asarray(spikes.spike_times, dtype=np.float64)
    n_channels = int(indices.max()) + 1 if indices.size else 1
    n_neurons, warnings = resolve_neuron_count(n_channels, settings)
    if not duration_s > 0:
        duration_s = float(times_s.max()) if times_s.size else 0.0
    if not duration_s > 0:
        raise ValueError("The spike data has no duration")
    keep = (indices < n_neurons) & (times_s >= 0) & (times_s < duration_s)
    return _spike_plan(
        indices[keep].astype(np.uint32),
        times_s[keep],
        n_neurons,
        float(duration_s),
        settings,
        warnings,
    )


def _spike_plan(
    indices: np.ndarray,
    times_s: np.ndarray,
    n_neurons: int,
    duration_s: float,
    settings: Bss2Settings,
    warnings: list[str],
) -> SpikeInputPlan:
    order = np.lexsort((times_s, indices))
    plan = SpikeInputPlan(
        indices=np.asarray(indices, dtype=np.uint32)[order],
        times_s=np.asarray(times_s, dtype=np.float64)[order],
        n_neurons=n_neurons,
        duration_s=float(duration_s),
        weight=int(np.clip(settings.spike_weight, 0, WEIGHT_LIMIT)),
        warnings=warnings,
    )
    if plan.n_spikes == 0:
        plan.warnings.append("No input spikes: the driven neurons receive no input.")
    return plan


Plan = ConstantCurrentPlan | SpikeInputPlan


def plan_summary(plan: Plan, settings: Bss2Settings) -> str:
    """Human-readable summary for the delivery confirmation dialog."""
    hw_ms = float(bio_s_to_hw_ms(plan.duration_s, settings.speedup))
    lines = [f"Neurons: {plan.n_neurons} (hardware neurons 0-{plan.n_neurons - 1})"]
    if isinstance(plan, ConstantCurrentPlan):
        lines.append(
            f"Input: constant current, {plan.n_segments} segment(s), "
            f"i_offset up to {settings.i_offset_max} at stim value "
            f"{plan.value_at_max:g}"
        )
    else:
        lines.append(
            f"Input: {plan.n_spikes} spike(s), excitatory weight {plan.weight}"
        )
    lines.append(
        f"Duration: {plan.duration_s:g} s biological = {hw_ms:g} ms on hardware "
        f"(speed-up {settings.speedup:g}x)"
    )
    lines.append(f"Membrane voltage recorded from neuron {settings.madc_neuron}")
    lines.extend(plan.warnings)
    return "\n".join(lines)


# ---- request / result ------------------------------------------------------


def build_run_arguments(plan: Plan, settings: Bss2Settings) -> dict[str, Any]:
    """The ``arguments`` of a ``run`` request (all times in hardware ms)."""
    if not 0 <= settings.madc_neuron < plan.n_neurons:
        raise ValueError(
            f"The recorded neuron {settings.madc_neuron} is not among the "
            f"{plan.n_neurons} driven neurons."
        )
    arguments: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "n_neurons": plan.n_neurons,
        "duration_ms": float(bio_s_to_hw_ms(plan.duration_s, settings.speedup)),
        "record_v_neuron": int(settings.madc_neuron),
        # HXNeuron overrides in hardware units; empty = the chip's nightly
        # calibration (see the helper's execute_run).
        "neuron_parameters": {},
    }
    if isinstance(plan, ConstantCurrentPlan):
        arguments["mode"] = INPUT_CONSTANT_CURRENT
        arguments["segments"] = {
            "durations_ms": encode_array(
                bio_s_to_hw_ms(plan.durations_s, settings.speedup), "<f4"
            ),
            "enable": encode_array(plan.enable, "|u1"),
            "offset": encode_array(plan.offset, "<u2"),
        }
    else:
        arguments["mode"] = INPUT_SPIKES
        arguments["spikes"] = {
            "indices": encode_array(plan.indices, "<u4"),
            "times_ms": encode_array(
                bio_s_to_hw_ms(plan.times_s, settings.speedup), "<f4"
            ),
            "weight": int(plan.weight),
        }
    return arguments


@dataclass
class RunReadback:
    """Decoded ``run`` result in biological time."""

    spikes: SpikeData
    trace: TraceData | None
    duration_s: float


def decode_run_result(
    result: Mapping[str, Any], speedup: float, *, label: str = "BSS-2"
) -> RunReadback:
    """Convert the helper's ``run`` result into NeuroSeq data (biological time).

    Spikes keep their hardware neuron index. The membrane voltage of the one
    MADC-recorded neuron is resampled onto a uniform grid (the MADC delivers
    irregular sample times) and stored as a single-row trace in arbitrary units.
    """
    spikes = result["spikes"]
    indices = decode_array(spikes["indices"]).astype(np.uint32)
    times_ms = decode_array(spikes["times_ms"]).astype(np.float64)
    if indices.shape != times_ms.shape:
        raise ValueError("Spike indices and times differ in length")
    duration_s = float(hw_ms_to_bio_s(float(result["duration_ms"]), speedup))
    spike_data = SpikeData(
        name=f"{label} spikes",
        spike_indices=indices,
        spike_times=hw_ms_to_bio_s(times_ms, speedup).astype(np.float32),
    )
    trace = None
    v = result.get("v")
    if v:
        neuron = int(v["neuron"])
        v_times = hw_ms_to_bio_s(decode_array(v["times_ms"]), speedup)
        values = decode_array(v["values"]).astype(np.float64)
        if v_times.shape != values.shape:
            raise ValueError("Membrane voltage times and values differ in length")
        if values.size >= 2:
            order = np.argsort(v_times, kind="stable")
            v_times, values = v_times[order], values[order]
            dt = float(np.median(np.diff(v_times)))
            if not dt > 0:
                raise ValueError("Membrane voltage samples have no time spacing")
            grid = (
                v_times[0]
                + np.arange(int(np.floor((v_times[-1] - v_times[0]) / dt)) + 1) * dt
            )
            resampled = np.interp(grid, v_times, values)
            if v_times[0] > 0:
                # Start the trace at t=0 like every other NeuroSeq trace.
                lead = int(np.floor(v_times[0] / dt))
                resampled = np.concatenate([np.full(lead, resampled[0]), resampled])
            trace = TraceData(
                name=f"{label} membrane voltage (neuron {neuron})",
                traces=resampled[None, :].astype(np.float32),
                indices=np.array([neuron], dtype=np.uint32),
                units="a.u.",
                dt=dt,
            )
    return RunReadback(spikes=spike_data, trace=trace, duration_s=duration_s)


def readback_experiment(
    readback: RunReadback, stim_name: str, when: datetime | None = None
) -> Experiment:
    """Wrap a readback into an experiment named ``BSS-2 <stim> <timestamp>``."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H-%M-%S")
    data: list[Data] = [readback.spikes]
    if readback.trace is not None:
        data.append(readback.trace)
    return Experiment(
        name=f"BSS-2 {stim_name} {stamp}",
        duration=readback.duration_s,
        timepoints=[],
        data=data,
    )


# ---- quiggeldy setups ------------------------------------------------------


@dataclass(frozen=True)
class QuiggeldySetup:
    host: str
    port: int
    identifier: str
    #: Short note shown in the picker, e.g. the software version it runs.
    description: str = ""

    @property
    def label(self) -> str:
        address = f"{self.host}:{self.port}"
        note = f", {self.description}" if self.description else ""
        if self.identifier:
            return f"{self.identifier} ({address}{note})"
        return f"{address} ({self.description})" if self.description else address


#: quiggeldy instances on the KIP host, from the BSS-2 collaborators. They run
#: different software versions; the chip identifier is reported on connect.
KIP_QUIGGELDY_HOST = "helvetica.kip.uni-heidelberg.de"
KIP_SETUPS: tuple[QuiggeldySetup, ...] = (
    QuiggeldySetup(KIP_QUIGGELDY_HOST, 11717, "", "stable"),
    QuiggeldySetup(KIP_QUIGGELDY_HOST, 11750, "", "experimental"),
    QuiggeldySetup(KIP_QUIGGELDY_HOST, 11780, "", "special"),
)


def combined_setups(
    downloaded: Iterable[QuiggeldySetup],
    extra: Iterable[QuiggeldySetup] = KIP_SETUPS,
) -> list[QuiggeldySetup]:
    """Downloaded setups first, then ``extra`` ones not already listed."""
    setups = list(downloaded)
    known = {(setup.host, setup.port) for setup in setups}
    for setup in extra:
        if (setup.host, setup.port) not in known:
            setups.append(setup)
            known.add((setup.host, setup.port))
    return setups


def parse_setups_csv(text: str) -> list[QuiggeldySetup]:
    """Parse the quiggeldy setup list (columns ``Host,Port,Identifier``)."""
    reader = csv.DictReader(io.StringIO(text.strip()), skipinitialspace=True)
    if reader.fieldnames is None:
        return []
    columns = {name.strip().lower(): name for name in reader.fieldnames}
    missing = {"host", "port", "identifier"} - columns.keys()
    if missing:
        raise ValueError(f"Setup list lacks column(s): {', '.join(sorted(missing))}")
    setups = []
    for row in reader:
        host = (row.get(columns["host"]) or "").strip()
        identifier = (row.get(columns["identifier"]) or "").strip()
        try:
            port = int((row.get(columns["port"]) or "").strip())
        except ValueError:
            continue
        if host and identifier and 0 < port < 65536:
            setups.append(QuiggeldySetup(host, port, identifier))
    return setups


def default_setup_index(username: str, n_setups: int) -> int:
    """The collaborators' default: ``sha256(username) mod len(setups)``."""
    if n_setups < 1:
        raise ValueError("No setups to choose from")
    return int(hashlib.sha256(username.encode()).hexdigest(), 16) % n_setups


def current_username() -> str:
    """``$USER`` (what quiggeldy is told via ``QUIGGELDY_USER_NO_MUNGE``)."""
    return os.environ.get("USER") or getpass.getuser()


# ---- container launch ------------------------------------------------------


@dataclass(frozen=True)
class LaunchConfig:
    """Everything needed to start the helper, apart from the token."""

    container_image: str
    container_app: str
    host: str
    port: int
    user: str
    singularity: str = "singularity"
    helper_path: Path = HELPER_PATH
    env_setup: str = CONTAINER_ENV_SETUP
    #: Folder for the cached spack activation script (see
    #: :func:`activation_cache_file`); ``None`` activates spack on every start.
    cache_dir: Path | None = None


@dataclass(frozen=True)
class LaunchCommand:
    program: str
    arguments: list[str]
    #: Variables to add to the process environment (contains the token).
    environment: dict[str, str]


_CONTAINER_ENV_PREFIXES = ("SINGULARITYENV_", "APPTAINERENV_")


def helper_environment(config: LaunchConfig, token: str) -> dict[str, str]:
    """Quiggeldy variables, both plain and ``SINGULARITYENV_``-prefixed.

    The prefixed copies are injected by Singularity/Apptainer even when the
    host environment is not inherited (``--cleanenv``).
    """
    plain = {
        "QUIGGELDY_ENABLED": "1",
        "QUIGGELDY_USER_NO_MUNGE": config.user,
        "QUIGGELDY_IP": config.host,
        "QUIGGELDY_PORT": str(int(config.port)),
        "PYTHONUNBUFFERED": "1",
    }
    if token:
        plain["QUIGGELDY_TOKEN"] = token
    environment = dict(plain)
    # Apptainer prefers APPTAINERENV_ (and logs an INFO line per
    # SINGULARITYENV_ variable otherwise); SingularityCE only reads
    # SINGULARITYENV_. Exporting both works quietly with either.
    for prefix in _CONTAINER_ENV_PREFIXES:
        environment.update({f"{prefix}{k}": v for k, v in plain.items()})
    return environment


#: Sourcing ``spack env activate --sh`` output takes seconds instead of the
#: ~50 s a full ``spack env activate`` needs on every start.
_SPACK_SETUP = "source /esd/vendor/spack/share/spack/setup-env.sh"


def activation_cache_file(cache_dir: Path, container_image: str) -> Path:
    """Cache file for one image; a rebuilt/updated image gets a new file."""
    image = Path(container_image)
    try:
        stamp = str(image.stat().st_mtime_ns)
    except OSError:
        stamp = "missing"
    key = hashlib.sha256(f"{image.resolve()}|{stamp}".encode()).hexdigest()[:16]
    return Path(cache_dir) / f"esd-activate-{key}.sh"


def _cached_env_setup(cache: Path) -> str:
    quoted = shlex.quote(str(cache))
    partial = shlex.quote(f"{cache}.partial")
    return (
        'export SYSTEMNAME="ebrainslab"; '
        f"if [ -s {quoted} ]; then source {quoted}; "
        f"else {_SPACK_SETUP} && spack env activate --sh esd > {partial} "
        f"&& mv {partial} {quoted} && source {quoted}; fi"
    )


def helper_launch_command(config: LaunchConfig, token: str) -> LaunchCommand:
    """``singularity exec --bind <dir> [--app <app>] <image> bash -c '...'``.

    The token only appears in the returned environment, never in the argv.
    """
    if not config.container_image.strip():
        raise ValueError("No container image configured")
    helper = Path(config.helper_path).resolve()
    env_setup = config.env_setup
    binds = [str(helper.parent)]
    if config.cache_dir is not None and env_setup == CONTAINER_ENV_SETUP:
        cache = activation_cache_file(config.cache_dir, config.container_image)
        env_setup = _cached_env_setup(cache)
        binds.append(str(Path(config.cache_dir).resolve()))
    script = f"{{ {env_setup}; }} 1>&2 && exec python -u {shlex.quote(str(helper))}"
    arguments = ["exec", "--bind", ",".join(binds)]
    if config.container_app.strip():
        arguments += ["--app", config.container_app.strip()]
    arguments += [config.container_image.strip(), "bash", "-c", script]
    return LaunchCommand(
        program=config.singularity,
        arguments=arguments,
        environment=helper_environment(config, token),
    )


_TOKEN_LIKE = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")


_ENCODED_ARRAY_KEYS = {"dtype", "shape", "data"}


def summarize_message(message: Any, limit: int = 4000) -> str:
    """Compact JSON of a protocol message for the debug log.

    Encoded arrays are shown as ``<array dtype shape>`` instead of base64.
    """

    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            if value.keys() >= _ENCODED_ARRAY_KEYS:
                return f"<array {value.get('dtype')} {value.get('shape')}>"
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    text = json.dumps(compact(message), default=str)
    if len(text) > limit:
        text = text[:limit] + f"… ({len(text) - limit} more characters)"
    return text


def describe_launch(command: LaunchCommand) -> list[str]:
    """Debug lines for a helper launch; the token's value is never shown."""
    lines = ["Launch: " + shlex.join([command.program, *command.arguments])]
    plain = {
        key: value
        for key, value in command.environment.items()
        if not key.startswith(_CONTAINER_ENV_PREFIXES)
    }
    for key, value in plain.items():
        if key == "QUIGGELDY_TOKEN":
            lines.append(
                f"  env QUIGGELDY_TOKEN (value hidden): {len(value)} characters"
            )
        else:
            lines.append(f"  env {key}={value}")
    if "QUIGGELDY_TOKEN" not in plain:
        lines.append("  env QUIGGELDY_TOKEN not set by NeuroSeq")
    if any(key.startswith(_CONTAINER_ENV_PREFIXES) for key in command.environment):
        lines.append(
            "  (each also exported with the SINGULARITYENV_ and APPTAINERENV_ prefixes)"
        )
    return lines


def debug_report_header(
    settings: Bss2Settings,
    *,
    token_given: bool,
    neuroseq_version: str,
    singularity_version: str,
    connection_state: str,
    identifier: str | None,
) -> list[str]:
    """Context lines prepended when the debug log is copied (no token)."""
    import platform

    lines = [
        "=== NeuroSeq BrainScaleS-2 debug log ===",
        f"copied:      {datetime.now().isoformat(timespec='seconds')}",
        f"neuroseq:    {neuroseq_version}",
        f"host:        {platform.node()} ({platform.platform()})",
        f"python:      {platform.python_version()}",
        f"singularity: {singularity_version}",
        f"user:        {current_username()}",
        f"state:       {connection_state}"
        + (f" (identifier {identifier})" if identifier else ""),
        f"token:       {'given' if token_given else 'not given'}",
        "settings:",
    ]
    lines += [f"  {key} = {value}" for key, value in settings.to_dict().items()]
    lines.append("=== log ===")
    return lines


def scrub_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask ``secrets`` and anything that looks like a JWT access token."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    text = _TOKEN_LIKE.sub("***", text)
    return re.sub(r"(QUIGGELDY_TOKEN\s*[=:]\s*)\S+", r"\1***", text)
