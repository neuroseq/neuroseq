"""NeuroSeq <-> BrainScaleS-2 helper; JSON lines over stdin/stdout.

Runs INSIDE the EBRAINS container (started by NeuroSeq with
``singularity exec ... bash -c '<spack env activation>; exec python -u <this file>'``).
It may only import the standard library, numpy and the BSS-2 stack
(``pynn_brainscales``, ``dlens_vx_v3``, ``pyNN``) - never NeuroSeq, which is
not installed in the container. The protocol is documented in
``neuroseq/device_control/bss2_protocol.py``.

stdout carries protocol messages only: at start-up the original stdout is
duplicated for the protocol and file descriptor 1 is pointed at stderr, so
anything the hardware stack prints (also from C++) ends up in stderr.

The EBRAINS token arrives in ``QUIGGELDY_TOKEN`` through the environment and
is never read into, or written out of, a protocol message.
"""

from __future__ import annotations

import base64
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from collections.abc import Iterable
from typing import IO
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
_ARRAY_DTYPES = {"<f4", "<u4", "<u2", "|u1"}
INPUT_CONSTANT_CURRENT = "constant_current"
INPUT_SPIKES = "spike_input"
MAX_NEURONS = 512
I_OFFSET_LIMIT = 1022
WEIGHT_LIMIT = 63

Progress = Callable[[str], None]


# ---- pure helpers (unit-tested from the NeuroSeq test suite) ---------------


def encode_array(array: Any, dtype: str) -> dict[str, Any]:
    if dtype not in _ARRAY_DTYPES:
        raise ValueError(f"Unsupported array dtype {dtype!r}")
    data = np.ascontiguousarray(np.asarray(array).astype(np.dtype(dtype)))
    return {
        "dtype": dtype,
        "shape": list(data.shape),
        "data": base64.b64encode(data.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = str(payload["dtype"])
    if dtype not in _ARRAY_DTYPES:
        raise ValueError(f"Unsupported array dtype {dtype!r}")
    shape = tuple(int(n) for n in payload["shape"])
    raw = base64.b64decode(payload["data"], validate=True)
    item = np.dtype(dtype)
    if any(n < 0 for n in shape) or len(raw) != int(np.prod(shape)) * item.itemsize:
        raise ValueError("Array payload size does not match its shape")
    return np.frombuffer(raw, dtype=item).reshape(shape).copy()


def parse_run_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``run`` request and decode its arrays."""
    if int(arguments.get("protocol", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")
    n = int(arguments["n_neurons"])
    if not 1 <= n <= MAX_NEURONS:
        raise ValueError(f"n_neurons must be in 1..{MAX_NEURONS}")
    duration_ms = float(arguments["duration_ms"])
    if not duration_ms > 0:
        raise ValueError("duration_ms must be > 0")
    record = int(arguments.get("record_v_neuron", 0))
    if not 0 <= record < n:
        raise ValueError("record_v_neuron is not a driven neuron")
    spec: dict[str, Any] = {
        "mode": arguments["mode"],
        "n_neurons": n,
        "duration_ms": duration_ms,
        "record_v_neuron": record,
        "neuron_parameters": dict(arguments.get("neuron_parameters") or {}),
    }
    if spec["mode"] == INPUT_CONSTANT_CURRENT:
        segments = arguments["segments"]
        durations = decode_array(segments["durations_ms"]).astype(np.float64)
        enable = decode_array(segments["enable"]).astype(bool)
        offset = decode_array(segments["offset"]).astype(np.int64)
        if durations.ndim != 1 or durations.size == 0:
            raise ValueError("At least one segment is required")
        if enable.shape != (durations.size, n) or offset.shape != enable.shape:
            raise ValueError("Segment arrays do not match (segments, n_neurons)")
        if np.any(durations <= 0):
            raise ValueError("Segment durations must be > 0")
        if np.any(offset < 0) or np.any(offset > I_OFFSET_LIMIT):
            raise ValueError(f"i_offset must be in 0..{I_OFFSET_LIMIT}")
        spec.update(durations_ms=durations, enable=enable, offset=offset)
    elif spec["mode"] == INPUT_SPIKES:
        spikes = arguments["spikes"]
        indices = decode_array(spikes["indices"]).astype(np.int64)
        times = decode_array(spikes["times_ms"]).astype(np.float64)
        if indices.shape != times.shape or indices.ndim != 1:
            raise ValueError("Spike indices and times differ in shape")
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Spike index outside the driven neurons")
        weight = int(spikes.get("weight", WEIGHT_LIMIT))
        if not 0 <= weight <= WEIGHT_LIMIT:
            raise ValueError(f"weight must be in 0..{WEIGHT_LIMIT}")
        spec.update(
            spike_trains=spike_trains(indices, times, n),
            weight=weight,
        )
    else:
        raise ValueError(f"Unknown input mode {spec['mode']!r}")
    return spec


def spike_trains(
    indices: np.ndarray, times_ms: np.ndarray, n: int
) -> list[list[float]]:
    """One sorted spike-time list (ms) per neuron."""
    trains: list[list[float]] = [[] for _ in range(n)]
    for index, time in sorted(zip(indices.tolist(), times_ms.tolist(), strict=True)):
        trains[index].append(float(time))
    return trains


def configure_quiggeldy(
    arguments: dict[str, Any], environ: dict[str, str] | Any
) -> list[str]:
    """Set the QUIGGELDY_* variables hxcomm reads; return any warnings.

    The token is only ever taken from the environment.
    """
    environ["QUIGGELDY_ENABLED"] = "1"
    user = str(arguments.get("user") or environ.get("QUIGGELDY_USER_NO_MUNGE") or "")
    if user:
        environ["QUIGGELDY_USER_NO_MUNGE"] = user
    host = str(arguments.get("host") or "")
    if host:
        environ["QUIGGELDY_IP"] = host
    if arguments.get("port"):
        environ["QUIGGELDY_PORT"] = str(int(arguments["port"]))
    warnings = []
    if not environ.get("QUIGGELDY_TOKEN"):
        warnings.append(
            "No QUIGGELDY_TOKEN set (fine for allow-listed IPs; needed for "
            "token-based access)."
        )
    if not environ.get("QUIGGELDY_IP"):
        warnings.append("No quiggeldy host configured.")
    return warnings


_REPORTED_VARIABLES = (
    "QUIGGELDY_ENABLED",
    "QUIGGELDY_IP",
    "QUIGGELDY_PORT",
    "QUIGGELDY_USER_NO_MUNGE",
    "QUIGGELDY_TOKEN",
    "SYSTEMNAME",
    "SPACK_ENV",
)
#: Distribution name -> import name (the BSS-2 stack ships without metadata).
_REPORTED_PACKAGES = {
    "pynn-brainscales": "pynn_brainscales",
    "dlens-vx-v3": "dlens_vx_v3",
    "PyNN": "pyNN",
    "neo": "neo",
    "numpy": "numpy",
}


def environment_report(environ: dict[str, str] | Any) -> dict[str, str]:
    """Debug view of the variables hxcomm uses; the token is never revealed."""
    report = {}
    for name in _REPORTED_VARIABLES:
        value = environ.get(name)
        if name == "QUIGGELDY_TOKEN":
            # A separate key, so log scrubbing of "QUIGGELDY_TOKEN=..." cannot
            # hide whether the token arrived.
            if value is None:
                status = "unset"
            elif not value:
                status = "empty"
            else:
                status = f"set ({len(value)} characters)"
            report["QUIGGELDY_TOKEN_STATUS"] = status
        else:
            report[name] = "<unset>" if value is None else value
    return report


def package_versions() -> dict[str, str]:
    """Installed versions of the BSS-2 stack, without importing it."""
    versions = {}
    for name, module in _REPORTED_PACKAGES.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = (
                "importable (no version metadata)"
                if _has_module(module)
                else "not installed"
            )
        except Exception as exc:  # broken metadata must not stop the helper
            versions[name] = f"unknown ({type(exc).__name__})"
    return versions


def _magnitude(quantity: Any, unit: str) -> np.ndarray:
    """Neo/quantities value in ``unit`` (plain arrays pass through)."""
    if hasattr(quantity, "rescale"):
        quantity = quantity.rescale(unit)
    return np.asarray(getattr(quantity, "magnitude", quantity), dtype=np.float64)


def collect_spikes(spiketrains: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Neo spike trains into (neuron index, time in ms) arrays."""
    indices: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for position, train in enumerate(spiketrains):
        annotations = getattr(train, "annotations", {}) or {}
        index = int(annotations.get("source_index", position))
        values = _magnitude(train, "ms").ravel()
        indices.append(np.full(values.size, index, dtype=np.uint32))
        times.append(values)
    if not indices:
        return np.zeros(0, np.uint32), np.zeros(0, np.float64)
    return np.concatenate(indices), np.concatenate(times)


def collect_signal(signal: Any) -> tuple[np.ndarray, np.ndarray]:
    """(times in ms, values) of a one-channel Neo (irregularly sampled) signal."""
    values = np.asarray(getattr(signal, "magnitude", signal), dtype=np.float64)
    if values.ndim == 2:
        values = values[:, 0]
    times = _magnitude(signal.times, "ms").ravel()
    if times.shape != values.shape:
        raise ValueError("Membrane recording times and values differ in length")
    return times, values


# ---- hardware access (exercised in tests with stubbed modules) -------------


def first_identifier(identifier: Any) -> str:
    """``get_unique_identifier()`` returns a list of identifiers; use the first."""
    if isinstance(identifier, (list, tuple)):
        return str(identifier[0]) if identifier else ""
    return str(identifier)


def query_identifier(progress: Progress) -> str:
    """Open a managed (quiggeldy) connection and ask for the chip identifier."""
    from dlens_vx_v3 import hxcomm

    progress(
        "Opening hxcomm connection to "
        f"{os.environ.get('QUIGGELDY_IP', '?')}:{os.environ.get('QUIGGELDY_PORT', '?')}"
    )
    with hxcomm.ManagedConnection() as connection:
        return first_identifier(connection.get_unique_identifier())


#: Fallback used by the EBRAINS demos when ``stable/latest`` is unavailable.
EBRAINS_CALIBRATION_URL = (
    "https://openproject.bioai.eu/data_calibration/hicann-dls-sr-hx/"
    "{identifier}/stable/ebrains-stable/spiking_cocolist.pbin"
)


def load_calibration(pynn: Any, identifier: str | None, progress: Progress) -> Any:
    """The chip's nightly calibration as a ``lola.Chip``.

    ``pynn.helper.chip_from_nightly()`` reads the local copy or downloads
    ``stable/latest``; if that fails, the EBRAINS demos' ``ebrains-stable``
    copy is downloaded for ``identifier``.
    """
    progress("Loading the nightly calibration of the chip")
    try:
        return pynn.helper.chip_from_nightly()
    except Exception as exc:
        if not identifier:
            raise
        url = EBRAINS_CALIBRATION_URL.format(identifier=identifier)
        progress(f"chip_from_nightly failed ({exc}); downloading {url}")
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        return pynn.helper.chip_from_portable_binary(data)


def execute_run(
    spec: dict[str, Any],
    progress: Progress,
    identifier: str | None = None,
    calibration_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the network, deliver the input, and read back spikes and ``v``.

    Neurons are ``HXNeuron``s configured from the chip's nightly calibration
    (as in the BrainScaleS-2 demos): only ``HXNeuron`` exposes
    ``constant_current_enable``/``constant_current_i_offset``.
    ``neuron_parameters`` are optional ``HXNeuron`` overrides in hardware units.
    """
    progress("Importing pynn_brainscales")
    import pynn_brainscales.brainscales2 as pynn

    n = spec["n_neurons"]
    k = spec["record_v_neuron"]
    cache = calibration_cache if calibration_cache is not None else {}
    if "chip" not in cache:
        cache["chip"] = load_calibration(pynn, identifier, progress)
    progress("Setting up the chip")
    pynn.setup(initial_config=cache["chip"])
    try:
        pop = pynn.Population(n, pynn.cells.HXNeuron(**spec["neuron_parameters"]))
        pop.record(["spikes"])
        # The MADC digitises one membrane at a time.
        recorded = pop[k : k + 1]
        recorded.record(["v"])
        if spec["mode"] == INPUT_CONSTANT_CURRENT:
            durations = spec["durations_ms"]
            progress(f"Executing {durations.size} constant-current segment(s)")
            for i, duration in enumerate(durations):
                pop.set(
                    constant_current_enable=[bool(v) for v in spec["enable"][i]],
                    constant_current_i_offset=[int(v) for v in spec["offset"][i]],
                )
                last = i == durations.size - 1
                command = pynn.RunCommand.EXECUTE if last else pynn.RunCommand.APPEND
                pynn.run(float(duration), command)
        else:
            trains = spec["spike_trains"]
            try:
                from pyNN.parameters import Sequence

                times: list[Any] = [Sequence(t) for t in trains]
            except ImportError:
                times = trains
            source = pynn.Population(n, pynn.cells.SpikeSourceArray(spike_times=times))
            pynn.Projection(
                source,
                pop,
                pynn.OneToOneConnector(),
                synapse_type=pynn.synapses.StaticSynapse(weight=spec["weight"]),
                receptor_type="excitatory",
            )
            progress("Executing the spike-input run")
            pynn.run(spec["duration_ms"])
        progress("Reading back spikes and membrane voltage")
        spike_segment = pop.get_data("spikes").segments[-1]
        indices, spike_times = collect_spikes(spike_segment.spiketrains)
        v_segment = recorded.get_data("v").segments[-1]
        signals = list(getattr(v_segment, "irregularlysampledsignals", [])) or list(
            getattr(v_segment, "analogsignals", [])
        )
        v_result = None
        if signals:
            v_times, v_values = collect_signal(signals[0])
            v_result = {
                "neuron": k,
                "times_ms": encode_array(v_times, "<f4"),
                "values": encode_array(v_values, "<f4"),
            }
    finally:
        pynn.end()
    return {
        "duration_ms": spec["duration_ms"],
        "spikes": {
            "indices": encode_array(indices, "<u4"),
            "times_ms": encode_array(spike_times, "<f4"),
        },
        "v": v_result,
    }


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return name in sys.modules


# ---- dispatch ----------------------------------------------------------------


class Helper:
    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.emit = emit
        self.identifier: str | None = None
        #: The nightly calibration is loaded once per helper session.
        self.calibration: dict[str, Any] = {}

    def dispatch(self, request_id: Any, command: str, arguments: dict) -> dict:
        def progress(message: str) -> None:
            self.emit({"id": request_id, "event": "progress", "message": message})

        if command == "ping":
            return {
                "pong": True,
                "protocol": PROTOCOL_VERSION,
                "pynn_brainscales": _has_module("pynn_brainscales"),
            }
        if command == "connect":
            for warning in configure_quiggeldy(arguments, os.environ):
                progress(warning)
            report = environment_report(os.environ)
            progress(
                "Connection environment: "
                + ", ".join(f"{key}={value}" for key, value in report.items())
            )
            self.identifier = query_identifier(progress)
            expected = str(arguments.get("identifier") or "")
            return {
                "identifier": self.identifier,
                "matches_expected": not expected or expected == self.identifier,
            }
        if command == "run":
            spec = parse_run_arguments(arguments)
            result = execute_run(spec, progress, self.identifier, self.calibration)
            result["identifier"] = self.identifier
            return result
        if command == "shutdown":
            return {"bye": True}
        raise ValueError(f"Unknown command {command!r}")


def serve(lines: Iterable[str], out: IO[str]) -> None:
    """Answer requests from ``lines`` until ``shutdown`` or end of input."""

    def emit(message: dict[str, Any]) -> None:
        out.write(json.dumps(message) + "\n")
        out.flush()

    emit(
        {
            "id": None,
            "event": "ready",
            "versions": {
                "protocol": PROTOCOL_VERSION,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "debug": {
                "python_executable": sys.executable,
                "hostname": platform.node(),
                "platform": platform.platform(),
                "packages": package_versions(),
                "environment": environment_report(os.environ),
            },
        }
    )
    helper = Helper(emit)
    for line in lines:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("Request must be an object")
            request = parsed
            result = helper.dispatch(
                request.get("id"),
                str(request["command"]),
                request.get("arguments") or {},
            )
            emit({"id": request.get("id"), "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        if request.get("command") == "shutdown":
            break


def main() -> None:  # pragma: no cover - needs real file descriptors
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(sys.stdin, protocol)


if __name__ == "__main__":  # pragma: no cover
    main()
