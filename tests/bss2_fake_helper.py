"""Stand-in for the container: runs the real BSS-2 helper on a fake BSS-2 stack.

Launched by the tests with ``sys.executable`` instead of ``singularity``. The
behaviour is chosen with ``FAKE_BSS2_MODE``:

``ok`` (default), ``exit`` (dies before start-up), ``fatal`` (Singularity-style
FATAL message and exit code 255), ``silent`` (never ready),
``connect_error``, ``run_error``, ``crash_on_run`` (dies mid-run),
``slow_run`` (run never returns).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "neuroseq"
    / "resources"
    / "bss2"
    / "neuroseq_bss2_helper.py"
)
MODE = os.environ.get("FAKE_BSS2_MODE", "ok")


class _Train:
    def __init__(self, times, index):
        self.magnitude = np.asarray(times, dtype=float)
        self.annotations = {"source_index": index}


class _Signal:
    def __init__(self, times, values):
        self.magnitude = np.asarray(values, dtype=float)[:, None]
        self.times = np.asarray(times, dtype=float)


class _Population:
    def __init__(self, stack, n):
        self.stack = stack
        self.n = n

    def record(self, _variables):
        pass

    def set(self, **_kwargs):
        pass

    def __getitem__(self, _item):
        return self

    def get_data(self, variable):
        if variable == "spikes":
            trains = [_Train([0.1 * (i + 1)], i) for i in range(self.n)]
            segment = types.SimpleNamespace(spiketrains=trains)
        else:
            t = self.stack.elapsed_ms
            signal = _Signal([0.0, t / 2, t], [1.0, 2.0, 3.0])
            segment = types.SimpleNamespace(irregularlysampledsignals=[signal])
        return types.SimpleNamespace(segments=[segment])


class _Pynn(types.ModuleType):
    RunCommand = types.SimpleNamespace(APPEND="append", EXECUTE="execute")

    def __init__(self):
        super().__init__("pynn_brainscales.brainscales2")
        self.elapsed_ms = 0.0
        self.cells = types.SimpleNamespace(
            CalibHXNeuronCoba=lambda **_kw: "neuron",
            SpikeSourceArray=lambda spike_times: "source",
        )
        self.synapses = types.SimpleNamespace(StaticSynapse=lambda weight: weight)

    def setup(self):
        self.elapsed_ms = 0.0

    def end(self):
        pass

    def Population(self, n, _cell_type):
        return _Population(self, n)

    def OneToOneConnector(self):
        return None

    def Projection(self, *_args, **_kwargs):
        pass

    def run(self, duration, _command=None):
        if MODE == "run_error":
            raise RuntimeError("fake chip failure")
        if MODE == "crash_on_run":
            sys.stderr.write("fatal: segmentation fault\n")
            sys.stderr.flush()
            os._exit(5)
        if MODE == "slow_run":
            time.sleep(60)
        self.elapsed_ms += float(duration)


class _ManagedConnection:
    def __enter__(self):
        if MODE == "connect_error":
            raise ConnectionError("quiggeldy refused the token")
        return self

    def __exit__(self, *_exc):
        return False

    def get_unique_identifier(self):
        has_token = "with-token" if os.environ.get("QUIGGELDY_TOKEN") else "no-token"
        return f"fakechip_{os.environ.get('QUIGGELDY_IP')}_{has_token}"


def main() -> None:
    if MODE == "exit":
        sys.exit(3)
    if MODE == "fatal":
        sys.stderr.write("FATAL:   could not open image /containers/stable/latest\n")
        sys.stderr.flush()
        sys.exit(255)
    if MODE == "silent":
        time.sleep(60)
    # Like spack, chatter on both streams before the protocol starts; the
    # token must be scrubbed from anything NeuroSeq shows.
    print("activating environment esd", flush=True)
    sys.stderr.write(f"debug QUIGGELDY_TOKEN={os.environ.get('QUIGGELDY_TOKEN')}\n")
    sys.stderr.write(f"token echo: {os.environ.get('QUIGGELDY_TOKEN')}\n")
    sys.stderr.flush()
    pynn = _Pynn()
    package = types.ModuleType("pynn_brainscales")
    package.brainscales2 = pynn  # type: ignore[attr-defined]
    sys.modules["pynn_brainscales"] = package
    sys.modules["pynn_brainscales.brainscales2"] = pynn
    dlens = types.ModuleType("dlens_vx_v3")
    dlens.hxcomm = types.SimpleNamespace(ManagedConnection=_ManagedConnection)  # type: ignore[attr-defined]
    sys.modules["dlens_vx_v3"] = dlens
    spec = importlib.util.spec_from_file_location("neuroseq_bss2_helper", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
