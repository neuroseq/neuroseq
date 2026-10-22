"""Deliver dynamic input blocks to BrainScaleS-2 neuromorphic hardware.

The BrainScaleS-2 counterpart of the Brian2 simulation host and of
:class:`~neuroseq.device_control.syscon2_stimulation_host.SysCon2StimulationHost`: a
closed-loop routine (rheobase) appends a block to the stim array and asks the host to run
it. Here that means one constant-current run of the block on the chip through the
workspace's :class:`~neuroseq.workspace_devices.Bss2Session`; its spikes (and the membrane
voltage of the probed neuron) are added to the project as a data group, and the routine
measures from them.

Stim neuron ``i`` drives hardware neuron ``i``. Block values are fractions (0-1) of the
BrainScaleS-2 tab's *i_offset max*, the same convention as SysCon2 blocks (fractions of
the target power), so one block's amplitude does not depend on the others.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from qtpy import QtCore

from neuroseq.device_control import bss2_protocol as bp
from neuroseq.device_control.bss2_connection import STATE_CONNECTED

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stimulation.stim import StimArray
    from neuroseq.workspace_devices import Bss2Session

logger = logging.getLogger(__name__)

#: Block value driving the chip's ``i_offset max`` (values are fractions of it).
FULL_SCALE_VALUE = 1.0


@dataclass
class _Run:
    request_id: str
    owner: object
    project: Project
    label: str
    speedup: float
    warnings: list[str]


class Bss2StimulationHost(QtCore.QObject):
    """Runs routine blocks on BrainScaleS-2 and adds each readback to the project.

    Signals (GUI thread), each carrying the ``owner`` passed to :meth:`send_block`:

    - ``block_completed(owner, payload)`` -- the run's data is in the project:
      ``payload["spikes"]`` names the spike dataset, ``payload["trace"]`` the probed
      neuron's membrane voltage (if read back), ``payload["experiment"]`` the data group.
      ``payload["warnings"]`` lists notes about the mapping onto the chip (e.g.
      channels beyond the tab's neuron limit that were not delivered).
    - ``block_failed(owner, str)`` -- the block could not be run or read back.
    - ``progress(str)`` -- a status line for the user.
    - ``experiment_added(str)`` -- a readback data group was added to the project.
    """

    block_completed = QtCore.Signal(object, object)
    block_failed = QtCore.Signal(object, str)
    progress = QtCore.Signal(str)
    experiment_added = QtCore.Signal(str)

    def __init__(
        self,
        session: Bss2Session,
        settings_provider: Callable[[], bp.Bss2Settings],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._settings_provider = settings_provider
        self._run: _Run | None = None
        session.run_finished.connect(self._on_run_finished)
        session.run_failed.connect(self._on_run_failed)
        session.run_progress.connect(self._on_run_progress)

    @property
    def busy(self) -> bool:
        """Whether a block of this host is running."""
        return self._run is not None

    def unavailable(self) -> str | None:
        """Why a block cannot be run now, or ``None``."""
        if self._session.state != STATE_CONNECTED:
            return "Connect to BrainScaleS-2 first (Stimulation setup > BrainScaleS-2)."
        if self._run is not None or self._session.busy:
            return "BrainScaleS-2 is still running another delivery."
        return None

    def current_settings(self) -> bp.Bss2Settings:
        """The BrainScaleS-2 tab's settings now; a routine keeps one snapshot for all
        its blocks, so the current scale cannot change in the middle of a search."""
        return self._settings_provider()

    def send_block(
        self,
        project: Project,
        stim_array: StimArray,
        start_sample: int,
        stop_sample: int,
        label: str = "block",
        *,
        owner: object = None,
        record_neuron: int = 0,
        settings: bp.Bss2Settings | None = None,
    ) -> bool:
        """Run ``stim_array[start:stop]`` as constant current on the chip.

        ``record_neuron`` is the neuron whose membrane voltage the MADC records;
        ``settings`` (default: the tab's current ones) give the scale and speed-up.
        Returns ``False`` (after emitting ``block_failed``) if it could not start.
        """
        reason = self.unavailable()
        if reason is not None:
            self.block_failed.emit(owner, reason)
            return False
        settings = replace(
            settings if settings is not None else self._settings_provider(),
            input_mode=bp.INPUT_CONSTANT_CURRENT,
            # One stim neuron per hardware neuron, and block values as fractions of
            # i_offset max rather than of each block's own maximum.
            n_neurons=0,
            value_at_max=FULL_SCALE_VALUE,
            madc_neuron=int(record_neuron),
        )
        try:
            if not 0 <= start_sample < stop_sample <= stim_array.values.shape[0]:
                raise ValueError(
                    f"Invalid sample range [{start_sample}, {stop_sample})."
                )
            plan = bp.plan_constant_current(
                stim_array.values[start_sample:stop_sample], stim_array.dt, settings
            )
            arguments = bp.build_run_arguments(plan, settings)
        except ValueError as exc:
            self.block_failed.emit(owner, str(exc))
            return False
        request_id = uuid4().hex
        if not self._session.run(request_id, arguments):
            self.block_failed.emit(
                owner, "BrainScaleS-2 is busy or no longer connected."
            )
            return False
        self._run = _Run(
            request_id, owner, project, label, settings.speedup, list(plan.warnings)
        )
        self.progress.emit(
            f"Running a {plan.duration_s:g} s block on BrainScaleS-2 "
            f"({plan.n_segments} segment(s))…"
        )
        return True

    def cancel(self, owner: object = None) -> None:
        """Forget the block in flight; with ``owner``, only if it is that routine's.

        The short hardware run is not interrupted (that would kill the shared helper);
        its result is ignored.
        """
        if self._run is not None and (owner is None or self._run.owner is owner):
            self._run = None

    # ---- session signal handlers ----

    def _on_run_progress(self, message: str) -> None:
        if self._run is not None:
            self.progress.emit(message)

    def _on_run_finished(self, request_id: str, result: object) -> None:
        run = self._run
        if run is None or request_id != run.request_id:
            return
        self._run = None
        try:
            if not isinstance(result, dict):
                raise ValueError("The helper returned no result.")
            experiment, readback = bp.add_readback_experiment(
                run.project, result, run.speedup, run.label, label=f"BSS-2 {run.label}"
            )
        except Exception as exc:
            # Whatever went wrong, the routine must hear about it, or it waits forever.
            logger.exception("Could not add the BrainScaleS-2 readback")
            self.block_failed.emit(
                run.owner, f"The run finished, but its data could not be added: {exc}"
            )
            return
        payload: dict[str, object] = {
            "experiment": experiment.name,
            "spikes": f"{experiment.name} :: {readback.spikes.name}",
            "warnings": run.warnings,
        }
        if readback.trace is not None:
            payload["trace"] = f"{experiment.name} :: {readback.trace.name}"
        self.experiment_added.emit(experiment.name)
        self.progress.emit(
            f"Added '{experiment.name}' ({readback.spikes.spike_indices.size} spike(s))."
        )
        self.block_completed.emit(run.owner, payload)

    def _on_run_failed(self, request_id: str, message: str) -> None:
        run = self._run
        if run is None or request_id != run.request_id:
            return
        self._run = None
        self.block_failed.emit(run.owner, message)
