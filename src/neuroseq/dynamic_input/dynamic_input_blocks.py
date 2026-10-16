from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

import brian2.numpy_ as np

if TYPE_CHECKING:
    from neuroseq.project import Project
    from neuroseq.stim import StimArray


class BlockEditor(Protocol):
    def apply_pattern(
        self, neuron_slice: slice, time_slice: slice, new_values: np.ndarray
    ) -> None: ...

    def reset_stim_array(self) -> None: ...

    def pad_stim_array_to(self, total_length_secs: float) -> None: ...

    def focus_time_range(self, start_seconds: float, stop_seconds: float) -> None: ...


class RoutineArrayCursor:
    """Tracks the live write position for a running dynamic-input routine.

    Blocks are appended contiguously from ``sample`` in the current stim
    array. Before writing, the array is padded ahead of that position by
    roughly ``BLOCKS_TO_PAD`` inputs' worth of (unsimulated) time, so the
    camera can hold a fixed view of a window the user can watch fill up
    with inputs - plus the traces and spikes below - without a resize (and
    therefore a camera jump) on every single block. When a block would
    overflow the padded window, the array is padded again by the same
    amount and the camera jumps right to the new window.

    The pad amount is based on the average length of the blocks placed so
    far this run, falling back to ``initial_block_duration_estimate_seconds``
    before any block exists - routines whose block length is not fixed
    (e.g. an adaptive pairing offset) have no better estimate at that
    point.
    """

    BLOCKS_TO_PAD = 10

    def __init__(
        self, start_sample: int, initial_block_duration_estimate_seconds: float
    ) -> None:
        self.sample = start_sample
        self._padded_until_sample: int | None = None
        self._initial_duration_estimate_seconds = max(
            initial_block_duration_estimate_seconds, 0.0
        )
        self._blocks_seen = 0
        self._total_block_seconds = 0.0

    def _duration_estimate_seconds(self, dt: float) -> float:
        if self._blocks_seen == 0:
            return max(self._initial_duration_estimate_seconds, dt)
        return max(self._total_block_seconds / self._blocks_seen, dt)

    def place_block(
        self, editor: BlockEditor, stim: StimArray, block: np.ndarray
    ) -> int:
        """Pad ahead of the cursor if needed, write ``block``, advance it."""
        start = self.sample
        end = start + block.shape[0]
        dt = float(stim.dt)
        if self._padded_until_sample is None or end > self._padded_until_sample:
            pad_seconds = self.BLOCKS_TO_PAD * self._duration_estimate_seconds(dt)
            editor.pad_stim_array_to(start * dt + pad_seconds)
            self._padded_until_sample = stim.values.shape[0]
            editor.focus_time_range(start * dt, self._padded_until_sample * dt)
        editor.apply_pattern(slice(0, stim.values.shape[1]), slice(start, end), block)
        self.sample = end
        self._total_block_seconds += block.shape[0] * dt
        self._blocks_seen += 1
        return start


def run_to_sample(project: Project, stim: StimArray, stop_sample: int) -> float:
    """Return the forward simulation duration needed to reach ``stop_sample``."""
    current_time = project.b2_model.get_sim_time_secs()
    return max(stim.dt, stop_sample * stim.dt - current_time)
