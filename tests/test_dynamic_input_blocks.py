from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq.dynamic_input_blocks import RoutineArrayCursor
from neuroseq.dynamic_input_blocks import run_to_sample


class _Editor:
    """A minimal stand-in for StimEditor's block-placement surface.

    pad_stim_array_to grows ``stim.values`` with zero rows exactly like
    StimEditor.pad_stim_array_to() does (a no-op if already long enough);
    apply_pattern writes into the existing array like StimEditor.apply_pattern();
    focus_time_range just records what it was asked to show.
    """

    def __init__(self, stim: SimpleNamespace) -> None:
        self.stim = stim
        self.pad_calls: list[float] = []
        self.focus_calls: list[tuple[float, float]] = []
        self.apply_calls: list[slice] = []

    def apply_pattern(self, _neurons: slice, times: slice, values: np.ndarray) -> None:
        self.apply_calls.append(times)
        self.stim.values[times.start : times.stop, :] = values

    def reset_stim_array(self) -> None:
        raise AssertionError("RoutineArrayCursor must not call reset_stim_array")

    def pad_stim_array_to(self, total_length_secs: float) -> None:
        self.pad_calls.append(total_length_secs)
        current_secs = self.stim.dt * self.stim.values.shape[0]
        if total_length_secs <= current_secs:
            return
        n_pad = int(round((total_length_secs - current_secs) / self.stim.dt))
        self.stim.values = np.pad(self.stim.values, ((0, n_pad), (0, 0)))

    def focus_time_range(self, start_seconds: float, stop_seconds: float) -> None:
        self.focus_calls.append((start_seconds, stop_seconds))


def _stim(n_samples: int = 0, dt: float = 0.001, n_neurons: int = 2) -> SimpleNamespace:
    return SimpleNamespace(values=np.zeros((n_samples, n_neurons)), dt=dt)


def test_run_to_sample_has_one_dt_floor() -> None:
    stim = SimpleNamespace(dt=0.001)
    project = SimpleNamespace(b2_model=SimpleNamespace(get_sim_time_secs=lambda: 0.01))
    assert run_to_sample(project, stim, 10) == 0.001  # type: ignore[arg-type]
    assert run_to_sample(project, stim, 25) == pytest.approx(0.015)  # type: ignore[arg-type]


def test_place_block_pads_ahead_by_ten_blocks_and_focuses_camera() -> None:
    stim = _stim()
    editor = _Editor(stim)
    cursor = RoutineArrayCursor(
        start_sample=0, initial_block_duration_estimate_seconds=0.003
    )
    block = np.ones((3, 2))

    start = cursor.place_block(editor, stim, block)  # type: ignore[arg-type]

    assert start == 0
    # 10 blocks of the initial 3 ms estimate = 30 ms = 30 samples at dt=1 ms.
    assert editor.pad_calls == [0.03]
    assert stim.values.shape[0] == 30
    assert editor.focus_calls == [(0.0, 0.03)]
    assert editor.apply_calls == [slice(0, 3)]
    assert np.all(stim.values[:3] == 1.0)
    assert cursor.sample == 3


def test_place_block_does_not_repad_or_refocus_while_room_remains() -> None:
    stim = _stim()
    editor = _Editor(stim)
    cursor = RoutineArrayCursor(
        start_sample=0, initial_block_duration_estimate_seconds=0.003
    )
    cursor.place_block(editor, stim, np.ones((3, 2)))  # type: ignore[arg-type]

    start = cursor.place_block(editor, stim, np.full((3, 2), 2.0))  # type: ignore[arg-type]

    assert start == 3
    # No second pad or camera move: the block still fits the padded window.
    assert editor.pad_calls == [0.03]
    assert editor.focus_calls == [(0.0, 0.03)]
    assert cursor.sample == 6


def test_place_block_repads_and_jumps_camera_when_block_overflows() -> None:
    stim = _stim()
    editor = _Editor(stim)
    cursor = RoutineArrayCursor(
        start_sample=0, initial_block_duration_estimate_seconds=0.003
    )
    # Consume the whole padded window (30 samples) with ten 3-sample blocks.
    for _ in range(10):
        cursor.place_block(editor, stim, np.ones((3, 2)))  # type: ignore[arg-type]
    assert cursor.sample == 30
    assert len(editor.pad_calls) == 1

    start = cursor.place_block(editor, stim, np.ones((3, 2)))  # type: ignore[arg-type]

    assert start == 30
    # A second pad, sized the same way (10 blocks) but anchored at the new
    # block's own position, plus a camera jump to that new window.
    assert len(editor.pad_calls) == 2
    assert editor.pad_calls[-1] == pytest.approx(0.06)
    assert len(editor.focus_calls) == 2
    assert editor.focus_calls[-1] == pytest.approx((0.03, 0.06))


def test_duration_estimate_switches_to_running_average_of_real_blocks() -> None:
    stim = _stim()
    editor = _Editor(stim)
    # A deliberately wrong initial guess (1 ms) - only used for the very
    # first pad, before any real block exists.
    cursor = RoutineArrayCursor(
        start_sample=0, initial_block_duration_estimate_seconds=0.001
    )
    cursor.place_block(editor, stim, np.ones((5, 2)))  # type: ignore[arg-type]
    assert editor.pad_calls == [0.01]  # 10 * 1 ms initial guess -> 10 ms padded.

    # The padded window (10 ms) exactly fits one more 5 ms block with no
    # room to spare, so the third block forces a re-pad.
    cursor.place_block(editor, stim, np.ones((5, 2)))  # type: ignore[arg-type]
    start = cursor.place_block(editor, stim, np.ones((5, 2)))  # type: ignore[arg-type]

    assert start == 10
    # Average of the two real 5 ms blocks placed so far (0.005 s) - not the
    # 1 ms initial guess, and not blended with it - times 10, anchored at
    # this block's own position (10 samples = 10 ms).
    assert editor.pad_calls[-1] == pytest.approx(0.01 + 10 * 0.005)
