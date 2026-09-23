from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neuroseq.dynamic_input_blocks import append_block
from neuroseq.dynamic_input_blocks import run_to_sample


class _Editor:
    def __init__(self, stim: SimpleNamespace | None = None, sim_head: int = 0) -> None:
        self.events: list[str] = []
        self.time_slice: slice | None = None
        self.stim = stim
        self.sim_head = sim_head

    def reset_stim_array(self) -> None:
        self.events.append("reset")
        if self.stim is not None:
            self.stim.values = self.stim.values[: self.sim_head]

    def apply_pattern(self, _neurons: slice, times: slice, _values: np.ndarray) -> None:
        self.events.append("apply")
        self.time_slice = times


def test_append_block_trims_before_writing_at_array_head() -> None:
    editor = _Editor()
    stim = SimpleNamespace(values=np.zeros((7, 2)), dt=0.001)
    start = append_block(editor, stim, np.ones((3, 2)))  # type: ignore[arg-type]
    assert start == 7
    assert editor.events == ["reset", "apply"]
    assert editor.time_slice == slice(7, 10)


def test_run_to_sample_has_one_dt_floor() -> None:
    stim = SimpleNamespace(dt=0.001)
    project = SimpleNamespace(b2_model=SimpleNamespace(get_sim_time_secs=lambda: 0.01))
    assert run_to_sample(project, stim, 10) == 0.001  # type: ignore[arg-type]
    assert run_to_sample(project, stim, 25) == pytest.approx(0.015)  # type: ignore[arg-type]


def test_append_block_absorbs_padding_beyond_simulation_head() -> None:
    stim = SimpleNamespace(values=np.zeros((50, 2)), dt=0.001)
    editor = _Editor(stim, sim_head=20)
    start = append_block(editor, stim, np.ones((5, 2)))  # type: ignore[arg-type]
    assert start == 20
    assert editor.time_slice == slice(20, 25)
