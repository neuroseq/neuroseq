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


def append_block(editor: BlockEditor, stim: StimArray, block: np.ndarray) -> int:
    """Trim unsimulated padding and append ``block`` at the true array head."""
    editor.reset_stim_array()
    start = stim.values.shape[0]
    editor.apply_pattern(
        slice(0, stim.values.shape[1]), slice(start, start + block.shape[0]), block
    )
    return start


def run_to_sample(project: Project, stim: StimArray, stop_sample: int) -> float:
    """Return the forward simulation duration needed to reach ``stop_sample``."""
    current_time = project.b2_model.get_sim_time_secs()
    return max(stim.dt, stop_sample * stim.dt - current_time)
