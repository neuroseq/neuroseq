import copy

import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _build_sinus_offset_pattern(
    tfinal_ms: int,
    n_units: int = 5,
    slot_ms: int = 100,
    active_offset_mv: float = 3.0,
) -> np.ndarray:
    """Piecewise offset pattern used by the sinus-offset external simulator.

    The pattern is zero most of the time, with rolling per-unit positive
    offsets so the additive stimulus contribution is easy to inspect.
    """
    stim = np.zeros((tfinal_ms, n_units), dtype=np.float32)
    period_ms = n_units * slot_ms
    for cycle_start in range(0, tfinal_ms, period_ms):
        for unit in range(n_units):
            i_start = cycle_start + unit * slot_ms
            i_end = min(i_start + slot_ms, tfinal_ms)
            stim[i_start:i_end, unit] = active_offset_mv
    return stim


@register_example_project("Sinus Offset External Simulator", section="Network")
def sinus_offset_project() -> Project:
    """Five-unit sinus-wave simulator with additive stimulation offsets.

    Uses ``neuroseq.example_models.sinus_offset`` through the external
    simulation interface so model parameters can be edited from the GUI.
    """
    from neuroseq.example_models.sinus_offset import DEFAULT_PARAMS

    project = Project()
    project.external_simulate_path = "neuroseq:sinus_offset"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_NN()

    tfinal_ms = 4000
    stim = _build_sinus_offset_pattern(tfinal_ms=tfinal_ms, n_units=5)
    project.add_stim_array(
        StimArray(
            name="offset",
            stim_type=StimType.External,
            values=stim,
            units="mV",
            dt=0.001,
        )
    )
    return project
