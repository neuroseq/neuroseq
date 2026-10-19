import copy

import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def _build_pynn_stim_pattern(
    tfinal_ms: int,
    n_neurons: int = 16,
    slot_ms: int = 100,
    active: float = 1.0,
) -> np.ndarray:
    """Rolling single-neuron stimulation pattern for the PyNN example.

    Each neuron is photo-stimulated in turn for ``slot_ms`` (a travelling
    holographic spot), so the evoked raster shows a clear diagonal sweep that
    repeats every ``n_neurons * slot_ms``.
    """
    stim = np.zeros((tfinal_ms, n_neurons), dtype=np.float32)
    period_ms = n_neurons * slot_ms
    for cycle_start in range(0, tfinal_ms, period_ms):
        for neuron in range(n_neurons):
            i_start = cycle_start + neuron * slot_ms
            i_end = min(i_start + slot_ms, tfinal_ms)
            stim[i_start:i_end, neuron] = active
    return stim


@register_example_project("PyNN AdEx Neuromorphic Network", section="Network")
def pynn_demo_project() -> Project:
    """AdEx spiking network run through PyNN (Brian2 backend by default).

    Uses ``neuroseq.example_models.pynn_demo`` through the external simulation
    interface. A rolling holographic stimulation sweeps the population one
    neuron at a time. The same network description can target BrainScaleS-2
    neuromorphic hardware by switching the ``backend`` parameter (requires the
    BSS-2 software stack; see the module docstring).

    Running this project needs ``pyNN`` installed (``pip install pyNN``); Brian2
    is already a NeuroSeq dependency. Building the project does not import pyNN.
    """
    from neuroseq.example_models.pynn_demo import DEFAULT_PARAMS

    project = Project()
    project.external_simulate_path = "neuroseq:pynn_demo"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_NN()

    n_neurons = int(DEFAULT_PARAMS["n_neurons"]["default"])
    tfinal_ms = 2 * n_neurons * 100  # two full sweeps, exactly periodic
    stim = _build_pynn_stim_pattern(tfinal_ms=tfinal_ms, n_neurons=n_neurons)
    project.add_stim_array(
        StimArray(
            name="opsin",
            stim_type=StimType.Excitation,
            values=stim,
            units="1",  # dimensionless activation drive
            dt=0.001,
        )
    )
    return project
