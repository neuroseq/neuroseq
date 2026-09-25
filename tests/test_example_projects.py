from __future__ import annotations

import pathlib

import brian2 as b2
import brian2.numpy_ as np
import pytest

import neuroseq.example_projects as example_projects
from neuroseq.example_projects import default_project
from neuroseq.example_projects import getting_started_project
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions

project_generators = [
    pytest.param(project_generator, id=f"{section}/{name}")
    for section, section_projects in example_projects.example_projects.items()
    for name, project_generator in section_projects.items()
]


@pytest.mark.parametrize("project_generator", project_generators)
def test_example_projects(project_generator, tmp_path: pathlib.Path) -> None:
    project = project_generator()
    # do a short simulation
    project.run_simulation(0.001)
    # check we can save and re-load the model
    save_options = SaveLoadOptions(True, True, True, True)
    project.save(tmp_path / "model.nseq", save_options)
    model2 = Project(tmp_path / "model.nseq")
    # check saved model matches original
    assert list(model2.stim_arrays.keys()) == list(project.stim_arrays.keys())
    assert list(model2.b2_model.parameters.keys()) == list(
        project.b2_model.parameters.keys()
    )
    assert list(model2.b2_model.neurongroups.keys()) == list(
        project.b2_model.neurongroups.keys()
    )
    assert list(model2.b2_model.synapses.keys()) == list(
        project.b2_model.synapses.keys()
    )
    assert list(model2.b2_model.monitors.keys()) == list(
        project.b2_model.monitors.keys()
    )
    assert list(model2.b2_model.initials.keys()) == list(
        project.b2_model.initials.keys()
    )
    assert list(model2.b2_model.functions.keys()) == list(
        project.b2_model.functions.keys()
    )


def test_default_project_uses_10_neurons_and_sequential_stimulus() -> None:
    project = default_project()

    stimulus = project.stim_arrays["stimulus"]
    assert list(project.stim_arrays.keys()) == ["stimulus"]
    assert stimulus.values.shape == (1180, 10)
    assert stimulus.units == "1"
    assert project.b2_model.neurongroups["P"]["neuronnumber"] == 10
    assert project.b2_model.parameters["g_exc"] / b2.nS == pytest.approx(400)
    assert project.b2_model.initials["Cee"]["wee"] == "1*nS"
    assert "g_exc * stimulus(t, i)" in project.b2_model.neurongroups["P"]["model"]
    assert "stimulus_inh" not in project.b2_model.neurongroups["P"]["model"]

    pulse_n = int(round(0.100 / stimulus.dt))
    gap_n = int(round(0.018 / stimulus.dt))
    for i in range(stimulus.values.shape[1]):
        t0 = i * (pulse_n + gap_n)
        assert np.all(stimulus.values[t0 : t0 + pulse_n, i] == 1.0)
        if t0 > 0:
            assert np.all(stimulus.values[t0 - gap_n : t0, i] == 0.0)
        for j in range(stimulus.values.shape[1]):
            if j != i:
                assert np.all(stimulus.values[t0 : t0 + pulse_n, j] == 0.0)


def test_default_project_stimulus_pulses_evoke_spikes() -> None:
    """Regression: g_exc was 0.25 nS (a leftover placeholder, ~1600x weaker
    than every other example project's identically-purposed optogenetic
    drive parameter), too weak to depolarize a neuron past threshold within
    its own 100ms stimulus pulse - the stims did not visibly "bump up"
    activity at all. Every neuron must spike during its own pulse window."""
    project = default_project()
    project.run_simulation(1.18)

    spike_monitor = project.b2_model.network["spike"]
    spike_indices = np.asarray(spike_monitor.i[:])
    spike_times = np.asarray(spike_monitor.t[:] / b2.second)

    stimulus = project.stim_arrays["stimulus"]
    pulse_n = int(round(0.100 / stimulus.dt))
    gap_n = int(round(0.018 / stimulus.dt))
    for i in range(stimulus.values.shape[1]):
        t0 = i * (pulse_n + gap_n)
        window_start = t0 * stimulus.dt
        window_end = window_start + pulse_n * stimulus.dt
        during_own_pulse = (
            (spike_indices == i)
            & (spike_times >= window_start)
            & (spike_times < window_end)
        )
        assert during_own_pulse.any(), f"neuron {i} never spiked during its own pulse"


def test_getting_started_project_bundles_a_cell_segmentation() -> None:
    """The Getting Started example ships with a real 2-photon FOV +
    Cellpose segmentation so its Cell segmentation tab isn't empty, mapped
    to the model's own neuron count rather than every detected cell - but
    plain default_project() (also used as the app's blank "New Project"
    template) must not carry it, or every new project would open with
    someone else's imaging data already attached."""
    assert default_project().cellpose_data is None

    project = getting_started_project()

    cellpose_data = project.cellpose_data
    assert cellpose_data is not None
    assert cellpose_data.segmentation_array is not None
    assert cellpose_data.overlay_image is not None
    assert cellpose_data.image_path.exists()

    n_neurons = project.b2_model.neurongroups["P"]["neuronnumber"]
    mapping = cellpose_data.stimulus_to_cellpose_mapping
    assert mapping is not None
    assert set(mapping.keys()) == set(range(n_neurons))
    # more cells are detected than the model has neurons - only the first
    # n_neurons should be mapped, not every one of them
    unique_ids = np.unique(cellpose_data.segmentation_array)
    n_detected_cells = int(np.sum(unique_ids >= 0))
    assert n_detected_cells > n_neurons
    # max_neurons must follow the mapping, not the full segmentation, or the
    # trace/stim/spike panels grow empty rows above the real neurons
    assert project.max_neurons == n_neurons


def test_hh_stdp_example_is_two_neuron_and_potentiates() -> None:
    project = example_projects.hh_neuron_with_plasticity()
    model = project.b2_model

    # 2-neuron layout: neuron 0 excitatory/pre (Pe), neuron 1 inhibitory/post (Pi)
    assert model.neurongroups["P"]["neuronnumber"] == 2
    for name, sa in project.stim_arrays.items():
        assert sa.values.shape[1] == 2, name

    model.assemble_network()
    assert model.network["Pe"].N == 1
    assert model.network["Pi"].N == 1
    # the plastic Ce synapse wires the single 0 -> 1 pair
    assert len(model.network["Ce"]) == 1

    # the pre-before-post pairing protocol potentiates the plastic weight
    project.run_simulation(project.suggested_simulation_time_seconds())
    we = np.asarray(model.network["synapse"].we).ravel() / float(b2.nS)
    assert we[-1] > we[0]


def _abs_synapse_pairs(model, name):
    """Return (list of (abs_source, target) pairs, autapse_count) for a synapse."""
    synapse = model.network[name]
    start = getattr(synapse.source, "start", 0)
    src = start + np.asarray(synapse.i)
    tgt = np.asarray(synapse.j)
    return list(zip(src.tolist(), tgt.tolist(), strict=True)), int((src == tgt).sum())


def test_hh_stdp_example_has_no_autapses_and_reciprocal_wiring() -> None:
    # The excitatory plastic synapse Ce runs 0 -> 1; the inhibitory Ci provides
    # feedback 1 -> 0. Neither should contain a self-connection (autapse).
    project = example_projects.hh_neuron_with_plasticity()
    model = project.b2_model
    model.assemble_network()

    ce_pairs, ce_autapses = _abs_synapse_pairs(model, "Ce")
    ci_pairs, ci_autapses = _abs_synapse_pairs(model, "Ci")

    assert ce_autapses == 0
    assert ci_autapses == 0
    assert (0, 1) in ce_pairs
    assert (1, 0) in ci_pairs


def test_hh_stdp_example_no_autapses_after_resize() -> None:
    # The gid-based condition must stay autapse-free when the network is resized
    # (subgroups start at non-zero offsets, which broke the old "i != j").
    project = example_projects.hh_neuron_with_plasticity()
    model = project.b2_model
    model.functions["neurongroups"]["P"]["neuronnumber"]["modify"](40)
    model.assemble_network()

    for name in ("Ce", "Ci"):
        _, autapses = _abs_synapse_pairs(model, name)
        assert autapses == 0, f"{name} has {autapses} autapses after resize"
