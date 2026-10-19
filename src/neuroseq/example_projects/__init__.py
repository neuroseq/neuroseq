"""Example projects grouped into neuron models and networks."""

# Import order preserves the existing example-project menu order.
# isort: off
from neuroseq.example_projects._registry import ExampleProjectGenerator
from neuroseq.example_projects._registry import example_projects
from neuroseq.example_projects._registry import register_example_project
from neuroseq.example_projects.networks.default_project import default_project
from neuroseq.example_projects.networks.getting_started_project import (
    getting_started_project,
)
from neuroseq.example_projects.networks.getting_started_project import (
    _getting_started_cellpose_data as _getting_started_cellpose_data,
)
from neuroseq.example_projects.networks.non_linear import non_linear
from neuroseq.example_projects.neurons.single_hh_neuron import single_hh_neuron
from neuroseq.example_projects.neurons.single_adex_neuron import single_adex_neuron
from neuroseq.example_projects.neurons.single_lif_neuron_many_stims import (
    single_lif_neuron_many_stims,
)
from neuroseq.example_projects.networks.hh_neuron_with_synapses import (
    hh_neuron_with_synapses,
)
from neuroseq.example_projects.networks.hh_neuron_with_synthetic_synapses import (
    hh_neuron_with_synthetic_synapses,
)
from neuroseq.example_projects.networks.hh_neurons_ei_model import hh_neurons_ei_model
from neuroseq.example_projects.neurons.single_lif_neuron import single_lif_neuron
from neuroseq.example_projects.neurons.single_lif_neuron_without_stim import (
    single_lif_neuron_without_stim,
)
from neuroseq.example_projects.networks.hh_neurons_with_plasticity import (
    hh_neurons_with_plasticity,
)
from neuroseq.example_projects.networks.two_hh_neurons_with_plasticity import (
    two_hh_neurons_with_plasticity,
)
from neuroseq.example_projects.networks.locally_connected_network import (
    locally_connected_network,
)
from neuroseq.example_projects.neurons.pyrho_6state_chr2_project import (
    pyrho_6state_chr2_project,
)
from neuroseq.example_projects.networks.elise_project import elise_project
from neuroseq.example_projects.networks.sinus_offset_project import sinus_offset_project
from neuroseq.example_projects.networks.sinus_offset_project import (
    _build_sinus_offset_pattern as _build_sinus_offset_pattern,
)
from neuroseq.example_projects.networks.pynn_demo_project import pynn_demo_project
from neuroseq.example_projects.networks.pynn_demo_project import (
    _build_pynn_stim_pattern as _build_pynn_stim_pattern,
)
# isort: on

# Public alias retained for existing scripts.
hh_neuron_with_plasticity = hh_neurons_with_plasticity

__all__ = [
    "ExampleProjectGenerator",
    "example_projects",
    "register_example_project",
    "default_project",
    "getting_started_project",
    "non_linear",
    "single_hh_neuron",
    "single_adex_neuron",
    "single_lif_neuron_many_stims",
    "hh_neuron_with_synapses",
    "hh_neuron_with_synthetic_synapses",
    "hh_neurons_ei_model",
    "single_lif_neuron",
    "single_lif_neuron_without_stim",
    "hh_neurons_with_plasticity",
    "two_hh_neurons_with_plasticity",
    "locally_connected_network",
    "pyrho_6state_chr2_project",
    "elise_project",
    "sinus_offset_project",
    "pynn_demo_project",
    "hh_neuron_with_plasticity",
]
