"""One-line descriptions for Brian2 model fields and common parameters.

These populate the "Description" column of the model editor. Two registries:

* ``STRUCTURAL_FIELD_DESCRIPTIONS`` — the fixed set of editor fields that every
  neurongroup / synapse / monitor exposes (same across all projects).
* ``COMMON_PARAM_DESCRIPTIONS`` — the standard parameter names reused across the
  built-in example projects (Hodgkin-Huxley, LIF, optogenetics, ...).

A model may override/extend these per-parameter via
``B2Model.parameter_descriptions`` (persisted in the ``.nseq`` archive), which
takes precedence over ``COMMON_PARAM_DESCRIPTIONS``.
"""

from __future__ import annotations

# Fields exposed by neurongroups, synapses, subgroups and monitors in the editor.
STRUCTURAL_FIELD_DESCRIPTIONS: dict[str, str] = {
    # neurongroup
    "neuronnumber": "Number of neurons in the group.",
    "model": "Differential/algebraic equations governing the group or synapse.",
    "th": "Threshold condition; the neuron spikes when it evaluates true.",
    "reset": "Statements applied to a neuron immediately after it spikes.",
    "refractory": "Refractory period (or condition) during which the neuron cannot spike.",
    "method": "Numerical integration method for the equations (e.g. euler, exact).",
    "dt": "Integration/recording time step for this element.",
    # subgroup
    "target": "Name of the group/neurongroup this element acts on.",
    "idxs": (
        "Neuron index range as (start, stop): a half-open interval selecting "
        "neurons start..stop-1 of the target group. E.g. (0, 8) is the first 8 "
        "neurons; (8, 10) is neurons 8 and 9. A single int n means just neuron n."
    ),
    # synapse
    "origin": "Presynaptic (source) neurongroup.",
    "on_pre": "Statements executed on the arrival of a presynaptic spike.",
    "on_post": "Statements executed when the postsynaptic neuron spikes.",
    "connection_condition": "Boolean expression selecting which pairs are connected.",
    "p": "Connection probability for each candidate pair (0-1) or an expression.",
    "connection_i": "Explicit presynaptic source indices (overrides the condition).",
    "connection_j": "Explicit postsynaptic target indices (overrides the condition).",
    "connection_n": "Number of synapses created per connected pair (multapses).",
    "skip_if_invalid": "Silently drop out-of-range indices instead of raising.",
    "delay": "Transmission delay applied to the presynaptic pathway.",
    # monitor
    "type": "Monitor type: StateMonitor (records variables) or SpikeMonitor (records spikes).",
    "source": "Network element (neurongroup or synapse) being recorded.",
    "variable": "Which variable of the source to record.",
}

# Standard parameter names shared across the built-in example projects.
COMMON_PARAM_DESCRIPTIONS: dict[str, str] = {
    # membrane / passive properties
    "area": "Membrane surface area of the neuron.",
    "Cm": "Membrane capacitance.",
    "gl": "Leak (resting) conductance.",
    "El": "Leak/resting reversal potential.",
    "E_m": "Resting membrane potential.",
    "R_m": "Membrane resistance.",
    "Rm": "Membrane resistance.",
    "tau": "Membrane time constant.",
    "tau_m": "Membrane time constant.",
    "v0": "Initial/resting membrane potential.",
    "v1": "Secondary reference membrane potential.",
    "v_t0": "Baseline spike threshold potential.",
    "VT": "Spike-threshold parameter of the Hodgkin-Huxley gating kinetics.",
    "t_ref": "Absolute refractory period.",
    "sigma": "Noise amplitude of the membrane potential.",
    "sigma_c": "Noise correlation / colouring parameter.",
    # ionic reversal potentials and conductances (Hodgkin-Huxley)
    "EK": "Potassium reversal potential.",
    "ENa": "Sodium reversal potential.",
    "E": "Reversal potential.",
    "g_na": "Maximal sodium (Na+) conductance.",
    "g_kd": "Maximal delayed-rectifier potassium (K+) conductance.",
    # synaptic reversal potentials, conductances, weights
    "Ee": "Excitatory synaptic reversal potential.",
    "Ei": "Inhibitory synaptic reversal potential.",
    "taue": "Excitatory synaptic conductance decay time constant.",
    "taui": "Inhibitory synaptic conductance decay time constant.",
    "we": "Excitatory synaptic weight (conductance increment per spike).",
    "wi": "Inhibitory synaptic weight (conductance increment per spike).",
    "w_exc": "Excitatory synaptic weight.",
    "w_inh": "Inhibitory synaptic weight.",
    "v_rev_ext": "Excitatory (external) synaptic reversal potential.",
    "v_rev_inh": "Inhibitory synaptic reversal potential.",
    "Rext": "Excitatory (external) input resistance/scaling.",
    "Rinh": "Inhibitory input resistance/scaling.",
    "Iext": "External injected current.",
    # STDP plasticity
    "taupre": "Time constant of the presynaptic STDP trace (Apre).",
    "taupost": "Time constant of the postsynaptic STDP trace (Apost).",
    "gmax": "Maximum (clipped) plastic synaptic weight.",
    "dApre": "Increment added to the pre-trace on a presynaptic spike (LTP step).",
    "dApost": "Increment added to the post-trace on a postsynaptic spike (LTD step).",
    # optogenetics
    "Eeopto": "Excitatory opsin (e.g. ChrimsonR) reversal potential.",
    "Eiopto": "Inhibitory opsin (e.g. GtACR) reversal potential.",
    "g_exc": "Maximal excitatory opsin photocurrent conductance.",
    "g_inh": "Maximal inhibitory opsin photocurrent conductance.",
    "opto_units": "Unit-carrying scale factor for the optogenetic input (power density).",
    "phi_m": "Photon flux at which the opsin activation half-saturates.",
    "phy": "Photon flux / illumination level driving the opsin.",
    # channelrhodopsin (multi-state opsin) kinetics
    "g0": "Single-channel/opsin unit conductance.",
    "gam": "Ratio of open-state 2 to open-state 1 conductance.",
    "Gd1": "Deactivation rate from open state O1.",
    "Gd2": "Deactivation rate from open state O2.",
    "Gf0": "Baseline O1->O2 transition rate.",
    "Gb0": "Baseline O2->O1 transition rate.",
    "Go1": "Light-independent C1->O1 opening rate.",
    "Go2": "Light-independent C2->O2 opening rate.",
    "Gr0": "Recovery (dark) rate from the closed desensitised state.",
    "k1": "Light-dependent activation rate constant (state 1).",
    "k2": "Light-dependent activation rate constant (state 2).",
    "k_f": "Forward O1->O2 rate scaling.",
    "k_b": "Backward O2->O1 rate scaling.",
    "q": "Hill coefficient of the opsin light-activation function.",
    # generic linear/rate-model constants
    "a": "Linear rate constant.",
    "b": "Linear rate constant.",
    "c": "Linear rate constant.",
    "p": "Connection probability or model parameter.",
}


def describe_field(field_name: str) -> str:
    """Return the description for a structural editor field, or ''."""
    return STRUCTURAL_FIELD_DESCRIPTIONS.get(field_name, "")


def describe_parameter(name: str, overrides: dict[str, str] | None = None) -> str:
    """Return the description for a parameter name.

    A per-model ``overrides`` mapping (``B2Model.parameter_descriptions``) takes
    precedence over the shared ``COMMON_PARAM_DESCRIPTIONS`` registry.
    """
    if overrides and name in overrides and overrides[name]:
        return overrides[name]
    return COMMON_PARAM_DESCRIPTIONS.get(name, "")
