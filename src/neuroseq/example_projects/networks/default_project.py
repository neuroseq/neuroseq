import brian2 as b2
import brian2.numpy_ as np

from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


def default_project() -> Project:
    """The network used both by the "Getting Started Project" example and,
    bare, as the app's blank "New Project" template (gui.py) - so it must
    stay usable without the Cellpose demo imagery getting_started_project()
    adds on top; a real 2-photon FOV has no business appearing in a project
    the user didn't ask for any imaging data in."""
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 10
    nneurons_exc = 8

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("gl", 10.0 * b2.nsiemens)
    b2_model.define_parameter("gmax", 2.0 * b2.nsiemens)

    b2_model.define_parameter("El", -60 * b2.mV)
    b2_model.define_parameter("EK", -90 * b2.mV)
    b2_model.define_parameter("ENa", 50 * b2.mV)
    b2_model.define_parameter("g_na", 20.0 * b2.usiemens)
    b2_model.define_parameter("g_kd", 6.0 * b2.usiemens)
    b2_model.define_parameter("VT", -63 * b2.mV)
    # Time constants
    b2_model.define_parameter("taue", 5 * b2.ms)
    b2_model.define_parameter("taui", 10 * b2.ms)
    b2_model.define_parameter("taupost", 20 * b2.ms)
    b2_model.define_parameter("taupre", 20 * b2.ms)
    ## Reversal potentials
    b2_model.define_parameter("Ee", 0 * b2.mV)
    b2_model.define_parameter("Ei", -80 * b2.mV)

    ### optogenetics parameters
    b2_model.define_parameter("Eeopto", 0 * b2.mV)
    b2_model.define_parameter(
        "g_exc", (2e-3 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    # b2_model.define_parameter("wee", 6 * b2.nS)
    # b2_model.define_parameter("wii", 67 * b2.nS)
    # b2_model.define_parameter("wei", 6 * b2.nS)
    # b2_model.define_parameter("wie", 67 * b2.nS)

    b2_model.define_parameter("dApost", -0.01 * 1.05 * b2.nsiemens)
    b2_model.define_parameter("dApre", 0.01 * b2.nsiemens)

    eqs = """
dv/dt = (gl*(El-v)+ge*(Ee-v)+geopto*(Eeopto-v)+gi*(Ei-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
geopto = g_exc * stimulus(t, i): siemens
alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    """

    # neuron group
    b2_model.define_neuronnumber("P", nneurons)
    b2_model.define_solver_method("P", "exponential_euler")
    b2_model.define_equation("P", eqs)
    b2_model.define_threshold("P", "v>-20*mV")
    b2_model.define_refractory("P", 3 * b2.ms)

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    # Excitatory -> Excitatory synapses

    ce_eq = """
wee : siemens
dApre/dt = -Apre / taupre : siemens (event-driven)
dApost/dt = -Apost / taupost : siemens (event-driven)
    """
    b2_model.define_synapse_equation("Cee", ce_eq)
    b2_model.define_synapse_connection("Cee", connection_condition="i != j")
    b2_model.define_synapse_origin("Cee", "Pe")
    b2_model.define_synapse_target("Cee", "Pe")

    ce_on_pre = """
ge += wee
Apre += dApre
wee = clip(wee + Apost, 0.0*nS, gmax)
"""
    b2_model.define_synapse_pre("Cee", ce_on_pre)

    ce_on_post = """
Apost += dApost
wee= clip(wee + Apre, 0.0*nS, gmax)
"""
    b2_model.define_synapse_post("Cee", ce_on_post)

    b2_model.define_init("Cee", "wee", "1*nS")

    # Inhibitory -> Excitatory synapses

    b2_model.define_synapse_equation("Cie", "wie = 5*nS : siemens")
    b2_model.define_synapse_connection("Cie", None)
    b2_model.define_synapse_origin("Cie", "Pi")
    b2_model.define_synapse_target("Cie", "Pe")
    b2_model.define_synapse_pre("Cie", "gi+=wie")

    # Excitatory -> Inhibitory synapses

    b2_model.define_synapse_equation("Cei", "wei = 1*nS : siemens")
    b2_model.define_synapse_connection("Cei", None)
    b2_model.define_synapse_origin("Cei", "Pe")
    b2_model.define_synapse_target("Cei", "Pi")
    b2_model.define_synapse_pre("Cei", "ge+=wei")

    # initial conditions
    b2_model.define_init("P", "v", "El")
    # b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    # b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("spike", "P")
    b2_model.define_monitor_type("spike", "SpikeMonitor")
    b2_model.define_monitor_variable("spike", "v")

    b2_model.define_monitor_source("synapse", "Cee")
    b2_model.define_monitor_type("synapse", "StateMonitor")
    b2_model.define_monitor_variable("synapse", "wee")

    # define stim arrays

    tfinal = 1000 * b2.ms

    stim_dt = 0.001
    stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    pulse_n = int(round(0.100 / stim_dt))
    gap_n = int(round(0.018 / stim_dt))
    total_n = nneurons * (pulse_n + gap_n)
    amp = 1.0
    new_len = max(stim.shape[0], total_n)
    stim = np.zeros((new_len, nneurons), dtype=stim.dtype)
    for i in range(nneurons):
        t0 = i * (pulse_n + gap_n)
        stim[t0 : t0 + pulse_n, i] = amp
    stim_array = StimArray(
        name="stimulus",
        stim_type=StimType.Excitation,
        values=stim,
        units="1",
        dt=stim_dt,
    )
    project.add_stim_array(stim_array)
    # Mirrors register_example_project()'s own post-build reset: built up
    # via Project's mutating API, which marks dirty as a side effect, but
    # a freshly built starting point (example or blank "New Project" alike)
    # shouldn't already look unsaved the moment it opens.
    project.dirty = False

    return project
