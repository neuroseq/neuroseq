import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


@register_example_project("Two Hodgkin-Huxley Neurons with STDP", section="Network")
def two_hh_neurons_with_plasticity() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    # Minimal 2-neuron STDP demo: neuron 0 is excitatory (subgroup Pe) and
    # presynaptic, neuron 1 is inhibitory (subgroup Pi) and postsynaptic. The
    # plastic synapse Ce (0 -> 1) potentiates when the pre-before-post stim
    # pattern (built below) drives them.
    nneurons = 2
    nneurons_exc = 2

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (10e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("El", -70 * b2.mV)
    b2_model.define_parameter("EK", -90 * b2.mV)
    b2_model.define_parameter("ENa", 50 * b2.mV)
    b2_model.define_parameter(
        "g_na", (100 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_kd", (30 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("VT", -63 * b2.mV)
    # Time constants
    b2_model.define_parameter("taue", 5 * b2.ms)
    b2_model.define_parameter("taui", 10 * b2.ms)
    ## Reversal potentials
    b2_model.define_parameter("Ee", 0 * b2.mV)
    b2_model.define_parameter("Ei", -80 * b2.mV)

    ### optogenetics parameters
    b2_model.define_parameter("Eeopto", 0 * b2.mV)
    b2_model.define_parameter("Eiopto", -80 * b2.mV)
    b2_model.define_parameter(
        "g_exc", (2e-3 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_inh", (3e-3 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("taupre", 20 * b2.ms)
    b2_model.define_parameter("taupost", 20 * b2.ms)
    b2_model.define_parameter("gmax", 60 * b2.nS)
    b2_model.define_parameter("dApre", 1 * b2.nS)
    b2_model.define_parameter("dApost", -1 * 1.05 * b2.nS)

    # b2_model.define_parameter("we", 6 * b2.nS)
    # b2_model.define_parameter("wi", 67 * b2.nS)

    opto_units = 1 * b2.mwatt / b2.mm2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (ge*(Ee-v)+gi*(Ei-v)+geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i)/opto_units: siemens
    giopto = g_inh * Inh_stimulation(t, i)/opto_units: siemens
    gid = i : integer  # absolute neuron index, for subgroup-safe connectivity
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dn/dt = alpha_n*(1-n)-beta_n*n : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
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

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    ce_eq = """
    we : siemens
    dApre/dt = -Apre / taupre : siemens (event-driven)
    dApost/dt = -Apost / taupost : siemens (event-driven)
    """

    b2_model.define_synapse_connection("Ce", "gid_pre != gid_post")
    b2_model.define_synapse_equation("Ce", ce_eq)
    b2_model.define_synapse_connection_probability("Ce", 1.0)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    ce_on_pre = """
    ge += we
    Apre += dApre
    we = clip(we + Apost, 0*nS, gmax)
    """
    b2_model.define_synapse_pre("Ce", ce_on_pre)

    ce_on_post = """
    Apost += dApost
    we= clip(we + Apre, 0*nS, gmax)
    """
    b2_model.define_synapse_post("Ce", ce_on_post)

    # b2_model.define_init("Ce", "we", "exp(0.25*randn())*6*nS")
    b2_model.define_init("Ce", "we", "1*nS")

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model.define_synapse_connection("Ci", "gid_pre != gid_post")
    b2_model.define_synapse_equation("Ci", """wi : siemens""")
    b2_model.define_synapse_connection_probability("Ci", 1.0)
    b2_model.define_synapse_origin("Ci", "Pi")
    b2_model.define_synapse_target("Ci", "P")

    b2_model.define_synapse_pre("Ci", "gi += clip(wi, 0*nS, 210*nS)")

    b2_model.define_init("Ci", "wi", "exp(0.25*randn())*67*nS")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "P")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    b2_model.define_monitor_source("synapse", "Ce")
    b2_model.define_monitor_type("synapse", "StateMonitor")
    b2_model.define_monitor_variable("synapse", "we")

    # define stim arrays
    #
    # STDP pairing protocol: on every cycle, drive neuron 0 (presynaptic) with
    # a short excitation pulse and then neuron 1 (postsynaptic) a few ms later.
    # Repeated pre-before-post pairings potentiate the plastic Ce synapse, so
    # its weight `we` climbs toward gmax over the run. Swapping the two offsets
    # (post before pre) would instead depress it.
    tfinal = 2000 * b2.ms
    stim_dt = 0.001  # s (1 ms)
    n_steps = int(tfinal / b2.ms)

    i_stim = np.zeros((n_steps, nneurons), dtype="float")
    exc_stim = np.zeros((n_steps, nneurons), dtype="float")
    inh_stim = np.zeros((n_steps, nneurons), dtype="float")

    pulse_ms = 1  # length of each drive pulse
    pre_post_gap_ms = 5  # neuron 1 fires ~5 ms after neuron 0 -> potentiation

    # Protocol phases:
    # 1) 0-500 ms: no stimulation
    # 2) 500-1000 ms: 5 pulses on neuron 0
    # 3) then 5 pairings (neuron 0 then neuron 1)
    # 4) then 5 pulses on neuron 0 again
    first_phase_start_ms = 500
    first_phase_period_ms = 100
    for t0 in range(
        first_phase_start_ms, first_phase_start_ms + 500, first_phase_period_ms
    ):
        exc_stim[t0 : t0 + pulse_ms, 0] = 1.0

    pairing_start_ms = 1000
    pairing_period_ms = 100
    for t0 in range(
        pairing_start_ms, pairing_start_ms + 5 * pairing_period_ms, pairing_period_ms
    ):
        exc_stim[t0 : t0 + pulse_ms, 0] = 1.0
        post_start = t0 + pre_post_gap_ms
        exc_stim[post_start : post_start + pulse_ms, 1] = 1.0

    last_phase_start_ms = pairing_start_ms + 5 * pairing_period_ms
    last_phase_period_ms = 100
    for t0 in range(
        last_phase_start_ms,
        last_phase_start_ms + 5 * last_phase_period_ms,
        last_phase_period_ms,
    ):
        exc_stim[t0 : t0 + pulse_ms, 0] = 1.0

    # Exc_stimulation carries the STDP pairing protocol, so add it first; the
    # unused I_external current is added last.
    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / mm2",
        dt=stim_dt,
    )
    project.add_stim_array(exc_stimulation)

    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / mm2",
        dt=stim_dt,
    )
    project.add_stim_array(inh_stimulation)

    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=stim_dt,
    )
    project.add_stim_array(i_external)

    return project
