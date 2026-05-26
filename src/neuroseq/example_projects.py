import copy
from collections.abc import Callable
from functools import wraps

import brian2 as b2
import brian2.numpy_ as np

from neuroseq.project import Project
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

ExampleProjectGenerator = Callable[[], Project]

example_projects: dict[str, dict[str, ExampleProjectGenerator]] = {}


def register_example_project(name: str, section: str):
    """
    Decorator used to register example project generator functions so that they will be displayed in the GUI

    Args:
        name: The name shown in the GUI (e.g., "Non-linear").
        section: Submenu section in "Open example project".
    """

    def decorator(func: ExampleProjectGenerator) -> ExampleProjectGenerator:
        @wraps(func)
        def wrapped() -> Project:
            project = func()
            project.project_name = name
            project.project_section = section
            return project

        example_projects.setdefault(section, {})[name] = wrapped
        return wrapped

    return decorator


@register_example_project("Getting Started Project", section="Basics")
def default_project() -> Project:
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
    b2_model.define_parameter("gmax", 60.0 * b2.nsiemens)

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
    b2_model.define_parameter("Eiopto", -80 * b2.mV)

    b2_model.define_parameter("we", 6 * b2.nS)
    b2_model.define_parameter("wi", 67 * b2.nS)

    b2_model.define_parameter("dApost", -6.3 * b2.nsiemens)
    b2_model.define_parameter("dApre", 6.0 * b2.nsiemens)

    eqs = """
dv/dt = (gl*(El-v)+ge*(Ee-v)+geopto*(Eeopto-v)+gi*(Ei-v)+giopto*(Eiopto-v)-g_na*(m*m*m)*h*(v-ENa)-g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
giopto = stimulus_inh(t, i): siemens
geopto = stimulus(t, i): siemens
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

    ce_eq = """
we : siemens
dApre/dt = -Apre / taupre : siemens (event-driven)
dApost/dt = -Apost / taupost : siemens (event-driven)
    """
    b2_model.define_synapse_equation("Ce", ce_eq)
    b2_model.define_synapse_connection("Ce", connection_condition="i != j")
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "Pe")

    b2_model.define_synapse_equation("Cie", "wi = 67*nS : siemens")
    b2_model.define_synapse_connection("Cie", None)
    b2_model.define_synapse_origin("Cie", "Pi")
    b2_model.define_synapse_target("Cie", "Pe")

    b2_model.define_synapse_equation("Cei", "we = 6*nS : siemens")
    b2_model.define_synapse_connection("Cei", None)
    b2_model.define_synapse_origin("Cei", "Pe")
    b2_model.define_synapse_target("Cei", "Pi")

    ce_on_pre = """
ge += we
Apre += dApre
we = clip(we + Apost, -gmax, gmax)
"""
    b2_model.define_synapse_pre("Ce", ce_on_pre)
    b2_model.define_synapse_pre("Cei", "ge+=we")
    b2_model.define_synapse_pre("Cie", "gi+=wi")

    ce_on_post = """
Apost += dApost
we= clip(we + Apre, -gmax, gmax)
"""
    b2_model.define_synapse_post("Ce", ce_on_post)

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("spike", "P")
    b2_model.define_monitor_type("spike", "SpikeMonitor")
    b2_model.define_monitor_variable("spike", "v")

    b2_model.define_monitor_source("synapse", "Ce")
    b2_model.define_monitor_type("synapse", "StateMonitor")
    b2_model.define_monitor_variable("synapse", "we")

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
        units="siemens",
        dt=stim_dt,
    )
    project.add_stim_array(stim_array)

    stim_inh = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    stim_inh[800:802, :nneurons_exc] = 1
    stim_array_inh = StimArray(
        name="stimulus_inh",
        stim_type=StimType.Inhibition,
        values=stim_inh,
        units="siemens",
        dt=0.001,
    )
    project.add_stim_array(stim_array_inh)

    return project


@register_example_project("non_linear", section="Basics")
def non_linear() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    b2_model.define_parameter("a", 1 / (10 * b2.ms))
    b2_model.define_parameter("b", 1 / (10 * b2.ms))
    b2_model.define_parameter("c", 1 / (10 * b2.ms))

    b2_model.define_neuronnumber("neuron_input", 2)
    b2_model.define_solver_method("neuron_input", "euler")
    b2_model.define_equation("neuron_input", "dv/dt = 1/(10*ms) : 1")
    b2_model.define_threshold("neuron_input", "v>1")
    b2_model.define_reset("neuron_input", "v=0")

    b2_model.define_neuronnumber("neurons", 1)
    b2_model.define_equation(
        "neurons",
        """dv/dt = (g-v)/(10*ms) : 1
                            g : 1""",
    )
    b2_model.define_solver_method("neurons", "exact")

    synapse_eq = """
        dg_syn/dt = -a*g_syn+b*x*(1-g_syn) : 1 (clock-driven)
        g_post = g_syn : 1 (summed)
        dx/dt=-c*x : 1 (clock-driven)
        w : 1 # synaptic weight
    """

    b2_model.define_synapse_equation("S", synapse_eq)
    b2_model.define_synapse_origin("S", "neuron_input")
    b2_model.define_synapse_target("S", "neurons")
    b2_model.define_synapse_pre("S", "x += w")

    b2_model.define_neuronsubgroup_target("input_subgroup", "neuron_input")
    b2_model.define_neuronsubgroup_idxs("input_subgroup", 1)

    b2_model.define_monitor_source("M", "S")
    b2_model.define_monitor_type("M", "StateMonitor")
    b2_model.define_monitor_variable("M", "g")

    b2_model.define_monitor_source("Mn", "neurons")
    b2_model.define_monitor_type("Mn", "StateMonitor")
    b2_model.define_monitor_variable("Mn", "g")

    b2_model.define_init("S", "w", [1.0, 10.0])
    b2_model.define_init("neuron_input", "v", [0.0, 0.5])

    return project


@register_example_project("Hodgkin-Huxley Neuron", section="Single Neuron")
def single_hh_neuron() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (5e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("El", -60 * b2.mV)
    b2_model.define_parameter("EK", -90 * b2.mV)
    b2_model.define_parameter("ENa", 50 * b2.mV)
    b2_model.define_parameter(
        "g_na", (100 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_kd", (30 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("VT", -40 * b2.mV)
    # Time constants
    b2_model.define_parameter("taue", 5 * b2.ms)
    b2_model.define_parameter("taui", 10 * b2.ms)
    # # Reversal potentials
    b2_model.define_parameter("Eeopto", 0 * b2.mV)
    b2_model.define_parameter("Eiopto", -80 * b2.mV)
    b2_model.define_parameter(
        "g_exc", (1e-4 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_inh", (1e-4 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    eqs = """
    dv/dt = (geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i): siemens
    giopto = g_inh * Inh_stimulation(t, i): siemens
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
    b2_model.define_neuronnumber("neuron", 1)
    b2_model.define_solver_method("neuron", "exponential_euler")
    b2_model.define_equation("neuron", eqs)
    b2_model.define_threshold("neuron", "v>-20*mV")
    # model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("neuron", 2 * b2.ms)
    b2_model.define_dt("neuron", 0.1 * b2.ms)

    # initial conditions
    b2_model.define_init("neuron", "v", "El + (randn() * 5 - 5)*mV")

    # monitors
    b2_model.define_monitor_source("trace", "neuron")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "neuron")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    # define stim arrays
    i_stim = np.zeros((3100, 1), dtype="float")
    i_stim[:, :] = 0
    i_stim[100:199, :] = np.arange(99)[:, np.newaxis] * 3
    i_stim[300:399, :] = np.arange(99)[:, np.newaxis] * 6
    i_stim[500:599, :] = 200
    i_stim[700:799, :] = 400
    i_stim[900:999, :] = 600
    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stim = np.zeros((3100, 1), dtype="float")
    exc_stim[:, :] = 0
    exc_stim[1100:1199, :] = np.arange(99)[:, np.newaxis] * 0.005
    exc_stim[1300:1399, :] = np.arange(99)[:, np.newaxis] * 0.01
    exc_stim[1500:1599, :] = 0.25
    exc_stim[1700:1799, :] = 0.5
    exc_stim[1900:1999, :] = 1.0
    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="1",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stim = np.zeros((3100, 1), dtype="float")
    inh_stim[:, :] = 0
    inh_stim[2100:2199, :] = np.arange(99)[:, np.newaxis] * 0.005
    inh_stim[2300:2399, :] = np.arange(99)[:, np.newaxis] * 0.01
    inh_stim[2500:2599, :] = 0.25
    inh_stim[2700:2799, :] = 0.5
    inh_stim[2900:2999, :] = 1.0
    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="1",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project


@register_example_project("Leaky-Integrate-and-Fire Neuron", section="Single Neuron")
def single_lif_neuron_many_stims() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    # parameters
    b2_model.define_parameter("v0", -70 * b2.mV)
    b2_model.define_parameter("tau", 10 * b2.ms)
    b2_model.define_parameter("Iext", 10 * b2.pA)
    b2_model.define_parameter("Rm", 100 * b2.Mohm)
    b2_model.define_parameter("sigma", 50 * b2.mV)

    # optogenetic parameters
    b2_model.define_parameter("v_rev_ext", 0 * b2.mV)
    b2_model.define_parameter("v_rev_inh", -80 * b2.mV)

    b2_model.define_parameter("Rext", 1 * b2.Mohm)
    b2_model.define_parameter("Rinh", 1 * b2.Mohm)

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
             dv/dt = -(v-v0)/tau + Rm*(I_stim)/tau: volt
             I_stim = I_ext + I_opto_ext + I_opto_inh: amp
             I_ext = I_external(t,i): amp
             I_opto_ext = Exc_stimulation(t,i)/opto_units*(v_rev_ext - v)/Rext: amp
             I_opto_inh = Inh_stimulation(t,i)/opto_units*(v_rev_inh - v)/Rinh: amp
          """

    # neuron group
    b2_model.define_neuronnumber("neuron", 1)
    b2_model.define_solver_method("neuron", "exponential_euler")
    b2_model.define_equation("neuron", eqs)
    b2_model.define_threshold("neuron", "v>-10*mV")
    b2_model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("neuron", 2 * b2.ms)
    b2_model.define_dt("neuron", 0.1 * b2.ms)
    b2_model._assemble_neurongroup("neuron")

    # initial conditions
    b2_model.define_init("neuron", "v", 5 * b2.mV)

    # monitors
    b2_model.define_monitor_source("trace", "neuron")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "neuron")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    # define stim arrays
    tfinal = 3100 * b2.ms

    i_stim = np.zeros((int(tfinal / b2.ms), 1), dtype="float")
    i_stim[:, :] = 0
    i_stim[100:199, :] = np.arange(99)[:, np.newaxis] * 3
    i_stim[300:399, :] = np.arange(99)[:, np.newaxis] * 6
    i_stim[500:599, :] = 200
    i_stim[700:799, :] = 400
    i_stim[900:999, :] = 600
    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stim = np.zeros((int(tfinal / b2.ms), 1), dtype="float")
    exc_stim[:, :] = 0
    exc_stim[1100:1199, :] = np.arange(99)[:, np.newaxis] * 0.005
    exc_stim[1300:1399, :] = np.arange(99)[:, np.newaxis] * 0.01
    exc_stim[1500:1599, :] = 0.25
    exc_stim[1700:1799, :] = 0.5
    exc_stim[1900:1999, :] = 1.0
    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stim = np.zeros((int(tfinal / b2.ms), 1), dtype="float")
    inh_stim[:, :] = 0
    inh_stim[2100:2199, :] = np.arange(99)[:, np.newaxis] * 0.005
    inh_stim[2300:2399, :] = np.arange(99)[:, np.newaxis] * 0.01
    inh_stim[2500:2599, :] = 0.25
    inh_stim[2700:2799, :] = 0.5
    inh_stim[2900:2999, :] = 1.0
    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)
    return project


@register_example_project("Network of Hodgkin-Huxley Neurons", section="Network")
def hh_neuron_with_synapses() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 4000
    nneurons_exc = 3200

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (5e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("El", -60 * b2.mV)
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
    # # Reversal potentials

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

    # b2_model.define_parameter("we", 6 * b2.nS)
    # b2_model.define_parameter("wi", 67 * b2.nS)

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (ge*(Ee-v)+gi*(Ei-v)+geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i)/opto_units: siemens
    giopto = g_inh * Inh_stimulation(t, i)/opto_units: siemens
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
    # model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("P", 3 * b2.ms)
    # model.define_dt("P", 0.1 * b2.ms)

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model.define_synapse_connection("Ce", None)
    # b2_model.define_synapse_connection("Ce", "i != j")
    b2_model.define_synapse_equation("Ce", """we : siemens""")
    b2_model.define_synapse_connection_probability("Ce", 0.02)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    b2_model.define_synapse_connection("Ci", None)
    # b2_model.define_synapse_connection("Ci", "i != j")
    b2_model.define_synapse_equation("Ci", """wi : siemens""")
    b2_model.define_synapse_connection_probability("Ci", 0.02)
    b2_model.define_synapse_origin("Ci", "Pi")
    b2_model.define_synapse_target("Ci", "P")

    b2_model.define_synapse_pre("Ce", "ge += clip(we, 0*nS, 18*nS)")
    b2_model.define_synapse_pre("Ci", "gi += clip(wi, 0*nS, 210*nS)")

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")

    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    b2_model.define_init("Ce", "we", "exp(0.25*randn())*6*nS")
    b2_model.define_init("Ci", "wi", "exp(0.25*randn())*67*nS")

    # b2_model.define_init("Ce", "we", "6*nS")
    # b2_model.define_init("Ci", "wi", "67*nS")

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
    # self..define_monitor_dt(1.0 * b2.s, "synapse")

    # define stim arrays

    tfinal = 2000 * b2.ms

    i_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    i_stim[:, :] = 0
    # i_stim[200:300, :] = 1

    exc_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    exc_stim[:, :] = 0
    # exc_stim[0:200, :] = 1

    inh_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    inh_stim[:, :] = 0
    # inh_stim[300:900, :] = 1

    sp = 10

    for n in range(0, 5):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n * 8 : n * 8 + 8] = 1

    sp = 100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) : 3200] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    sp = 1100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) :] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project


@register_example_project(
    "Network of Hodgkin-Huxley Neurons with Predefined Synapses", section="Network"
)
def hh_neuron_with_synthetic_synapses() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 4000
    nneurons_exc = 3200

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (5e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("El", -60 * b2.mV)
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
    # # Reversal potentials

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

    # b2_model.define_parameter("we", 6 * b2.nS)
    # b2_model.define_parameter("wi", 67 * b2.nS)

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (ge*(Ee-v)+gi*(Ei-v)+geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i)/opto_units: siemens
    giopto = g_inh * Inh_stimulation(t, i)/opto_units: siemens
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
    # model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("P", 3 * b2.ms)
    # model.define_dt("P", 0.1 * b2.ms)

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model.define_synapse_connection("Ce", None)
    # b2_model.define_synapse_connection("Ce", "i != j")
    b2_model.define_synapse_equation("Ce", """we : siemens""")
    b2_model.define_synapse_connection_probability("Ce", 0.0001)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    b2_model.define_synapse_connection("Ci", None)
    # b2_model.define_synapse_connection("Ci", "i != j")
    b2_model.define_synapse_equation("Ci", """wi : siemens""")
    b2_model.define_synapse_connection_probability("Ci", 0.0001)
    b2_model.define_synapse_origin("Ci", "Pi")
    b2_model.define_synapse_target("Ci", "P")

    b2_model.define_synapse_pre("Ce", "ge += clip(we, 0*nS, 18*nS)")
    b2_model.define_synapse_pre("Ci", "gi += clip(wi, 0*nS, 210*nS)")

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")

    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    b2_model.define_init("Ce", "we", "6*nS")
    b2_model.define_init("Ci", "wi", "67*nS")

    # b2_model.define_init("Ce", "we", "6*nS")
    # b2_model.define_init("Ci", "wi", "67*nS")

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
    # self..define_monitor_dt(1.0 * b2.s, "synapse")

    # define stim arrays

    tfinal = 2000 * b2.ms

    i_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    i_stim[:, :] = 0
    # i_stim[200:300, :] = 1

    exc_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    exc_stim[:, :] = 0
    # exc_stim[0:200, :] = 1

    inh_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    inh_stim[:, :] = 0
    # inh_stim[300:900, :] = 1

    sp = 10

    for n in range(0, 5):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n * 8 : n * 8 + 8] = 1

    sp = 100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) : 3200] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    sp = 1100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) :] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project


def hh_neurons_ei_model() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 30
    nneurons_exc = 24

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (5e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("El", -60 * b2.mV)
    b2_model.define_parameter("EK", -90 * b2.mV)
    b2_model.define_parameter("ENa", 50 * b2.mV)
    b2_model.define_parameter(
        "g_na", (100 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_kd", (30 * b2.msiemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter("VT", -40 * b2.mV)
    # Time constants
    b2_model.define_parameter("taue", 5 * b2.ms)
    b2_model.define_parameter("taui", 10 * b2.ms)
    # # Reversal potentials
    b2_model.define_parameter("Eeopto", 0 * b2.mV)
    b2_model.define_parameter("Eiopto", -80 * b2.mV)
    b2_model.define_parameter(
        "g_exc", (2e-3 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "g_inh", (3e-3 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i)/opto_units: siemens
    giopto = g_inh * Inh_stimulation(t, i)/opto_units: siemens
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
    # model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("P", 2 * b2.ms)
    b2_model.define_dt("P", 0.1 * b2.ms)
    b2_model._assemble_neurongroup("P")

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model._assemble_neuronsubgroup("Pi")
    b2_model._assemble_neuronsubgroup("Pe")

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "P")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    # define stim arrays

    tfinal = 100 * b2.ms

    i_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    i_stim[:, :] = 0
    i_stim[200:300, :] = 1
    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    exc_stim[:, :] = 0
    exc_stim[0:200, :] = 1
    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    inh_stim[:, :] = 0
    inh_stim[300:900, :] = 1
    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project


def single_lif_neuron() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    # parameters
    b2_model.define_parameter("v0", -70 * b2.mV)
    b2_model.define_parameter("tau", 10 * b2.ms)
    b2_model.define_parameter("Iext", 10 * b2.pA)
    b2_model.define_parameter("Rm", 100 * b2.Mohm)
    b2_model.define_parameter("sigma", 50 * b2.mV)

    eqs = """
             dv/dt = -(v-v0)/tau + Rm*Istim/tau : volt
             Istim = stimulus(t,i) : amp
          """
    # neuron group
    b2_model.define_neuronnumber("neuron", 1)
    b2_model.define_solver_method("neuron", "exponential_euler")
    b2_model.define_equation("neuron", eqs)
    b2_model.define_threshold("neuron", "v>10*mV")
    b2_model.define_reset("neuron", "v=-65*mV")
    b2_model.define_refractory("neuron", 2 * b2.ms)
    b2_model.define_dt("neuron", 0.1 * b2.ms)
    b2_model._assemble_neurongroup("neuron")

    # initial conditions
    b2_model.define_init("neuron", "v", 5 * b2.mV)

    # monitors
    b2_model.define_monitor_source("trace", "neuron")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "neuron")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    # define stim arrays
    tfinal = 1000 * b2.ms

    stim = np.zeros((int(tfinal / b2.ms), 1), dtype="float")
    stim[:, :] = 0
    stim[200:300, :] = 1000
    stim_array = StimArray(
        name="stimulus",
        stim_type=StimType.External,
        values=stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(stim_array)

    return project


def single_lif_neuron_without_stim() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    # parameters
    b2_model.define_parameter("v0", -65 * b2.mV)
    b2_model.define_parameter("tau", 10 * b2.ms)
    b2_model.define_parameter("Iext", 10 * b2.pA)
    b2_model.define_parameter("Rm", 100 * b2.Mohm)
    b2_model.define_parameter("sigma", 50 * b2.mV)

    eqs = """
    dv/dt = -(v-v0)/tau + Rm*Iext/tau : volt
    """
    # neuron group
    b2_model.define_neuronnumber("neuron", 1)
    b2_model.define_solver_method("neuron", "exponential_euler")
    b2_model.define_equation("neuron", eqs)
    b2_model.define_threshold("neuron", "v>10*mV")
    b2_model.define_reset("neuron", "v=-70*mV")
    b2_model.define_refractory("neuron", 2 * b2.ms)
    b2_model.define_dt("neuron", 0.1 * b2.ms)
    b2_model._assemble_neurongroup("neuron")

    # initial conditions
    b2_model.define_init("neuron", "v", -50 * b2.mV)

    # monitors
    b2_model.define_monitor_source("trace", "neuron")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "neuron")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    return project


@register_example_project(
    "Network of Hodgkin-Huxley Neurons with STDP", section="Network"
)
def hh_neuron_with_plasticity() -> Project:
    project = Project()

    # define Brian2 model
    b2_model = project.b2_model

    nneurons = 40
    nneurons_exc = 32

    b2_model.define_parameter("area", 20000 * b2.umetre**2)
    b2_model.define_parameter(
        "Cm", (1 * b2.ufarad * b2.cm**-2) * b2_model.parameters["area"]
    )
    b2_model.define_parameter(
        "gl", (5e-5 * b2.siemens * b2.cm**-2) * b2_model.parameters["area"]
    )

    b2_model.define_parameter("El", -60 * b2.mV)
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
    b2_model.define_parameter("dApre", 6 * b2.nS)
    b2_model.define_parameter("dApost", -6 * 1.05 * b2.nS)

    # b2_model.define_parameter("we", 6 * b2.nS)
    # b2_model.define_parameter("wi", 67 * b2.nS)

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (ge*(Ee-v)+gi*(Ei-v)+geopto*(Eeopto-v)+giopto*(Eiopto-v)+gl*(El-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK)+Iext)/Cm : volt
    Iext = I_external(t,i) : amp
    geopto = g_exc * Exc_stimulation(t, i)/opto_units: siemens
    giopto = g_inh * Inh_stimulation(t, i)/opto_units: siemens
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

    b2_model.define_synapse_connection("Ce", None)
    b2_model.define_synapse_connection("Ce", "i != j")
    # b2_model.define_synapse_equation("Ce", """we : siemens""")
    b2_model.define_synapse_equation("Ce", ce_eq)
    b2_model.define_synapse_connection_probability("Ce", 0.02)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    b2_model.define_synapse_pre("Ce", "ge += clip(we, 0*nS, 18*nS)")

    ce_on_pre = """
    ge += we
    Apre += dApre
    we = clip(we + Apost, -gmax, gmax)
    """
    b2_model.define_synapse_pre("Ce", ce_on_pre)

    ce_on_post = """
    Apost += dApost
    we= clip(we + Apre, -gmax, gmax)
    """
    b2_model.define_synapse_post("Ce", ce_on_post)

    b2_model.define_init("Ce", "we", "exp(0.25*randn())*6*nS")

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model.define_synapse_connection("Ci", None)
    b2_model.define_synapse_connection("Ci", "i != j")
    b2_model.define_synapse_equation("Ci", """wi : siemens""")
    b2_model.define_synapse_connection_probability("Ci", 0.02)
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

    tfinal = 2000 * b2.ms

    i_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    i_stim[:, :] = 0
    # i_stim[200:300, :] = 1

    exc_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    exc_stim[:, :] = 0
    # exc_stim[0:200, :] = 1

    inh_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    inh_stim[:, :] = 0
    # inh_stim[300:900, :] = 1

    sp = 10

    for n in range(0, 5):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n * 8 : n * 8 + 8] = 1

    sp = 100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) : 3200] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    sp = 1100

    for n in range(1, 40):
        exc_stim[sp + 10 * n : sp + 10 * (n + 1), n] = 1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), (n + 1) :] = 0.1
        inh_stim[sp + 10 * n : sp + 10 * (n + 1), 1:n] = 0.1

    i_external = StimArray(
        name="I_external",
        stim_type=StimType.External,
        values=i_stim,
        units="pA",
        dt=0.001,
    )
    project.add_stim_array(i_external)

    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(exc_stimulation)

    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="mwatt / um2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project


@register_example_project("ChR2 in HEK293", section="Optogenetics")
def pyrho_6state_chr2_project() -> Project:
    """
    PyRho 6-state ChR2 optogenetic model based on published parameters.
    Implements optogenetic stimulation with detailed channelrhodopsin dynamics.
    """
    project = Project()
    b2_model = project.b2_model

    dt = 0.1 * b2.ms
    tfinal = 1000 * b2.ms

    # Number of neurons
    N = 10

    # Define network parameters (PyRho Article version)
    b2_model.define_parameter("tau_m", 10 * b2.ms)
    b2_model.define_parameter("R_m", 7000 * b2.Mohm)
    b2_model.define_parameter("E_m", -70 * b2.mV)
    b2_model.define_parameter("v_t0", 10000 * b2.mV)
    b2_model.define_parameter("sigma", 10 * b2.mV)
    b2_model.define_parameter("t_ref", 4 * b2.ms)

    # Optogenetic parameters
    b2_model.define_parameter("phy", 5e17 * b2.mm**-2 * b2.Hz)
    b2_model.define_parameter("g0", 27.700 * b2.nS)
    b2_model.define_parameter("gam", 0.05)
    b2_model.define_parameter("phi_m", 5e17 * b2.mm**-2 * b2.Hz)

    # Kinetic parameters
    b2_model.define_parameter("k1", 18.2 * b2.ms**-1)
    b2_model.define_parameter("k2", 4.07 * b2.ms**-1)
    b2_model.define_parameter("p", 0.981)
    b2_model.define_parameter("Gf0", 0.037 * b2.ms**-1)
    b2_model.define_parameter("k_f", 0.121 * b2.ms**-1)
    b2_model.define_parameter("Gb0", 0.0143 * b2.ms**-1)
    b2_model.define_parameter("k_b", 0.0851 * b2.ms**-1)
    b2_model.define_parameter("q", 1.45)
    b2_model.define_parameter("Go1", 1.93 * b2.ms**-1)
    b2_model.define_parameter("Go2", 3.38 * b2.ms**-1)
    b2_model.define_parameter("Gd1", 0.115 * b2.ms**-1)
    b2_model.define_parameter("Gd2", 0.011 * b2.ms**-1)
    b2_model.define_parameter("Gr0", 0.00033 * b2.ms**-1)
    b2_model.define_parameter("E", -0.5 * b2.mV)
    b2_model.define_parameter("v0", 43 * b2.mV)
    b2_model.define_parameter("v1", 17.1 * b2.mV)

    # PyRho 6-state model equations
    pyRho = """
        dC_1/dt = Gd1*O_1 + Gr0*C_2 - Ga1*C_1            : 1
        dI_1/dt = Ga1*C_1 - Go1*I_1                      : 1
        dO_1/dt = Go1*I_1 + Gb*O_2 - (Gd1+Gf)*O_1        : 1
        dO_2/dt = Go2*I_2 + Gf*O_1 - (Gd2+Gb)*O_2        : 1
        dI_2/dt = Ga2*C_2 - Go2*I_2                      : 1
        C_2 = 1 - C_1 - I_1 - O_1 - O_2 - I_2            : 1
        Theta = int(phy*stim(t,i) > 0*phy*stim(t,i))     : 1
        H_p   = Theta*(((phy*stim(t,i))**p)/((phy*stim(t,i))**p+phi_m**p)) : 1
        H_q   = Theta*(((phy*stim(t,i))**q)/((phy*stim(t,i))**q+phi_m**q)) : 1
        Ga1   = k1*H_p                                   : Hz
        Ga2   = k2*H_p                                   : Hz
        Gf    = k_f*H_q + Gf0                            : Hz
        Gb    = k_b*H_q + Gb0                            : Hz
        f_v   = (1-exp(-(v-E)/v0))/((v-E)/v1)            : 1
        f_phi = O_1+gam*O_2                              : 1
        I     = g0*f_phi*f_v*(v-E)                       : amp
        """

    # Complete neuron equations (voltage clamp mode)
    eqs = "dv/dt = (E_vc-v)/tau_m : volt\nE_vc : volt\n" + pyRho

    # Define neuron group
    b2_model.define_neuronnumber("G0", N)
    b2_model.define_equation("G0", eqs)
    b2_model.define_solver_method("G0", "euler")
    b2_model.define_threshold("G0", "v>v_t0")
    b2_model.define_reset("G0", "v=E_m")
    b2_model.define_refractory("G0", "t_ref")
    b2_model.define_dt("G0", dt)

    # Initial conditions
    b2_model.define_init(
        "G0", "E_vc", np.arange(-100, 100, 20) * b2.mV
    )  # voltage clamp values per neuron
    b2_model.define_init(
        "G0", "v", np.arange(-100, 100, 20) * b2.mV
    )  # -100mV to 80mV with 10 steps

    b2_model.define_init("G0", "C_1", 1)
    b2_model.define_init("G0", "O_1", 0)
    b2_model.define_init("G0", "O_2", 0)
    b2_model.define_init("G0", "I_1", 0)
    b2_model.define_init("G0", "I_2", 0)

    # Define monitors for voltage
    b2_model.define_monitor_source("trace_v", "G0")
    b2_model.define_monitor_type("trace_v", "StateMonitor")
    b2_model.define_monitor_variable("trace_v", "v")

    # Define monitors for current
    b2_model.define_monitor_source("trace_I", "G0")
    b2_model.define_monitor_type("trace_I", "StateMonitor")
    b2_model.define_monitor_variable("trace_I", "I")

    # Define monitors for channel states
    b2_model.define_monitor_source("C_1_state", "G0")
    b2_model.define_monitor_type("C_1_state", "StateMonitor")
    b2_model.define_monitor_variable("C_1_state", "C_1")

    b2_model.define_monitor_source("C_2_state", "G0")
    b2_model.define_monitor_type("C_2_state", "StateMonitor")
    b2_model.define_monitor_variable("C_2_state", "C_2")

    b2_model.define_monitor_source("O_1_state", "G0")
    b2_model.define_monitor_type("O_1_state", "StateMonitor")
    b2_model.define_monitor_variable("O_1_state", "O_1")

    b2_model.define_monitor_source("O_2_state", "G0")
    b2_model.define_monitor_type("O_2_state", "StateMonitor")
    b2_model.define_monitor_variable("O_2_state", "O_2")

    b2_model.define_monitor_source("I_1_state", "G0")
    b2_model.define_monitor_type("I_1_state", "StateMonitor")
    b2_model.define_monitor_variable("I_1_state", "I_1")

    b2_model.define_monitor_source("I_2_state", "G0")
    b2_model.define_monitor_type("I_2_state", "StateMonitor")
    b2_model.define_monitor_variable("I_2_state", "I_2")

    # Define stimulation protocol
    st = np.zeros((int(tfinal / b2.ms), N), dtype="float")

    st[:, :] = 0
    # Stimulation: 0.00442 from 100ms to 700ms (indices 100:700 at 1 ms dt)
    st[100:700, :] = 0.00442  # at 2.21/500 × 10^15

    stim_array = StimArray(
        name="stim",
        stim_type=StimType.Excitation,
        values=st,
        units="1",
        dt=0.001,
    )
    project.add_stim_array(stim_array)

    return project


# Beethoven "Für Elise" opening phrase as used in Kriener et al., 2024 (Fig. 2).
# Each entry is a pitch index (int) or a tuple of two simultaneous pitch indices
# (chord). Max pitch index is 12, so 13 visible neurons cover all pitches.
# Source: sequence-learning/scripts/patterns/fuer_elise_original.txt
_FUER_ELISE_INTRO: list[int | tuple[int, ...]] = [
    0,
    1,
    0,
    1,
    0,
    4,
    2,
    3,
    (5, 11),
    (5, 10),
    8,
    7,
    6,
    5,
    (4, 12),
    (4, 10),
    (4, 9),
    7,
    3,
    4,
    (5, 11),
    (5, 10),
    (5, 8),
    5,
]
_FUER_ELISE_N_PITCHES = 13
# Normalized so g_exc + g_inh = 1 in each state while preserving the
# ratios that reproduce u_tgt = -50 mV (active) and -70 mV (rest).
_FUER_ELISE_OPSIN_EXC_REST: float = 1.0 / 15.0
_FUER_ELISE_OPSIN_INH_REST: float = 14.0 / 15.0
_FUER_ELISE_OPSIN_EXC_ACTIVE: float = 1.0 / 3.0
_FUER_ELISE_OPSIN_INH_ACTIVE: float = 2.0 / 3.0


def _build_fuer_elise_opsin_stims(
    tfinal_ms: int = 1920,
    event_ms: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Two opsin-activation arrays encoding the Für Elise intro.

    Returns ``(stim_exc, stim_inh)`` of shape ``(tfinal_ms, 13)``. The
    wrapper's matching-potential math turns these into ``u_tgt = -50 mV`` for
    active pitches and ``u_tgt = -70 mV`` for inactive pitches, reproducing
    the same teaching as the previous single-stim pattern.

    Background voltages match the upstream ``to_biounits`` transform:
    rest = ``E_l = -70 mV``, active = ``E_l + biounits_scale = -50 mV``.
    """
    n_pitches = _FUER_ELISE_N_PITCHES
    stim_exc = np.full(
        (tfinal_ms, n_pitches), _FUER_ELISE_OPSIN_EXC_REST, dtype=np.float32
    )
    stim_inh = np.full(
        (tfinal_ms, n_pitches), _FUER_ELISE_OPSIN_INH_REST, dtype=np.float32
    )
    cycle_ms = len(_FUER_ELISE_INTRO) * event_ms

    # Only emit complete cycles; any remainder is left at the rest ratio so the
    # trailing tail isn't a half-phrase.
    for cycle_start in range(0, tfinal_ms - cycle_ms + 1, cycle_ms):
        for event_idx, event in enumerate(_FUER_ELISE_INTRO):
            t_start = cycle_start + event_idx * event_ms
            t_end = t_start + event_ms
            pitches = event if isinstance(event, tuple) else (event,)
            for pitch in pitches:
                stim_exc[t_start:t_end, pitch] = _FUER_ELISE_OPSIN_EXC_ACTIVE
                stim_inh[t_start:t_end, pitch] = _FUER_ELISE_OPSIN_INH_ACTIVE
    return stim_exc, stim_inh


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


@register_example_project("ELISE Sequence Learning Network", section="Network")
def elise_project() -> Project:
    """ELISE rate-coded sequence-learning network (Kriener et al., 2024).

    Visible neurons are driven by two virtual opsins (``opsin_exc`` and
    ``opsin_inh``); the wrapper computes the per-pitch matching potential
    from their ratio each step. Release (no teaching) is expressed by
    zeroing both stim arrays in the time range where you want the network
    to ring out.
    """
    # Deferred so opening unrelated example projects doesn't pull in elise/numba.
    from neuroseq.example_models.elise import DEFAULT_PARAMS

    project = Project()
    project.external_simulate_path = "neuroseq:elise"
    project.external_simulate_params = copy.deepcopy(DEFAULT_PARAMS)
    project.load_NN()

    tfinal_ms = 1920  # 8 × 240 ms cycles, exactly periodic
    stim_exc, stim_inh = _build_fuer_elise_opsin_stims(tfinal_ms=tfinal_ms)
    project.add_stim_array(
        StimArray(
            name="opsin_exc",
            stim_type=StimType.Excitation,
            values=stim_exc,
            units="1",
            dt=0.001,
        )
    )
    project.add_stim_array(
        StimArray(
            name="opsin_inh",
            stim_type=StimType.Inhibition,
            values=stim_inh,
            units="1",
            dt=0.001,
        )
    )
    return project


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
