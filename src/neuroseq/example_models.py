import brian2 as b2
import brian2.numpy_ as np

from neuroseq.model import Model
from neuroseq.stim import StimArray
from neuroseq.stim import StimType

example_models = {}


def register_example_model(name: str):
    """
    Decorator used to register example model generator functions so that they will be displayed in the GUI

    Args:
        name: The name shown in the GUI (e.g., "Non-linear").
    """

    def decorator(func):
        example_models[name] = func
        return func

    return decorator


@register_example_model("Default model")
def default_model() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

    nneurons = 30
    nneurons_exc = 24

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

    stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    stim[120, :nneurons_exc:2] = 1
    stim[121, 1:nneurons_exc:2] = 1
    stim[320, :nneurons_exc:2] = 1
    stim[321, 1:nneurons_exc:2] = 1
    stim[520, :nneurons_exc:2] = 1
    stim[521, 1:nneurons_exc:2] = 1
    stim_array = StimArray(
        name="stimulus",
        stim_type=StimType.Excitation,
        values=stim,
        units="siemens",
        dt=0.001,
    )
    model.add_stim_array(stim_array)

    stim_inh = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    stim_inh[800:802, :nneurons_exc] = 1
    stim_array_inh = StimArray(
        name="stimulus_inh",
        stim_type=StimType.Inhibition,
        values=stim_inh,
        units="siemens",
        dt=0.001,
    )
    model.add_stim_array(stim_array_inh)

    return model


def non_linear() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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

    return model


@register_example_model("Hodgkin-Huxley Neuron")
def single_hh_neuron() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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
    model.add_stim_array(i_external)

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
    model.add_stim_array(exc_stimulation)

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
    model.add_stim_array(inh_stimulation)

    return model


@register_example_model("HH Neuron with plasticity")
def hh_neuron_with_plasticity() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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

    b2_model.define_parameter("we", 6 * b2.nS)
    b2_model.define_parameter("wi", 67 * b2.nS)

    opto_units = 1 * b2.mwatt / b2.um2

    b2_model.define_parameter("opto_units", opto_units)

    eqs = """
    dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK))/Cm : volt
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

    b2_model.define_neuronsubgroup_target("Pe", "P")
    b2_model.define_neuronsubgroup_idxs("Pe", (0, nneurons_exc))

    b2_model.define_neuronsubgroup_target("Pi", "P")
    b2_model.define_neuronsubgroup_idxs("Pi", (nneurons_exc, nneurons))

    b2_model.define_synapse_connection("Ce", None)
    b2_model.define_synapse_connection_probability("Ce", 0.02)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    b2_model.define_synapse_connection("Ci", None)
    b2_model.define_synapse_connection_probability("Ci", 0.02)
    b2_model.define_synapse_origin("Ci", "Pi")
    b2_model.define_synapse_target("Ci", "P")

    b2_model.define_synapse_pre("Ce", "ge += we")
    b2_model.define_synapse_pre("Ci", "gi += wi")

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("mon", "P")
    b2_model.define_monitor_type("mon", "SpikeMonitor")
    b2_model.define_monitor_variable("mon", "v")

    # define stim arrays

    tfinal = 1000 * b2.ms

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
    model.add_stim_array(i_external)

    exc_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    exc_stim[:, :] = 0
    exc_stim[0:200, :] = 1
    exc_stimulation = StimArray(
        name="Exc_stimulation",
        stim_type=StimType.Excitation,
        values=exc_stim,
        units="1",
        dt=0.001,
    )
    model.add_stim_array(exc_stimulation)

    inh_stim = np.zeros((int(tfinal / b2.ms), nneurons), dtype="float")
    inh_stim[:, :] = 0
    inh_stim[300:900, :] = 1
    inh_stimulation = StimArray(
        name="Inh_stimulation",
        stim_type=StimType.Inhibition,
        values=inh_stim,
        units="1",
        dt=0.001,
    )
    model.add_stim_array(inh_stimulation)

    return model


@register_example_model("Network of Hodgkin-Huxley Neurons")
def hh_neuron_with_synapses() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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

    b2_model.define_parameter("we", 6 * b2.nS)
    b2_model.define_parameter("wi", 67 * b2.nS)

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
    b2_model.define_synapse_equation("Ce", """we = 6*nS : siemens""")
    b2_model.define_synapse_connection_probability("Ce", 0.02)
    b2_model.define_synapse_origin("Ce", "Pe")
    b2_model.define_synapse_target("Ce", "P")

    b2_model.define_synapse_connection("Ci", None)
    b2_model.define_synapse_equation("Ci", """wi = 67*nS : siemens""")
    b2_model.define_synapse_connection_probability("Ci", 0.02)
    b2_model.define_synapse_origin("Ci", "Pi")
    b2_model.define_synapse_target("Ci", "P")

    b2_model.define_synapse_pre("Ce", "ge += we")
    b2_model.define_synapse_pre("Ci", "gi += wi")

    # initial conditions
    b2_model.define_init("P", "v", "El + (randn() * 5 - 5)*mV")
    b2_model.define_init("P", "ge", "(randn() * 1.5 + 4) * 10.*nS")
    b2_model.define_init("P", "gi", "(randn() * 12 + 20) * 10.*nS")

    # b2_model.define_init("Ce", "we", "3.*nS")
    # b2_model.define_init("Ci", "wi", "10.*nS")

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

    tfinal = 1000 * b2.ms

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
    model.add_stim_array(i_external)

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
    model.add_stim_array(exc_stimulation)

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
    model.add_stim_array(inh_stimulation)

    return model


def hh_neurons_ei_model() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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
    model.add_stim_array(i_external)

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
    model.add_stim_array(exc_stimulation)

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
    model.add_stim_array(inh_stimulation)

    return model


def single_lif_neuron() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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
    model.add_stim_array(stim_array)

    return model


def single_lif_neuron_without_stim() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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

    return model


@register_example_model("Leaky-Integrate-and-Fire Neuron")
def single_lif_neuron_many_stims() -> Model:
    model = Model()

    # define Brian2 model
    b2_model = model.b2_model

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
    model.add_stim_array(i_external)

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
    model.add_stim_array(exc_stimulation)

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
    model.add_stim_array(inh_stimulation)
    return model
