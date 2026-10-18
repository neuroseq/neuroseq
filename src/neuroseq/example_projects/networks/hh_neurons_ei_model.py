import brian2 as b2
import brian2.numpy_ as np

from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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

    opto_units = 1 * b2.mwatt / b2.mm2

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
        units="mwatt / mm2",
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
        units="mwatt / mm2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)

    return project
