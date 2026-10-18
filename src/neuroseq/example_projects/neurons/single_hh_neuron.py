import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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
