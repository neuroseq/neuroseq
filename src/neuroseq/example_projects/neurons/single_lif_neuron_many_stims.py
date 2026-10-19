import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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

    opto_units = 1 * b2.mwatt / b2.mm2

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
        units="mwatt / mm2",
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
        units="mwatt / mm2",
        dt=0.001,
    )
    project.add_stim_array(inh_stimulation)
    return project
