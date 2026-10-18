import brian2 as b2

from neuroseq.project import Project


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
