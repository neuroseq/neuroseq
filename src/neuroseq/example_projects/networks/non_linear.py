import brian2 as b2

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project


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
