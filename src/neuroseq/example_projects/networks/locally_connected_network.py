import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


@register_example_project("Locally Connected Network", section="Network")
def locally_connected_network() -> Project:
    """A 1-D network whose connectivity depends on the distance between neurons.

    This example showcases the structured-connectivity options of the synapse
    editor (origin/target, ``connection_condition``, ``p``, ``connection_i`` /
    ``connection_j``, ``connection_n``, ``skip_if_invalid``):

    * **Local excitation (``E``)** uses a distance-dependent connection
      *probability*: ``p`` is a string expression, so each neuron connects to
      its neighbours with a Gaussian fall-off in index space.
    * **Surround inhibition (``I``)** uses a generator expression for
      ``connection_j`` together with ``skip_if_invalid`` to wire a fixed window
      of more distant neighbours (a "Mexican-hat" centre-surround motif).

    A brief stimulus to a central group of neurons seeds activity that the
    structured connectivity then shapes. Adapted from the Brian2
    ``spatial_connections`` and ``efficient_gaussian_connectivity`` examples
    (https://brian2.readthedocs.io/en/stable/user/synapses.html).
    """
    project = Project()
    b2_model = project.b2_model

    n_neurons = 60

    # neuron / network parameters
    b2_model.define_parameter("v0", -70 * b2.mV)  # resting potential
    b2_model.define_parameter("tau", 12 * b2.ms)  # membrane time constant
    b2_model.define_parameter("Rm", 100 * b2.Mohm)  # membrane resistance
    b2_model.define_parameter("w_exc", 7 * b2.mV)  # excitatory PSP amplitude
    b2_model.define_parameter("w_inh", -2 * b2.mV)  # inhibitory PSP amplitude
    b2_model.define_parameter("sigma_c", 3.5)  # excitation width (in neurons)

    eqs = """
    dv/dt = -(v - v0)/tau + Rm*I_drive/tau : volt (unless refractory)
    I_drive = I_external(t, i) : amp
    """

    b2_model.define_neuronnumber("P", n_neurons)
    b2_model.define_solver_method("P", "exact")
    b2_model.define_equation("P", eqs)
    b2_model.define_threshold("P", "v > -50*mV")
    b2_model.define_reset("P", "v = -70*mV")
    b2_model.define_refractory("P", 4 * b2.ms)
    b2_model.define_dt("P", 0.1 * b2.ms)

    # local excitation: distance-dependent connection probability (string `p`)
    b2_model.define_synapse_equation("E", "w : volt")
    b2_model.define_synapse_origin("E", "P")
    b2_model.define_synapse_target("E", "P")
    b2_model.define_synapse_connection("E", "i != j")
    b2_model.define_synapse_connection_probability(
        "E", "exp(-(i - j)**2 / (2.0 * sigma_c**2))"
    )
    b2_model.define_synapse_pre("E", "v_post += w")
    b2_model.define_init("E", "w", "w_exc")

    # surround inhibition: a generator window for `j`, with skip_if_invalid for
    # the neurons near the edges whose window runs off the end of the line
    b2_model.define_synapse_equation("I", "w : volt")
    b2_model.define_synapse_origin("I", "P")
    b2_model.define_synapse_target("I", "P")
    b2_model.define_synapse_j("I", "k for k in range(i - 12, i + 13) if abs(k - i) > 4")
    b2_model.define_synapse_skip_if_invalid("I", True)
    b2_model.define_synapse_pre("I", "v_post += w")
    b2_model.define_init("I", "w", "w_inh")

    # initial conditions
    b2_model.define_init("P", "v", "v0")

    # monitors
    b2_model.define_monitor_source("trace", "P")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")

    b2_model.define_monitor_source("spike", "P")
    b2_model.define_monitor_type("spike", "SpikeMonitor")
    b2_model.define_monitor_variable("spike", "v")

    b2_model.define_monitor_source("synapse", "E")
    b2_model.define_monitor_type("synapse", "StateMonitor")
    b2_model.define_monitor_variable("synapse", "w")

    # brief stimulus to a central group of neurons to seed activity
    tfinal = 300 * b2.ms
    i_stim = np.zeros((int(tfinal / b2.ms), n_neurons), dtype="float")
    i_stim[10:40, 28:32] = 600.0
    project.add_stim_array(
        StimArray(
            name="I_external",
            stim_type=StimType.External,
            values=i_stim,
            units="pA",
            dt=0.001,
        )
    )

    return project
