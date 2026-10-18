import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


@register_example_project(
    "Adaptive Exponential Integrate-and-Fire Neuron", section="Single Neuron"
)
def single_adex_neuron() -> Project:
    """A regular-spiking AdEx neuron driven by a sustained current step."""
    project = Project()
    b2_model = project.b2_model

    # Regular-spiking parameters from Brette and Gerstner (2005).  The
    # adaptation current ``w`` grows both below threshold and at every spike,
    # making the firing rate visibly slow during the current step.
    b2_model.define_parameter("C", 281 * b2.pF)
    b2_model.define_parameter("gL", 30 * b2.nS)
    b2_model.define_parameter("EL", -70.6 * b2.mV)
    b2_model.define_parameter("VT_adex", -50.4 * b2.mV)
    b2_model.define_parameter("DeltaT", 2 * b2.mV)
    b2_model.define_parameter("tau_w", 144 * b2.ms)
    b2_model.define_parameter("a_adex", 4 * b2.nS)
    b2_model.define_parameter("b_adex", 80.5 * b2.pA)
    b2_model.define_parameter("V_reset", -70.6 * b2.mV)

    b2_model.parameter_descriptions.update(
        {
            "C": "Membrane capacitance.",
            "gL": "Leak conductance.",
            "EL": "Leak reversal potential.",
            "VT_adex": "AdEx exponential threshold potential.",
            "DeltaT": "AdEx spike-slope factor.",
            "tau_w": "Adaptation-current time constant.",
            "a_adex": "Subthreshold adaptation conductance.",
            "b_adex": "Spike-triggered adaptation-current increment.",
            "V_reset": "Membrane potential applied after a spike.",
        }
    )

    equations = """
    dv/dt = (gL*(EL-v) + gL*DeltaT*exp((v-VT_adex)/DeltaT) - w
             + I_external(t, i))/C : volt (unless refractory)
    dw/dt = (a_adex*(v-EL) - w)/tau_w : amp
    """
    b2_model.define_neuronnumber("neuron", 1)
    b2_model.define_equation("neuron", equations)
    b2_model.define_solver_method("neuron", "euler")
    b2_model.define_threshold("neuron", "v > VT_adex + 5*DeltaT")
    b2_model.define_reset("neuron", "v = V_reset; w += b_adex")
    b2_model.define_refractory("neuron", 2 * b2.ms)
    b2_model.define_dt("neuron", 0.1 * b2.ms)
    b2_model.define_init("neuron", "v", "EL")
    b2_model.define_init("neuron", "w", "0*pA")

    b2_model.define_monitor_source("trace", "neuron")
    b2_model.define_monitor_type("trace", "StateMonitor")
    b2_model.define_monitor_variable("trace", "v")
    b2_model.define_monitor_source("adaptation", "neuron")
    b2_model.define_monitor_type("adaptation", "StateMonitor")
    b2_model.define_monitor_variable("adaptation", "w")
    b2_model.define_monitor_source("spikes", "neuron")
    b2_model.define_monitor_type("spikes", "SpikeMonitor")
    b2_model.define_monitor_variable("spikes", "v")

    current = np.zeros((1000, 1), dtype="float")
    current[200:800, 0] = 700
    project.add_stim_array(
        StimArray(
            name="I_external",
            stim_type=StimType.External,
            values=current,
            units="pA",
            dt=0.001,
        )
    )
    return project
