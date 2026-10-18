import brian2 as b2
import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.project import Project
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType


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
