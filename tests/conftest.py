import brian2 as b2
import numpy as np
import pytest


@pytest.fixture()
def reference_system(tmp_path):
    # this is a version of the basic system from https://github.com/formozov/NBC_with_STDP/blob/main/NBC_large_24E6I_beta_version_COBAHH_mod_STDP_version_stable_w_plasticity_disconnected_inhibition.py

    data = np.array(
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype="int"
    )
    data_inv = np.array(~np.array(data, dtype="bool"), dtype="int")

    tfinal = 2000 * b2.ms

    defaultclock.dt = 0.1 * b2.ms

    stim = np.zeros((int(tfinal / b2.ms), 30), dtype="float")
    stim[:, :] = 0
    stim_constant = 300

    for l in range(3):
        for k in range(20):
            stim[200 * l + 120 : 200 * l + 121, k + 4] = stim_constant * data[k]
            stim[200 * l + 118 : 200 * l + 119, k + 4] = stim_constant * data_inv[k]
        stim[200 * l + 119 : 200 * l + 120, 0] = stim_constant

    shift = 20

    stimulus = b2.TimedArray(stim * b2.nS, dt=1 * b2.ms)

    stim_inh = np.zeros(int(tfinal / b2.ms), dtype="float")

    stim_inh[0:500] = 0
    stim_inh[500:-1] = 0

    stimulus_inh = b2.TimedArray(
        stim_inh * b2.nS, dt=1 * b2.ms
    )  # in principle, I can introduce a control of inhibitory neurons and then release them directly (!)

    area = 20000 * b2.umetre**2
    Cm = (1 * b2.ufarad * b2.cm**-2) * area
    gl = (5e-5 * b2.siemens * b2.cm**-2) * area

    El = -60 * b2.mV
    EK = -90 * b2.mV
    ENa = 50 * b2.mV
    g_na = (100 * b2.msiemens * b2.cm**-2) * area
    g_kd = (30 * b2.msiemens * b2.cm**-2) * area
    VT = -63 * b2.mV
    # Time constants
    taue = 5 * b2.ms
    taui = 10 * b2.ms
    # Reversal potentials
    Ee = 0 * b2.mV
    Ei = -80 * b2.mV

    Eiopto = -80 * b2.mV  # −69.5*mV
    Eeopto = (
        0 * b2.mV
    )  # Theoretical Analysis of Low-power Bidirectional Optogenetic Control of High-frequency Neural Codes with Single Spike Resolution

    eqs = b2.Equations(
        """
    dv/dt = (gl*(El-v)+ge*(Ee-v)+geopto*(Eeopto-v)+gi*(Ei-v)+giopto*(Eiopto-v)-
            g_na*(m*m*m)*h*(v-ENa)-
            g_kd*(n*n*n*n)*(v-EK))/Cm : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dn/dt = alpha_n*(1-n)-beta_n*n : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dge/dt = -ge*(1./taue) : siemens
    dgi/dt = -gi*(1./taui) : siemens
    giopto = stimulus_inh(t): siemens
    geopto = stimulus(t, i): siemens
    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    """
    )

    P = b2.NeuronGroup(
        30,
        model=eqs,
        threshold="v>-20*mV",
        refractory=3 * b2.ms,
        method="exponential_euler",
    )
    Pe = P[:24]
    Pi = P[24:]

    taupre = 20 * b2.ms
    taupost = taupre

    gmax = 60 * b2.nS
    dApre = 6 * b2.nS

    dApost = -dApre * taupre / taupost * 1.05

    # connectivity?
    Ce = b2.Synapses(
        Pe,
        Pe,
        """we : siemens
                    dApre/dt = -Apre / taupre : siemens (event-driven)
                    dApost/dt = -Apost / taupost : siemens (event-driven)""",
        on_pre="""ge += we
                        Apre += dApre
                        we = clip(we + Apost, -gmax, gmax)""",
        on_post="""Apost += dApost
                        we= clip(we + Apre, -gmax, gmax)""",
    )

    # input and output connections?
    Cei = b2.Synapses(Pe, Pi, """we = 6*nS : siemens""", on_pre="ge+=we")
    Cie = b2.Synapses(Pi, Pe, """wi = 67*nS : siemens""", on_pre="gi+=wi")

    Ce.connect("i != j", p=1)
    # Ci.connect('i != j',p=1)   # I disconneted inhibition (with 'i !- j there was winner take all machanism in inhibition')
    Cei.connect(p=1)
    Cie.connect(p=1)

    b2.seed(4321)
    P.v = "El + (randn() * 5 - 5)*mV"
    P.ge = "(randn() * 1.5 + 4) * 10.*nS"
    Pe.gi = "(randn() * 12 + 20) * 10.*nS"

    Traces_v = b2.StateMonitor(P, "v", record=True)

    spikemon = b2.SpikeMonitor(P)

    synmon = b2.StateMonitor(Ce, "we", record=True)

    b2.run(tfinal, report="text")

    yield Traces_v, spikemon, synmon, tmp_path
