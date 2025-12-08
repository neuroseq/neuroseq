# Example of an external simulation function that can be imported into neuroseq

import numpy as np
from brian2 import *

from neuroseq.data import TraceData, SpikeData, SynapseData, Experiment


def simulate(stims_dict: dict[str, np.ndarray], simulation_time: float) -> Experiment:
    # This function is provided by neuroseq with a dict of stimuli, and the desired simulation time.
    # stims_dict is a dict of key:data pairs, where data contains
    #   - "array": 2d numpy array of timestamps x neurons
    #   - "dt": the time step of the stimulus in seconds as a float

    simulation_time = int(
        1000 * simulation_time
    )  # convertion of seconds to milliseconds

    tfinal = simulation_time * ms

    defaultclock.dt = 0.1 * ms

    stim = stims_dict["stimulus"]["array"]  ###TODO: remove magic numebers
    stim = 10e9 * stim

    # stim = np.zeros((int(tfinal / ms), 30), dtype="float")
    # stim[:, :] = 0

    # stim_constant = 300

    ##### STORAGE #####

    # bw_train = 400
    # bw_num = 40

    # for l in range(4):
    #     for k in range(10):
    #         stim[
    #             bw_train * l + bw_num * k + 120 : bw_train * l + bw_num * k + 121, k + 4
    #         ] = stim_constant
    #         if (k % 2) == 0:
    #             stim[
    #                 bw_train * l + bw_num * k + 121 : bw_train * l + bw_num * k + 122, 0
    #             ] = stim_constant
    #         if (k % 2) == 1:
    #             stim[
    #                 bw_train * l + bw_num * k + 121 : bw_train * l + bw_num * k + 122, 1
    #             ] = stim_constant

    stimulus = TimedArray(stim * nS, dt=1 * ms)

    stim_inh = np.zeros(int(tfinal / ms), dtype="float")

    stimulus_inh = TimedArray(
        stim_inh * nS, dt=1 * ms
    )  # in principle, I can introduce a control of inhibitory neurons and then release them directly (!)

    # Parameters

    area = 20000 * umetre**2
    Cm = (1 * ufarad * cm**-2) * area
    gl = (5e-5 * siemens * cm**-2) * area

    El = -60 * mV
    EK = -90 * mV
    ENa = 50 * mV
    g_na = (100 * msiemens * cm**-2) * area
    g_kd = (30 * msiemens * cm**-2) * area
    VT = -63 * mV
    # Time constants
    taue = 5 * ms
    taui = 10 * ms
    # Reversal potentials
    Ee = 0 * mV
    Ei = -80 * mV

    Eiopto = -80 * mV  # −69.5*mV
    Eeopto = (
        0 * mV
    )  # Theoretical Analysis of Low-power Bidirectional Optogenetic Control of High-frequency Neural Codes with Single Spike Resolution

    # we = 6*nS  # excitatory synaptic weight
    # wi = 67*nS  # inhibitory synaptic weight

    # The model
    eqs = Equations("""
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
    """)

    P = NeuronGroup(
        30,
        model=eqs,
        threshold="v>-20*mV",
        refractory=3 * ms,
        method="exponential_euler",
    )
    Pe = P[:24]
    Pi = P[24:]

    taupre = 20 * ms
    taupost = taupre

    gmax = 60 * nS
    dApre = 6 * nS

    dApost = -dApre * taupre / taupost * 1.05

    Ce = Synapses(
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

    # Ce = Synapses(Pe, Pe,  '''we = 6*nS : siemens''', on_pre='ge+=we')
    # Ci = Synapses(Pi, Pi, '''wi = 67*nS : siemens''', on_pre='gi+=wi')

    Cei = Synapses(Pe, Pi, """we = 6*nS : siemens""", on_pre="ge+=we")
    Cie = Synapses(Pi, Pe, """wi = 67*nS : siemens""", on_pre="gi+=wi")

    Ce.connect("i != j", p=1)
    #
    # Ci.connect('i != j',p=1)   # I disconneted inhibition (with 'i !- j there was winner take all machanism in inhibition')
    Cei.connect(p=1)
    Cie.connect(p=1)

    seed(4321)

    # Initialization
    P.v = "El + (randn() * 5 - 5)*mV"
    P.ge = "(randn() * 1.5 + 4) * 10.*nS"
    Pe.gi = "(randn() * 12 + 20) * 10.*nS"

    # Record traces

    Traces_v = StateMonitor(P, "v", record=True)
    spikemon = SpikeMonitor(P)
    synmon = StateMonitor(Ce, "we", record=True, dt=10 * ms)  # ,

    run(tfinal, report="text")

    dt_trace = 0.1  # ms

    results = []

    results.append(
        TraceData(
            name="trace_example",
            traces=np.array(Traces_v.v),
            indices=np.array(Traces_v.record),
            dt=dt_trace / 1000.0,  # conversion to seconds
            units="",
        )
    )

    results.append(
        SpikeData(
            name="spike_example",
            spike_indices=np.array(spikemon.i),
            spike_times=np.array(spikemon.t / second),
        )
    )

    n_neurons_heatmap = 24
    dt_heatmap = 10
    n_timestamps_heatmap = int(float(simulation_time) / dt_heatmap)

    connect_mat = np.array(synmon.we.tolist())
    sq_connect_mat = np.zeros(
        (n_neurons_heatmap, n_neurons_heatmap, n_timestamps_heatmap), dtype="float"
    )
    shift = 0

    for i in range(n_neurons_heatmap):
        for j in range(n_neurons_heatmap):
            if i == j:
                shift = shift + 1
                sq_connect_mat[i, j, :] = 0
            else:
                for tt in range(n_timestamps_heatmap):
                    sq_connect_mat[i, j, tt] = connect_mat[
                        j + i * n_neurons_heatmap - shift, tt
                    ]

    results.append(
        SynapseData(
            name="heatmap_example",
            values=sq_connect_mat,
            timestamps=np.linspace(0, simulation_time / 1000.0, n_timestamps_heatmap),
        )
    )

    return Experiment("external brian2 example", simulation_time / 1000.0, [], results)
