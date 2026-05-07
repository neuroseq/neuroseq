#!/usr/bin/env python3
import pickle
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt

from .rate_buffer import Buffer
from .weights import DendriticWeights, SomaticWeights

# disable numba jit for debugging etc.
numba.config.DISABLE_JIT = False
np.seterr(all="raise")


class Network:
    def __init__(
        self,
        network_params: dict,
        neuron_params: dict,
        dendritic_weights: DendriticWeights,
        somatic_weights: SomaticWeights,
        rate_buffer: Buffer,
    ):
        self.neuron_params = neuron_params
        self.num_lat = network_params.num_lat
        self.num_vis = network_params.num_vis
        self.num_all = self.num_lat + self.num_vis

        self.dendritic_weights, self.dendritic_delays = dendritic_weights(
            num_vis=self.num_vis, num_lat=self.num_lat
        )
        self.somatic_weights, self.somatic_delays = somatic_weights(
            num_vis=self.num_vis, num_lat=self.num_lat
        )
        self.interneuron_delays = somatic_weights.create_interneuron_delays()

        self.r_rest = eq_phi(
            self.neuron_params.E_l, self.neuron_params.a, self.neuron_params.b
        )

        # Dynamical variables
        self.v = np.ones(self.num_all) * self.neuron_params.E_l
        self.u = np.ones(self.num_all) * self.neuron_params.E_l
        self.r_bar = np.ones(self.num_all) * self.r_rest
        self.r = np.ones(self.num_all) * self.r_rest
        self.r_den = np.ones(self.num_all) * self.r_rest
        self.r_exc = np.ones(self.num_all) * self.r_rest
        self.r_inh = np.ones(self.num_all) * self.r_rest

        self.view_visible = np.s_[: self.num_vis]
        self.view_latent = np.s_[self.num_vis :]
        self.view_vis_lat = np.s_[: self.num_vis, self.num_vis :]
        self.view_vis_vis = np.s_[: self.num_vis, : self.num_vis]
        self.view_lat_vis = np.s_[self.num_vis :, : self.num_vis]
        self.view_lat_lat = np.s_[self.num_vis :, self.num_vis :]

        self.views = {
            "all": np.s_[:],
            "visible": self.view_visible,
            "latent": self.view_latent,
            "vis_lat": self.view_vis_lat,
            "vis_vis": self.view_vis_vis,
            "lat_vis": self.view_lat_vis,
            "lat_lat": self.view_lat_lat,
        }

        self.dt = None

    def set_visible_activity(self, activity: int):
        """
        Set the activity of the visible neurons.

        :param activity: Activity to set the visible neurons to.
        :type activity: int
        """
        self.u[self.view_visible] = activity

    def reset_activity(self):
        """
        Reset the activity of the network to the resting state.
        """
        self.rate_buffer = Buffer(
            self.num_all, self._compute_buffer_depth(self.dt), self.r_rest
        )
        self.v.fill(self.neuron_params.E_l)
        self.u.fill(self.neuron_params.E_l)
        self.r_bar.fill(self.r_rest)
        self.r.fill(self.r_rest)
        self.r_den.fill(self.r_rest)
        self.r_exc.fill(self.r_rest)
        self.r_inh.fill(self.r_rest)

    def get_val(self, attribute_name, view="all"):  #
        """
        Retrieve the specified attribute for the given neuron type.

        :param attribute_name: Name attribute to retrieve (e.g., 'v', 'u', 'r', 'r_bar')
        :param neuron_type: Type of neurons to retrieve ('all', 'visible', or 'latent')
        :returns: Copy of the requested attribute values
        :rtype: numpy.ndarray

        :raises AttributeError: If the specified attribute doesn't exist
        :raises ValueError: If an invalid neuron_type is provided
        """
        if not hasattr(self, attribute_name):
            raise AttributeError(f"Attribute '{attribute_name}' does not exist")

        attribute = getattr(self, attribute_name)

        if not isinstance(attribute, np.ndarray):
            raise AttributeError(f"Attribute '{attribute_name}' is not an array")

        if view not in self.views:
            raise ValueError(
                "Invalid view. Must be 'all', 'visible', 'latent', \
                'vis_lat', 'vis_vis', 'lat_vis', or 'lat_lat'"
            )

        return np.copy(attribute[self.views[view]])

    def _compute_buffer_depth(self, dt):
        max_buffer_ms = max(max(self.dendritic_delays), max(self.interneuron_delays))
        buffer_depth = int(max_buffer_ms / dt)

        return buffer_depth

    def prepare_for_simulation(self, dt, optimizer_vis, optimizer_lat):
        # Configure rate buffer for simulation
        self.rate_buffer = Buffer(
            self.num_all, self._compute_buffer_depth(dt), self.r_rest
        )

        self.dt = dt
        self.optimizer_vis = optimizer_vis
        self.optimizer_lat = optimizer_lat
        self.dt_dendritic_delays = (self.dendritic_delays / dt).astype(int)
        self.dt_somatic_delays = (self.somatic_delays / dt).astype(int)
        self.dt_interneuron_delays = (self.interneuron_delays / dt).astype(int)

    def _compute_update(self, u_inp):
        # Compute delayed rates

        self.u_inp = u_inp
        self.r_den = self.rate_buffer.get(self.dt_dendritic_delays)
        self.r_exc = self.rate_buffer.get(self.dt_somatic_delays)
        self.r_inh = self.rate_buffer.get(self.dt_interneuron_delays)

        dudt, dvdt, dwdt, dr_bar_dt = total_diff_eq(
            u=self.u,
            v=self.v,
            w_den=self.dendritic_weights,
            r_bar=self.r_bar,
            r_den=self.r_den,
            r_exc=self.r_exc,
            r_inh=self.r_inh,
            u_inp=self.u_inp,
            w_som=self.somatic_weights,
            C_v=self.neuron_params.C_v,
            C_u=self.neuron_params.C_u,
            E_l=self.neuron_params.E_l,
            E_exc=self.neuron_params.E_exc,
            E_inh=self.neuron_params.E_inh,
            g_l=self.neuron_params.g_l,
            g_den=self.neuron_params.g_den,
            g_exc_0=self.neuron_params.g_exc_0,
            g_inh_0=self.neuron_params.g_inh_0,
            a=self.neuron_params.a,
            b=self.neuron_params.b,
            lam=self.neuron_params.lam,
        )

        return dudt, dvdt, dwdt, dr_bar_dt

    def _update_weights(self, dwdt):
        dwdt_full = self.optimizer_lat.get_update(self.dendritic_weights, dwdt)
        dwdt_vis = self.optimizer_vis.get_update(
            self.dendritic_weights[self.view_vis_vis], dwdt[self.view_vis_vis]
        )
        dwdt_full[self.view_vis_vis] = dwdt_vis

        self.dendritic_weights += dwdt_full * self.dt

    def _update_dynamic_variables(self, dudt, dvdt, dr_bar_dt):
        self.u += dudt * self.dt
        self.v += dvdt * self.dt
        self.r_bar += dr_bar_dt * self.dt

    def _update_rates_and_buffer(self):
        new_r = eq_phi(self.u, self.neuron_params.a, self.neuron_params.b)
        self.r = new_r
        self.rate_buffer.roll(new_r)

    def __call__(self, u_inp, learn: bool = True, u_noise_gen=None, w_noise_gen=None):
        dudt, dvdt, dwdt, dr_bar_dt = self._compute_update(u_inp)

        if u_noise_gen is not None:
            dudt = u_noise_gen(dudt, dt=self.dt)

        if w_noise_gen is not None:
            dwdt = w_noise_gen(dwdt, dt=self.dt)

        self._update_dynamic_variables(dudt, dvdt, dr_bar_dt)
        if learn:
            self._update_weights(dwdt)
        self._update_rates_and_buffer()

    def save(self, path: str) -> None:
        """
        Save the tracker object to a file.

        :param path: File path to save the tracker.
        :type path: str
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a tracker object from a file.

        :param path: File path to load the tracker from.
        :type path: str
        :return: The loaded tracker object.
        :rtype: Tracker
        """
        with open(path, "rb") as f:
            return pickle.load(f)


#####################################
# equations of the dynamical system #
#####################################


@numba.njit
def eq_phi(u: npt.NDArray | float, a: float, b: float) -> npt.NDArray:
    """Activation function."""
    return 1.0 / (1 + np.exp(a * (b - u)))


@numba.njit()
def eq_i_den(r_delayed: npt.NDArray, w_den: npt.NDArray) -> npt.NDArray:
    """Dendritic synaptic current."""
    i_den = np.dot(w_den, r_delayed)
    return i_den


@numba.vectorize()
def eq_syn_cond(r_delay: float, g_0: float, phi_el: float) -> float:
    """Calc the synaptic conductances for the somatic weights."""
    if r_delay > phi_el:
        return r_delay * g_0
    else:
        return phi_el * g_0


@numba.njit()
def eq_rescale_v(v: npt.NDArray, g_l: float, g_den: float, E_l: float) -> npt.NDArray:
    """Rescale dendritic voltage to somatic for plasticity."""
    return (g_l * E_l + g_den * v) / (g_l + g_den)


@numba.njit()
def eq_drbar_dt(
    r_bar: npt.NDArray, r_delay: npt.NDArray, g_l: float, g_den: float
) -> npt.NDArray:
    """Diff. eq. for the lowpass filtered rate used in the plasticity."""
    return -g_l * r_bar + (g_l * g_den) / (g_l + g_den) * r_delay


@numba.njit()
def eq_i_som(
    w_som: npt.NDArray,
    g_exc: npt.NDArray,
    g_inh: npt.NDArray,
    u: npt.NDArray,
    E_exc: float,
    E_inh: float,
) -> npt.NDArray:
    """Calculate the somatic input current w/o teaching synapses."""
    i_som = np.dot(w_som, g_exc) * (E_exc - u) + np.dot(w_som, g_inh) * (E_inh - u)
    return i_som


@numba.njit()
def eq_cond_exc_inp(
    u_inp: npt.NDArray, lam: float, g_l: float, g_den: float, E_exc: float, E_inh: float
) -> npt.NDArray:
    """Calc the axcitatory conductances for the input."""
    g_exc = lam * (g_l + g_den) * (E_inh - u_inp) / ((1 - lam) * (E_inh - E_exc))
    return g_exc


@numba.njit()
def eq_cond_inh_inp(
    u_inp: npt.NDArray, lam: float, g_l: float, g_den: float, E_exc: float, E_inh: float
) -> npt.NDArray:
    """Calc the excitatory conductances for the input."""
    g_exc = lam * (g_l + g_den) * (u_inp - E_exc) / ((1 - lam) * (E_inh - E_exc))
    return g_exc


@numba.njit()
def eq_i_inp(
    g_exc_inp: npt.NDArray,
    g_inh_inp: npt.NDArray,
    u: npt.NDArray,
    E_exc: float,
    E_inh: float,
) -> npt.NDArray:
    """Calc the synaptic current of the input synapses.

    This is added only to the visible neurons.
    """
    i_inp = g_exc_inp * (E_exc - u) + g_inh_inp * (E_inh - u)
    return i_inp


@numba.njit()
def eq_dvdt(
    v: npt.NDArray, i_den: npt.NDArray, C_m: float, g_l: float, E_l: float
) -> npt.NDArray:
    """Diff. eq. for the dendritic potential v."""
    dvdt = (-g_l * (v - E_l) + i_den) / C_m
    return dvdt


@numba.njit()
def eq_dudt(
    u: npt.NDArray,
    v: npt.NDArray,
    i_som: npt.NDArray,
    C_m: float,
    g_l: float,
    g_den: float,
    E_l: float,
) -> npt.NDArray:
    """Diff. eq. for the somatic potential."""
    dudt = (-g_l * (u - E_l) + g_den * (v - u) + i_som) / C_m
    return dudt


@numba.njit()
def eq_dwdt(phi_u: npt.NDArray, phi_v: npt.NDArray, r_bar: npt.NDArray) -> npt.NDArray:
    """Urbanczik-Senn plasiticity rule."""
    dwdt = np.outer(phi_u - phi_v, r_bar)
    np.fill_diagonal(dwdt, 0)

    return dwdt


@numba.njit()
def total_diff_eq(
    u: npt.NDArray,
    v: npt.NDArray,
    w_den: npt.NDArray,
    r_bar: npt.NDArray,
    r_den: npt.NDArray,
    r_exc: npt.NDArray,
    r_inh: npt.NDArray,
    u_inp: npt.NDArray,
    w_som: npt.NDArray,
    C_u: float,
    C_v: float,
    g_l: float,
    g_den: float,
    g_exc_0: float,
    g_inh_0: float,
    E_l: float,
    E_exc: float,
    E_inh: float,
    a: float,
    b: float,
    lam: float,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Calculate the total differential equations for a neural network model.

    This function computes the time derivatives of membrane potentials, synaptic
    weights, and firing rates for a network of neurons with dendritic and somatic
    compartments.

    :param u: Somatic membrane potentials, dynamic model variable (1-dim vector of size N)  # noqa
    :type u: numpy.ndarray
    :param v: Dendritic membrane potentials, dynamic model variable (1-dim vector of size N)  # noqa
    :type v: numpy.ndarray
    :param w_den: Dendritic synaptic weights, dynamic model variable (2-dim matrix of size NxN)  # noqa
    :type w_den: numpy.ndarray
    :param r_bar: Average firing rates, dynamic model variable (1-dim vector of size N)  # noqa
    :type r_bar: numpy.ndarray
    :param r_den: Delayed dendritic firing rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_den: numpy.ndarray
    :param r_exc: delayed somatic excitatory rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_exc: numpy.ndarray
    :param r_inh: delayed somatic inhibitory rates, instanteous variable (1-dim vector of size N)  # noqa
    :type r_inh: numpy.ndarray
    :param u_inp: Input membrane potentials, instanteous variable (1-dim vector of size N_visible)  # noqa
    :type u_inp: numpy.ndarray
    :param w_som: Sparse somatic synaptic weights, fixed model parameter (2-dim matrix of size NxN)  # noqa
    :type w_som: numpy.ndarray
    :param C_u: Somatic membrane capacitance, fixed model parameter
    :type C_u: float
    :param C_v: Dendritic membrane capacitance, fixed model parameter
    :type C_v: float
    :param g_l: Leak conductance, fixed model parameter
    :type g_l: float
    :param g_den: Dendritic coupling conductance, fixed model parameter
    :type g_den: float
    :param g_exc_0: Baseline excitatory conductance, fixed model parameter
    :type g_exc_0: float
    :param g_inh_0: Baseline inhibitory conductance, fixed model parameter
    :type g_inh_0: float
    :param E_l: Leak reversal potential, fixed model parameter
    :type E_l: float
    :param E_exc: Excitatory reversal potential, fixed model parameter
    :type E_exc: float
    :param E_inh: Inhibitory reversal potential, fixed model parameter
    :type E_inh: float
    :param a: Activation function parameter, fixed model parameter
    :type a: float
    :param b: Activation function parameter, fixed model parameter
    :type b: float
    :param lam: Input scaling factor, fixed model parameter
    :type lam: float

    :return: Tuple containing time derivatives of u, v, w, and r_bar
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

    This function implements various equations for neural dynamics, including:
    - Dendritic and somatic currents
    - Synaptic conductances
    - Membrane potential dynamics
    - Firing rate dynamics
    - Synaptic plasticity

    The function uses helper functions (not shown here) to compute individual components
    of the model, such as eq_i_den, eq_syn_cond, eq_i_som, etc.
    """
    phi_el = eq_phi(E_l, a, b)

    # somato-dendritic connections
    i_den = eq_i_den(r_den, w_den)  # eq. 8

    # somato-somatic conentions
    g_exc = eq_syn_cond(r_exc, g_exc_0, phi_el)  # eq. 10
    g_inh = eq_syn_cond(r_inh, g_inh_0, phi_el)  # eq. 11
    i_som = eq_i_som(w_som, g_exc, g_inh, u, E_exc, E_inh)

    if u_inp is not None:
        # input to visible layer
        num_vis = len(u_inp)
        g_exc_inp = eq_cond_exc_inp(u_inp, lam, g_l, g_den, E_exc, E_inh)
        g_inh_inp = eq_cond_inh_inp(u_inp, lam, g_l, g_den, E_exc, E_inh)
        i_som_inp = eq_i_inp(g_exc_inp, g_inh_inp, u[:num_vis], E_exc, E_inh)
        # total somatic input current:
        i_som[:num_vis] += i_som_inp

    # differential equations:
    dvdt = eq_dvdt(v, i_den, C_v, g_l, E_l)
    dudt = eq_dudt(u, v, i_som, C_u, g_l, g_den, E_l)
    dr_bar_dt = eq_drbar_dt(r_bar, r_den, g_l, g_den)
    # plasticity
    phi_u = eq_phi(u, a, b)
    v_rescaled = eq_rescale_v(v, g_l, g_den, E_l)
    phi_v = eq_phi(v_rescaled, a, b)
    dwdt = eq_dwdt(phi_u, phi_v, r_bar)

    return dudt, dvdt, dwdt, dr_bar_dt
