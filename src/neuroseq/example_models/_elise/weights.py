#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt


class Weights(ABC):
    """
    Base class for weight matrices in neural networks.
    """

    @abstractmethod
    def __init__(self, weight_params: dict):
        self.weight_matrix = None
        self.delays = None
        pass

    def __call__(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Create weight matrix.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created weight matrix and associated delays.
        :rtype: Tuple[npt.NDArray]
        """
        self.weight_matrix = self._create_weight_matrix(num_vis, num_lat)
        self.delays = self._create_delays(num_vis, num_lat)

        return self.weight_matrix, self.delays

    def _create_delays(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        "Create the delays associated with the weight matrix"
        """
        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: A tuple containing the created delays.
        :rtype: Tuple[npt.NDArray]
        """

        num_total = num_vis + num_lat
        d_min = self.d_range[0]
        d_max = self.d_range[1]
        delays = self.rng_d.integers(d_min, d_max, num_total)

        return delays

    @abstractmethod
    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        pass


class DendriticWeights(Weights):
    """
    Class for creating dendritic weight matrices.

    This class extends the base Weights class and provides functionality to create
    dendritic weight matrices based on the given configuration.

    :param weight_params: Configuration object containing weight parameters.
    :type weight_params: WeightConfig
    """

    def __init__(self, weight_params: dict, rng_w, rng_d):
        """
        Initialize the DendriticWeights object.

        :param weight_params: Configuration object containing weight parameters.
        :type weight_params: WeightConfig
        """
        super().__init__(weight_params)
        self.delays = None
        self.W_vis_vis = weight_params.W_vis_vis
        self.W_vis_lat = weight_params.W_vis_lat
        self.W_lat_vis = weight_params.W_lat_vis
        self.W_lat_lat = weight_params.W_lat_lat
        self.d_range = weight_params.d_den
        self.rng_w = rng_w
        self.rng_d = rng_d

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the dendritic weight matrix.

        This method initializes a weight matrix and fills it with random values
        based on the configuration parameters.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        # Initialize weight matrix
        weights = np.zeros((num_vis + num_lat, num_vis + num_lat))
        weights[num_vis:, num_vis:] = self.rng_w.uniform(
            self.W_lat_lat[0], self.W_lat_lat[1], (num_lat, num_lat)
        )  # Lat to Lat
        weights[num_vis:, :num_vis] = self.rng_w.uniform(
            self.W_lat_vis[0], self.W_lat_vis[1], (num_lat, num_vis)
        )  # Lat to Vis
        weights[:num_vis, num_vis:] = self.rng_w.uniform(
            self.W_vis_lat[0], self.W_vis_lat[1], (num_vis, num_lat)
        )  # Vis to Lat
        weights[:num_vis, :num_vis] = self.rng_w.uniform(
            self.W_vis_vis[0], self.W_vis_vis[1], (num_vis, num_vis)
        )  # Vis to Vis

        # remove self connections
        np.fill_diagonal(weights, 0)

        return weights


class RandomSomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.
    Creates random connections based on a fixed probability.

    :param weight_params: Configuration object containing weight parameters.
    :type weight_params: WeightConfig
    """

    def __init__(self, weight_params: dict, rng_w, rng_d):
        """
        Initialize the SomaticWeights object.

        :param weight_params: Configuration object containing weight parameters.
        :type weight_params: WeightConfig
        """
        super().__init__(weight_params)
        self.delays = None
        self.d_range = weight_params.d_som
        self.p = weight_params.p
        self.rng_w = rng_w
        self.rng_d = rng_d
        self.inh_delay = weight_params.d_int

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the somatic weight matrix based on probabilistic connection rules.

        This method implements a complex algorithm to create connections between
        neurons based on various probabilities and rules.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        """

        num_total = num_vis + num_lat
        weight_matrix = np.zeros((num_total, num_total))
        for pre_idx in range(num_total):
            for post_idx in range(num_vis, num_total):
                if pre_idx != post_idx:
                    formation = self.rng_w.binomial(1, self.p)
                    if formation:
                        weight_matrix[post_idx, pre_idx] = 1

        return weight_matrix

    def create_interneuron_delays(self) -> npt.NDArray:
        # Take self.delays and add the interneuron delay to all entries
        self.inh_delay = self.delays + self.inh_delay

        return self.inh_delay


class SomaticWeights(Weights):
    """
    Class for creating somatic weight matrices.

    This class extends the base Weights class and provides functionality to create
    somatic weight matrices based on probabilistic connection rules.

    :param weight_params: Configuration object containing weight parameters.
    :type weight_params: WeightConfig
    """

    def __init__(self, weight_params: dict, rng_w, rng_d):
        """
        Initialize the SomaticWeights object.

        :param weight_params: Configuration object containing weight parameters.
        :type weight_params: WeightConfig
        """
        super().__init__(weight_params)
        self.delays = None
        self.d_range = weight_params.d_som
        self.p = weight_params.p
        self.q = weight_params.q
        self.p0 = weight_params.p0
        self.p_first = 1 - self.p0
        self.rng_w = rng_w
        self.rng_d = rng_d
        self.inh_delay = weight_params.d_int

    def _create_weight_matrix(self, num_vis: int, num_lat: int) -> Tuple[npt.NDArray]:
        """
        Create the somatic weight matrix based on probabilistic connection rules.

        This method implements a complex algorithm to create connections between
        neurons based on various probabilities and rules.

        :param num_vis: Number of visible neurons.
        :type num_vis: int
        :param num_lat: Number of lateral neurons.
        :type num_lat: int
        :return: The created weight matrix.
        :rtype: Tuple[npt.NDArray]
        """
        num_total = num_vis + num_lat
        neurons_outgoing = np.arange(num_total)
        neurons_incoming = np.arange(num_vis, num_total)
        weight_matrix = np.zeros((num_total, num_total))

        # Incoming connections
        connections_in = np.zeros(num_total)
        connections_in[:num_vis] = 1

        # Outgoing connections
        connections_out = np.zeros(num_total)

        # Neurons that can make a connection
        neurons_unspent = np.arange(num_total)
        neurons_looking = neurons_outgoing[connections_in > 0]

        while len(neurons_looking) > 0:
            for idx_pre in neurons_looking:
                while np.isin(idx_pre, neurons_looking):
                    # Probability for forming connection
                    if connections_out[idx_pre] == 0:
                        prob_out = 1 - self.p0
                    else:
                        prob_out = np.power(self.p, connections_out[idx_pre])

                    # Test for formation
                    formation = self.rng_w.binomial(1, prob_out)
                    if not formation:
                        # Remove neuron pre from list of unspent neurons
                        neurons_unspent = neurons_unspent[neurons_unspent != idx_pre]
                        neurons_looking = np.delete(neurons_looking, 0)
                    else:
                        # Exclude connections to self and to neurons already connected
                        possible_post = neurons_incoming[neurons_incoming != idx_pre]
                        neurons_connected = np.where(weight_matrix[:, idx_pre] == 1)[0]
                        possible_post = np.setdiff1d(possible_post, neurons_connected)

                        if self.q == 0:
                            possible_post = possible_post[
                                connections_in[possible_post] == 0
                            ]

                        if len(possible_post) == 0:
                            neurons_unspent = neurons_unspent[
                                neurons_unspent != idx_pre
                            ]
                            neurons_looking = neurons_looking[
                                neurons_looking != idx_pre
                            ]
                            break

                        max_attempts = 100
                        attempt = 0
                        formed = 0
                        while not formed and attempt < max_attempts:
                            attempt += 1
                            post_idx = self.rng_w.choice(possible_post)
                            prob_in = np.power(self.q, connections_in[post_idx])
                            accept = self.rng_w.binomial(1, prob_in)
                            if accept:
                                # Add connection to matrix
                                weight_matrix[post_idx, idx_pre] = 1
                                connections_out[idx_pre] += 1
                                connections_in[post_idx] += 1
                                if np.isin(post_idx, neurons_unspent):
                                    neurons_unspent = neurons_unspent[
                                        neurons_unspent != post_idx
                                    ]
                                    neurons_looking = np.append(
                                        neurons_looking, post_idx
                                    )
                                formed = 1
                            else:
                                continue

                        if not formed:
                            neurons_unspent = neurons_unspent[
                                neurons_unspent != idx_pre
                            ]
                            neurons_looking = neurons_looking[
                                neurons_looking != idx_pre
                            ]

        if np.sum(weight_matrix) != np.sum(connections_in) - num_vis:
            raise ValueError(
                "weight matrix sum does not match expected connection count."
            )

        return weight_matrix

    def create_interneuron_delays(self) -> npt.NDArray:
        # Take self.delays and add the interneuron delay to all entries
        self.inh_delay = self.delays + self.inh_delay

        return self.inh_delay
