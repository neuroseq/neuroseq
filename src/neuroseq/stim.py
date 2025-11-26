from __future__ import annotations

from dataclasses import dataclass

import brian2.numpy_ as np

from neuroseq.pretty_print_enum import PrettyPrintIntEnum


class StimType(PrettyPrintIntEnum):
    Excitation = 1
    Inhibition = 2
    External = 3


@dataclass
class StimArray:
    """A data class representing a set of stimuli.

    Attributes:
        name (str): The name of the data set.
        stim_type (StimType): The type of the stimuli (Excitation, Inhibition, External).
        values (np.ndarray): A 2D array of stimuli values where the first index is the neuron index and the second index is the time index.
        units (str): The unit of the stimuli (e.g., 'mV', 'pA').
        dt (float): The time step in seconds between consecutive timestamps.
    """

    name: str
    stim_type: StimType
    values: np.ndarray
    units: str
    dt: float


class DrawStimType(PrettyPrintIntEnum):
    Constant = 1
    Linear = 2
    Logarithmic = 3
    Random = 4


@dataclass
class DrawStim:
    draw_stim_type: DrawStimType
    value1: float
    value2: float
