from __future__ import annotations

from dataclasses import dataclass

from neuroseq.pretty_print_enum import PrettyPrintEnum


class StimType(PrettyPrintEnum):
    Excitation = 1
    Inhibition = 2
    External = 3


class DrawStimType(PrettyPrintEnum):
    Constant = 1
    Linear = 2
    Logarithmic = 3
    Random = 4


@dataclass
class DrawStim:
    draw_stim_type: DrawStimType
    value1: float
    value2: float
