import numpy as np

from neuroseq.rheobase import bisection_converged
from neuroseq.rheobase import count_spikes
from neuroseq.rheobase import next_bracket


def test_count_spikes_counts_upward_crossings() -> None:
    trace = np.array([-70.0, -30.0, -10.0, 30.0, 10.0, -65.0, -19.0, 20.0])

    assert count_spikes(trace, threshold_mv=-20.0) == 2


def test_count_spikes_counts_plateau_once() -> None:
    trace = np.array([-70.0, -19.0, -18.0, -17.0, -65.0])

    assert count_spikes(trace, threshold_mv=-20.0) == 1


def test_count_spikes_ignores_subthreshold_bumps() -> None:
    trace = np.array([-70.0, -45.0, -25.0, -50.0])

    assert count_spikes(trace, threshold_mv=-20.0) == 0


def test_count_spikes_short_or_empty_returns_zero() -> None:
    assert count_spikes(np.array([])) == 0
    assert count_spikes(np.array([-70.0])) == 0


def test_next_bracket_moves_high_down_on_spike() -> None:
    assert next_bracket(0.0, 4.0, 2.0, spiked=True) == (0.0, 2.0)


def test_next_bracket_moves_low_up_on_no_spike() -> None:
    assert next_bracket(0.0, 4.0, 2.0, spiked=False) == (2.0, 4.0)


def test_bisection_converged() -> None:
    assert bisection_converged(1.0, 1.04, 0.05)
    assert not bisection_converged(1.0, 1.06, 0.05)
