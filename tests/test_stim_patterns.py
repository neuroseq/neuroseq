from __future__ import annotations

import numpy as np
import pytest

from neuroseq.stim import DrawStimType
from neuroseq.stim_patterns import generate_fuer_elise
from neuroseq.stim_patterns import generate_per_neuron
from neuroseq.stim_patterns import generate_ramp
from neuroseq.stim_patterns import generate_sequential
from neuroseq.stim_patterns import generate_simultaneous


def test_sequential_default_matches_getting_started() -> None:
    result = generate_sequential(
        n_neurons=10,
        pulse_ms=100,
        ipi_ms=18,
        magnitude=1.0,
        dt_seconds=0.001,
    )
    assert result.shape == (1180, 10)
    for i in range(10):
        t0 = i * 118
        assert np.all(result[t0 : t0 + 100, i] == 1.0)
        assert np.all(result[t0 + 100 : t0 + 118, i] == 0.0)
        for j in range(10):
            if j != i:
                assert np.all(result[t0 : t0 + 100, j] == 0.0)


def test_sequential_includes_trailing_ipi() -> None:
    result = generate_sequential(3, 10, 5, 1.0, 0.001)
    assert np.all(result[-5:] == 0.0)
    assert result.shape == (45, 3)


def test_simultaneous_all_columns_fire_together() -> None:
    result = generate_simultaneous(4, 50, 10, 0.7, repeats=3, dt_seconds=0.001)
    assert result.shape == (180, 4)
    for repeat_index in range(3):
        t0 = repeat_index * 60
        assert np.all(result[t0 : t0 + 50, :] == 0.7)
        assert np.all(result[t0 + 50 : t0 + 60, :] == 0.0)


def test_fuer_elise_uses_first_13_neurons() -> None:
    result = generate_fuer_elise(13, 100, 18, 1.0, repeats=1, dt_seconds=0.001)
    assert result.shape == (24 * 118, 13)
    assert np.all(result[:100, 0] == 1.0)
    assert np.all(result[:100, 1:] == 0.0)
    t0 = 8 * 118
    assert np.all(result[t0 : t0 + 100, 5] == 1.0)
    assert np.all(result[t0 : t0 + 100, 11] == 1.0)
    for col in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12]:
        assert np.all(result[t0 : t0 + 100, col] == 0.0)


def test_fuer_elise_clips_to_available_neurons() -> None:
    result = generate_fuer_elise(6, 100, 18, 1.0, repeats=1, dt_seconds=0.001)
    assert result.shape == (24 * 118, 6)
    t0 = 8 * 118
    assert np.all(result[t0 : t0 + 100, 5] == 1.0)
    for col in range(5):
        assert np.all(result[t0 : t0 + 100, col] == 0.0)


def test_fuer_elise_repeats_with_trailing_ipi_between() -> None:
    result = generate_fuer_elise(13, 10, 5, 1.0, repeats=2, dt_seconds=0.001)
    assert result.shape == (720, 13)
    assert np.all(result[24 * 15 - 5 : 24 * 15, :] == 0.0)


def test_per_neuron_constant() -> None:
    result = generate_per_neuron(
        neuron_index=2,
        n_neurons=4,
        pulse_ms=100,
        ipi_ms=18,
        profile=DrawStimType.Constant,
        value1=0.5,
        value2=0.0,
        repeats=3,
        dt_seconds=0.001,
    )
    assert result.shape == (3 * 118, 4)
    for repeat_index in range(3):
        t0 = repeat_index * 118
        assert np.all(result[t0 : t0 + 100, 2] == 0.5)
        assert np.all(result[t0 + 100 : t0 + 118, 2] == 0.0)
    for col in [0, 1, 3]:
        assert np.all(result[:, col] == 0.0)


def test_per_neuron_linear_intensity_ramps() -> None:
    result = generate_per_neuron(
        0,
        1,
        10,
        5,
        DrawStimType.Linear,
        value1=0.0,
        value2=1.0,
        repeats=5,
        dt_seconds=0.001,
    )
    expected = np.linspace(0.0, 1.0, 5)
    for repeat_index in range(5):
        t0 = repeat_index * 15
        assert np.all(result[t0 : t0 + 10, 0] == expected[repeat_index])


def test_per_neuron_logarithmic_requires_positive() -> None:
    with pytest.raises(ValueError):
        generate_per_neuron(
            0,
            1,
            10,
            5,
            DrawStimType.Logarithmic,
            value1=0.0,
            value2=1.0,
            repeats=3,
            dt_seconds=0.001,
        )


def test_per_neuron_random_seeded_reproducible() -> None:
    np.random.seed(123)
    result = generate_per_neuron(
        0,
        1,
        10,
        5,
        DrawStimType.Random,
        value1=0.2,
        value2=0.8,
        repeats=10,
        dt_seconds=0.001,
    )
    for repeat_index in range(10):
        value = result[repeat_index * 15, 0]
        assert 0.2 <= value <= 0.8


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        generate_sequential(5, 0, 5, 1.0, 0.001)
    with pytest.raises(ValueError):
        generate_sequential(5, 10, -1, 1.0, 0.001)
    with pytest.raises(ValueError):
        generate_simultaneous(5, 10, 5, 1.0, 0, 0.001)


def test_ramp_all_neurons_broadcasts_same_ramp() -> None:
    result = generate_ramp(
        n_neurons=4,
        ramp_ms=10,
        ipi_ms=5,
        start_magnitude=0.0,
        end_magnitude=1.0,
        repeats=2,
        dt_seconds=0.001,
    )
    assert result.shape == (2 * (10 + 5), 4)
    expected = np.linspace(0.0, 1.0, 10)
    for repeat_index in range(2):
        t0 = repeat_index * 15
        for col in range(4):
            np.testing.assert_allclose(result[t0 : t0 + 10, col], expected)
        assert np.all(result[t0 + 10 : t0 + 15] == 0.0)


def test_ramp_specific_neuron_only_target_column() -> None:
    result = generate_ramp(
        n_neurons=4,
        ramp_ms=10,
        ipi_ms=5,
        start_magnitude=0.0,
        end_magnitude=1.0,
        repeats=2,
        dt_seconds=0.001,
        neuron_index=2,
    )
    assert result.shape == (30, 4)
    expected = np.linspace(0.0, 1.0, 10)
    for repeat_index in range(2):
        t0 = repeat_index * 15
        np.testing.assert_allclose(result[t0 : t0 + 10, 2], expected)
        for col in [0, 1, 3]:
            assert np.all(result[t0 : t0 + 10, col] == 0.0)


def test_ramp_includes_trailing_ipi() -> None:
    result = generate_ramp(
        n_neurons=2,
        ramp_ms=10,
        ipi_ms=5,
        start_magnitude=0.0,
        end_magnitude=1.0,
        repeats=1,
        dt_seconds=0.001,
    )
    assert result.shape == (15, 2)
    assert np.all(result[10:15] == 0.0)


def test_ramp_invalid_neuron_index_raises() -> None:
    with pytest.raises(ValueError):
        generate_ramp(
            n_neurons=2,
            ramp_ms=10,
            ipi_ms=5,
            start_magnitude=0.0,
            end_magnitude=1.0,
            repeats=1,
            dt_seconds=0.001,
            neuron_index=5,
        )
