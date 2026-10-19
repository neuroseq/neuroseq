from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from neuroseq.stimulation.static_input.stim_patterns import generate_fuer_elise
from neuroseq.stimulation.static_input.stim_patterns import generate_per_neuron
from neuroseq.stimulation.static_input.stim_patterns import generate_ramp
from neuroseq.stimulation.static_input.stim_patterns import generate_sequential
from neuroseq.stimulation.static_input.stim_patterns import generate_simultaneous
from neuroseq.stimulation.static_input.stim_patterns import parse_neuron_indices
from neuroseq.stimulation.stim import DrawStimType


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


def test_sequential_specific_neurons_fire_in_given_order() -> None:
    """Feature (#582): a neuron subset fires in the order given, not just
    the full 0..n_neurons-1 range."""
    result = generate_sequential(
        n_neurons=5,
        pulse_ms=10,
        ipi_ms=5,
        magnitude=1.0,
        dt_seconds=0.001,
        neuron_index=[3, 1],
    )
    assert result.shape == (30, 5)
    assert np.all(result[0:10, 3] == 1.0)
    assert np.all(result[10:15, :] == 0.0)
    assert np.all(result[15:25, 1] == 1.0)
    for col in [0, 2, 4]:
        assert np.all(result[:, col] == 0.0)
    assert np.all(result[0:10, [0, 1, 2, 4]] == 0.0)
    assert np.all(result[15:25, [0, 2, 3, 4]] == 0.0)


def test_sequential_single_neuron_index() -> None:
    result = generate_sequential(
        n_neurons=4,
        pulse_ms=10,
        ipi_ms=5,
        magnitude=1.0,
        dt_seconds=0.001,
        neuron_index=2,
    )
    assert result.shape == (15, 4)
    assert np.all(result[0:10, 2] == 1.0)
    for col in [0, 1, 3]:
        assert np.all(result[:, col] == 0.0)


def test_sequential_invalid_neuron_index_raises() -> None:
    with pytest.raises(ValueError):
        generate_sequential(
            n_neurons=3,
            pulse_ms=10,
            ipi_ms=5,
            magnitude=1.0,
            dt_seconds=0.001,
            neuron_index=[0, 5],
        )


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


def test_fuer_elise_pulse_repetitions_repeat_each_event() -> None:
    result = generate_fuer_elise(
        13,
        10,
        5,
        1.0,
        repeats=2,
        dt_seconds=0.001,
        pulse_repetitions=3,
    )

    assert result.shape == (2 * 24 * 3 * 15, 13)
    for pulse_index in range(3):
        t0 = pulse_index * 15
        assert np.all(result[t0 : t0 + 10, 0] == 1.0)
        assert np.all(result[t0 + 10 : t0 + 15, :] == 0.0)
    # The second musical event starts only after all repetitions of the first.
    assert np.all(result[45:55, 1] == 1.0)


def test_fuer_elise_rejects_invalid_pulse_repetitions() -> None:
    with pytest.raises(ValueError, match="pulse_repetitions"):
        generate_fuer_elise(
            13,
            10,
            5,
            1.0,
            repeats=1,
            dt_seconds=0.001,
            pulse_repetitions=0,
        )


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


def test_per_neuron_multiple_indices_sets_each_column() -> None:
    result = generate_per_neuron(
        neuron_index=[0, 2, 3],
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
        for col in [0, 2, 3]:
            assert np.all(result[t0 : t0 + 100, col] == 0.5)
    assert np.all(result[:, 1] == 0.0)


def test_per_neuron_empty_indices_raises() -> None:
    with pytest.raises(ValueError):
        generate_per_neuron(
            neuron_index=[],
            n_neurons=4,
            pulse_ms=100,
            ipi_ms=18,
            profile=DrawStimType.Constant,
            value1=0.5,
            value2=0.0,
            repeats=1,
            dt_seconds=0.001,
        )


def test_ramp_multiple_neurons_sets_each_column() -> None:
    result = generate_ramp(
        n_neurons=5,
        ramp_ms=10,
        ipi_ms=5,
        start_magnitude=0.0,
        end_magnitude=1.0,
        repeats=2,
        dt_seconds=0.001,
        neuron_index=[1, 3],
    )
    assert result.shape == (30, 5)
    expected = np.linspace(0.0, 1.0, 10)
    for repeat_index in range(2):
        t0 = repeat_index * 15
        for col in [1, 3]:
            np.testing.assert_allclose(result[t0 : t0 + 10, col], expected)
        for col in [0, 2, 4]:
            assert np.all(result[t0 : t0 + 10, col] == 0.0)


def test_ramp_multiple_neurons_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        generate_ramp(
            n_neurons=4,
            ramp_ms=10,
            ipi_ms=5,
            start_magnitude=0.0,
            end_magnitude=1.0,
            repeats=1,
            dt_seconds=0.001,
            neuron_index=[1, 9],
        )


def test_parse_neuron_indices_mixed_indices_and_ranges() -> None:
    assert parse_neuron_indices("0, 2, 5-8", 10) == [0, 2, 5, 6, 7, 8]


def test_parse_neuron_indices_dedupes_and_sorts() -> None:
    assert parse_neuron_indices("3 1 2-3 1", 10) == [1, 2, 3]


def test_parse_neuron_indices_reversed_range() -> None:
    assert parse_neuron_indices("8-5", 10) == [5, 6, 7, 8]


def test_parse_neuron_indices_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_neuron_indices("   ", 10)


def test_parse_neuron_indices_non_integer_raises() -> None:
    with pytest.raises(ValueError):
        parse_neuron_indices("0, x", 10)


def test_parse_neuron_indices_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        parse_neuron_indices("0, 10", 10)


def test_generate_sequential_repeats_tile_the_whole_sequence() -> None:
    single = generate_sequential(3, 10, 5, 1.0, 0.001)
    repeated = generate_sequential(3, 10, 5, 1.0, 0.001, repeats=2)

    assert repeated.shape[0] == 2 * single.shape[0]
    np.testing.assert_array_equal(repeated[: single.shape[0]], single)
    np.testing.assert_array_equal(repeated[single.shape[0] :], single)


def test_generate_sequential_repeats_with_specific_neurons() -> None:
    single = generate_sequential(5, 10, 5, 1.0, 0.001, neuron_index=[4, 1])
    repeated = generate_sequential(5, 10, 5, 1.0, 0.001, neuron_index=[4, 1], repeats=3)

    np.testing.assert_array_equal(repeated, np.tile(single, (3, 1)))


def test_generate_sequential_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        generate_sequential(3, 10, 5, 1.0, 0.001, repeats=0)


def test_generate_sequential_pulses_per_neuron() -> None:
    from neuroseq.stimulation.static_input.stim_patterns import generate_sequential

    values = generate_sequential(
        n_neurons=2,
        pulse_ms=1,
        ipi_ms=1,
        magnitude=1.0,
        dt_seconds=0.001,
        pulse_repetitions=2,
    )
    np.testing.assert_array_equal(values[:, 0], [1, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(values[:, 1], [0, 0, 0, 0, 1, 0, 1, 0])


def test_stim_patterns_does_not_import_elise_simulator() -> None:
    # Opening the static designer must not pull in the ELiSe model/simulator.
    code = (
        "import sys\n"
        "import neuroseq.stimulation.static_input.stim_patterns\n"
        "assert 'neuroseq.example_models.elise' not in sys.modules\n"
        "assert 'elise' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_elise_model_reuses_stim_pattern_note_sequence() -> None:
    from neuroseq.example_models import elise as elise_model
    from neuroseq.stimulation.static_input import stim_patterns

    assert elise_model._FUER_ELISE_INTRO is stim_patterns.FUER_ELISE_INTRO
