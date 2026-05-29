"""Tests for the ELiSe target MSE entry point (paper-style replay metric)."""

from __future__ import annotations

import brian2.numpy_ as np
import pytest

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.example_models.elise import _detect_pattern_cycle
from neuroseq.example_models.elise import build_target_rate_pattern
from neuroseq.example_projects import _build_fuer_elise_opsin_stims
from neuroseq.replay_metrics import compute_performance_against_target
from neuroseq.replay_metrics import save_replay_metrics_csv

# ---------- compute_performance_against_target ----------


def test_target_mse_matches_perfect_replay() -> None:
    rng = np.random.default_rng(0)
    T = 64
    n_signals = 3
    target = rng.uniform(0.0, 1.0, size=(n_signals, T))
    traces = np.tile(target, (1, 4))

    mses, corrs, shifts = compute_performance_against_target(traces, target)

    assert mses.shape == (3,)
    np.testing.assert_array_almost_equal(mses, np.zeros(3), decimal=8)
    np.testing.assert_array_almost_equal(corrs, np.ones(3), decimal=8)
    np.testing.assert_array_equal(shifts, np.zeros(3, dtype=np.int64))


def test_target_mse_recovers_shifted_signal() -> None:
    rng = np.random.default_rng(1)
    T = 32
    n_signals = 2
    target = rng.uniform(0.0, 1.0, size=(n_signals, T))

    # Stream the target shifted by +5 samples for 3 cycles.
    shifted_stream = np.roll(np.tile(target, (1, 3)), 5, axis=1)
    traces = np.concatenate([target, shifted_stream], axis=1)

    mses, corrs, shifts = compute_performance_against_target(traces, target)

    assert mses.shape == (3,)
    assert int(shifts[0]) == 0
    assert mses[0] == pytest.approx(0.0, abs=1e-10)
    for k in (1, 2):
        assert int(shifts[k]) == 5
        assert mses[k] == pytest.approx(0.0, abs=1e-8)
        assert corrs[k] == pytest.approx(1.0, abs=1e-8)


def test_target_mse_rejects_shape_mismatch() -> None:
    target = np.zeros((3, 32))
    traces = np.zeros((4, 128))  # 4 signals vs target's 3.
    with pytest.raises(ValueError, match="number of signals"):
        compute_performance_against_target(traces, target)


def test_target_mse_rejects_too_short_traces() -> None:
    target = np.zeros((2, 32))
    traces = np.zeros((2, 40))  # < 2T
    with pytest.raises(ValueError, match="two whole cycles"):
        compute_performance_against_target(traces, target)


# ---------- _detect_pattern_cycle / build_target_rate_pattern ----------


def test_detect_pattern_cycle_for_fuer_elise() -> None:
    stim_exc, _ = _build_fuer_elise_opsin_stims(tfinal_ms=1920)
    T = _detect_pattern_cycle(stim_exc)
    # Für Elise intro: 24 events × 10 ms = 240 ms = 240 stim samples (dt=1 ms).
    assert T == 240


def test_detect_pattern_cycle_falls_back_to_full_length_when_aperiodic() -> None:
    rng = np.random.default_rng(2)
    stim = rng.standard_normal((97, 3))  # prime length, random → no period.
    T = _detect_pattern_cycle(stim)
    assert T == 97  # falls back to full length


def test_build_target_rate_pattern_maps_binary_to_phi() -> None:
    # Construct a stim where pitch 0 is on for samples 0..4 then off, pitch 1
    # is off then on for 5..9. Cycle length = 10.
    stim = np.zeros((20, 2), dtype=np.float32)
    stim[:5, 0] = 1.0
    stim[5:10, 1] = 1.0
    stim[10:15, 0] = 1.0  # second cycle, identical to first.
    stim[15:20, 1] = 1.0

    target, T = build_target_rate_pattern(stim)

    assert T == 10
    assert target.shape == (2, 10)
    # With default params (E_l=-70, a=0.3, b=-58, u_active=−50): φ(−70) ≈ 0.027,
    # φ(−50) ≈ 0.917.
    on_rate = 1.0 / (1.0 + np.exp(0.3 * (-58 - (-50))))
    off_rate = 1.0 / (1.0 + np.exp(0.3 * (-58 - (-70))))
    np.testing.assert_allclose(target[0, :5], on_rate, atol=1e-10)
    np.testing.assert_allclose(target[0, 5:], off_rate, atol=1e-10)
    np.testing.assert_allclose(target[1, :5], off_rate, atol=1e-10)
    np.testing.assert_allclose(target[1, 5:], on_rate, atol=1e-10)


def test_build_target_rate_pattern_ignores_trailing_ringout() -> None:
    """A trailing all-zero "ring-out" tail must not hide the training period."""
    stim, _ = _build_fuer_elise_opsin_stims(tfinal_ms=1920)
    ringout = np.zeros_like(stim)
    stim_with_ringout = np.concatenate([stim, ringout], axis=0)

    target, T = build_target_rate_pattern(stim_with_ringout)

    # Still 240 stim samples = one Für Elise phrase, despite the silence tail.
    assert T == 240
    assert target.shape == (13, 240)


def test_build_target_rate_pattern_rejects_all_zero_stim() -> None:
    stim = np.zeros((100, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="no active samples"):
        build_target_rate_pattern(stim)


def test_build_target_rate_pattern_honours_explicit_cycle_length() -> None:
    stim = np.zeros((40, 1), dtype=np.float32)
    stim[::4, 0] = 1.0  # impulses every 4 samples.
    target, T = build_target_rate_pattern(stim, cycle_length=4)
    assert T == 4
    # First sample on, rest off in the returned single-cycle slice.
    assert target[0, 0] > 0.9
    assert all(target[0, 1:] < 0.1)


# ---------- CSV export ----------


def test_save_replay_metrics_csv_round_trip(tmp_path) -> None:
    import csv

    mses = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    corrs = np.array([0.99, 0.95, 0.9], dtype=np.float64)
    shifts = np.array([0, 3, 7], dtype=np.int64)

    out = save_replay_metrics_csv(
        tmp_path / "analysis.csv",
        mses,
        corrs,
        shifts,
        cycle_dt_ms=0.1,
        cycle_length_samples=2400,
        n_output_neurons=13,
        source="/tmp/fake.nseq",
    )

    text = out.read_text()
    # Header sanity: metadata lines, expected fields populated.
    assert "# Source: /tmp/fake.nseq" in text
    assert "# Output neurons: 13" in text
    assert "# Pattern cycle: 2400 samples (240 ms)" in text
    assert "# Mean MSE: 0.02" in text

    # Data table reads back with the right columns and values.
    rows = [
        r for r in csv.reader(out.read_text().splitlines()) if not r[0].startswith("#")
    ]
    assert rows[0] == [
        "cycle_index",
        "start_time_ms",
        "mse",
        "corr_coeff",
        "shift_samples",
        "shift_ms",
    ]
    assert len(rows) == 1 + 3
    # Cycle 1: start_time_ms = 1 * 240 = 240; shift 3 samples = 0.3 ms.
    assert rows[2][0] == "1"
    assert float(rows[2][1]) == pytest.approx(240.0)
    assert float(rows[2][2]) == pytest.approx(0.02)
    assert int(rows[2][4]) == 3
    assert float(rows[2][5]) == pytest.approx(0.3)


# ---------- plugin registration ----------


def test_elise_target_mse_plugin_is_discoverable() -> None:
    plugin_ids = {p.id for p in discover_analysis_plugins()}
    assert "elise_target_mse" in plugin_ids
