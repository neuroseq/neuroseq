"""Tests for the Replay MSE analysis plugin."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import brian2.numpy_ as np
import pytest
from pytestqt.qtbot import QtBot

from neuroseq.analysis_plugins import discover_analysis_plugins
from neuroseq.data import TraceData
from neuroseq.visualization.custom_analysis_widget import CustomAnalysisWidget


def _load_plugin_module():
    """Load the plugin module directly (so we can test the pure helper)."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "neuroseq"
        / "resources"
        / "analysis_plugins"
        / "replay_mse.py"
    )
    spec = importlib.util.spec_from_file_location("replay_mse_under_test", src)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_analysis_plugins_contains_replay_mse() -> None:
    plugin_ids = {p.id for p in discover_analysis_plugins()}
    assert "replay_mse" in plugin_ids


def test_compute_cycle_mse_reference_cycle_is_zero() -> None:
    mod = _load_plugin_module()
    # 2 neurons × 30 samples = 3 cycles of length 10.
    rng = np.random.default_rng(0)
    cycle = rng.standard_normal((2, 10)).astype(np.float32)
    traces = np.tile(cycle, (1, 3))

    mses = mod.compute_cycle_mse(traces, n_cycle_samples=10)

    assert mses.shape == (3,)
    np.testing.assert_array_almost_equal(mses, np.zeros(3), decimal=6)


def test_compute_cycle_mse_detects_difference_between_cycles() -> None:
    mod = _load_plugin_module()
    # Three cycles: identical, +1 offset, +2 offset.
    reference = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    cycle1 = reference + 1.0
    cycle2 = reference + 2.0
    traces = np.concatenate([reference, cycle1, cycle2], axis=1)

    mses = mod.compute_cycle_mse(traces, n_cycle_samples=4)

    np.testing.assert_array_almost_equal(mses, [0.0, 1.0, 4.0], decimal=6)


def test_compute_cycle_mse_drops_partial_trailing_cycle() -> None:
    mod = _load_plugin_module()
    # 14 samples with cycle length 4 → 3 whole cycles, 2 leftover.
    cycle = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    traces = np.concatenate([cycle, cycle, cycle, cycle[:, :2]], axis=1)

    mses = mod.compute_cycle_mse(traces, n_cycle_samples=4)

    assert mses.shape == (3,)
    np.testing.assert_array_almost_equal(mses, np.zeros(3), decimal=6)


def test_compute_cycle_mse_rejects_too_short_selection() -> None:
    mod = _load_plugin_module()
    traces = np.zeros((2, 5), dtype=np.float32)  # only one cycle of length 5
    with pytest.raises(ValueError, match="two whole cycles"):
        mod.compute_cycle_mse(traces, n_cycle_samples=5)


def test_compute_cycle_mse_rejects_one_d_input() -> None:
    mod = _load_plugin_module()
    with pytest.raises(ValueError, match="2-D"):
        mod.compute_cycle_mse(np.zeros(20, dtype=np.float32), n_cycle_samples=4)


def test_compute_cycle_performance_periodic_signal_is_optimal() -> None:
    mod = _load_plugin_module()
    rng = np.random.default_rng(0)
    cycle = rng.standard_normal((3, 64)).astype(np.float64)
    traces = np.tile(cycle, (1, 4))  # 4 identical cycles

    mses, corrs, shifts = mod.compute_cycle_performance(traces, n_cycle_samples=64)

    # Last cycle is dropped due to shift slack → 3 evaluable cycles.
    assert mses.shape == (3,)
    assert corrs.shape == (3,)
    assert shifts.shape == (3,)
    np.testing.assert_array_almost_equal(mses, np.zeros(3), decimal=8)
    np.testing.assert_array_almost_equal(corrs, np.ones(3), decimal=8)
    # Shifts should be zero - modulo the period of 64
    np.testing.assert_array_equal(shifts % 64, np.zeros(3, dtype=np.int64))


def test_compute_cycle_performance_recovers_time_shifted_cycle() -> None:
    mod = _load_plugin_module()
    rng = np.random.default_rng(1)
    T = 32
    n_cycles = 4
    cycle = rng.standard_normal((2, T)).astype(np.float64)

    # The "replay" part (cycles 1..n_cycles-1) is a continuous periodic
    # signal that is identical to the reference but shifted by +5 samples.
    # The optimal alignment for every replay cycle is therefore s=5.
    shifted_part = np.roll(np.tile(cycle, (1, n_cycles - 1)), 5, axis=1)
    traces = np.concatenate([cycle, shifted_part], axis=1)

    mses, corrs, shifts = mod.compute_cycle_performance(traces, n_cycle_samples=T)

    # 4 cycles, last one dropped → 3 evaluable.
    assert mses.shape == (3,)
    # k=0 is the trivial reference cycle.
    assert int(shifts[0]) == 0
    assert mses[0] == pytest.approx(0.0, abs=1e-10)
    assert corrs[0] == pytest.approx(1.0, abs=1e-10)
    # k=1 and k=2 recover the +5-sample shift exactly.
    assert int(shifts[1]) == 5
    assert int(shifts[2]) == 5
    assert mses[1] == pytest.approx(0.0, abs=1e-8)
    assert mses[2] == pytest.approx(0.0, abs=1e-8)
    assert corrs[1] == pytest.approx(1.0, abs=1e-8)
    assert corrs[2] == pytest.approx(1.0, abs=1e-8)


def test_compute_cycle_performance_detects_uncorrelated_noise() -> None:
    mod = _load_plugin_module()
    rng = np.random.default_rng(2)
    T = 64
    reference = rng.standard_normal((4, T)).astype(np.float64)
    # 4 cycles: reference, noise, noise, noise. With shift slack, only
    # cycles 0..2 are evaluable.
    noise = rng.standard_normal((4, 3 * T)).astype(np.float64)
    traces = np.concatenate([reference, noise], axis=1)

    mses, corrs, _ = mod.compute_cycle_performance(traces, n_cycle_samples=T)

    assert mses.shape == (3,)
    # Cycle 0 is trivial.
    assert mses[0] == pytest.approx(0.0, abs=1e-10)
    assert corrs[0] == pytest.approx(1.0, abs=1e-10)
    # Cycles 1, 2: independent noise → MSE bounded away from 0 and
    # correlation near 0 (low magnitude).
    assert mses[1] > 0.5
    assert mses[2] > 0.5
    assert abs(corrs[1]) < 0.5
    assert abs(corrs[2]) < 0.5


def test_compute_cycle_performance_rejects_too_short_selection() -> None:
    mod = _load_plugin_module()
    traces = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="two whole cycles"):
        mod.compute_cycle_performance(traces, n_cycle_samples=5)


def _make_periodic_trace(
    n_neurons: int, cycle_samples: int, n_cycles: int, dt: float, noise: float = 0.0
) -> TraceData:
    rng = np.random.default_rng(42)
    cycle = rng.standard_normal((n_neurons, cycle_samples)).astype(np.float32)
    traces = np.tile(cycle, (1, n_cycles)).astype(np.float32)
    if noise:
        traces[:, cycle_samples:] += (
            rng.standard_normal((n_neurons, cycle_samples * (n_cycles - 1))).astype(
                np.float32
            )
            * noise
        )
    return TraceData(
        name="replay_test",
        traces=traces,
        indices=np.arange(n_neurons, dtype=np.uint32),
        units="mV",
        dt=dt,
    )


def test_widget_renders_replay_mse_bar_chart(qtbot: QtBot) -> None:
    plugin = next(p for p in discover_analysis_plugins() if p.id == "replay_mse")
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget.isVisible())

    # 3 cycles of 1920 samples at dt=1 ms → 5.76 s, identical cycles.
    data = _make_periodic_trace(n_neurons=4, cycle_samples=1920, n_cycles=3, dt=0.001)
    widget.set_data(data, plugin, slice(0, 4), slice(0, 1920 * 3))
    qtbot.wait(20)

    # Shift-corrected MSE drops the last cycle (no shift slack), so we
    # expect n_cycles - 1 = 2 bars; both should be ≈ 0 since the signal
    # is perfectly periodic.
    bars = [patch for patch in widget.subplot.patches if hasattr(patch, "get_height")]
    assert len(bars) == 2
    heights = [float(b.get_height()) for b in bars]
    assert all(h == pytest.approx(0.0, abs=1e-6) for h in heights)


def test_widget_shows_helpful_message_on_too_short_selection(
    qtbot: QtBot,
) -> None:
    plugin = next(p for p in discover_analysis_plugins() if p.id == "replay_mse")
    widget = CustomAnalysisWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget.isVisible())

    # 0.5 s of data at dt=1 ms; default cycle is 1920 ms → too short.
    data = _make_periodic_trace(n_neurons=3, cycle_samples=500, n_cycles=1, dt=0.001)
    widget.set_data(data, plugin, slice(0, 3), slice(0, 500))
    qtbot.wait(20)

    texts = [t.get_text() for t in widget.subplot.texts]
    assert any("at least 2 full cycles" in t for t in texts)
