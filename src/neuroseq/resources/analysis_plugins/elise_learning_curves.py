"""ELiSe learning curves — MSE & correlation over training and replay.

Renders the four panels of Kriener et al. (2024), Fig. 2b/c: mean squared
error and Pearson correlation of the output-population replay against the
teaching target ``y = φ(u_tgt)`` (Eq. 6), measured on teacher-off
validation cycles during training and on the free replay phase afterwards.

This plugin does not compute anything itself — it plots the per-epoch
curves produced by :meth:`neuroseq.example_models.elise.Elise.run_protocol`.
The dedicated "Run ELiSe protocol" toolbar action runs the protocol and
attaches the result dict to the analysis widget before rendering. Triggering
this plugin from the right-click menu (no curves attached) shows a helpful
message instead of failing.
"""

from __future__ import annotations

import brian2.numpy_ as np

from neuroseq.elise_support.plugin_keys import CURVES_ATTR
from neuroseq.elise_support.plugin_keys import LEARNING_CURVES_PLUGIN_ID

PLUGIN_ID = LEARNING_CURVES_PLUGIN_ID
PLUGIN_NAME = "ELiSe learning curves (train + replay)"
PLUGIN_ICON = "mdi6.chart-line"


def _show_message(widget, message: str) -> None:
    fig = widget.subplot.figure
    fig.clear()
    ax = fig.add_subplot(111)
    widget.subplot = ax
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True
    )
    widget._redraw()


def _plot(widget, curves: dict) -> None:
    val_mse = np.asarray(curves.get("val_mse", []), dtype=np.float64)
    val_corr = np.asarray(curves.get("val_corr", []), dtype=np.float64)
    rep_mse = np.asarray(curves.get("rep_mse", []), dtype=np.float64)
    rep_corr = np.asarray(curves.get("rep_corr", []), dtype=np.float64)
    cfg = curves.get("config", {}) or {}

    fig = widget.subplot.figure
    fig.clear()
    axes = fig.subplots(2, 2)
    # Keep a live single-axes reference so theme recolouring still works.
    widget.subplot = axes[0, 0]

    def _line(ax, y, color, ylabel, title, corr=False):
        x = np.arange(y.size)
        ax.plot(x, y, color=color, marker="o" if y.size <= 60 else None, ms=3, lw=1.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        if corr:
            ax.axhline(1.0, color=color, ls=":", lw=0.6, alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
        elif y.size:
            ax.set_ylim(0.0, max(float(y.max()) * 1.1, 1e-6))

    _line(axes[0, 0], val_mse, "tab:blue", "MSE [rate²]", "Training — validation MSE")
    _line(
        axes[0, 1],
        val_corr,
        "tab:blue",
        "corr. coeff.",
        "Training — validation corr.",
        corr=True,
    )
    _line(axes[1, 0], rep_mse, "tab:orange", "MSE [rate²]", "Replay — MSE per epoch")
    _line(
        axes[1, 1],
        rep_corr,
        "tab:orange",
        "corr. coeff.",
        "Replay — corr. per epoch",
        corr=True,
    )

    mode = cfg.get("teacher_mode", "?")
    mean_rep = float(np.mean(rep_mse)) if rep_mse.size else float("nan")
    fig.suptitle(
        f"ELiSe learning curves (teacher={mode}) — "
        f"{val_mse.size} train epochs × {cfg.get('training_cycles', '?')} cyc · "
        f"final val MSE {val_mse[-1]:.4g} · mean replay MSE {mean_rep:.4g}"
        if val_mse.size
        else "ELiSe learning curves",
        fontsize=10,
    )
    fig.tight_layout()
    widget._redraw()


def render_analysis(widget, data, neuron_slice, time_slice) -> None:
    curves = getattr(widget, CURVES_ATTR, None)
    if not isinstance(curves, dict):
        _show_message(
            widget,
            "No ELiSe protocol results attached. Run the 'Run ELiSe "
            "protocol' toolbar action to train the network and generate "
            "the training + replay learning curves.",
        )
        return
    _plot(widget, curves)
