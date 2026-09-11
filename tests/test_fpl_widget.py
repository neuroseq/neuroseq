from __future__ import annotations

import pygfx
import pytest
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.fpl_widget import FPLWidget


def test_fpl_widget_apply_theme_updates_background(qtbot: QtBot) -> None:
    widget = FPLWidget()
    qtbot.addWidget(widget)

    widget.apply_theme(theme.DARK)
    assert widget._theme == theme.DARK
    dark_bg, _ = theme.fpl_colors(theme.DARK)
    assert tuple(widget.subplot.background_color[0]) == tuple(pygfx.Color(dark_bg))
    assert all(
        tuple(dock.background_color[0]) == tuple(pygfx.Color(dark_bg))
        for dock in widget.subplot.docks.values()
    )
    assert tuple(widget.subplot.frame.plane.material.color) == tuple(
        pygfx.Color(dark_bg)
    )

    widget.apply_theme(theme.LIGHT)
    light_bg, _ = theme.fpl_colors(theme.LIGHT)
    assert tuple(widget.subplot.background_color[0]) == tuple(pygfx.Color(light_bg))

    # Unknown theme falls back to light.
    widget.apply_theme("not-a-theme")
    assert widget._theme == theme.LIGHT


def test_fpl_widget_removes_subplot_title_band(qtbot: QtBot) -> None:
    widget = FPLWidget()
    qtbot.addWidget(widget)

    assert widget.subplot.title.text == ""
    assert not widget.subplot.title.visible
    _, frame_y, _, _ = widget.subplot.frame.rect
    _, render_y, _, _ = widget.subplot.frame.get_render_rect()
    assert render_y == pytest.approx(frame_y)


def test_fpl_widget_skips_rendering_until_canvas_is_sized(qtbot: QtBot) -> None:
    widget = FPLWidget()
    qtbot.addWidget(widget)

    rendered: list[bool] = []
    widget._fpl_figure_render = lambda draw=True: rendered.append(True)

    assert widget._fpl_canvas_ready() is False
    rendered.clear()
    widget._fpl_guarded_render()
    # invalid viewport so guarded renderer does not call the real renderer:
    assert rendered == []

    widget.show()
    qtbot.waitExposed(widget)
    qtbot.waitUntil(widget._fpl_canvas_ready)
    rendered.clear()
    widget._fpl_guarded_render()
    # valid viewport so guarded renderer calls the real renderer:
    assert rendered == [True]


class _DummyCanvas:
    def request_draw(self, draw_function=None) -> None:
        return

    def set_update_mode(self, update_mode, **kwargs) -> None:
        return


class _DummyFigure:
    def __init__(self, calls: list[dict[str, object]], **kwargs) -> None:
        self._calls = calls
        self._calls.append(kwargs)
        self.canvas = _DummyCanvas()

    def show(self) -> QtWidgets.QWidget:
        return QtWidgets.QWidget()

    def _render(self, draw: bool = True) -> None:
        return

    def __getitem__(self, key: tuple[int, int]) -> object:
        return object()

    def close(self) -> None:
        return


def test_fpl_widget_uses_bitmap_present_method_for_headless_tests(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("NEUROSEQ_HEADLESS_TEST", "1")
    monkeypatch.setattr(
        "neuroseq.fpl_widget.fpl.Figure",
        lambda **kwargs: _DummyFigure(calls, **kwargs),
    )
    monkeypatch.setattr(FPLWidget, "_apply_fpl_theme", lambda self: None)

    widget = FPLWidget()
    qtbot.addWidget(widget)

    assert calls[0]["canvas_kwargs"] == {
        "parent": widget,
        "present_method": "bitmap",
    }


def test_fpl_widget_uses_screen_present_method_by_default(
    monkeypatch: MonkeyPatch,
    qtbot: QtBot,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.delenv("NEUROSEQ_HEADLESS_TEST", raising=False)
    monkeypatch.setattr(
        "neuroseq.fpl_widget.fpl.Figure",
        lambda **kwargs: _DummyFigure(calls, **kwargs),
    )
    monkeypatch.setattr(FPLWidget, "_apply_fpl_theme", lambda self: None)

    widget = FPLWidget()
    qtbot.addWidget(widget)

    assert calls[0]["canvas_kwargs"] == {"parent": widget}
