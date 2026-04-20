from __future__ import annotations

from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq.fpl_widget import FPLWidget


class _DummyFigure:
    def __init__(self, calls: list[dict[str, object]], **kwargs) -> None:
        self._calls = calls
        self._calls.append(kwargs)

    def show(self) -> QtWidgets.QWidget:
        return QtWidgets.QWidget()

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
    monkeypatch.setattr(FPLWidget, "_set_light_mode", lambda self: None)

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
    monkeypatch.setattr(FPLWidget, "_set_light_mode", lambda self: None)

    widget = FPLWidget()
    qtbot.addWidget(widget)

    assert calls[0]["canvas_kwargs"] == {"parent": widget}
