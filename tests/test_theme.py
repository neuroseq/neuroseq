from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme


def test_normalize_theme_valid_passthrough() -> None:
    assert theme.normalize_theme(theme.LIGHT) == theme.LIGHT
    assert theme.normalize_theme(theme.DARK) == theme.DARK


@pytest.mark.parametrize("value", ["", "blue", "Dark", None])
def test_normalize_theme_invalid_falls_back_to_light(value: object) -> None:
    assert theme.normalize_theme(value) == theme.LIGHT  # type: ignore[arg-type]


def test_fpl_colors_differ_between_themes() -> None:
    light_bg, light_fg = theme.fpl_colors(theme.LIGHT)
    dark_bg, dark_fg = theme.fpl_colors(theme.DARK)
    assert light_bg != dark_bg
    assert light_fg != dark_fg
    # Unknown theme behaves like light.
    assert theme.fpl_colors("nope") == (light_bg, light_fg)


def test_mpl_colors_differ_between_themes() -> None:
    assert theme.mpl_colors(theme.LIGHT) != theme.mpl_colors(theme.DARK)


def test_qt_palette_window_color_differs() -> None:
    light = theme.qt_palette(theme.LIGHT)
    dark = theme.qt_palette(theme.DARK)
    role = QtGui.QPalette.ColorRole.Window
    assert light.color(role).name() != dark.color(role).name()


def test_apply_app_theme_sets_palette_and_fusion_style(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    theme.apply_app_theme(app, theme.DARK)
    assert app.style().objectName().lower() == "fusion"
    dark_window = app.palette().color(QtGui.QPalette.ColorRole.Window).name()
    theme.apply_app_theme(app, theme.LIGHT)
    light_window = app.palette().color(QtGui.QPalette.ColorRole.Window).name()
    assert dark_window != light_window


def test_apply_app_theme_none_app_is_noop() -> None:
    # Should not raise when there is no application instance to theme.
    theme.apply_app_theme(None, theme.DARK)


def test_apply_app_theme_propagates_to_menu_bar(qtbot: QtBot) -> None:
    # Regression: QApplication.setPalette does not reach an already-polished
    # menu bar, so apply_app_theme must push the palette to every widget.
    # The menu bar carries a stylesheet (as in the real app), which is painted
    # via QStyleSheetStyle and would otherwise lag one theme behind; the
    # stylesheet must survive the theme switch.
    app = QtWidgets.QApplication.instance()
    assert app is not None
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    menu_bar = window.menuBar()
    menu_bar.addMenu("File")
    menu_bar_style = "QMenuBar::item { padding: 3px 6px; margin: 0px; }"
    menu_bar.setStyleSheet(menu_bar_style)
    role = QtGui.QPalette.ColorRole.Window

    theme.apply_app_theme(app, theme.DARK)
    dark = menu_bar.palette().color(role).name()
    assert menu_bar.styleSheet() == menu_bar_style
    theme.apply_app_theme(app, theme.LIGHT)
    light = menu_bar.palette().color(role).name()
    assert menu_bar.styleSheet() == menu_bar_style

    assert dark != light
    assert dark == theme.qt_palette(theme.DARK).color(role).name()
    assert light == theme.qt_palette(theme.LIGHT).color(role).name()
