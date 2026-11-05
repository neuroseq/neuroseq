from __future__ import annotations

import pytest
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot
from qtpy import QtGui
from qtpy import QtWidgets

import neuroseq.theme as theme


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_normalize_theme_valid_passthrough() -> None:
    assert theme.normalize_theme(theme.LIGHT) == theme.LIGHT
    assert theme.normalize_theme(theme.DARK) == theme.DARK
    assert theme.normalize_theme(theme.RETRO) == theme.RETRO


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


def test_retro_theme_colors_are_distinct() -> None:
    retro_fpl = theme.fpl_colors(theme.RETRO)
    assert retro_fpl not in (
        theme.fpl_colors(theme.LIGHT),
        theme.fpl_colors(theme.DARK),
    )
    assert theme.mpl_colors(theme.RETRO) == retro_fpl
    assert theme.icon_color(theme.RETRO) not in (
        theme.icon_color(theme.LIGHT),
        theme.icon_color(theme.DARK),
    )
    role = QtGui.QPalette.ColorRole.Window
    retro_window = theme.qt_palette(theme.RETRO).color(role).name()
    assert retro_window not in (
        theme.qt_palette(theme.LIGHT).color(role).name(),
        theme.qt_palette(theme.DARK).color(role).name(),
    )


def test_app_stylesheet_only_set_for_retro() -> None:
    assert theme.app_stylesheet(theme.LIGHT) == ""
    assert theme.app_stylesheet(theme.DARK) == ""
    assert theme.app_stylesheet("nope") == ""
    assert "QToolBar" in theme.app_stylesheet(theme.RETRO)


def test_retro_stylesheet_covers_interactive_subcontrols_and_focus() -> None:
    stylesheet = theme.app_stylesheet(theme.RETRO)
    for selector in (
        "QSpinBox::up-button",
        "QSpinBox::up-arrow",
        "QDoubleSpinBox::down-arrow",
        "QComboBox::down-arrow",
        "QCheckBox::indicator:checked",
        "QCheckBox::indicator:indeterminate",
        "QRadioButton::indicator:checked",
        "QTabWidget::pane",
        "QTabBar::close-button",
        "QPushButton:focus",
        "QCheckBox:focus",
    ):
        assert selector in stylesheet
    assert "%(" not in stylesheet


def test_retro_combo_box_is_toned_into_the_chrome_but_stays_flat() -> None:
    stylesheet = theme.app_stylesheet(theme.RETRO)
    combo_rule = stylesheet.split("\nQComboBox {")[1].split("}")[0]
    # Toned into the surrounding metal rather than left as a pale text field.
    assert "background: #ccd7e4" in combo_rule
    # ...and deliberately NOT a gradient: Qt reuses the closed combo's
    # background for the popup row under the pointer, and a gradient there
    # paints it black. See the comment on the rule in retro.qss.
    assert "qlineargradient" not in combo_rule
    # The gradient belongs on the drop-down sub-control, which is safe.
    drop_down_rule = stylesheet.split("\nQComboBox::drop-down {")[1].split("}")[0]
    assert "qlineargradient" in drop_down_rule


def test_missing_retro_stylesheet_resource_degrades_to_no_skin(
    monkeypatch: MonkeyPatch,
) -> None:
    def missing_resource(_package: str) -> None:
        raise FileNotFoundError

    theme._retro_stylesheet.cache_clear()
    monkeypatch.setattr(theme.resources, "files", missing_resource)
    try:
        assert theme.app_stylesheet(theme.RETRO) == ""
    finally:
        theme._retro_stylesheet.cache_clear()


def test_retro_disabled_text_meets_wcag_contrast() -> None:
    palette = theme.qt_palette(theme.RETRO)
    disabled = palette.color(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.WindowText,
    ).name()
    assert _contrast_ratio(disabled, "#cfd9e5") >= 4.5
    assert _contrast_ratio(disabled, "#b8c5d5") >= 4.5


def test_plot_overlay_and_semantic_colors_follow_theme() -> None:
    light_overlays = theme.plot_overlay_colors(theme.LIGHT)
    assert light_overlays == {
        "grid": (0.3, 0.3, 0.3, 0.4),
        "shading": (0.0, 0.0, 0.0, 0.05),
        "roi_fill": (0.5, 0.5, 0.5, 0.2),
        "roi_text": (0.0, 0.0, 0.0, 1.0),
    }
    assert theme.plot_overlay_colors(theme.DARK) != light_overlays
    assert theme.plot_overlay_colors(theme.RETRO) != light_overlays
    assert theme.plot_overlay_colors("unknown") == light_overlays
    for accessor in (
        theme.muted_color,
        theme.success_color,
        theme.warning_color,
        theme.error_color,
        theme.accent_color,
    ):
        assert accessor(theme.LIGHT) != accessor(theme.DARK)
        assert accessor("unknown") == accessor(theme.LIGHT)


def test_retro_theme_does_not_increase_toolbar_height(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    window = QtWidgets.QMainWindow()
    toolbar = window.addToolBar("Test")
    toolbar.addAction("Action")
    qtbot.addWidget(window)
    try:
        theme.apply_app_theme(app, theme.LIGHT)
        light_height = toolbar.sizeHint().height()
        theme.apply_app_theme(app, theme.RETRO)
        retro_height = toolbar.sizeHint().height()
        # retro.qss deliberately frames QToolBar with a 1px border and a
        # "2px 1px" padding (top+bottom: 2*1 + 2*2 = 6px). Under Linux's
        # Fusion style that exactly cancels out against the native chrome
        # it replaces, but native styles on other platforms (e.g. macOS)
        # don't reserve the same space, so the two don't cancel there and
        # the full 6px shows up. Allow exactly that much slack rather than
        # asserting byte-for-byte parity across native styles - anything
        # beyond it would be a real regression.
        retro_extra_chrome_px = 2 * (1 + 2)
        assert light_height <= retro_height <= light_height + retro_extra_chrome_px
    finally:
        theme.apply_app_theme(app, theme.LIGHT)


def test_apply_app_theme_sets_and_clears_retro_stylesheet(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    try:
        theme.apply_app_theme(app, theme.RETRO)
        assert app.styleSheet() == theme.app_stylesheet(theme.RETRO)
        theme.apply_app_theme(app, theme.LIGHT)
        assert app.styleSheet() == ""
    finally:
        theme.apply_app_theme(app, theme.LIGHT)


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

    for name in (theme.RETRO, theme.DARK, theme.LIGHT, theme.RETRO, theme.LIGHT):
        theme.apply_app_theme(app, name)
        expected = theme.qt_palette(name).color(role).name()
        assert menu_bar.palette().color(role).name() == expected, name


def test_reapplying_the_current_theme_skips_restyling(
    qtbot: QtBot, monkeypatch: MonkeyPatch
) -> None:
    # Every new project workspace applies the theme; restyling every existing
    # widget each time made opening a workspace slower the more were open.
    app = QtWidgets.QApplication.instance()
    assert app is not None
    try:
        for name in (theme.DARK, theme.RETRO):
            theme.apply_app_theme(app, name)
            set_style_calls: list[str] = []
            monkeypatch.setattr(app, "setStyle", set_style_calls.append)
            theme.apply_app_theme(app, name)
            assert set_style_calls == []
            monkeypatch.undo()
    finally:
        theme.apply_app_theme(app, theme.LIGHT)


def test_widgets_created_after_the_theme_was_applied_get_it(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    role = QtGui.QPalette.ColorRole.Window
    dark = theme.qt_palette(theme.DARK).color(role).name()
    try:
        theme.apply_app_theme(app, theme.DARK)
        window = QtWidgets.QMainWindow()
        qtbot.addWidget(window)
        menu_bar = window.menuBar()
        menu_bar.setStyleSheet("QMenuBar::item { padding: 3px 6px; margin: 0px; }")
        theme.apply_app_theme(app, theme.DARK)
        window.show()
        assert menu_bar.palette().color(role).name() == dark
    finally:
        theme.apply_app_theme(app, theme.LIGHT)


def test_theme_is_reapplied_after_the_palette_was_changed_elsewhere(
    qtbot: QtBot,
) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    role = QtGui.QPalette.ColorRole.Window
    try:
        theme.apply_app_theme(app, theme.DARK)
        app.setPalette(theme.qt_palette(theme.LIGHT))
        theme.apply_app_theme(app, theme.DARK)
        assert app.palette().color(role) == theme.qt_palette(theme.DARK).color(role)
    finally:
        theme.apply_app_theme(app, theme.LIGHT)
