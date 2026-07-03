from __future__ import annotations

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

LIGHT = "light"
DARK = "dark"
THEMES = (LIGHT, DARK)

# Background / foreground colors for the fastplotlib (GPU) canvases. These do
# not follow the Qt palette, so they are applied explicitly per widget.
_FPL_COLORS = {
    LIGHT: {"background": "#ffffff", "foreground": "#000000"},
    DARK: {"background": "#1e1e1e", "foreground": "#dcdcdc"},
}

# Matplotlib face colors for the custom-analysis widget, mirroring the fpl ones.
_MPL_COLORS = {
    LIGHT: {"face": "#ffffff", "text": "#000000"},
    DARK: {"face": "#1e1e1e", "text": "#dcdcdc"},
}

# Foreground color for qtawesome toolbar/menu icons per theme. Dark glyphs read
# well on the light chrome; light glyphs are needed on the dark chrome.
_ICON_COLORS = {
    LIGHT: "#000000",
    DARK: "#dcdcdc",
}


def normalize_theme(theme: str | None) -> str:
    """Return a valid theme name, falling back to LIGHT for anything unknown."""
    return theme if theme in THEMES else LIGHT


def fpl_colors(theme: str) -> tuple[str, str]:
    """(background, foreground) colors for fastplotlib canvases of this theme."""
    colors = _FPL_COLORS[normalize_theme(theme)]
    return colors["background"], colors["foreground"]


def mpl_colors(theme: str) -> tuple[str, str]:
    """(face, text) colors for matplotlib figures of this theme."""
    colors = _MPL_COLORS[normalize_theme(theme)]
    return colors["face"], colors["text"]


def icon_color(theme: str) -> str:
    """Foreground color for qtawesome icons of this theme."""
    return _ICON_COLORS[normalize_theme(theme)]


def _light_palette() -> QtGui.QPalette:
    Role = QtGui.QPalette.ColorRole
    Group = QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    palette.setColor(Role.Window, QtGui.QColor("#f0f0f0"))
    palette.setColor(Role.WindowText, QtGui.QColor("#000000"))
    palette.setColor(Role.Base, QtGui.QColor("#ffffff"))
    palette.setColor(Role.AlternateBase, QtGui.QColor("#e9e9e9"))
    palette.setColor(Role.ToolTipBase, QtGui.QColor("#ffffdc"))
    palette.setColor(Role.ToolTipText, QtGui.QColor("#000000"))
    palette.setColor(Role.Text, QtGui.QColor("#000000"))
    palette.setColor(Role.Button, QtGui.QColor("#f0f0f0"))
    palette.setColor(Role.ButtonText, QtGui.QColor("#000000"))
    palette.setColor(Role.BrightText, QtGui.QColor("#ff0000"))
    palette.setColor(Role.Link, QtGui.QColor("#2a82da"))
    palette.setColor(Role.Highlight, QtGui.QColor("#308cc6"))
    palette.setColor(Role.HighlightedText, QtGui.QColor("#ffffff"))
    disabled = QtGui.QColor("#a0a0a0")
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        palette.setColor(Group.Disabled, role, disabled)
    return palette


def _dark_palette() -> QtGui.QPalette:
    Role = QtGui.QPalette.ColorRole
    Group = QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    palette.setColor(Role.Window, QtGui.QColor("#353535"))
    palette.setColor(Role.WindowText, QtGui.QColor("#dcdcdc"))
    palette.setColor(Role.Base, QtGui.QColor("#232323"))
    palette.setColor(Role.AlternateBase, QtGui.QColor("#353535"))
    palette.setColor(Role.ToolTipBase, QtGui.QColor("#353535"))
    palette.setColor(Role.ToolTipText, QtGui.QColor("#dcdcdc"))
    palette.setColor(Role.Text, QtGui.QColor("#dcdcdc"))
    palette.setColor(Role.Button, QtGui.QColor("#353535"))
    palette.setColor(Role.ButtonText, QtGui.QColor("#dcdcdc"))
    palette.setColor(Role.BrightText, QtGui.QColor("#ff5555"))
    palette.setColor(Role.Link, QtGui.QColor("#2a82da"))
    palette.setColor(Role.Highlight, QtGui.QColor("#2a82da"))
    palette.setColor(Role.HighlightedText, QtGui.QColor("#000000"))
    disabled = QtGui.QColor("#7f7f7f")
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        palette.setColor(Group.Disabled, role, disabled)
    return palette


def qt_palette(theme: str) -> QtGui.QPalette:
    """Return the QPalette for the given theme."""
    if normalize_theme(theme) == DARK:
        return _dark_palette()
    return _light_palette()


def apply_app_theme(app: QtCore.QCoreApplication | None, theme: str) -> None:
    """Apply ``theme`` to the whole Qt application.

    Uses the Fusion style so the custom palette is honored consistently across
    platforms, then installs the matching palette. A no-op when there is no
    GUI application instance (e.g. a bare QCoreApplication or ``None``).
    """
    if not isinstance(app, QtWidgets.QApplication):
        return
    palette = qt_palette(normalize_theme(theme))
    app.setPalette(palette)
    # QApplication.setPalette does not reach already-polished menu bars, menus,
    # or tool bars, so push the palette to every existing widget as well. In
    # addition, widgets that carry a stylesheet are painted through
    # QStyleSheetStyle, which caches palette-derived brushes and is not
    # invalidated by a palette change alone; re-applying the same stylesheet
    # forces it to re-resolve against the new palette (otherwise the top bar
    # lags one theme behind).
    for widget in app.allWidgets():
        widget.setPalette(palette)
        style_sheet = widget.styleSheet()
        if style_sheet:
            widget.setStyleSheet(style_sheet)
    # Re-assert the Fusion style last so every widget is repolished and
    # repainted with the new palette already in place.
    app.setStyle("Fusion")
