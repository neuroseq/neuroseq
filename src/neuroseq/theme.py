from __future__ import annotations

import logging
from functools import cache
from importlib import resources

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

logger = logging.getLogger(__name__)

LIGHT = "light"
DARK = "dark"
RETRO = "retro"
THEMES = (LIGHT, DARK, RETRO)

# Theme a fresh install starts with. Note this is deliberately not what
# ``normalize_theme`` falls back to: an unrecognized theme name is a bug or a
# hand-edited config, and the plain light theme is the safer thing to show
# there than a skin.
DEFAULT = RETRO

# Retro skin colors: a light "metallic" chrome (brushed blue-grey bevels) with
# dark LCD read-outs, in the spirit of the classic Winamp light skins.
_RETRO_CHROME_TOP = "#eaf0f7"
_RETRO_CHROME_UPPER = "#ccd7e4"
_RETRO_CHROME_LOWER = "#b8c5d5"
_RETRO_CHROME_BOTTOM = "#d8e0ec"
_RETRO_BEVEL_LIGHT = "#ffffff"
_RETRO_BORDER = "#7d8b9d"
_RETRO_BORDER_DARK = "#5b6878"
_RETRO_TEXT = "#17222e"
_RETRO_LCD_BG = "#0c1a2a"
_RETRO_LCD_FG = "#7ef0a8"
_RETRO_LCD_DIM = "#3d7a5c"
_RETRO_DISABLED_TEXT = "#414d5b"

# Background / foreground colors for the fastplotlib (GPU) canvases. These do
# not follow the Qt palette, so they are applied explicitly per widget.
_FPL_COLORS = {
    LIGHT: {"background": "#ffffff", "foreground": "#000000"},
    DARK: {"background": "#1e1e1e", "foreground": "#dcdcdc"},
    RETRO: {"background": _RETRO_LCD_BG, "foreground": _RETRO_LCD_FG},
}

# Matplotlib face colors for the custom-analysis widget, mirroring the fpl ones.
_MPL_COLORS = {
    LIGHT: {"face": "#ffffff", "text": "#000000"},
    DARK: {"face": "#1e1e1e", "text": "#dcdcdc"},
    RETRO: {"face": _RETRO_LCD_BG, "text": _RETRO_LCD_FG},
}

# Foreground color for qtawesome toolbar/menu icons per theme. Dark glyphs read
# well on the light chrome; light glyphs are needed on the dark chrome.
_ICON_COLORS = {
    LIGHT: "#000000",
    DARK: "#dcdcdc",
    RETRO: "#16232f",
}

# Plot overlays are independent pygfx objects and therefore need explicit
# colors just like the plot canvas itself. The light values preserve the
# original appearance.
#
# NOTE on "shading" alpha: pygfx/wgpu composites alpha-blended meshes in
# LINEAR light space, not in sRGB space. For a *dark* background, blending a
# bright overlay at a given alpha in linear space lifts the sRGB result far
# more than the same alpha would in a naive sRGB-space blend (verified: DARK's
# 0.06 rendered as if it were ~0.06 -> ~0.10 sRGB-equivalent; RETRO's 0.07
# rendered as if ~0.07 -> ~0.19 sRGB-equivalent). The "shading" alphas below
# for DARK and RETRO are therefore deliberately much smaller than LIGHT's, so
# the rendered lift is comparable in magnitude across themes.
_PLOT_OVERLAY_COLORS = {
    LIGHT: {
        "grid": (0.3, 0.3, 0.3, 0.4),
        "shading": (0.0, 0.0, 0.0, 0.05),
        "roi_fill": (0.5, 0.5, 0.5, 0.2),
        "roi_text": (0.0, 0.0, 0.0, 1.0),
    },
    DARK: {
        "grid": (0.78, 0.78, 0.78, 0.38),
        "shading": (1.0, 1.0, 1.0, 0.012),
        "roi_fill": (0.19, 0.55, 0.78, 0.32),
        "roi_text": (0.86, 0.86, 0.86, 1.0),
    },
    RETRO: {
        "grid": (0.49, 0.94, 0.66, 0.38),
        "shading": (0.49, 0.94, 0.66, 0.02),
        "roi_fill": (0.17, 0.31, 0.49, 0.55),
        "roi_text": (0.49, 0.94, 0.66, 1.0),
    },
}

_SEMANTIC_COLORS = {
    LIGHT: {
        "muted": "#666666",
        "success": "#2a7a2a",
        "warning": "#8a5a00",
        "error": "#c62828",
        "accent": "#2a6fbb",
    },
    DARK: {
        "muted": "#aaaaaa",
        "success": "#70c878",
        "warning": "#f0bd60",
        "error": "#ff6b6b",
        "accent": "#6ca9e8",
    },
    RETRO: {
        "muted": "#4d5968",
        "success": "#176b3a",
        "warning": "#765000",
        "error": "#a51414",
        "accent": "#1a5fb4",
    },
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


def plot_overlay_colors(theme: str) -> dict[str, tuple[float, float, float, float]]:
    """Colors for grids, row shading, and selection overlays on plot canvases."""
    return _PLOT_OVERLAY_COLORS[normalize_theme(theme)].copy()


def muted_color(theme: str) -> str:
    """Muted informational text color for ``theme``."""
    return _SEMANTIC_COLORS[normalize_theme(theme)]["muted"]


def success_color(theme: str) -> str:
    """Success-state text color for ``theme``."""
    return _SEMANTIC_COLORS[normalize_theme(theme)]["success"]


def warning_color(theme: str) -> str:
    """Warning-state text color for ``theme``."""
    return _SEMANTIC_COLORS[normalize_theme(theme)]["warning"]


def error_color(theme: str) -> str:
    """Error-state text color for ``theme``."""
    return _SEMANTIC_COLORS[normalize_theme(theme)]["error"]


def accent_color(theme: str) -> str:
    """Accent text color for ``theme``."""
    return _SEMANTIC_COLORS[normalize_theme(theme)]["accent"]


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


def _retro_palette() -> QtGui.QPalette:
    Role = QtGui.QPalette.ColorRole
    Group = QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    palette.setColor(Role.Window, QtGui.QColor("#cfd9e5"))
    palette.setColor(Role.WindowText, QtGui.QColor(_RETRO_TEXT))
    palette.setColor(Role.Base, QtGui.QColor("#eef3f9"))
    palette.setColor(Role.AlternateBase, QtGui.QColor("#dce4ee"))
    palette.setColor(Role.ToolTipBase, QtGui.QColor(_RETRO_LCD_BG))
    palette.setColor(Role.ToolTipText, QtGui.QColor(_RETRO_LCD_FG))
    palette.setColor(Role.Text, QtGui.QColor(_RETRO_TEXT))
    palette.setColor(Role.Button, QtGui.QColor("#d4dde9"))
    palette.setColor(Role.ButtonText, QtGui.QColor(_RETRO_TEXT))
    palette.setColor(Role.BrightText, QtGui.QColor("#d02020"))
    palette.setColor(Role.Link, QtGui.QColor("#1a5fb4"))
    palette.setColor(Role.Highlight, QtGui.QColor("#2b4f7e"))
    palette.setColor(Role.HighlightedText, QtGui.QColor("#ffffff"))
    disabled = QtGui.QColor(_RETRO_DISABLED_TEXT)
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        palette.setColor(Group.Disabled, role, disabled)
    return palette


def qt_palette(theme: str) -> QtGui.QPalette:
    """Return the QPalette for the given theme."""
    theme = normalize_theme(theme)
    if theme == DARK:
        return _dark_palette()
    if theme == RETRO:
        return _retro_palette()
    return _light_palette()


_CHROME_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
    f" stop:0 {_RETRO_CHROME_TOP}, stop:0.48 {_RETRO_CHROME_UPPER},"
    f" stop:0.52 {_RETRO_CHROME_LOWER}, stop:1 {_RETRO_CHROME_BOTTOM})"
)
_BUTTON_GRADIENT = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
    f" stop:0 {_RETRO_BEVEL_LIGHT}, stop:0.45 {_RETRO_CHROME_UPPER},"
    f" stop:0.55 {_RETRO_CHROME_LOWER}, stop:1 #c8d3e1)"
)
_BUTTON_GRADIENT_SUNKEN = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
    " stop:0 #a3b2c4, stop:0.5 #b6c3d3, stop:1 #d0d9e5)"
)


@cache
def _retro_stylesheet() -> str:
    """Load and expand the packaged retro stylesheet once."""
    try:
        resource_root = resources.files("neuroseq.resources")
        stylesheet = resource_root.joinpath("retro.qss").read_text(encoding="utf-8")

        def resource_path(name: str) -> str:
            return str(resource_root.joinpath(name)).replace("\\", "/")

        tokens = {
            "accent": "#2b4f7e",
            "arrow_down_icon": resource_path("retro_arrow_down.svg"),
            "arrow_down_icon_disabled": resource_path("retro_arrow_down_disabled.svg"),
            "arrow_up_icon": resource_path("retro_arrow_up.svg"),
            "arrow_up_icon_disabled": resource_path("retro_arrow_up_disabled.svg"),
            "bevel_light": _RETRO_BEVEL_LIGHT,
            "border": _RETRO_BORDER,
            "border_dark": _RETRO_BORDER_DARK,
            "button_gradient": _BUTTON_GRADIENT,
            "button_gradient_sunken": _BUTTON_GRADIENT_SUNKEN,
            "check_icon": resource_path("retro_check.svg"),
            "check_icon_disabled": resource_path("retro_check_disabled.svg"),
            "checkbox_icon": resource_path("retro_checkbox.svg"),
            "checkbox_icon_disabled": resource_path("retro_checkbox_disabled.svg"),
            "chrome_gradient": _CHROME_GRADIENT,
            "close_icon": resource_path("retro_close.svg"),
            "disabled_text": _RETRO_DISABLED_TEXT,
            "indeterminate_icon": resource_path("retro_indeterminate.svg"),
            "indeterminate_icon_disabled": resource_path(
                "retro_indeterminate_disabled.svg"
            ),
            "lcd_bg": _RETRO_LCD_BG,
            "lcd_dim": _RETRO_LCD_DIM,
            "lcd_fg": _RETRO_LCD_FG,
            "radio_icon": resource_path("retro_radio.svg"),
            "radio_icon_disabled": resource_path("retro_radio_disabled.svg"),
            "radio_off_icon": resource_path("retro_radio_off.svg"),
            "radio_off_icon_disabled": resource_path("retro_radio_off_disabled.svg"),
            "text": _RETRO_TEXT,
        }
        return stylesheet % tokens
    except (KeyError, ModuleNotFoundError, OSError, ValueError) as exc:
        logger.warning("Could not load retro theme stylesheet: %s", exc)
        return ""


def app_stylesheet(theme: str) -> str:
    """Application-wide stylesheet for ``theme`` (empty for light/dark)."""
    return _retro_stylesheet() if normalize_theme(theme) == RETRO else ""


_APPLIED_THEME_PROPERTY = "neuroseq_applied_theme"


def apply_app_theme(app: QtCore.QCoreApplication | None, theme: str) -> None:
    """Apply ``theme`` to the whole Qt application.

    Uses the Fusion style so the custom palette is honored consistently across
    platforms, then installs the matching palette. A no-op when there is no
    GUI application instance (e.g. a bare QCoreApplication or ``None``).
    """
    if not isinstance(app, QtWidgets.QApplication):
        return
    theme = normalize_theme(theme)
    palette = qt_palette(theme)
    stylesheet = app_stylesheet(theme)
    if (
        app.property(_APPLIED_THEME_PROPERTY) == theme
        and app.palette() == palette
        and app.styleSheet() == stylesheet
    ):
        # Already applied, and widgets created since then get it when they
        # are polished: re-applying it would only re-style every widget.
        return
    app.setPalette(palette)
    # The retro skin needs an application-wide stylesheet on top of the
    # palette; switching back to light/dark clears it again.
    app.setStyleSheet(stylesheet)
    # QApplication.setPalette does not reach already-polished menu bars, menus,
    # or tool bars, so push the palette to every existing widget as well. In
    # addition, widgets that carry a stylesheet are painted through
    # QStyleSheetStyle, which caches palette-derived brushes and is not
    # invalidated by a palette change alone; re-applying the same stylesheet
    # forces it to re-resolve against the new palette (otherwise the top bar
    # lags one theme behind).
    for widget in app.allWidgets():
        style_sheet = widget.styleSheet()
        if style_sheet:
            widget.setStyleSheet(style_sheet)
        # Set after re-polishing, which can restore the previous palette
        widget.setPalette(palette)
    # Re-assert the Fusion style last so every widget is repolished and
    # repainted with the new palette already in place.
    app.setStyle("Fusion")
    app.setProperty(_APPLIED_THEME_PROPERTY, theme)
