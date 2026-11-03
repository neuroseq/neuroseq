from __future__ import annotations

import logging
from collections.abc import Callable

import qtawesome as qta
from qtpy import QtGui

from neuroseq import theme as theme_module

logger = logging.getLogger(__name__)


class IconManager:
    """Creates qtawesome icons in the current theme color and recolors them.

    qtawesome bakes a fixed color into each rendered ``QIcon``, so icons do not
    follow the Qt palette. This manager remembers every icon target together
    with the icon name(s) used to build it, and re-renders them in the new
    theme's foreground color whenever the theme changes.
    """

    def __init__(self) -> None:
        self._theme = theme_module.LIGHT
        # (target, apply_fn, names, kwargs)
        self._entries: list[
            tuple[object, Callable[[object, QtGui.QIcon], None], tuple[str, ...], dict]
        ] = []

    def _build(self, names: tuple[str, ...], kwargs: dict) -> QtGui.QIcon:
        options = dict(kwargs)
        options.setdefault("color", theme_module.icon_color(self._theme))
        return qta.icon(*names, **options)

    def icon(self, *names: str, **kwargs: object) -> QtGui.QIcon:
        """Build a one-off themed icon that is not tracked for recoloring."""
        return self._build(names, dict(kwargs))

    def _register(
        self,
        target: object,
        apply_fn: Callable[[object, QtGui.QIcon], None],
        names: tuple[str, ...],
        kwargs: dict,
    ) -> None:
        apply_fn(target, self._build(names, kwargs))
        self._entries.append((target, apply_fn, names, kwargs))
        destroyed = getattr(target, "destroyed", None)
        if destroyed is not None:
            # Forget the target once Qt deletes it: the entry otherwise keeps
            # its Python wrapper, and everything its slots capture, alive.
            key = id(target)
            destroyed.connect(lambda *_: self._forget(key))

    def _forget(self, key: int) -> None:
        self._entries = [entry for entry in self._entries if id(entry[0]) != key]

    def set_action_icon(self, action: object, *names: str, **kwargs: object) -> None:
        self._register(
            action,
            lambda t, ic: t.setIcon(ic),  # type: ignore[attr-defined]
            names,
            dict(kwargs),
        )

    def set_widget_icon(self, widget: object, *names: str, **kwargs: object) -> None:
        self._register(
            widget,
            lambda t, ic: t.setIcon(ic),  # type: ignore[attr-defined]
            names,
            dict(kwargs),
        )

    def set_window_icon(self, window: object, *names: str, **kwargs: object) -> None:
        self._register(
            window,
            lambda t, ic: t.setWindowIcon(ic),  # type: ignore[attr-defined]
            names,
            dict(kwargs),
        )

    def apply_theme(self, theme: str) -> None:
        """Recolor every tracked icon for ``theme`` and drop dead targets."""
        self._theme = theme_module.normalize_theme(theme)
        live: list[
            tuple[object, Callable[[object, QtGui.QIcon], None], tuple[str, ...], dict]
        ] = []
        for target, apply_fn, names, kwargs in self._entries:
            try:
                apply_fn(target, self._build(names, kwargs))
            except RuntimeError:
                # The underlying C++ object has been deleted; drop the entry.
                continue
            live.append((target, apply_fn, names, kwargs))
        self._entries = live


# Application-wide singleton used by add_action and the widgets.
icons = IconManager()
