from __future__ import annotations

import gc
import weakref

from pytestqt.qtbot import QtBot
from qtpy import QtWidgets

from neuroseq import theme
from neuroseq.icon_manager import IconManager
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _icon_luminance(icon) -> float:
    """Mean luminance of the icon's opaque pixels (0=black .. 255=white)."""
    image = icon.pixmap(32, 32).toImage()
    total = 0.0
    count = 0
    for x in range(0, 32, 4):
        for y in range(0, 32, 4):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() > 10:
                total += (pixel.red() + pixel.green() + pixel.blue()) / 3
                count += 1
    return total / count if count else -1.0


def test_icon_manager_action_recolors_between_themes(qtbot: QtBot) -> None:
    manager = IconManager()
    action = QtWidgets.QAction("x")
    manager.set_action_icon(action, "mdi6.file")

    manager.apply_theme(theme.LIGHT)
    light = _icon_luminance(action.icon())
    manager.apply_theme(theme.DARK)
    dark = _icon_luminance(action.icon())

    # Light theme -> dark glyph; dark theme -> light glyph.
    assert light >= 0 and dark >= 0
    assert dark > light


def test_icon_manager_prunes_deleted_targets(qtbot: QtBot) -> None:
    manager = IconManager()
    action = QtWidgets.QAction("x")
    manager.set_action_icon(action, "mdi6.file")
    assert len(manager._entries) == 1

    action.deleteLater()
    action = None  # drop the last Python reference
    qtbot.wait(10)

    # Should not raise even though the underlying target may be gone, and it
    # must not grow the registry with dead entries.
    manager.apply_theme(theme.DARK)
    assert len(manager._entries) <= 1


def test_icon_manager_forgets_deleted_targets_without_theme_change(
    qtbot: QtBot,
) -> None:
    class Owner:
        pass

    def make_action(parent: QtWidgets.QWidget, owner: Owner) -> QtWidgets.QAction:
        action = QtWidgets.QAction("x", parent)
        action.triggered.connect(lambda: owner)
        return action

    manager = IconManager()
    parent = QtWidgets.QWidget()
    owner = Owner()
    action = make_action(parent, owner)
    manager.set_action_icon(action, "mdi6.file")
    assert len(manager._entries) == 1
    owner_ref = weakref.ref(owner)

    parent.deleteLater()
    del parent, action, owner
    qtbot.waitUntil(lambda: not manager._entries, timeout=WAIT_TIMEOUT_MS)
    gc.collect()
    # The registry no longer keeps the action, or what its slots capture, alive.
    assert owner_ref() is None


def test_icon_manager_can_be_collected_before_its_targets(qtbot: QtBot) -> None:
    with qtbot.captureExceptions() as exceptions:
        manager = IconManager()
        action = QtWidgets.QAction("x")
        manager.set_action_icon(action, "mdi6.file")
        del manager, action
        gc.collect()
        qtbot.wait(10)
    assert not exceptions


def test_icon_manager_widget_and_window_icons(qtbot: QtBot) -> None:
    manager = IconManager()
    button = QtWidgets.QPushButton()
    qtbot.addWidget(button)
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    manager.set_widget_icon(button, "mdi6.folder-open")
    manager.set_window_icon(window, "fa6s.brain")

    assert not button.icon().isNull()
    assert not window.windowIcon().isNull()

    # Recoloring both target kinds should not raise.
    manager.apply_theme(theme.DARK)
    manager.apply_theme(theme.LIGHT)
    assert not button.icon().isNull()
