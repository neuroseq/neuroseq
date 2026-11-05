from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq import theme
from tests.wait_timeout import WAIT_TIMEOUT_MS


def _relative_luminance(color: QtGui.QColor) -> float:
    channels = [c / 255 for c in (color.red(), color.green(), color.blue())]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: QtGui.QColor, second: QtGui.QColor) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _control_gallery() -> QtWidgets.QDialog:
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Theme control gallery")
    dialog.setFixedSize(560, 510)
    dialog.setFont(QtGui.QFont("DejaVu Sans", 10))
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    fields = QtWidgets.QGridLayout()
    spin_box = QtWidgets.QSpinBox()
    spin_box.setRange(0, 20)
    spin_box.setValue(7)
    double_spin_box = QtWidgets.QDoubleSpinBox()
    double_spin_box.setRange(0.0, 20.0)
    double_spin_box.setValue(3.5)
    combo_box = QtWidgets.QComboBox()
    combo_box.addItems(["Stim data", "Neuron data", "Spike data"])
    line_edit = QtWidgets.QLineEdit("Focused field")
    fields.addWidget(QtWidgets.QLabel("Integer"), 0, 0)
    fields.addWidget(spin_box, 0, 1)
    fields.addWidget(QtWidgets.QLabel("Decimal"), 0, 2)
    fields.addWidget(double_spin_box, 0, 3)
    fields.addWidget(QtWidgets.QLabel("Dataset"), 1, 0)
    fields.addWidget(combo_box, 1, 1)
    fields.addWidget(QtWidgets.QLabel("Name"), 1, 2)
    fields.addWidget(line_edit, 1, 3)
    layout.addLayout(fields)

    choices = QtWidgets.QHBoxLayout()
    for label, state in (
        ("Unchecked", QtCore.Qt.CheckState.Unchecked),
        ("Checked", QtCore.Qt.CheckState.Checked),
        ("Mixed", QtCore.Qt.CheckState.PartiallyChecked),
    ):
        check_box = QtWidgets.QCheckBox(label)
        check_box.setTristate(state == QtCore.Qt.CheckState.PartiallyChecked)
        check_box.setCheckState(state)
        choices.addWidget(check_box)
    radio_off = QtWidgets.QRadioButton("Off")
    radio_on = QtWidgets.QRadioButton("On")
    radio_on.setChecked(True)
    choices.addWidget(radio_off)
    choices.addWidget(radio_on)
    layout.addLayout(choices)

    tabs = QtWidgets.QTabWidget()
    tabs.setTabsClosable(True)
    first_page = QtWidgets.QWidget()
    first_layout = QtWidgets.QVBoxLayout(first_page)
    group_box = QtWidgets.QGroupBox("Metal group")
    group_layout = QtWidgets.QVBoxLayout(group_box)
    group_layout.addWidget(QtWidgets.QLabel("Grouped controls retain their bevel."))
    first_layout.addWidget(group_box)
    tabs.addTab(first_page, "Selected")
    tabs.addTab(QtWidgets.QWidget(), "Recessed")
    layout.addWidget(tabs)

    tree = QtWidgets.QTreeWidget()
    tree.setHeaderLabels(["Control", "State"])
    tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(["Plot canvas", "Themed"]))
    tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(["Selection", "Visible"]))
    tree.setFixedHeight(105)
    layout.addWidget(tree)

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    disabled_button = button_box.button(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    disabled_button.setDisabled(True)
    layout.addWidget(button_box)

    line_edit.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    return dialog


@pytest.mark.screenshot
@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_theme_control_gallery_screenshot(
    qtbot: QtBot,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    theme_name: str,
) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    dialog = _control_gallery()
    qtbot.addWidget(dialog)
    reference_path = (
        Path(__file__).parent / "sampledata" / f"theme_controls_{theme_name}.png"
    )
    actual_path = tmp_path / f"theme_controls_{theme_name}.png"

    try:
        theme.apply_app_theme(app, theme_name)
        dialog.show()
        qtbot.waitExposed(dialog, timeout=WAIT_TIMEOUT_MS)
        app.processEvents()
        pixmap = dialog.grab()
        assert pixmap.save(str(actual_path))

        if request.config.getoption("--update-screenshot"):
            assert pixmap.save(str(reference_path))
            return

        assert reference_path.exists()
        actual = np.asarray(Image.open(actual_path).convert("RGB"), dtype=np.int16)
        reference = np.asarray(
            Image.open(reference_path).convert("RGB"), dtype=np.int16
        )
        assert actual.shape == reference.shape
        difference = np.abs(actual - reference)
        assert float(np.mean(difference)) <= 3.0
        assert float(np.quantile(difference, 0.995)) <= 32.0
    finally:
        dialog.close()
        theme.apply_app_theme(app, theme.LIGHT)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "This is a real pixel grab of a genuine QComboBox popup window, not "
        "a palette computation - and CI's macOS runners use the offscreen "
        "QPA platform, whose popup-window rendering is not reliable there "
        "the way it is under Linux's offscreen backend. On macOS CI the "
        "highlighted and plain rows come back pixel-identical even though "
        "the same palette.setColor(Highlight, ...) call (applied to all "
        "QPalette color groups, so it is not an active/inactive-focus "
        "issue) is in effect - indicating the popup itself is not painted "
        "the same way there, not a real readability regression."
    ),
)
def test_retro_combo_popup_selection_is_readable(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout(dialog)
    combo_box = QtWidgets.QComboBox()
    combo_box.addItems(["Stim data", "Neuron data", "Spike data"])
    combo_box.setFixedWidth(220)
    layout.addWidget(combo_box)
    qtbot.addWidget(dialog)

    try:
        theme.apply_app_theme(app, theme.RETRO)
        dialog.show()
        qtbot.waitExposed(dialog, timeout=WAIT_TIMEOUT_MS)
        combo_box.showPopup()
        app.processEvents()

        view = combo_box.view()
        text_color = theme.qt_palette(theme.RETRO).color(QtGui.QPalette.ColorRole.Text)
        model = combo_box.model()
        # Compare the highlighted row against a plain one inside a SINGLE
        # frame: grabbing after a synthetic hover races the repaint and is
        # flaky, whereas the current index is applied synchronously.
        for row in range(combo_box.count()):
            view.setCurrentIndex(model.index(row, 0))
            app.processEvents()
            qtbot.wait(20)
            image = view.viewport().grab().toImage()
            sample_x = image.width() - 8
            highlighted = image.pixelColor(
                sample_x, view.visualRect(model.index(row, 0)).center().y()
            )
            other_row = (row + 1) % combo_box.count()
            plain = image.pixelColor(
                sample_x, view.visualRect(model.index(other_row, 0)).center().y()
            )
            # The highlighted row has to be visibly picked out...
            assert highlighted != plain
            # ...and its label has to stay legible on it. Styling the closed
            # combo makes Qt paint this row with the combo's own background
            # instead of the palette highlight, so a gradient there turns the
            # row black - which is what this contrast floor catches.
            assert _contrast_ratio(highlighted, text_color) >= 4.5
    finally:
        combo_box.hidePopup()
        dialog.close()
        theme.apply_app_theme(app, theme.LIGHT)


def test_retro_button_focus_does_not_look_pressed(qtbot: QtBot) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout(dialog)
    button = QtWidgets.QPushButton("Select file for segmentation")
    button.setFixedSize(220, 32)
    other_control = QtWidgets.QLineEdit()
    layout.addWidget(button)
    layout.addWidget(other_control)
    qtbot.addWidget(dialog)

    def button_surface_lightness() -> int:
        image = button.grab().toImage()
        return image.pixelColor(button.width() - 10, 5).lightness()

    try:
        theme.apply_app_theme(app, theme.RETRO)
        dialog.show()
        qtbot.waitExposed(dialog, timeout=WAIT_TIMEOUT_MS)

        other_control.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
        app.processEvents()
        normal_lightness = button_surface_lightness()

        button.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
        app.processEvents()
        focused_lightness = button_surface_lightness()

        qtbot.mousePress(button, QtCore.Qt.MouseButton.LeftButton)
        app.processEvents()
        pressed_lightness = button_surface_lightness()
        qtbot.mouseRelease(button, QtCore.Qt.MouseButton.LeftButton)

        assert abs(focused_lightness - normal_lightness) <= 5
        assert focused_lightness - pressed_lightness >= 20
    finally:
        dialog.close()
        theme.apply_app_theme(app, theme.LIGHT)
