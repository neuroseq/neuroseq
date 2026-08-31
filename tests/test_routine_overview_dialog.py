from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.routine_overview_dialog import ROUTINE_OVERVIEWS
from neuroseq.routine_overview_dialog import RoutineSelectorDialog
from neuroseq.routine_overview_dialog import routine_overview
from neuroseq.stim_editor import StimEditor


def test_every_dynamic_routine_has_an_overview() -> None:
    # Each routine offered by the selector must have a registered overview so
    # the always-on preview never falls back to "no preview available".
    for routine in StimEditor.DYNAMIC_INPUT_ROUTINES:
        overview = routine_overview(routine)
        assert overview is not None
        assert overview.goal
        assert overview.parameters
        # Exactly one preview source: a matplotlib plot or a static image.
        assert (overview.plot is not None) != (overview.image_path is not None)


def test_routine_overview_unknown_routine_returns_none() -> None:
    assert routine_overview("not a routine") is None


def test_selector_shows_preview_for_initial_selection(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    first = StimEditor.DYNAMIC_INPUT_ROUTINES[0]
    overview = routine_overview(first)
    assert overview is not None
    # Preview is populated immediately (always on), in the same window.
    assert dialog.lbl_title.text() == overview.title
    assert dialog.lbl_goal.text() == overview.goal
    assert dialog._axes.get_title() == overview.plot_title
    assert dialog._axes.lines
    dialog.close()


def test_selector_preview_updates_when_selection_changes(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    for routine in StimEditor.DYNAMIC_INPUT_ROUTINES:
        dialog.cmb_routine.setCurrentText(routine)
        overview = routine_overview(routine)
        assert overview is not None
        assert dialog.lbl_title.text() == overview.title
        if overview.image_path is not None:
            # Image-backed routines show the SVG widget, not the plot.
            assert not dialog.image_widget.isHidden()
            assert dialog.image_widget.is_valid()
            assert dialog.plot_widget.isHidden()
        else:
            assert not dialog.plot_widget.isHidden()
            assert dialog.image_widget.isHidden()
            assert dialog._axes.get_title() == overview.plot_title
            assert dialog._axes.lines
    dialog.close()


def test_rheobase_preview_uses_bundled_svg(qtbot: QtBot) -> None:
    overview = routine_overview(StimEditor.RHEOBASE_ROUTINE)
    assert overview is not None
    # Rheobase previews the bundled strength-duration SVG diagram.
    assert overview.plot is None
    assert overview.image_path is not None
    assert overview.image_path.name == "rheobase_chronaxie.svg"
    assert overview.image_path.is_file()

    dialog = RoutineSelectorDialog([StimEditor.RHEOBASE_ROUTINE])
    qtbot.addWidget(dialog)
    assert not dialog.image_widget.isHidden()
    assert dialog.image_widget.is_valid()
    assert dialog.plot_widget.isHidden()
    dialog.close()


def test_selector_continue_emits_selected_routine(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(StimEditor.DYNAMIC_INPUT_ROUTINES))
    qtbot.addWidget(dialog)

    chosen: list[str] = []
    dialog.routine_chosen.connect(chosen.append)
    dialog.cmb_routine.setCurrentText(StimEditor.STDP_WINDOW_ROUTINE)
    dialog._on_continue()
    assert chosen == [StimEditor.STDP_WINDOW_ROUTINE]
    dialog.close()


def test_selector_apply_theme_runs(qtbot: QtBot) -> None:
    dialog = RoutineSelectorDialog(list(ROUTINE_OVERVIEWS))
    qtbot.addWidget(dialog)
    # Recoloring for either theme must not raise.
    dialog.apply_theme("dark")
    dialog.apply_theme("light")
    dialog.close()
