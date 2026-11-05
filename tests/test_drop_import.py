"""Importing files dropped onto the main window (issue #597)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from neuroseq.project_workspace import DROP_SEGMENT
from neuroseq.project_workspace import DROP_TRACES
from neuroseq.project_workspace import DROP_VIDEO
from neuroseq.project_workspace import ProjectWorkspace
from tests.wait_timeout import WAIT_TIMEOUT_MS

LEGACY_SEQ = Path(__file__).parent / "data" / "legacy_cellpose_v1" / "legacy_export.seq"


@pytest.fixture
def gui(qtbot: QtBot):
    window = ProjectWorkspace()
    qtbot.addWidget(window)
    yield window
    window._close_confirmed = True
    window.close()


@pytest.fixture
def calls(
    gui: ProjectWorkspace, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, dict]]:
    """Record the import handlers a drop reaches instead of opening dialogs."""
    recorded: list[tuple[str, dict]] = []
    for name in (
        "load_project",
        "btn_import_syscon2_seq_clicked",
        "btn_import_patch_clamp_input_clicked",
        "btn_import_patch_clamp_output_clicked",
        "btn_import_cellpose_data_clicked",
        "btn_import_tif_clicked",
        "_on_calcium_video_import_requested",
    ):

        def _record(*args: Any, _name: str = name, **kwargs: Any) -> None:
            recorded.append((_name, {"args": args, **kwargs}))

        monkeypatch.setattr(gui, name, _record)
    return recorded


def test_file_types_go_to_their_import(gui: ProjectWorkspace, calls: list) -> None:
    gui.import_dropped_files(
        [Path("p.nseq"), Path("s.seq"), Path("w.ibw"), Path("t.csv")], target=None
    )
    assert calls == [
        ("load_project", {"args": (Path("p.nseq"),)}),
        ("btn_import_syscon2_seq_clicked", {"args": (), "path": Path("s.seq")}),
        (
            "btn_import_patch_clamp_output_clicked",
            {"args": (), "path": Path("w.ibw")},
        ),
        (
            "btn_import_patch_clamp_output_clicked",
            {"args": (), "path": Path("t.csv")},
        ),
    ]


def test_waves_dropped_on_the_input_array_become_inputs(
    gui: ProjectWorkspace, calls: list
) -> None:
    gui.import_dropped_files([Path("w.pxp"), Path("t.txt")], target="input")
    assert [name for name, _ in calls] == [
        "btn_import_patch_clamp_input_clicked",
        "btn_import_patch_clamp_input_clicked",
    ]
    assert calls[1][1]["path"] == Path("t.txt")


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("segmentation", "btn_import_cellpose_data_clicked"),
        ("traces", "btn_import_tif_clicked"),
        ("imaging", "_on_calcium_video_import_requested"),
    ],
)
def test_imaging_files_follow_the_panel(
    gui: ProjectWorkspace,
    calls: list,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected: str,
) -> None:
    prompts: list[str] = []

    def _get_item(_parent, title, _label, items, *args):
        prompts.append(title)
        return items[1], True  # the second Cellpose model

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", _get_item)
    gui.import_dropped_files([Path("fov.tif")], target=target)

    assert [name for name, _ in calls] == [expected]
    if expected == "btn_import_cellpose_data_clicked":
        assert prompts == ["Segment cells"]  # only the model is asked for
        assert calls[0][1]["segment"] == Path("fov.tif")
        assert calls[0][1]["model"] == "cpsam_v2"
    else:
        assert prompts == []


def test_imaging_files_elsewhere_ask_what_to_do(
    gui: ProjectWorkspace, calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    offered: list[list[str]] = []

    def _get_item(_parent, _title, _label, items, *args):
        offered.append(list(items))
        return DROP_VIDEO, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", _get_item)
    gui.import_dropped_files([Path("a.tif"), Path("b.gif")], target=None)
    assert offered == [[DROP_SEGMENT, DROP_TRACES, DROP_VIDEO]]
    assert calls == [
        (
            "_on_calcium_video_import_requested",
            {"args": (), "paths": [Path("a.tif"), Path("b.gif")]},
        )
    ]

    offered.clear()
    gui.import_dropped_files([Path("snapshot.png")], target="imaging")
    assert offered == [[DROP_SEGMENT]]  # a PNG is no movie


def test_unknown_files_are_reported(
    gui: ProjectWorkspace, calls: list, qmessagebox_calls: list[dict[str, str]]
) -> None:
    gui.import_dropped_files([Path("notes.docx")], target=None)
    assert calls == []
    assert qmessagebox_calls[-1]["title"] == "Cannot import these files"
    assert "notes.docx" in qmessagebox_calls[-1]["text"]


def test_dropping_on_a_panel(
    gui: ProjectWorkspace, calls: list, qtbot: QtBot, tmp_path: Path
) -> None:
    """A real drop event, routed by the panel under the cursor."""
    with qtbot.waitExposed(gui, timeout=WAIT_TIMEOUT_MS):
        gui.show()
    gui._auto_show_widget(gui.stim_editor)
    qtbot.wait(50)
    path = tmp_path / "wave.ibw"
    path.write_bytes(b"")
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
    centre = gui.stim_editor.mapTo(gui, gui.stim_editor.rect().center())
    assert gui._drop_target(centre) == "input"

    enter = QtGui.QDragEnterEvent(
        centre,
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(gui, enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(
        QtCore.QPointF(centre),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(gui, drop)
    qtbot.waitUntil(lambda: bool(calls), timeout=WAIT_TIMEOUT_MS)
    assert calls == [
        ("btn_import_patch_clamp_input_clicked", {"args": (), "path": path})
    ]


def test_dropped_seq_file_is_loaded_in_the_dialog(
    gui: ProjectWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuroseq.importers.syscon2_import_dialog import Syscon2ImportDialog

    seen: list[str] = []

    def _exec(dialog: Syscon2ImportDialog) -> int:
        seen.append(dialog.file_path)
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(Syscon2ImportDialog, "exec", _exec)
    gui.btn_import_syscon2_seq_clicked(path=LEGACY_SEQ)
    assert seen == [str(LEGACY_SEQ)]


def test_menu_imports_still_ask_for_files(
    gui: ProjectWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The menu actions pass Qt's "checked" flag; the new keyword-only file
    parameters must not pick it up."""
    from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
    from neuroseq.importers.tif_import_dialog import TifImportDialog

    opened: list[str] = []

    def _cellpose_exec(dialog: CellposeImportDialog) -> int:
        opened.append("cellpose")
        return QtWidgets.QDialog.DialogCode.Rejected

    def _tif_exec(dialog: TifImportDialog) -> int:
        opened.append(f"tif:{len(dialog._tif_paths)}")
        return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(CellposeImportDialog, "exec", _cellpose_exec)
    monkeypatch.setattr(TifImportDialog, "exec", _tif_exec)
    actions = {action.text(): action for action in gui.findChildren(QtWidgets.QAction)}
    actions["Cell &segmentation"].trigger()
    actions["&Calcium Imaging (raw .tif)"].trigger()
    assert opened == ["cellpose", "tif:0"]
