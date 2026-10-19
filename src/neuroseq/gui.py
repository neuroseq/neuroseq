from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import psutil
import pynvml
import qtawesome as qta
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel

from neuroseq.example_projects import getting_started_project
from neuroseq.gui_settings import GuiSettings
from neuroseq.project import Project
from neuroseq.project_workspace import ProjectWorkspace
from neuroseq.project_workspace import WorkspaceCloseGate
from neuroseq.workspace_devices import WorkspaceDevices


class ResourceMonitorLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.process.cpu_percent(interval=None)
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError as e:
            self.logger.warning(f"Could not initialize NVML: {e}")
            self.handle = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)
        self.update_stats()

    def update_stats(self):
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            ram_gb = self.process.memory_info().rss / (1024 * 1024 * 1024)
            text = f"CPU: {cpu_percent:.1f}% | RAM: {ram_gb:.2f} GB "
            if self.handle is not None:
                gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                text += f"| GPU: {gpu_percent:.1f}% "
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                    if p.pid == self.pid and p.usedGpuMemory is not None:
                        vram_gb = p.usedGpuMemory / (1024 * 1024 * 1024)
                        text += f"| VRAM: {vram_gb:.2f} GB "
            self.setText(text)
        except Exception as e:
            self.logger.exception(e)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)


class _ProjectTabs(QtWidgets.QStackedWidget):
    """Keep project pages together with a tab bar placed in their header row."""

    tabCloseRequested = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._tab_bar = QtWidgets.QTabBar(self)
        self._tab_bar.setDocumentMode(True)
        self._tab_bar.setMovable(True)
        self._tab_bar.setTabsClosable(True)
        self._tab_bar.setExpanding(False)
        self._tab_bar.setStyleSheet("QTabBar::tab { padding: 0px 8px; }")
        self._tab_bar.setFixedHeight(max(22, self._tab_bar.fontMetrics().height() + 4))
        self._tab_bar.currentChanged.connect(self.setCurrentIndex)
        self.currentChanged.connect(self._tab_bar.setCurrentIndex)
        self._tab_bar.tabCloseRequested.connect(self.tabCloseRequested.emit)
        self._tab_bar.tabMoved.connect(self._move_tab)

    def tabBar(self) -> QtWidgets.QTabBar:
        return self._tab_bar

    def addTab(self, workspace: ProjectWorkspace, label: str) -> int:
        with QtCore.QSignalBlocker(self._tab_bar):
            self._tab_bar.addTab(label)
            index = self.addWidget(workspace)
            self._tab_bar.setCurrentIndex(self.currentIndex())
        return index

    def setTabText(self, index: int, label: str) -> None:
        self._tab_bar.setTabText(index, label)

    def removeTab(self, index: int) -> None:
        workspace = self.widget(index)
        if workspace is None:
            return
        if workspace.isAncestorOf(self._tab_bar):
            # Keep the shared bar alive when its current workspace is deleted.
            self._tab_bar.setParent(self)
        with QtCore.QSignalBlocker(self._tab_bar):
            self._tab_bar.removeTab(index)
            self.removeWidget(workspace)
            self._tab_bar.setCurrentIndex(self.currentIndex())

    def _move_tab(self, before: int, after: int) -> None:
        workspace = self.widget(before)
        current = self.currentWidget()
        assert workspace is not None and current is not None
        with QtCore.QSignalBlocker(self):
            self.removeWidget(workspace)
            self.insertWidget(after, workspace)
            self.setCurrentWidget(current)
        self.currentChanged.emit(self.currentIndex())


class Gui(QtWidgets.QMainWindow):
    """Application shell; every tab owns a complete ProjectWorkspace."""

    def __init__(self, filename: pathlib.Path | str | Project | None = None) -> None:
        super().__init__()
        self.settings = GuiSettings()
        self.devices = WorkspaceDevices(self)
        self._closing_discarded: set[ProjectWorkspace] = set()
        self._close_gate = WorkspaceCloseGate(self)
        self._disposed = False
        self._visible_workspace: ProjectWorkspace | None = None
        self.project_tabs = _ProjectTabs(self)
        self.project_tabs.currentChanged.connect(self._activate_workspace)
        self.project_tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.project_tabs)
        self.status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(ResourceMonitorLabel())
        self.setWindowIcon(qta.icon("fa6s.brain"))
        self._add_workspace(filename)
        self.showMaximized()

    @property
    def active_workspace(self) -> ProjectWorkspace:
        workspace = self.project_tabs.currentWidget()
        assert isinstance(workspace, ProjectWorkspace)
        return workspace

    @property
    def workspaces(self) -> list[ProjectWorkspace]:
        workspaces = []
        for index in range(self.project_tabs.count()):
            workspace = self.project_tabs.widget(index)
            assert isinstance(workspace, ProjectWorkspace)
            workspaces.append(workspace)
        return workspaces

    @property
    def project(self) -> Project:
        return self.active_workspace.project

    def __getattr__(self, name: str) -> Any:
        # Keep read access to the active workspace convenient for integrations.
        # Signals and jobs always bind directly to a workspace, never to this proxy.
        tabs = self.__dict__.get("project_tabs")
        if tabs is not None:
            workspace = tabs.currentWidget()
            if workspace is not None:
                return getattr(workspace, name)
        raise AttributeError(name)

    def _add_workspace(
        self, project: pathlib.Path | str | Project | None = None
    ) -> ProjectWorkspace:
        for existing in self.workspaces:
            if existing.project is project:
                self.project_tabs.setCurrentWidget(existing)
                return existing
        workspace = ProjectWorkspace(
            project,
            parent=self.project_tabs,
            settings=self.settings,
            devices=self.devices,
        )
        workspace.open_project_requested.connect(self._add_workspace)
        workspace.exit_requested.connect(self.close)
        workspace.windowTitleChanged.connect(
            lambda _title: self._refresh_title(workspace)
        )
        workspace.theme_changed.connect(self._apply_workspace_theme)
        index = self.project_tabs.addTab(
            workspace, workspace._tab_label_for(workspace.project)
        )
        self.project_tabs.setCurrentIndex(index)
        self._refresh_title(workspace)
        return workspace

    def load_project(self, project: pathlib.Path | str | Project) -> None:
        # The workspace performs file loading asynchronously and requests a new
        # workspace only once it has a complete Project object.
        self.active_workspace.load_project(project)

    def _activate_workspace(self, _index: int) -> None:
        if self.project_tabs.currentWidget() is self._visible_workspace:
            return
        if self._visible_workspace is not None:
            self._visible_workspace.set_active(False)
        if self.project_tabs.currentWidget() is None:
            self._visible_workspace = None
            return
        self._visible_workspace = self.active_workspace
        self._visible_workspace.set_project_tab_bar(self.project_tabs.tabBar())
        self._visible_workspace.set_active(True)
        self._refresh_title(self._visible_workspace)

    def _refresh_title(self, workspace: ProjectWorkspace) -> None:
        index = self.project_tabs.indexOf(workspace)
        if index >= 0:
            self.project_tabs.setTabText(
                index, workspace._tab_label_for(workspace.project)
            )
            if workspace is self.project_tabs.currentWidget():
                self.setWindowTitle(workspace.windowTitle())

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        # Workspace shortcuts normally handle input in their own subtree.
        # Route shortcuts from shell controls to the active workspace as well.
        if event is not None:
            sequence = QtGui.QKeySequence(event.keyCombination())
            for action in self.active_workspace.actions():
                if action.isEnabled() and sequence in action.shortcuts():
                    action.trigger()
                    event.accept()
                    return
        super().keyPressEvent(event)

    def _apply_workspace_theme(self, theme: str) -> None:
        for workspace in self.workspaces:
            # Theme is an application preference; avoid rebroadcasting it.
            with QtCore.QSignalBlocker(workspace):
                workspace.apply_theme(theme)

    def _close_tab(self, index: int) -> None:
        workspace = self.project_tabs.widget(index)
        if isinstance(workspace, ProjectWorkspace):
            self._close_workspace(workspace)

    def _close_workspace(self, workspace: ProjectWorkspace) -> None:
        if self.project_tabs.indexOf(workspace) < 0:
            return
        self.project_tabs.setCurrentWidget(workspace)
        if not self._close_gate.ready(
            [workspace], "Close tab", lambda: self._close_workspace(workspace)
        ):
            return
        workspace._prompt_save_before_action(
            "Close tab",
            "Closing this tab will discard its unsaved changes.\n\nSave the project first?",
            lambda: self._remove_workspace(workspace),
        )

    def _remove_workspace(self, workspace: ProjectWorkspace) -> None:
        if self.project_tabs.indexOf(workspace) < 0:
            return
        # Work may have started while a save prompted by the close was running.
        if not self._close_gate.ready(
            [workspace], "Close tab", lambda: self._remove_workspace(workspace)
        ):
            return
        if not workspace.stop_background_job():
            return
        index = self.project_tabs.indexOf(workspace)
        # Deactivate before disposal; changing the current tab must never call
        # set_active() on a workspace whose plots have already been closed.
        if self._visible_workspace is workspace:
            workspace.set_active(False)
            self._visible_workspace = None
        self.project_tabs.removeTab(index)
        workspace.dispose()
        workspace.deleteLater()
        self._closing_discarded.discard(workspace)
        if self.project_tabs.count() == 0:
            self._add_workspace(getting_started_project())

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if event is None:
            return
        if self._disposed:
            event.accept()
            return
        # Check every workspace before stopping any: a refused close must not
        # have cancelled or dropped work in another tab.
        if not self._close_gate.ready(self.workspaces, "Close NeuroSeq", self.close):
            self._closing_discarded.clear()
            event.ignore()
            return
        for workspace in self.workspaces:
            if not workspace.project.dirty or workspace in self._closing_discarded:
                continue
            self.project_tabs.setCurrentWidget(workspace)
            answer = QtWidgets.QMessageBox.warning(
                self,
                "Close NeuroSeq",
                f"Save changes to '{workspace._tab_label_for(workspace.project)}' before closing?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                self._closing_discarded.clear()
                event.ignore()
                return
            if answer == QtWidgets.QMessageBox.StandardButton.Save:
                event.ignore()
                # Keep discarded projects unchanged while another tab saves.
                self.setEnabled(False)

                def saved(_result: object) -> None:
                    self.setEnabled(True)
                    self.close()

                def failed(_message: str) -> None:
                    self.setEnabled(True)
                    self._closing_discarded.clear()

                if not workspace._save_project(on_success=saved, on_failure=failed):
                    failed("")
                return
            self._closing_discarded.add(workspace)
        # Jobs have already ended; this drains chats before destroying any
        # workspace, including hidden ones.
        for workspace in self.workspaces:
            if not workspace.stop_background_job():
                self._closing_discarded.clear()
                event.ignore()
                return
        for workspace in self.workspaces:
            workspace.dispose()
        self.devices.shutdown()
        self._disposed = True
        super().closeEvent(event)
