# Contributing

The NeuroSeq is an open-source project. Everyone can suggest and implement improvements as well as improve overall reliability and stability of its operation.

## Development pipeline

1. Fork and clone repository.
2. Create a feature branch.
3. [Install](developer_installation.md) in editable mode.
4. Modify the code.
5. Test new functionality.
6. Run [pytests](testing.md) before opening a pull request.

When using AI tools, follow the guidance on
[responsible use of AI](responsible_ai.md), including human review and scientific
validation of the resulting changes.

## Project workspaces

`gui.py` contains the application shell. Each tab embeds a separate
`ProjectWorkspace` from `project_workspace.py`, which owns its project's widgets,
controllers, console namespace, undo history, input designers, detached windows,
and background jobs. The workspace uses a child `QMainWindow` to lay out its own
menus and toolbars. `Gui.active_workspace` is the explicit access point for the
current tab; `Gui.project` and forwarded attribute reads remain available for
integrations.

Switching tabs changes the visible workspace without calling `update_project()`
or copying widget state into `Project`. Playback pauses when leaving a tab;
background jobs continue and deliver their results to their owning workspace.
`update_project()` refreshes data within a workspace, for example after imports
or model edits. GUI state remains in memory and is not serialized to `.nseq`.

Settings and physical connections are application resources. `workspace_devices.py`
shares the ScanImage connection and SysCon2 transport. Each SysCon2 panel receives
a session that routes command replies to the requesting workspace; connection
status is shared. Only the active workspace receives live ScanImage frames.
Acquisition results are matched by run ID to the workspace that requested them.
Closing a tab stops its jobs and releases its resources without disconnecting
other tabs. Application shutdown checks unsaved projects individually and stops
all workspaces before shutting down the shared connections.

Run `uv run pytest tests/test_project_workspaces.py` for tab isolation, background
result routing, device sessions, and workspace lifecycle tests. Existing widget
and workflow tests exercise `ProjectWorkspace` directly.
