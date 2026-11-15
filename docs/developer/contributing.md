# Contributing

NeuroSeq is a neuronal sequencer: one input and readout interface for digital,
biological and neuromorphic networks. Contributions are welcome.

## Plugins first, core by discussion

Extend NeuroSeq without touching the core whenever possible:

| Extension                    | Mechanism                                                 |
| ---------------------------- | --------------------------------------------------------- |
| Network backend or substrate | [Custom simulator](custom_simulators.md) file             |
| Analysis pipeline            | [Analysis plugin](analysis_plugins.md) file               |
| Input protocol               | Script in the embedded console that writes an input array |

There is no plugin interface for closed-loop routines yet; they live in
`stimulation/dynamic_input/`. Changes to core code (anything under
`src/neuroseq/`) need prior discussion with the core developers, in an issue or
draft pull request.

## Workflow

1. Fork, clone and [install](developer_installation.md) in editable mode.
2. Create a feature branch.
3. Make the change and add tests, including a
   [scientific validation test](testing.md#scientific-validation-tests) when it
   affects results.
4. Run `uv run pytest` and `prek run -a`.
5. Open a pull request that explains the user-visible change, links the issue,
   lists the commands you ran and includes screenshots for UI changes.

Use short, imperative commit subjects. Do not commit generated files
(`.coverage`, `coverage.xml`, caches, `.venv/`). If you use AI tools, follow
[Responsible use of AI](responsible_ai.md).

## Code organisation

`CLAUDE.md` and `AGENTS.md` at the repository root are the authoritative
description of the architecture and grouping rules. In short:

- Group code by workflow (`importers/`, `export/`, `device_control/`,
  `stimulation/`, `visualization/`), not by kind of file.
- Keep Qt-free logic in its own module; put widget/project coordination in a controller.
- Change project data only through `Project` methods so
  [undo history](project_history.md) stays consistent.
- Run long tasks through `background_jobs.BackgroundJob`.
- Use `qtpy`, not `PyQt6`/`PySide6`, directly.

## Project workspaces

`gui.py` is the tab shell. Each tab holds a `ProjectWorkspace`
(`project_workspace.py`) that owns its widgets, controllers, console namespace,
undo history and background jobs; `Gui.active_workspace` is the current tab.
Switching tabs does not copy state; playback pauses but jobs keep running and
report to their own workspace. GUI state is not saved in `.nseq` files.

Device connections are shared application resources (`workspace_devices.py`).
SysCon2 replies and ScanImage acquisition results are routed to the requesting
workspace; only the active one gets live ScanImage frames. Closing a tab stops
its jobs without disconnecting other tabs. Tests: `tests/test_project_workspaces.py`.
