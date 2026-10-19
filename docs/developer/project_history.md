# Project data undo and redo

`src/neuroseq/project_history.py` contains the Qt-independent `ProjectHistory`
and `ProjectCommand` classes. Each `Project` owns a `history`; stimulation
modifiers use that same instance. Edits across input arrays and experimental
data are undone in chronological order, regardless of the selected input.

The shared history records stimulation value edits, generated patterns, padding
and trimming, and experimental-data additions, appends, replacements and
deletions. Deleting several data groups is one command. Appending retains the
previous arrays and timing metadata, including data overwritten by overlapping
time ranges. A failed append restores all affected datasets without recording
a command. Redo uses the retained data and does not repeat file import.

Commands provide a label and `redo()` / `undo()` callbacks. Use
`history.execute(command)` to apply a new command, or
`history.record_applied(command)` after a successful mutation. Callbacks should
validate before mutation or roll back partial failures. Commands must operate on
project data without retaining Qt widgets. History listeners refresh the views;
bound-method listeners are weak and observers can explicitly unsubscribe.

Experiment mutators retain array buffers by reference and copy metadata, rather
than copying a whole project or its simulator. These mutators replace buffers
when changing data; writing into retained buffers in place would invalidate a
snapshot. Direct console mutations do not automatically create commands.

Refreshing an existing workspace preserves history. Histories belong to each
project independently, remain available after saving, and are not serialized in
`.nseq` archives. Loaded projects start with an empty history. Undo and redo mark
the project dirty, even when undo happens to return to its last saved content.

Existing non-undoable input restructuring operations (creation, renaming, deletion,
resampling, neuron remapping and looping) clear the history because older
commands can reference obsolete names or sample indices. Importing project
components also starts a new baseline. Simulator checkpoints, device state and
hardware activity remain outside this history. Workspace undo and redo are
disabled while a background operation is running.
