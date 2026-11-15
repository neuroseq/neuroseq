# Project undo and redo

`src/neuroseq/project_history.py` holds the Qt-free `ProjectHistory` and
`ProjectCommand`. Each `Project` owns one `history`, shared by all its input
arrays and experiments, so undo follows chronological order across them.

## What is recorded

Input value edits, generated patterns, padding and trimming, and experiment
additions, appends, replacements and deletions (deleting several groups is one
command). Redo reuses retained data and never repeats a file import.

These clear the history, because older commands could reference obsolete names
or sample indices: creating, renaming, deleting, resampling, remapping or looping
inputs, and importing project components. Simulator checkpoints, device state and
console mutations are not recorded. Undo/redo is disabled while a background job runs.

## Writing a command

A `ProjectCommand` has a `label` and `redo`/`undo` callables. Use
`history.execute(command)` to apply it, or `history.record_applied(command)` after
the mutation already succeeded. Callbacks must:

- validate before mutating, or roll back partial failures;
- not hold Qt widgets;
- raise `StaleCommandError` before changing anything if the target data changed
  outside the history (the history then drops all commands).

Experiment mutators keep array buffers by reference, so replace buffers instead
of writing into them in place. Listeners refresh views; bound-method listeners
are held weakly.

## Lifetime

History is per project, survives saving and workspace refresh, and is not stored
in `.nseq` files: a loaded project starts empty. Undo and redo mark the project dirty.
