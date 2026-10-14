# Windows desktop shortcut

On a Windows PC that runs NeuroSeq from a git clone with `uv`, a desktop
shortcut can keep that clone up to date with `main` on GitHub and start
NeuroSeq from it, so nobody has to open a prompt.

## Install

Run once, from a prompt where `uv --version` works (for example the Anaconda
Prompt with the environment that provides `uv` active), in the NeuroSeq clone:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install_shortcut.ps1
```

This creates **NeuroSeq** on the desktop. It also removes an older desktop
shortcut that starts a pip-installed `neuroseq.exe`, since that copy does not
follow the clone. Pass `-KeepOld` to keep it.

Run the command again after moving the clone or `uv`.

## What the shortcut does

Each time it starts NeuroSeq, the shortcut:

1. Updates the clone from GitHub. This only happens when the clone is on
   `main`, has no changes to tracked files, and the update is a fast-forward.
   Untracked files, such as data, do not prevent it.
2. Starts NeuroSeq with `uv run`, which first brings the clone's `.venv` in
   line with `uv.lock`. This takes longer than usual only after an update
   changed the dependencies.

If the update cannot run, NeuroSeq still starts with the current version, and
a notice that closes by itself says why. The reasons are: the clone is on
another branch, it has local changes, GitHub could not be reached, or the
local `main` has commits that are not on GitHub.

A clone on a feature branch is never moved to `main` automatically. Switch it
back yourself once that work is merged:

```bat
git switch main
```

## Logs

Output of the last start is in `%LOCALAPPDATA%\neuroseq`:

- `launcher.log`: the update steps and the commit that was started.
- `neuroseq.out.log` and `neuroseq.err.log`: NeuroSeq's own output. If
  NeuroSeq stops with an error, the end of `neuroseq.err.log` is also shown in a
  message.

To start without updating, for example during an experiment, run:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\launch_neuroseq.ps1 -NoUpdate
```
