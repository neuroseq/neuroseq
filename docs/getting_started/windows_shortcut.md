# Windows desktop shortcut

On Windows, NeuroSeq can be started from a desktop shortcut that first updates
it from `main` on GitHub. NeuroSeq then runs from a git clone with
[uv](https://docs.astral.sh/uv/), so the installed version always matches the
code, and nobody has to open a prompt to start or update it.

## Install

### 1. Install Git and uv

- **Git:** install [Git for Windows](https://git-scm.com/download/win) with
  the default options.
- **uv:** in PowerShell, run the
  [standalone installer](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer):

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  If you already use conda, `pip install uv` inside a conda environment works
  as well.

Open a new prompt afterwards and check that both work:

```bat
git --version
uv --version
```

### 2. Get NeuroSeq

Clone the repository, for example into `Documents\GitHub`:

```bat
cd %USERPROFILE%\Documents
mkdir GitHub
cd GitHub
git clone https://github.com/neuroseq/neuroseq.git
cd neuroseq
```

The HTTPS address needs no GitHub login, so the shortcut can update without
asking for a password.

### 3. Create the shortcut

In the same prompt, inside the clone, run:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install_shortcut.ps1
```

It creates **NeuroSeq** on the desktop and prints the clone and the `uv` it
will use. If `uv` came from a conda environment, run this from the Anaconda
Prompt with that environment active, because the shortcut remembers that `uv`.

The script also removes an older desktop shortcut that starts a pip-installed
`neuroseq.exe`, since that copy does not follow the clone. Pass `-KeepOld` to
keep it. Run the script again after moving the clone or `uv`.

### 4. First start

The first install downloads all dependencies, including PyTorch with CUDA
support (several GB), which can take many minutes. Run it once in the same
prompt so you can follow its progress:

```bat
uv sync
```

Then double-click **NeuroSeq**. Later starts are quick, unless an update
changed the dependencies; the shortcut shows no progress while it installs
them, so give it time instead of starting it again.

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
