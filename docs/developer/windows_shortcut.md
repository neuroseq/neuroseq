# Windows desktop shortcut

## Installed release (pip)

Point a shortcut at the installed executable; no terminal or activation is needed.

1. In the environment where NeuroSeq is installed, print the executable path:

   ```bat
   python -c "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_path('scripts')) / 'neuroseq.exe')"
   ```

   In Conda, use `conda run` instead so Conda sets up library paths. Find
   `conda.exe` with `python -c "import os; print(os.environ['CONDA_EXE'])"`.

2. Right-click the desktop, choose **New > Shortcut**, and enter the location
   in double quotes, for example:

   ```text
   "C:\Users\YourName\neuroseq-env\Scripts\neuroseq.exe"
   "C:\Users\YourName\anaconda3\Scripts\conda.exe" run --no-capture-output -n neuroseq neuroseq
   ```

3. Name it **NeuroSeq** and click **Finish**.

Recreate the shortcut if you move the environment. Update with
`python -m pip install --upgrade neuroseq` in that environment.

## Development version (Git and uv)

This shortcut updates a git clone from `main` and starts it with `uv run`, so
lab machines always run the current code.

1. Install [Git for Windows](https://git-scm.com/download/win) and
   [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
   (`pip install uv` in a Conda environment also works). Check `git --version`
   and `uv --version` in a new prompt.
2. Clone over HTTPS (no login needed for updates):

   ```bat
   cd %USERPROFILE%\Documents
   git clone https://github.com/neuroseq/neuroseq.git
   cd neuroseq
   ```

3. Create the shortcut from inside the clone (from the Anaconda Prompt with the
   right environment active if uv came from Conda):

   ```bat
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install_shortcut.ps1
   ```

   It also removes an older shortcut to a pip-installed `neuroseq.exe` unless you
   pass `-KeepOld`. Rerun it after moving the clone or uv.

4. Run `uv sync` once in the prompt to watch the first, multi-GB install
   (PyTorch with CUDA). Afterwards, double-click **NeuroSeq**. After an update
   that changes dependencies, the shortcut installs them silently; wait for it.

On each start, the shortcut fast-forwards the clone from GitHub, then runs
`uv run neuroseq`. It skips the update (and shows a notice) when the clone is
not on `main`, has changes to tracked files, has unpushed commits, or GitHub is
unreachable. It never switches branches; run `git switch main` yourself.

Logs of the last start are in `%LOCALAPPDATA%\neuroseq`: `launcher.log` (update
steps, started commit), `neuroseq.out.log` and `neuroseq.err.log` (the end of the
error log is shown if NeuroSeq fails). To start without updating, for example
during an experiment:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\launch_neuroseq.ps1 -NoUpdate
```
