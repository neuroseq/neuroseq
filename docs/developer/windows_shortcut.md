# Miscellaneous

## Windows desktop shortcut

### Pip-installed release

After installing with pip, you can create a shortcut to launch the installed
release without opening a terminal each time. Git and uv are not required.

#### With venv or a regular Python installation

1. In the environment where you installed NeuroSeq, find its executable:

   ```bat
   python -c "import sysconfig; from pathlib import Path; print(Path(sysconfig.get_path('scripts')) / 'neuroseq.exe')"
   ```

2. Right-click the Windows desktop and choose **New > Shortcut**. Paste the
   full path printed above, enclosed in double quotes, into the location field.
   For example:

   ```text
   "C:\Users\YourName\neuroseq-env\Scripts\neuroseq.exe"
   ```

3. Click **Next**, name the shortcut **NeuroSeq**, and click **Finish**.

Double-click the shortcut to start NeuroSeq. A
[venv does not need to be activated](https://docs.python.org/3.12/library/venv.html#how-venvs-work)
when using the full path to an installed executable. Keep the environment in
the same location; recreate the shortcut if you move or replace it.

#### With Conda

For a pip installation inside Conda, use `conda run` in the shortcut so that
[Conda sets up the environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#activating-an-environment),
including Windows library paths.

In an Anaconda Prompt with your environment activated, find `conda.exe`:

```bat
python -c "import os; print(os.environ['CONDA_EXE'])"
```

Create a shortcut as above, using this location instead:

```text
"C:\Users\YourName\anaconda3\Scripts\conda.exe" run --no-capture-output -n neuroseq neuroseq
```

Replace the example path with your own `conda.exe` path and, if needed, replace
the environment name after `-n` with the name you used during installation.

Both shortcuts start the installed release. To update it, activate the same
environment and run `python -m pip install --upgrade neuroseq`.

### Development version (Git and uv)

On Windows, NeuroSeq can be started from a desktop shortcut that first updates
it from `main` on GitHub. NeuroSeq then runs from a git clone with
[uv](https://docs.astral.sh/uv/), so the installed version always matches the
code, and nobody has to open a prompt to start or update it.

#### Install

##### 1. Install Git and uv

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

##### 2. Get NeuroSeq

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

##### 3. Create the shortcut

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

##### 4. First start

The first install downloads all dependencies, including PyTorch with CUDA
support (several GB), which can take many minutes. Run it once in the same
prompt so you can follow its progress:

```bat
uv sync
```

Then double-click **NeuroSeq**. Later starts are quick, unless an update
changed the dependencies; the shortcut shows no progress while it installs
them, so give it time instead of starting it again.

#### What the shortcut does

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

#### Logs

Output of the last start is in `%LOCALAPPDATA%\neuroseq`:

- `launcher.log`: the update steps and the commit that was started.
- `neuroseq.out.log` and `neuroseq.err.log`: NeuroSeq's own output. If
  NeuroSeq stops with an error, the end of `neuroseq.err.log` is also shown in a
  message.

To start without updating, for example during an experiment, run:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\launch_neuroseq.ps1 -NoUpdate
```

## Connection to Holo4D profile

The syscon2 path to user profile with .seq files:

C:\ProgramData\ROE\SysCon 2\2.7.2\profiles\Holo4D\Sequence Manager\Experiments\NeuroSeq
