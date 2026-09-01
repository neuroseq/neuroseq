## Issues during installation

### Long paths in windows

During installation on windows one can encounter the following error:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\af24\\AppData\\Local\\Temp\\pip-install-i3syxwx0\\imgui-bundle_3462a851e1e44f9caa7b8510206f002e\\external\\hello_imgui\\hello_imgui\\hello_imgui_cmake\\android\\apkCMake\\templates\\sdl\\gradle_template\\app\\src\\main\\java\\org\\libsdl\\app\\HIDDeviceBLESteamController.java'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths
```

The issues can be fixed by entering Registry Editor and modifying _LongPathsEnabled_ variable under _HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem_ from "0" to "1".

Refer to [this webpage](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later) on the instructions to fix it.

## Imaging / segmentation

### Cellpose "use GPU" is greyed out (segmentation is very slow)

Cellpose (and Suite2p) only offer GPU acceleration when **PyTorch** exposes a
GPU backend — **CUDA** (NVIDIA) or **MPS** (Apple Silicon). A **CPU-only**
PyTorch build disables the "use GPU" checkbox. This is common on **Windows**,
where PyPI's default `torch` wheel is CPU-only.

Check the installed build (in the NeuroSeq environment):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`…+cpu` / `False` means CPU-only. If the machine has an NVIDIA (or
Apple-Silicon) GPU, install a GPU build and relaunch:

```bash
uv pip install --torch-backend=auto torch
```

Not using `uv` (e.g. NeuroSeq is installed into a conda environment)? Install
directly with pip instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If the machine has **only Intel integrated graphics**, Cellpose cannot use it
(CUDA/MPS only) — run segmentation on a GPU machine and import its `_seg.npy`
instead. See
[Importing imaging data → GPU acceleration](user_guide/importing_imaging_data.md#gpu-acceleration-cellpose-suite2p)
for details.
