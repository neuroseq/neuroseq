# Troubleshooting

## Installation

### Long path error during install

`pip install` fails with `OSError: [Errno 2] No such file or directory` on a
very long path and the hint _"this system does not have Windows Long Path
support enabled"_. Fix: in the Registry Editor set `LongPathsEnabled` under
`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` to `1`
([Microsoft instructions](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later)).

## Imaging and segmentation

### GPU option greyed out in cell segmentation

Cellpose and Suite2p use the GPU only if PyTorch has CUDA (NVIDIA) or MPS
(Apple Silicon). PyPI's default `torch` on Windows is CPU-only. Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`+cpu` or `False` means CPU-only. On a machine with a supported GPU, install a
GPU build and relaunch:

```bash
uv pip install --torch-backend=auto torch
# without uv (e.g. Conda):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Intel integrated graphics are not supported: segment on a GPU machine and
import its `_seg.npy`. See [Imaging data](readout/imaging_data.md).
