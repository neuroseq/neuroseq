"""Programmatic Cellpose annotation for mean-fluorescence images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from neuroseq.gpu import torch_gpu_status


def annotate_mean_fluorescence_tiff(
    image_path: str | Path,
    *,
    model_name: str = "cpsam",
) -> tuple[Path, Path]:
    """Annotate a TIFF and save Cellpose GUI and mask outputs beside it.

    Returns the standard ``*_seg.npy`` file and the ``*_cp_masks.png`` mask.
    Cellpose is imported lazily so NeuroSeq can still show a useful GUI error if
    the optional dependency is unavailable.
    """
    gpu_available, gpu_message = torch_gpu_status()
    if not gpu_available:
        raise RuntimeError(
            "Cellpose annotation requires GPU acceleration.\n\n" + gpu_message
        )

    from cellpose import io
    from cellpose import models

    path = Path(image_path).resolve()
    with Image.open(path) as image:
        image_array = np.asarray(image)

    # GPU availability was verified above. Passing True is intentional: never
    # silently downgrade this interactive annotation workflow to the CPU.
    model = models.CellposeModel(gpu=True, pretrained_model=model_name)
    masks, flows, _styles = model.eval(image_array)

    # Save the editable Cellpose GUI payload as well as an ordinary labelled mask.
    io.masks_flows_to_seg(image_array, masks, flows, str(path))
    io.save_masks(
        image_array,
        masks,
        flows,
        str(path),
        png=True,
        tif=False,
        suffix="_cp_masks",
    )

    segmentation_path = path.with_name(f"{path.stem}_seg.npy")
    mask_path = path.with_name(f"{path.stem}_cp_masks.png")
    if not segmentation_path.exists():
        raise RuntimeError(f"Cellpose did not create {segmentation_path.name}")
    return segmentation_path, mask_path
