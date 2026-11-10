"""Programmatic Cellpose annotation for mean-fluorescence images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from neuroseq.gpu import torch_gpu_status
from neuroseq.visualization_support.brightness_contrast import DisplayRange
from neuroseq.visualization_support.brightness_contrast import apply_display_range
from neuroseq.visualization_support.brightness_contrast import to_8bit

MAX_SOURCE_FRAMES = 300
FRAME_STEP = 2

TIFF_SUFFIXES = (".tif", ".tiff")
# .jpg/.jpeg deliberately excluded - lossy compression isn't appropriate for
# quantitative segmentation (see issue #568).
SEGMENTATION_IMAGE_SUFFIXES = (*TIFF_SUFFIXES, ".png")

# Model identifiers accepted by cellpose.models.CellposeModel(pretrained_model=...)
# in the pinned cellpose version. "cpsam" is listed first (rather than cellpose's
# own new default of "cpsam_v2") so NeuroSeq's one-button segmentation keeps
# producing the same results existing users already rely on unless they
# deliberately switch models.
CELLPOSE_MODEL_NAMES = ("cpsam", "cpsam_v2", "cpdino", "cpdino-vitb")


def tiff_frame_count(image_path: str | Path) -> int:
    """Return the number of pages in the selected TIFF."""
    with tifffile.TiffFile(Path(image_path)) as tif:
        return len(tif.pages)


def load_annotation_image(image_path: str | Path) -> np.ndarray:
    """Load a single image, or average sampled pages from a TIFF stack.

    The result is writable: Cellpose normalises its input in place, and a
    single image read through Pillow would be read-only.
    """
    path = Path(image_path)
    if path.suffix.lower() not in TIFF_SUFFIXES:
        # A single still image (e.g. PNG) - nothing to average over.
        with Image.open(path) as image:
            return np.array(image)

    frame_count = tiff_frame_count(path)
    if frame_count <= 1:
        with Image.open(path) as image:
            return np.array(image)

    frame_indices = range(0, min(frame_count, MAX_SOURCE_FRAMES), FRAME_STEP)
    frames = tifffile.imread(path, key=frame_indices)
    mean = np.mean(frames, axis=0)
    if frames.dtype == np.uint8:
        # Stays 8-bit, so brightness_contrast.to_8bit keeps it on its own scale
        # as the Cellpose importer does.
        return np.rint(mean).astype(np.uint8)
    return mean


def adjusted_annotation_image(
    image: np.ndarray, display_range: DisplayRange | None
) -> np.ndarray:
    """The image Cellpose segments: unchanged, or with a brightness/contrast
    window applied to its 8-bit display version (issue #611)."""
    if display_range is None or display_range.is_identity:
        return image
    return apply_display_range(to_8bit(image), display_range)


def annotate_mean_fluorescence_tiff(
    image_path: str | Path,
    *,
    model_name: str = "cpsam",
    display_range: DisplayRange | None = None,
) -> tuple[Path, Path]:
    """Annotate a TIFF or PNG image and save Cellpose GUI and mask outputs
    beside it.

    Returns the ``*_<model>_seg.npy`` file and the ``*_<model>_cp_masks.png``
    mask. The model name in the file names keeps the results of different
    models apart (issue #598), and an ``*_seg.npy`` made in the Cellpose GUI is
    never overwritten. Cellpose is imported lazily so NeuroSeq can still show a useful GUI error if
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
    image_array = adjusted_annotation_image(load_annotation_image(path), display_range)

    # GPU availability was verified above. Passing True is intentional: never
    # silently downgrade this interactive annotation workflow to the CPU.
    model = models.CellposeModel(gpu=True, pretrained_model=model_name)
    masks, flows, _styles = model.eval(image_array)

    segmentation_path = path.with_name(f"{path.stem}_{model_name}_seg.npy")
    mask_path = path.with_name(f"{path.stem}_{model_name}_cp_masks.png")
    # Cellpose names its outputs after the image, so let it write them in a
    # scratch folder next to the image and move them to their final names.
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        scratch_image = Path(scratch) / path.name
        # Save the editable Cellpose GUI payload as well as an ordinary
        # labelled mask.
        io.masks_flows_to_seg(image_array, masks, flows, str(scratch_image))
        io.save_masks(
            image_array,
            masks,
            flows,
            str(scratch_image),
            png=True,
            tif=False,
            suffix="_cp_masks",
        )
        scratch_segmentation = Path(scratch) / f"{path.stem}_seg.npy"
        if not scratch_segmentation.exists():
            raise RuntimeError(f"Cellpose did not create {segmentation_path.name}")
        # The Cellpose GUI finds the image through the stored file name.
        segmentation = np.load(scratch_segmentation, allow_pickle=True).item()
        segmentation["filename"] = str(path)
        segmentation["model_name"] = model_name
        if display_range is not None and not display_range.is_identity:
            segmentation["display_range"] = (
                display_range.minimum,
                display_range.maximum,
            )
        np.save(segmentation_path, segmentation, allow_pickle=True)
        scratch_mask = Path(scratch) / f"{path.stem}_cp_masks.png"
        if scratch_mask.exists():
            os.replace(scratch_mask, mask_path)
    return segmentation_path, mask_path
