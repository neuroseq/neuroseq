from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from neuroseq.cellpose_annotation import annotate_mean_fluorescence_tiff
from neuroseq.cellpose_annotation import load_annotation_image
from neuroseq.cellpose_annotation import tiff_frame_count


def test_annotation_requires_gpu(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "neuroseq.cellpose_annotation.torch_gpu_status",
        return_value=(False, "No CUDA/MPS backend."),
    )

    with pytest.raises(RuntimeError, match="requires GPU acceleration"):
        annotate_mean_fluorescence_tiff(tmp_path / "mean.tif")


def test_annotation_constructs_cellpose_with_gpu_enabled(
    tmp_path: Path, mocker, monkeypatch
) -> None:
    image_path = tmp_path / "mean.tif"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)
    model_init: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            model_init.update(kwargs)

        def eval(self, image):
            return np.zeros(image.shape, dtype=np.int32), [np.zeros(image.shape)], None

    def save_seg(_image, masks, _flows, filename) -> None:
        np.save(Path(filename).with_name(f"{Path(filename).stem}_seg.npy"), masks)

    def save_masks(_image, masks, _flows, filename, **_kwargs) -> None:
        Image.fromarray(masks.astype(np.uint8)).save(
            Path(filename).with_name(f"{Path(filename).stem}_cp_masks.png")
        )

    fake_cellpose = SimpleNamespace(
        models=SimpleNamespace(CellposeModel=FakeModel),
        io=SimpleNamespace(masks_flows_to_seg=save_seg, save_masks=save_masks),
    )
    monkeypatch.setitem(__import__("sys").modules, "cellpose", fake_cellpose)
    mocker.patch(
        "neuroseq.cellpose_annotation.torch_gpu_status",
        return_value=(True, "CUDA available."),
    )

    annotate_mean_fluorescence_tiff(image_path)

    assert model_init == {"gpu": True, "pretrained_model": "cpsam"}


def test_multiframe_tiff_averages_every_second_of_first_300_frames(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "recording.tif"
    frames = np.arange(302, dtype=np.uint16)[:, None, None] * np.ones(
        (1, 2, 3), dtype=np.uint16
    )
    tifffile.imwrite(image_path, frames, photometric="minisblack")

    image = load_annotation_image(image_path)

    assert tiff_frame_count(image_path) == 302
    assert image.shape == (2, 3)
    np.testing.assert_array_equal(image, np.full((2, 3), 149.0))
