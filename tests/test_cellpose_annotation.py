from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from neuroseq.cellpose_annotation import annotate_mean_fluorescence_tiff


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
