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
        np.save(
            Path(filename).with_name(f"{Path(filename).stem}_seg.npy"),
            np.array({"masks": masks, "filename": filename}, dtype=object),
        )

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

    # A segmentation made by hand in the Cellpose GUI must survive.
    manual = tmp_path / "mean_seg.npy"
    manual.write_bytes(b"hand-edited")

    seg_path, mask_path = annotate_mean_fluorescence_tiff(image_path)
    assert model_init == {"gpu": True, "pretrained_model": "cpsam"}
    assert (seg_path.name, mask_path.name) == (
        "mean_cpsam_seg.npy",
        "mean_cpsam_cp_masks.png",
    )

    # Issue #598: another model's results are saved beside, not over, them.
    v2_seg, v2_mask = annotate_mean_fluorescence_tiff(image_path, model_name="cpsam_v2")
    assert (v2_seg.name, v2_mask.name) == (
        "mean_cpsam_v2_seg.npy",
        "mean_cpsam_v2_cp_masks.png",
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mean.tif",
        "mean_cpsam_cp_masks.png",
        "mean_cpsam_seg.npy",
        "mean_cpsam_v2_cp_masks.png",
        "mean_cpsam_v2_seg.npy",
        "mean_seg.npy",
    ]
    assert manual.read_bytes() == b"hand-edited"
    saved = np.load(v2_seg, allow_pickle=True).item()
    assert saved["filename"] == str(image_path.resolve())
    assert saved["model_name"] == "cpsam_v2"


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


def test_load_annotation_image_accepts_a_png_snapshot(tmp_path: Path) -> None:
    """Feature (#566): a PNG is a single still image, not a stack - it
    should load directly without going through the TIFF frame-count path."""
    image_path = tmp_path / "snapshot.png"
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    Image.fromarray(pixels).save(image_path)

    image = load_annotation_image(image_path)

    np.testing.assert_array_equal(image, pixels)


def test_adjusted_annotation_image() -> None:
    """Issue #611: the window applies to the image's 8-bit display version;
    without one, Cellpose gets the original image."""
    from neuroseq.cellpose_annotation import adjusted_annotation_image
    from neuroseq.visualization_support.brightness_contrast import DisplayRange

    image = np.array([[0.0, 500.0], [1000.0, 2000.0]], dtype=np.float32)
    assert adjusted_annotation_image(image, None) is image
    assert adjusted_annotation_image(image, DisplayRange()) is image
    adjusted = adjusted_annotation_image(image, DisplayRange(0, 128))
    assert adjusted.dtype == np.uint8
    # 500 of 2000 is level 64 of 255; the window 0-128 doubles it.
    np.testing.assert_array_equal(adjusted, [[0, 127], [255, 255]])
