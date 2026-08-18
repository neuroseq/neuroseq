"""Unit tests for the raw-TIFF Suite2p pipeline glue code.

The full pipeline calls into Suite2p (heavy + GPU) which we do not exercise
here; we cover the pure-Python pieces and confirm the wrapper passes the right
parameters through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from neuroseq.importer_tif import TifPipelineParams
from neuroseq.importer_tif import _masks_to_stat


def _make_mask_image(shape=(32, 32)) -> np.ndarray:
    """Three labelled blobs on a background of zeros."""
    masks = np.zeros(shape, dtype=np.int32)
    masks[4:10, 4:10] = 1
    masks[15:20, 12:18] = 2
    masks[22:28, 22:30] = 3
    return masks


def _suite2p_available() -> bool:
    """Suite2p 0.x raises ValueError at import on Python 3.11 — be defensive."""
    try:
        import suite2p.detection  # noqa: F401

        return True
    except Exception:
        return False


def test_masks_to_stat_emits_one_entry_per_cell():
    if not _suite2p_available():
        pytest.skip("suite2p v1.x not installed")

    masks = _make_mask_image()
    stat = _masks_to_stat(masks, masks.shape[0], masks.shape[1])

    assert len(stat) == 3
    for entry in stat:
        assert {"ypix", "xpix", "npix", "lam", "med"} <= set(entry.keys())
        assert entry["npix"] == len(entry["xpix"])
        assert entry["lam"].shape == entry["xpix"].shape


def test_tif_pipeline_params_defaults_target_jgcamp8m():
    params = TifPipelineParams()
    assert params.fs == pytest.approx(30.0)
    assert params.tau == pytest.approx(0.5)
    assert params.nchannels == 2
    assert params.functional_chan == 1
    assert params.neuropil_coefficient == pytest.approx(0.7)
    assert params.batch_size == 200
    assert params.baseline == "maximin"


def test_run_suite2p_pipeline_rejects_empty_tif_list():
    # ValueError check fires before suite2p import — no skip needed.
    from neuroseq.importer_tif import run_suite2p_pipeline

    with pytest.raises(ValueError, match="at least one TIFF"):
        run_suite2p_pipeline([], _make_mask_image(), TifPipelineParams())


def test_run_roi_average_pipeline_extracts_and_concatenates_traces(tmp_path):
    import tifffile

    from neuroseq.importer_tif import run_roi_average_pipeline

    masks = np.array([[1, 1], [0, 2]], dtype=np.int32)
    first = np.array([[[2, 4], [99, 10]], [[6, 8], [99, 14]]], dtype=np.uint16)
    second = np.array([[[10, 12], [99, 18]]], dtype=np.uint16)
    first_path = tmp_path / "first.tif"
    second_path = tmp_path / "second.tif"
    tifffile.imwrite(first_path, first, photometric="minisblack")
    tifffile.imwrite(second_path, second, photometric="minisblack")

    result = run_roi_average_pipeline(
        [first_path, second_path], masks, TifPipelineParams(nchannels=1)
    )

    np.testing.assert_allclose(result.F, [[3, 7, 11], [10, 14, 18]])
    assert result.extra["pipeline"] == "roi_average"


def test_run_roi_average_pipeline_rejects_mismatched_masks(tmp_path):
    import tifffile

    from neuroseq.importer_tif import run_roi_average_pipeline

    path = tmp_path / "stack.tif"
    tifffile.imwrite(
        path, np.zeros((2, 3, 4), dtype=np.uint8), photometric="minisblack"
    )
    with pytest.raises(ValueError, match="does not match TIFF frame shape"):
        run_roi_average_pipeline(
            [path], np.ones((2, 2), dtype=np.int32), TifPipelineParams(nchannels=1)
        )


def test_run_roi_average_pipeline_uses_selected_interleaved_channel(tmp_path):
    import tifffile

    from neuroseq.importer_tif import run_roi_average_pipeline

    masks = np.ones((2, 2), dtype=np.int32)
    stack = np.array(
        [
            [[1, 1], [1, 1]],
            [[10, 10], [10, 10]],
            [[2, 2], [2, 2]],
            [[20, 20], [20, 20]],
        ],
        dtype=np.uint16,
    )
    path = tmp_path / "two_channels.tif"
    tifffile.imwrite(path, stack, photometric="minisblack")

    result = run_roi_average_pipeline(
        [path], masks, TifPipelineParams(nchannels=2, functional_chan=2)
    )

    np.testing.assert_allclose(result.F, [[10, 20]])
    assert result.extra["functional_chan"] == 2


def test_run_roi_average_pipeline_uses_channel_axis(tmp_path):
    import tifffile

    from neuroseq.importer_tif import run_roi_average_pipeline

    masks = np.ones((2, 2), dtype=np.int32)
    stack = np.zeros((3, 2, 2, 2), dtype=np.uint16)
    stack[:, 0, :, :] = np.array([1, 2, 3])[:, None, None]
    stack[:, 1, :, :] = np.array([10, 20, 30])[:, None, None]
    path = tmp_path / "channel_axis.tif"
    tifffile.imwrite(path, stack, photometric="minisblack")

    result = run_roi_average_pipeline(
        [path], masks, TifPipelineParams(nchannels=2, functional_chan=2)
    )

    np.testing.assert_allclose(result.F, [[10, 20, 30]])


def test_run_suite2p_pipeline_rejects_non_2d_masks():
    from neuroseq.importer_tif import run_suite2p_pipeline

    bad_masks = np.zeros((4, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="masks must be 2D"):
        run_suite2p_pipeline(
            [Path("/tmp/nonexistent.tif")], bad_masks, TifPipelineParams()
        )


def test_cellpose_result_to_masks_round_trips_background_and_labels():
    from neuroseq.tif_import_dialog import _cellpose_result_to_masks

    seg = np.array(
        [
            [-1, -1, 0, 0],
            [-1, 0, 0, 1],
            [-1, -1, 1, 1],
        ],
        dtype=np.int32,
    )
    result = MagicMock()
    result.segmentation_array = seg

    masks = _cellpose_result_to_masks(result)
    # 0-based -> 1-based, -1 -> 0
    np.testing.assert_array_equal(
        masks,
        np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 2],
                [0, 0, 2, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_load_masks_from_seg_file_unwraps_cellpose_dict(tmp_path: Path):
    from neuroseq.tif_import_dialog import _load_masks_from_seg_file

    masks = _make_mask_image()
    seg_path = tmp_path / "image_seg.npy"
    np.save(seg_path, np.array({"masks": masks, "outlines": None}, dtype=object))

    loaded = _load_masks_from_seg_file(seg_path)
    np.testing.assert_array_equal(loaded, masks)


def test_load_masks_from_seg_file_accepts_plain_array(tmp_path: Path):
    from neuroseq.tif_import_dialog import _load_masks_from_seg_file

    masks = _make_mask_image()
    seg_path = tmp_path / "plain.npy"
    np.save(seg_path, masks)

    loaded = _load_masks_from_seg_file(seg_path)
    np.testing.assert_array_equal(loaded, masks)


def test_run_suite2p_pipeline_orchestration(monkeypatch, tmp_path: Path):
    """Exercise run_suite2p_pipeline with fully-mocked Suite2p modules.

    Validates the orchestration: db assembly, channel handling, stat building,
    extraction/preprocess/oasis call sequence, and cleanup of the tempdir.
    """
    import sys
    import types

    from neuroseq.importer_tif import TifPipelineParams
    from neuroseq.importer_tif import run_suite2p_pipeline

    Ly, Lx, nframes, n_neurons = 16, 16, 32, 3
    masks = _make_mask_image((Ly, Lx))

    # Fabricate a minimal suite2p surface area.
    def default_db():
        return {"placeholder": 0}

    def default_settings():
        return {
            "extraction": {"neuropil_coefficient": 0.7, "batch_size": 200},
        }

    def get_file_list(db):
        return ["/tmp/x.tif"], np.array([1], dtype=np.int64)

    def init_dbs(db):
        return [
            {
                "reg_file": str(tmp_path / "reg.bin"),
                "reg_file_chan2": str(tmp_path / "reg2.bin"),
                "Ly": Ly,
                "Lx": Lx,
                "nframes": nframes,
            }
        ]

    def tiff_to_binary(dbs, settings, files, files_chan2):
        return dbs

    class FakeBinaryFile:
        def __init__(self, Ly, Lx, filename):
            self.Ly = Ly
            self.Lx = Lx
            self.data = np.zeros((nframes, Ly, Lx), dtype=np.float32)

    def assign_overlaps(stat, Ly_, Lx_):
        return stat

    def roi_stats(stat, Ly_, Lx_):
        return stat

    def extraction_wrapper(stat, f_bin, f_reg_chan2, settings, device):
        F = np.full((n_neurons, nframes), 5.0, dtype=np.float32)
        Fneu = np.full((n_neurons, nframes), 1.0, dtype=np.float32)
        return F, Fneu, None, None

    def preprocess(F, fs, batch_size, device, **kwargs):
        return F

    def oasis(F, batch_size, tau, fs):
        return np.zeros_like(F)

    class FakeClassfile:
        pass

    def classify(stat, classfile):
        return np.zeros((len(stat), 2), dtype=np.float32)

    # Assemble fake modules (use setattr to keep mypy quiet about attribute
    # assignments on ModuleType for names that resolve to real packages).
    def _module(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    suite2p_mod = _module(
        "suite2p", default_db=default_db, default_settings=default_settings
    )
    s2io_mod = _module(
        "suite2p.io",
        get_file_list=get_file_list,
        init_dbs=init_dbs,
        tiff_to_binary=tiff_to_binary,
        BinaryFile=FakeBinaryFile,
    )
    s2detection_mod = _module(
        "suite2p.detection",
        assign_overlaps=assign_overlaps,
        roi_stats=roi_stats,
    )
    s2extraction_mod = _module(
        "suite2p.extraction",
        extraction_wrapper=extraction_wrapper,
        preprocess=preprocess,
        oasis=oasis,
    )
    s2classification_mod = _module(
        "suite2p.classification",
        classify=classify,
        builtin_classfile=FakeClassfile(),
    )
    torch_mod = _module(
        "torch",
        cuda=_FakeCuda(),
        device=lambda x: f"device:{x}",
        # Distinct dummy so scipy's array-api probe sees a Tensor type that
        # numpy arrays are (correctly) not instances of.
        Tensor=type("Tensor", (), {}),
    )

    monkeypatch.setitem(sys.modules, "suite2p", suite2p_mod)
    monkeypatch.setitem(sys.modules, "suite2p.io", s2io_mod)
    monkeypatch.setitem(sys.modules, "suite2p.detection", s2detection_mod)
    monkeypatch.setitem(sys.modules, "suite2p.extraction", s2extraction_mod)
    monkeypatch.setitem(sys.modules, "suite2p.classification", s2classification_mod)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    tif = tmp_path / "data" / "stack.tif"
    tif.parent.mkdir()
    tif.write_bytes(b"")

    progress_calls: list[tuple[str, float]] = []
    result = run_suite2p_pipeline(
        [tif],
        masks,
        TifPipelineParams(nchannels=2),
        progress=lambda msg, frac: progress_calls.append((msg, frac)),
    )

    assert result.F.shape == (n_neurons, nframes)
    assert result.Fneu.shape == (n_neurons, nframes)
    assert result.spks.shape == (n_neurons, nframes)
    assert result.mean_image.shape == (Ly, Lx)
    # All Cellpose ROIs were marked as cells.
    assert np.all(result.iscell[:, 0] == 1)
    # Stat entries got snr/skew/std appended.
    for entry in result.stat:
        assert {"snr", "skew", "std"} <= set(entry.keys())
    # Progress callback was invoked for the pipeline phases.
    assert progress_calls
    assert progress_calls[-1] == ("Done", 1.0)
    # Temp dir cleaned up.
    children = [d for d in tmp_path.iterdir() if d.name.startswith("neuroseq_suite2p_")]
    assert children == []


def test_run_suite2p_pipeline_filters_mixed_fov_tiffs(monkeypatch, tmp_path: Path):
    """When the data folder contains TIFFs from other FOVs, only the selected
    files should reach `db["file_list"]`, and `first_files` should be re-anchored
    so each surviving folder still has a True entry."""
    import sys
    import types

    from neuroseq.importer_tif import TifPipelineParams
    from neuroseq.importer_tif import run_suite2p_pipeline

    Ly, Lx, nframes, n_neurons = 16, 16, 8, 3
    masks = _make_mask_image((Ly, Lx))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    selected_a = data_dir / "fov1_run1.tif"
    selected_b = data_dir / "fov1_run2.tif"
    other_fov = data_dir / "fov2_run1.tif"
    for p in (selected_a, selected_b, other_fov):
        p.write_bytes(b"")

    # Suite2p sees all three files; first_files marks the first per folder.
    discovered_files = [str(other_fov), str(selected_a), str(selected_b)]
    discovered_first = np.array([True, False, False], dtype=bool)

    captured: dict[str, Any] = {}

    def get_file_list(db):
        return list(discovered_files), discovered_first.copy()

    def init_dbs(db):
        captured["file_list"] = list(db["file_list"])
        captured["first_files"] = np.asarray(db["first_files"]).copy()
        return [
            {
                "reg_file": str(tmp_path / "reg.bin"),
                "reg_file_chan2": str(tmp_path / "reg2.bin"),
                "Ly": Ly,
                "Lx": Lx,
                "nframes": nframes,
            }
        ]

    class FakeBinaryFile:
        def __init__(self, Ly, Lx, filename):
            self.data = np.zeros((nframes, Ly, Lx), dtype=np.float32)

    def extraction_wrapper(stat, f_bin, f_reg_chan2, settings, device):
        F = np.full((n_neurons, nframes), 5.0, dtype=np.float32)
        Fneu = np.full((n_neurons, nframes), 1.0, dtype=np.float32)
        return F, Fneu, None, None

    def _module(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    suite2p_mod = _module(
        "suite2p",
        default_db=lambda: {},
        default_settings=lambda: {
            "extraction": {"neuropil_coefficient": 0.7, "batch_size": 1},
        },
    )
    s2io_mod = _module(
        "suite2p.io",
        get_file_list=get_file_list,
        init_dbs=init_dbs,
        tiff_to_binary=lambda dbs, *a, **kw: dbs,
        BinaryFile=FakeBinaryFile,
    )
    s2det = _module(
        "suite2p.detection",
        assign_overlaps=lambda s, *a, **kw: s,
        roi_stats=lambda s, *a, **kw: s,
    )
    s2ext = _module(
        "suite2p.extraction",
        extraction_wrapper=extraction_wrapper,
        preprocess=lambda F, **kw: F,
        oasis=lambda F, **kw: np.zeros_like(F),
    )
    s2cls = _module(
        "suite2p.classification",
        classify=lambda stat, classfile: np.zeros((len(stat), 2), dtype=np.float32),
        builtin_classfile=None,
    )
    torch_mod = _module(
        "torch",
        cuda=type("C", (), {"is_available": staticmethod(lambda: False)})(),
        device=lambda x: x,
        Tensor=type("Tensor", (), {}),
    )

    for name, mod in {
        "suite2p": suite2p_mod,
        "suite2p.io": s2io_mod,
        "suite2p.detection": s2det,
        "suite2p.extraction": s2ext,
        "suite2p.classification": s2cls,
        "torch": torch_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    run_suite2p_pipeline(
        [selected_a, selected_b], masks, TifPipelineParams(nchannels=1)
    )

    # Only the two selected files reached Suite2p; the other-FOV TIFF was dropped.
    surviving = {str(Path(p).resolve()) for p in captured["file_list"]}
    assert surviving == {str(selected_a.resolve()), str(selected_b.resolve())}
    # first_files was repaired: the first surviving entry per folder is True.
    first = captured["first_files"]
    assert len(first) == len(captured["file_list"])
    assert bool(first[0]) is True
    # Only one True per folder; both files live in the same folder here.
    assert int(np.count_nonzero(first)) == 1


def test_run_suite2p_pipeline_rejects_mismatched_mask_shape(
    monkeypatch, tmp_path: Path
):
    import sys
    import types

    from neuroseq.importer_tif import TifPipelineParams
    from neuroseq.importer_tif import run_suite2p_pipeline

    masks = np.zeros((8, 8), dtype=np.int32)

    def _module(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        return mod

    suite2p_mod = _module(
        "suite2p",
        default_db=lambda: {},
        default_settings=lambda: {
            "extraction": {"neuropil_coefficient": 0.7, "batch_size": 1},
        },
    )
    s2io_mod = _module(
        "suite2p.io",
        get_file_list=lambda db: ([], np.array([1])),
        init_dbs=lambda db: [
            {
                "reg_file": str(tmp_path / "r.bin"),
                "reg_file_chan2": str(tmp_path / "r2.bin"),
                "Ly": 16,
                "Lx": 16,
                "nframes": 1,
            }
        ],
        tiff_to_binary=lambda dbs, *a, **kw: dbs,
        BinaryFile=type(
            "BF",
            (),
            {
                "__init__": lambda self, **kw: setattr(
                    self, "data", np.zeros((1, 16, 16), dtype=np.float32)
                )
            },
        ),
    )
    s2det = _module(
        "suite2p.detection",
        assign_overlaps=lambda s, *a, **kw: s,
        roi_stats=lambda s, *a, **kw: s,
    )
    s2ext = _module(
        "suite2p.extraction",
        extraction_wrapper=lambda *a, **kw: (None, None, None, None),
        preprocess=lambda **kw: None,
        oasis=lambda **kw: None,
    )
    s2cls = _module(
        "suite2p.classification",
        classify=lambda **kw: None,
        builtin_classfile=None,
    )
    torch_mod = _module(
        "torch",
        cuda=type("C", (), {"is_available": staticmethod(lambda: False)})(),
        device=lambda x: x,
        Tensor=type("Tensor", (), {}),
    )

    for name, mod in {
        "suite2p": suite2p_mod,
        "suite2p.io": s2io_mod,
        "suite2p.detection": s2det,
        "suite2p.extraction": s2ext,
        "suite2p.classification": s2cls,
        "torch": torch_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    tif = tmp_path / "stack.tif"
    tif.write_bytes(b"")
    with pytest.raises(ValueError, match="does not match TIFF shape"):
        run_suite2p_pipeline([tif], masks, TifPipelineParams(nchannels=1))


# ---------- GPU device auto-selection ----------


def _fake_module(name: str, **attrs: Any):
    """Build a stand-in module. Uses ``setattr`` with a variable key so neither
    ruff (no constant-setattr) nor mypy (no attr-defined on ModuleType) complain.
    """
    import types

    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _fake_torch(*, cuda: bool, mps: bool | None):
    """Build a minimal fake ``torch`` module for select_torch_device tests.

    ``mps=None`` omits the ``backends`` attribute entirely (mimics older torch).
    """
    import types

    attrs: dict[str, Any] = {
        "cuda": type("C", (), {"is_available": staticmethod(lambda: cuda)})(),
        "device": lambda x: f"device:{x}",
    }
    if mps is not None:
        attrs["backends"] = types.SimpleNamespace(
            mps=type("M", (), {"is_available": staticmethod(lambda: mps)})()
        )
    return _fake_module("torch", **attrs)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, False, "GPU (CUDA)"),
        (True, True, "GPU (CUDA)"),  # CUDA preferred over MPS
        (False, True, "GPU (Apple MPS)"),
        (False, False, "CPU"),
        (False, None, "CPU"),  # torch without a backends/mps attribute
    ],
)
def test_select_torch_device_prefers_available_gpu(
    monkeypatch, cuda, mps, expected
) -> None:
    import sys

    from neuroseq.importer_tif import select_torch_device

    monkeypatch.delenv("NEUROSEQ_FORCE_CPU", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=cuda, mps=mps))
    device, label = select_torch_device()
    assert label == expected


def test_select_torch_device_force_cpu_env_overrides_gpu(monkeypatch) -> None:
    import sys

    from neuroseq.importer_tif import select_torch_device

    monkeypatch.setenv("NEUROSEQ_FORCE_CPU", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    device, label = select_torch_device()
    assert label == "CPU (forced)"
    assert device == "device:cpu"


def test_run_with_cpu_fallback_retries_on_cpu(monkeypatch) -> None:
    import sys
    import types

    from neuroseq.importer_tif import _run_with_cpu_fallback

    # Fake torch whose device objects carry a ``.type`` (as the real ones do).
    torch_mod = _fake_module(
        "torch", device=lambda name: types.SimpleNamespace(type=name)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    gpu = types.SimpleNamespace(type="cuda")
    calls: list[str] = []

    def fn(dev):
        calls.append(dev.type)
        if dev.type != "cpu":
            raise RuntimeError("op not implemented on this GPU")
        return "ok"

    result, device, label = _run_with_cpu_fallback(fn, gpu, "GPU (CUDA)")
    assert result == "ok"
    assert device.type == "cpu"
    assert label == "CPU (GPU fallback)"
    assert calls == ["cuda", "cpu"]  # tried GPU, then CPU


def test_run_with_cpu_fallback_reraises_on_cpu_failure(monkeypatch) -> None:
    import sys
    import types

    from neuroseq.importer_tif import _run_with_cpu_fallback

    # Hide torch so the CPU-reraise path must not import it (regression guard:
    # torch is absent in the conda-forge CI env).
    monkeypatch.setitem(sys.modules, "torch", None)

    cpu = types.SimpleNamespace(type="cpu")

    def fn(dev):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_with_cpu_fallback(fn, cpu, "CPU")
