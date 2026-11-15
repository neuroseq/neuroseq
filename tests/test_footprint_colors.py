from pathlib import Path

import brian2.numpy_ as np

from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.utils import neuron_indices_to_colors
from neuroseq.visualization_support.footprint_colors import compose_footprint_overlay
from neuroseq.visualization_support.footprint_colors import footprint_legend
from neuroseq.visualization_support.footprint_colors import footprint_neuron_ids


def _result(**kwargs) -> CellposeImportResult:
    return CellposeImportResult(
        image_path=Path("cellpose.png"),
        segmentation_array=np.zeros((1, 1), dtype=np.int32),
        overlay_image=np.zeros((1, 1, 3), dtype=np.uint8),
        trace_data=None,
        spike_data=None,
        **kwargs,
    )


def test_footprint_neuron_ids_without_data_are_the_cellpose_ids() -> None:
    assert footprint_neuron_ids(np.array([4, 9]), None).tolist() == [4, 9]
    assert footprint_neuron_ids(np.array([], dtype=int), _result()).size == 0


def test_footprint_neuron_ids_follow_linear_offset_and_gaps() -> None:
    result = _result(
        original_cellpose_ids=np.array([0, 3, 7]),
        mapping_strategy="Linear",
        linear_offset=3,
    )
    # Linear offset 3 starts the neuron ids at Cellpose id 3, wrapping 0.
    assert footprint_neuron_ids(np.array([0, 3, 7]), result).tolist() == [2, 0, 1]


def test_footprint_neuron_ids_fall_back_when_mapping_collides() -> None:
    result = _result(stimulus_to_cellpose_mapping={0: 1, 1: 1})
    assert footprint_neuron_ids(np.array([0, 1]), result).tolist() == [0, 1]


def test_footprint_legend_skips_background_and_uses_neuron_colours() -> None:
    segmentation = np.array([[-1, 2], [5, 5]])
    result = _result(stimulus_to_cellpose_mapping={0: 5, 1: 2})
    legend = dict(footprint_legend(segmentation, result))
    assert sorted(legend) == [2, 5]
    expected = neuron_indices_to_colors(np.array([1, 0]))
    assert legend[2] == tuple(int(c * 255) for c in expected[0][:3])
    assert legend[5] == tuple(int(c * 255) for c in expected[1][:3])
    assert footprint_legend(np.full((2, 2), -1)) == []


def test_compose_footprint_overlay_blends_only_inside_known_cells() -> None:
    background = np.full((1, 4, 3), 100, dtype=np.uint8)
    segmentation = np.array([[-1, 0, 1, 2]])
    overlay = compose_footprint_overlay(
        background, segmentation, [(0, (200, 0, 0)), (1, (0, 50, 250))], 0.5
    )
    assert overlay.dtype == np.uint8
    assert overlay[0, 0].tolist() == [100, 100, 100]  # background
    assert overlay[0, 1].tolist() == [150, 50, 50]
    assert overlay[0, 2].tolist() == [50, 75, 175]
    assert overlay[0, 3].tolist() == [100, 100, 100]  # cell without a colour
    assert np.array_equal(
        compose_footprint_overlay(background, segmentation, [], 0.5), background
    )
