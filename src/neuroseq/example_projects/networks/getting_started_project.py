from importlib import resources
from pathlib import Path

import brian2.numpy_ as np

from neuroseq.example_projects._registry import register_example_project
from neuroseq.example_projects.networks.default_project import default_project
from neuroseq.importers.importer_cellpose import CellPoseImporter
from neuroseq.importers.importer_cellpose import CellposeImportResult
from neuroseq.importers.importer_cellpose import generate_linear_stimulus_mapping
from neuroseq.project import Project


def _getting_started_cellpose_data(n_neurons: int) -> CellposeImportResult:
    """Load the bundled 2-photon FOV + Cellpose segmentation and map the
    first ``n_neurons`` detected cells to the model's neurons, so the
    Getting Started project has real spatial footprints to show off rather
    than opening with an empty Cell segmentation tab."""
    resource_root = resources.files("neuroseq.resources").joinpath("example_data")
    image_path = Path(str(resource_root.joinpath("2p_imaging_FOV.png")))
    segmentation_path = Path(str(resource_root.joinpath("2p_imaging_FOV_seg.npy")))

    importer = CellPoseImporter()
    importer.load_image_file(image_path)
    importer.load_segmentation_file(segmentation_path)
    importer.generate_overlay()

    assert importer.segmentation_data is not None
    unique_ids = np.unique(importer.segmentation_data)
    original_ids = unique_ids[unique_ids >= 0]
    full_mapping = generate_linear_stimulus_mapping(original_ids, offset=0)
    mapping = {stim_id: full_mapping[stim_id] for stim_id in range(n_neurons)}

    return importer.create_result(
        image_path,
        mapping,
        mapping_strategy="Linear",
        segmentation_path=segmentation_path,
    )


@register_example_project("Getting Started Project", section="Basics")
def getting_started_project() -> Project:
    project = default_project()
    project.cellpose_data = _getting_started_cellpose_data(
        project.b2_model.get_defined_neuron_count()
    )
    return project
