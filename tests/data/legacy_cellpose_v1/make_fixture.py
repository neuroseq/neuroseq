"""Generate the legacy (vertically flipped Cellpose storage) fixture.

Run once with NeuroSeq at commit 5f91023, before issue #601 removed the flip:
    uv run python tests/data/legacy_cellpose_v1/make_fixture.py
The outputs are committed; do not regenerate them with newer code.
"""

import json
import pathlib

import numpy as np
from PIL import Image

from neuroseq.export.exporter_syscon2 import generate_syscon2_sequence_text
from neuroseq.importers.importer_cellpose import AreasOfInterestConfig
from neuroseq.importers.importer_cellpose import CellPoseImporter
from neuroseq.importers.importer_cellpose import CircleAOI
from neuroseq.importers.importer_cellpose import SquareAOI
from neuroseq.importers.importer_cellpose import generate_aoi_stimulus_mapping
from neuroseq.project import Project
from neuroseq.save_load_options import SaveLoadOptions
from neuroseq.stimulation.stim import StimArray
from neuroseq.stimulation.stim import StimType
from neuroseq.visualization_support.cellpose_ids import compute_cell_centroids
from tests.test_exporter_syscon2 import _template_header

here = pathlib.Path(__file__).parent
height, width = 24, 40  # non-square on purpose
rows, cols = np.mgrid[:height, :width]

# Raw Cellpose masks: 0 background, cells 1..3, top-left origin.
masks = np.zeros((height, width), np.int32)
masks[(rows - 4) ** 2 + (cols - 6) ** 2 <= 9] = 1  # near the top-left
masks[(rows - 17) ** 2 + (cols - 30) ** 2 <= 16] = 2  # lower right
masks[2:7, 25:33] = 3  # top right, rectangular
np.save(here / "masks.npy", masks)
image = (rows * 4 + cols * 2).astype(np.uint8)
image[masks > 0] = 220
Image.fromarray(image).convert("RGB").save(here / "image.png")

importer = CellPoseImporter()
importer.load_image_file(here / "image.png")
importer.load_segmentation_file(here / "masks.npy")
importer.generate_overlay()
aoi = AreasOfInterestConfig(
    square=SquareAOI(True, "preferential", 0.0, 0.0, 12.0),  # top-left 12x12
    circle=CircleAOI(True, "exclusion", 30.0, 17.0, 12.0),  # around cell 2
)
centers = compute_cell_centroids(importer.segmentation_data)
ids = np.unique(importer.segmentation_data)
ids = ids[ids >= 0]
aoi_mapping = generate_aoi_stimulus_mapping(ids, centers, aoi, height)
result = importer.create_result(
    here / "image.png",
    aoi_mapping,
    mapping_strategy="AOI",
    aoi_config=aoi,
    segmentation_path=here / "masks.npy",
)

values = np.zeros((60, 3), dtype=np.float32)
values[10:12, 0] = 1.0
values[25:27, 1] = 1.0
values[40:42, 2] = 1.0
stim = StimArray(
    name="stim", stim_type=StimType.Excitation, values=values, units="1", dt=520.8125e-6
)
project = Project()
project.stim_arrays = {"stim": stim}
project.cellpose_data = result
project.save(here / "legacy.nseq", SaveLoadOptions(True, True, False, True))

seq_text, _ = generate_syscon2_sequence_text(
    stim,
    result,
    template_text=_template_header(),
    spiralling=False,
    stimulation_spots_per_roi=3,
)
(here / "legacy_export.seq").write_text(seq_text)
(here / "aoi_mapping.json").write_text(
    json.dumps({str(k): int(v) for k, v in aoi_mapping.items()})
)
print("aoi mapping", aoi_mapping)
print(
    "stored segmentation flipped:",
    np.array_equal(result.segmentation_array, np.flipud(masks - 1)),
)
