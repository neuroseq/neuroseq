"""Generate legacy_aoi_dialog.nseq: AOI shapes saved through the real dialog.

Run with NeuroSeq from before issue #601 (e.g. an origin/main worktree at
5f91023's parent), from this directory, offscreen:
    cd tests/data/legacy_cellpose_v1
    QT_QPA_PLATFORM=offscreen PYTHONPATH=<old checkout>/src python make_aoi_fixture.py
Files are loaded by relative name so the project stores "image.png", not a
machine path. The output is committed; do not regenerate it with newer code.
"""

from qtpy import QtWidgets

from neuroseq.gui import Gui
from neuroseq.importers.cellpose_import_dialog import CUSTOM_AOI_STRATEGY
from neuroseq.importers.cellpose_import_dialog import CellposeImportDialog
from neuroseq.importers.importer_cellpose import AOIRole
from neuroseq.importers.importer_cellpose import AreasOfInterestConfig
from neuroseq.importers.importer_cellpose import CircleAOI
from neuroseq.importers.importer_cellpose import SquareAOI
from neuroseq.save_load_options import SaveLoadOptions

app = QtWidgets.QApplication([])
# Modal confirmations would block an unattended run.
QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: None)
gui = Gui()
dialog = CellposeImportDialog(gui)
dialog._load_image_file("image.png")
dialog._load_segmentation_file("masks.npy")
for button in dialog.mapping_buttons.buttons():
    if button.text() == CUSTOM_AOI_STRATEGY:
        button.setChecked(True)
dialog._set_aoi_controls(
    AreasOfInterestConfig(
        square=SquareAOI(True, AOIRole.PREFERENTIAL, 0.0, 0.0, 12.0),
        circle=CircleAOI(True, AOIRole.EXCLUSION, 30.0, 17.0, 12.0),
        coordinate_mode="pixels",
    )
)
dialog._regenerate_aoi_mapping()
dialog.import_button.click()
gui._apply_cellpose_import_result(dialog.get_imported_data())
gui.project.save("legacy_aoi_dialog.nseq", SaveLoadOptions(True, True, False, True))
result = gui.project.cellpose_data
print(result.mapping_strategy, result.stimulus_to_cellpose_mapping, result.aoi_config)
gui._close_confirmed = True
gui.close()
