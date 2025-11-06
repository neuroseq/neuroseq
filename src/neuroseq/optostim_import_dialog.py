from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets


# This file expects to import a Mat file, for now just one of the Spikes. This comes from the OptoStim repository
# Brian2 spike monitor. That's why most values are close to 0.
# It will also import the Neuron Traces, once the statemonitor is added to that code in the Optostim repository.

class OptoStimReadingsImportDialog(QtWidgets.QDialog):
    """Dialog for importing optostim readings data from .mat files"""
    
    # Signal emitted when data is successfully imported
    # Emits dict with traces data compatible with Neuroseq format
    # Nothing for now as we need to sort this out on OptoStim repo first and produce TWO outputs and consider what Ui to have to import them
    # those two being for spikes and traces respectively. Most likely we will use two modals, once for each.
    data_imported = QtCore.Signal(dict)
    
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Import OptoStim Readings Data")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Not implemented message
        message = QtWidgets.QLabel("Not implemented yet")
        message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)
        
        # OK button
        button_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)