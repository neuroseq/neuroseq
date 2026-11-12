"""The "BrainScaleS-2" tab of the Stimulation setup pane.

Delivers the current stim array to BrainScaleS-2 neuromorphic hardware through
the container helper (see :mod:`neuroseq.device_control.bss2_connection`) and
adds the readback - spikes of all driven neurons and the membrane voltage of
one neuron - to the project as an undoable experiment.

The EBRAINS token typed here lives only in this widget for the session; it is
passed to the helper through its environment and never stored or logged.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.compat import getexistingdirectory
from qtpy.compat import getopenfilename

from neuroseq import __version__ as NEUROSEQ_VERSION
from neuroseq.background_jobs import BackgroundIOWorker
from neuroseq.background_jobs import BackgroundJob
from neuroseq.data import SpikeData
from neuroseq.device_control import bss2_protocol as bp
from neuroseq.device_control.bss2_connection import STATE_CONNECTED
from neuroseq.device_control.bss2_connection import STATE_CONNECTING
from neuroseq.device_control.bss2_connection import STATE_DISCONNECTED
from neuroseq.device_control.bss2_connection import STATE_STARTING
from neuroseq.device_control.bss2_stimulation_host import Bss2StimulationHost
from neuroseq.qt_helpers import WheelToScrollFilter
from neuroseq.stimulation.stim import StimArray

if TYPE_CHECKING:
    from neuroseq.gui_settings import GuiSettings
    from neuroseq.project import Project
    from neuroseq.workspace_devices import Bss2Session

logger = logging.getLogger(__name__)

_INPUT_MODES = [
    ("Constant current (input array)", bp.INPUT_CONSTANT_CURRENT),
    ("Spike input", bp.INPUT_SPIKES),
]
_SPIKE_SOURCES = [
    ("From input array (threshold)", bp.SPIKE_SOURCE_STIM),
    ("From project spike data", bp.SPIKE_SOURCE_EXPERIMENT),
]
_STARTING_STATES = (STATE_STARTING, STATE_CONNECTING)
_SETUP_DOWNLOAD_TIMEOUT_S = 10.0
_MAX_LOG_LINES = 5000


def fetch_text(url: str) -> str:
    """Download a small text file (runs on a background thread)."""
    with urllib.request.urlopen(url, timeout=_SETUP_DOWNLOAD_TIMEOUT_S) as reply:
        return reply.read(1_000_000).decode("utf-8", errors="replace")


class Bss2Panel(QtWidgets.QWidget):
    #: Emitted with the experiment name after a readback was added to the project.
    experiment_added = QtCore.Signal(str)

    def __init__(
        self,
        session: Bss2Session,
        parent: QtWidgets.QWidget | None = None,
        *,
        settings: GuiSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._gui_settings = settings
        self._project: Project | None = None
        self._current_stim_provider: Callable[[], StimArray | None] | None = None
        #: Downloads the quiggeldy setup list; replaced in tests.
        self.setup_fetcher: Callable[[str], str] = fetch_text
        self._setup_job = BackgroundJob(self)
        self._setups: list[bp.QuiggeldySetup] = []
        #: Connect was clicked without a host: connect once the list arrives.
        self._connect_after_setups = False
        #: (request id, stim name, speed-up) of this tab's run in flight.
        self._run: tuple[str, str, float] | None = None
        self._loading = False
        self._closed = False
        self._build_ui()
        self._load_settings()
        self._wire_session()
        self._update_ui()
        #: Runs dynamic input routine blocks with this tab's settings.
        self.stimulation_host = Bss2StimulationHost(
            session,
            self.current_settings,
            parent=self,
            connect_hardware=self.connect_hardware,
        )
        self.stimulation_host.progress.connect(self._log)
        self.stimulation_host.experiment_added.connect(self._on_host_experiment_added)

    # ---- workspace API --------------------------------------------------------

    def set_project(self, project: Project | None) -> None:
        """Show ``project``'s spike datasets (also called after data changes)."""
        self._project = project
        self._refresh_experiments()

    def set_current_stim_provider(
        self, provider: Callable[[], StimArray | None]
    ) -> None:
        """Tell the panel how to look up the stim array currently shown in the editor."""
        self._current_stim_provider = provider

    @property
    def run_in_progress(self) -> bool:
        """Whether a run of this workspace (this tab's or a dynamic input routine's,
        even one the routine stopped waiting for) is still executing."""
        return self._run is not None or self._session.owns_run

    def shutdown(self) -> None:
        """Detach from the shared helper (it keeps running for other tabs)."""
        if self._closed:
            return
        self._closed = True
        self._setup_job.stop(timeout_ms=int(_SETUP_DOWNLOAD_TIMEOUT_S * 1000) + 1000)
        self._session.shutdown()
        self.token_edit.clear()

    # ---- UI ------------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_delivery_group())
        layout.addWidget(self._build_debug_group())
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        wheel_filter = WheelToScrollFilter(scroll)
        value_widgets: list[QtWidgets.QWidget] = [
            *content.findChildren(QtWidgets.QAbstractSpinBox),
            *content.findChildren(QtWidgets.QComboBox),
        ]
        for widget in value_widgets:
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(wheel_filter)

    def _spin(self, low: int, high: int, value: int = 0) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.valueChanged.connect(self._save_settings)
        return spin

    def _double_spin(
        self, low: float, high: float, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self._save_settings)
        return spin

    def _build_connection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Container and connection")
        form = QtWidgets.QFormLayout(group)

        self.image_edit = QtWidgets.QLineEdit()
        self.image_edit.setPlaceholderText("Singularity image (.sif) or sandbox")
        self.image_edit.editingFinished.connect(self._save_settings)
        self.btn_browse_image = QtWidgets.QPushButton("Browse…")
        browse_menu = QtWidgets.QMenu(self.btn_browse_image)
        browse_menu.addAction("Image file (.sif)…", self._on_browse_image)
        browse_menu.addAction("Sandbox folder…", self._on_browse_sandbox)
        self.btn_browse_image.setMenu(browse_menu)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.image_edit, 1)
        row.addWidget(self.btn_browse_image)
        form.addRow("Container image", row)

        self.app_edit = QtWidgets.QLineEdit()
        self.app_edit.setPlaceholderText("none")
        self.app_edit.setToolTip("Optional singularity --app name (e.g. dls).")
        self.app_edit.editingFinished.connect(self._save_settings)
        form.addRow("Container app", self.app_edit)

        self.setup_combo = QtWidgets.QComboBox()
        self.setup_combo.setToolTip(
            "Quiggeldy instances on quiggeldy.bioai.eu (stable 11717, "
            "experimental 11750, special 11780; they need an allow-listed IP) "
            "and, after Refresh, those from the EBRAINS list."
        )
        self.setup_combo.activated.connect(self._on_setup_chosen)
        self.btn_refresh_setups = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_setups.setToolTip(f"Download {bp.QUIGGELDY_SETUPS_URL}")
        self.btn_refresh_setups.clicked.connect(self.refresh_setups)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.setup_combo, 1)
        row.addWidget(self.btn_refresh_setups)
        form.addRow("Setup", row)

        self.host_edit = QtWidgets.QLineEdit()
        self.host_edit.editingFinished.connect(self._save_settings)
        self.port_spin = self._spin(1, 65535, 11717)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.host_edit, 1)
        row.addWidget(QtWidgets.QLabel("Port"))
        row.addWidget(self.port_spin)
        form.addRow("Quiggeldy host", row)
        self.identifier_edit = QtWidgets.QLineEdit()
        self.identifier_edit.setPlaceholderText("expected chip identifier (optional)")
        self.identifier_edit.editingFinished.connect(self._save_settings)
        form.addRow("Identifier", self.identifier_edit)
        form.addRow("User", QtWidgets.QLabel(bp.current_username()))

        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText(
            "optional: EBRAINS token (kept in memory only)"
            if not os.environ.get("QUIGGELDY_TOKEN")
            else "using QUIGGELDY_TOKEN from the environment"
        )
        self.token_edit.setToolTip(
            "Optional. Not needed for the quiggeldy instances reached from an "
            "allow-listed IP; access through EBRAINS may need the output of "
            "clb_oauth.get_token(). Passed to the container through its "
            "environment only; never saved."
        )
        form.addRow("Token (optional)", self.token_edit)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self.connect_hardware)
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.clicked.connect(self._session.disconnect_helper)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_disconnect)
        row.addStretch(1)
        form.addRow(row)
        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Status", self.status_label)
        self.identifier_label = QtWidgets.QLabel("–")
        self.identifier_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Connected chip", self.identifier_label)
        return group

    def _build_input_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Input and readback")
        form = QtWidgets.QFormLayout(group)
        self.mode_combo = QtWidgets.QComboBox()
        for label, value in _INPUT_MODES:
            self.mode_combo.addItem(label, value)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Input mode", self.mode_combo)

        self.neurons_spin = self._spin(0, 4096)
        self.neurons_spin.setSpecialValueText("auto (input channels)")
        self.neurons_spin.setToolTip(
            "Input channel i drives hardware neuron i. Extra channels are dropped, "
            "missing ones receive no input."
        )
        form.addRow("Neurons", self.neurons_spin)
        self.max_neurons_spin = self._spin(1, 4096, bp.MAX_HARDWARE_NEURONS)
        form.addRow("Max neurons", self.max_neurons_spin)

        self.current_box = QtWidgets.QWidget()
        current_form = QtWidgets.QFormLayout(self.current_box)
        current_form.setContentsMargins(0, 0, 0, 0)
        self.i_offset_spin = self._spin(0, bp.I_OFFSET_LIMIT, 500)
        self.i_offset_spin.setToolTip(
            "constant_current_i_offset (hardware units) at the maximum input value."
        )
        current_form.addRow("Max i_offset", self.i_offset_spin)
        self.value_at_max_spin = self._double_spin(0.0, 1e9, 4)
        self.value_at_max_spin.setSpecialValueText("array maximum")
        self.value_at_max_spin.setToolTip("Input value mapped to the maximum i_offset.")
        current_form.addRow("Input value at max", self.value_at_max_spin)
        form.addRow(self.current_box)

        self.spike_box = QtWidgets.QWidget()
        spike_form = QtWidgets.QFormLayout(self.spike_box)
        spike_form.setContentsMargins(0, 0, 0, 0)
        self.spike_source_combo = QtWidgets.QComboBox()
        for label, value in _SPIKE_SOURCES:
            self.spike_source_combo.addItem(label, value)
        self.spike_source_combo.currentIndexChanged.connect(self._on_mode_changed)
        spike_form.addRow("Spike source", self.spike_source_combo)
        self.threshold_spin = self._double_spin(-1e9, 1e9, 4)
        spike_form.addRow("Threshold", self.threshold_spin)
        self.event_ms_spin = self._double_spin(0.01, 1e6, 2)
        self.event_ms_spin.setSuffix(" ms")
        self.event_ms_spin.setToolTip(
            "Biological spacing of the grid on which the input array is sampled."
        )
        spike_form.addRow("Event interval", self.event_ms_spin)
        self.experiment_combo = QtWidgets.QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self._refresh_datasets)
        spike_form.addRow("Data group", self.experiment_combo)
        self.dataset_combo = QtWidgets.QComboBox()
        spike_form.addRow("Spike dataset", self.dataset_combo)
        self.weight_spin = self._spin(0, bp.WEIGHT_LIMIT, bp.WEIGHT_LIMIT)
        spike_form.addRow("Weight", self.weight_spin)
        form.addRow(self.spike_box)

        self.madc_spin = self._spin(0, 4095)
        self.madc_spin.setToolTip(
            "The MADC records the membrane voltage of one neuron per run."
        )
        form.addRow("Record v of neuron", self.madc_spin)
        self.speedup_spin = self._double_spin(1.0, 1e6, 1)
        self.speedup_spin.setSuffix(" x")
        self.speedup_spin.setToolTip(
            "Biological time / hardware time. BSS-2 runs ~1000x faster than "
            "biology: with 1000, 1 s of input is a 1 ms hardware run."
        )
        form.addRow("Speed-up", self.speedup_spin)
        return group

    def _build_delivery_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Delivery")
        layout = QtWidgets.QVBoxLayout(group)
        row = QtWidgets.QHBoxLayout()
        self.btn_deliver = QtWidgets.QPushButton("Deliver current input array")
        self.btn_deliver.clicked.connect(self.deliver)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setToolTip(
            "Terminates the helper; a run already sent to the chip cannot be "
            "recalled and its results are discarded."
        )
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_deliver)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def _build_debug_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Debug log")
        layout = QtWidgets.QVBoxLayout(group)
        row = QtWidgets.QHBoxLayout()
        self.chk_protocol_trace = QtWidgets.QCheckBox("Show protocol traffic")
        self.chk_protocol_trace.setChecked(True)
        self.chk_protocol_trace.setToolTip(
            "Also log the launch command, every message exchanged with the "
            "container helper and the helper process lifecycle. The EBRAINS "
            "token is never shown."
        )
        self.btn_copy_log = QtWidgets.QPushButton("Copy log")
        self.btn_copy_log.setToolTip(
            "Copy the log, with NeuroSeq/host/settings details, to the clipboard."
        )
        self.btn_copy_log.clicked.connect(self.copy_log)
        self.btn_clear_log = QtWidgets.QPushButton("Clear")
        self.btn_clear_log.clicked.connect(self.clear_log)
        row.addWidget(self.chk_protocol_trace)
        row.addStretch(1)
        row.addWidget(self.btn_copy_log)
        row.addWidget(self.btn_clear_log)
        layout.addLayout(row)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.log_view.setMaximumBlockCount(_MAX_LOG_LINES)
        self.log_view.setMinimumHeight(160)
        layout.addWidget(self.log_view)
        return group

    # ---- settings ------------------------------------------------------------------

    def current_settings(self) -> bp.Bss2Settings:
        return bp.Bss2Settings(
            container_image=self.image_edit.text().strip(),
            container_app=self.app_edit.text().strip(),
            setup_host=self.host_edit.text().strip(),
            setup_port=self.port_spin.value(),
            setup_identifier=self.identifier_edit.text().strip(),
            input_mode=self.mode_combo.currentData(),
            n_neurons=self.neurons_spin.value(),
            max_neurons=self.max_neurons_spin.value(),
            i_offset_max=self.i_offset_spin.value(),
            value_at_max=self.value_at_max_spin.value(),
            spike_source=self.spike_source_combo.currentData(),
            spike_threshold=self.threshold_spin.value(),
            spike_event_ms=self.event_ms_spin.value(),
            spike_weight=self.weight_spin.value(),
            madc_neuron=self.madc_spin.value(),
            speedup=self.speedup_spin.value(),
        )

    def _load_settings(self) -> None:
        stored = self._gui_settings.bss2_settings if self._gui_settings else {}
        settings = bp.Bss2Settings.from_dict(stored)
        self._loading = True
        try:
            self.image_edit.setText(settings.container_image)
            self.app_edit.setText(settings.container_app)
            self.host_edit.setText(settings.setup_host)
            self.port_spin.setValue(settings.setup_port)
            self.identifier_edit.setText(settings.setup_identifier)
            self.mode_combo.setCurrentIndex(
                max(0, self.mode_combo.findData(settings.input_mode))
            )
            self.neurons_spin.setValue(settings.n_neurons)
            self.max_neurons_spin.setValue(settings.max_neurons)
            self.i_offset_spin.setValue(settings.i_offset_max)
            self.value_at_max_spin.setValue(settings.value_at_max)
            self.spike_source_combo.setCurrentIndex(
                max(0, self.spike_source_combo.findData(settings.spike_source))
            )
            self.threshold_spin.setValue(settings.spike_threshold)
            self.event_ms_spin.setValue(settings.spike_event_ms)
            self.weight_spin.setValue(settings.spike_weight)
            self.madc_spin.setValue(settings.madc_neuron)
            self.speedup_spin.setValue(settings.speedup)
            # Built-in KIP instances, plus the saved setup if it is another one.
            setups = list(bp.KIP_SETUPS)
            selected = -1
            if settings.setup_host:
                address = (settings.setup_host, settings.setup_port)
                matches = [
                    i for i, s in enumerate(setups) if (s.host, s.port) == address
                ]
                if matches:
                    selected = matches[0]
                else:
                    saved = bp.QuiggeldySetup(*address, settings.setup_identifier)
                    setups.insert(0, saved)
                    selected = 0
            self._set_setups(setups)
            if selected < 0:
                # Nothing saved yet: preselect the instance known to work, so
                # the host is never empty.
                selected = bp.DEFAULT_KIP_SETUP
                default = setups[selected]
                self.host_edit.setText(default.host)
                self.port_spin.setValue(default.port)
                self.identifier_edit.setText(default.identifier)
            self.setup_combo.setCurrentIndex(selected)
            self._on_mode_changed()
        finally:
            self._loading = False

    def _save_settings(self, *_args: object) -> None:
        if self._loading or self._gui_settings is None:
            return
        values = self.current_settings().to_dict()
        if values != self._gui_settings.bss2_settings:
            self._gui_settings.bss2_settings = values

    def _on_browse_image(self) -> None:
        path, _ = getopenfilename(
            self,
            "Choose the BrainScaleS-2 container image",
            self.image_edit.text(),
            "Singularity images (*.sif *.simg *.img);;All files (*)",
        )
        if path:
            self.image_edit.setText(path)
            self._save_settings()

    def _on_browse_sandbox(self) -> None:
        """An unpacked image (``singularity build --sandbox``) is a folder."""
        path = getexistingdirectory(
            self,
            "Choose the BrainScaleS-2 container sandbox folder",
            self.image_edit.text(),
        )
        if path:
            self.image_edit.setText(path)
            self._save_settings()

    # ---- quiggeldy setups ----------------------------------------------------------

    def refresh_setups(self) -> None:
        """Download the setup list off the GUI thread."""
        if self._setup_job.busy:
            return
        fetch = self.setup_fetcher
        self.btn_refresh_setups.setEnabled(False)
        self._log("Downloading the quiggeldy setup list…")
        self._setup_job.start(
            BackgroundIOWorker(lambda _report: fetch(bp.QUIGGELDY_SETUPS_URL)),
            on_progress=lambda _value: None,
            on_finished=self._on_setups_downloaded,
            on_failed=self._on_setups_failed,
        )

    def _on_setups_downloaded(self, text: object) -> None:
        self.btn_refresh_setups.setEnabled(True)
        try:
            setups = bp.parse_setups_csv(str(text))
        except ValueError as exc:
            self._on_setups_failed(str(exc))
            return
        if not setups:
            self._on_setups_failed("The setup list is empty.")
            return
        # Keep the current instance if it is still listed (Refresh must not
        # switch away from a working setup); otherwise use the collaborators'
        # default pick (hash of the user name) among the downloaded setups.
        current = (self.host_edit.text().strip(), self.port_spin.value())
        default_index = bp.default_setup_index(bp.current_username(), len(setups))
        setups = bp.combined_setups(setups)
        self._set_setups(setups)
        listed = [i for i, s in enumerate(setups) if (s.host, s.port) == current]
        index = listed[0] if listed else default_index
        self.setup_combo.setCurrentIndex(index)
        self._on_setup_chosen(index)
        self._log(f"{len(setups)} setup(s) available; selected: {setups[index].label}")
        if self._connect_after_setups:
            self._connect_after_setups = False
            self.connect_hardware()

    def _on_setups_failed(self, message: str) -> None:
        self.btn_refresh_setups.setEnabled(True)
        self._log(f"Could not load the setup list: {message}")
        if self._connect_after_setups:
            self._connect_after_setups = False
            QtWidgets.QMessageBox.warning(
                self,
                "BrainScaleS-2",
                f"Could not load the quiggeldy setup list ({message}).\n\n"
                "Enter the quiggeldy host and port by hand, then connect again.",
            )

    def _set_setups(self, setups: list[bp.QuiggeldySetup]) -> None:
        self._setups = list(setups)
        self.setup_combo.clear()
        for setup in setups:
            self.setup_combo.addItem(setup.label)

    def _on_setup_chosen(self, index: int) -> None:
        if not 0 <= index < len(self._setups):
            return
        setup = self._setups[index]
        self.host_edit.setText(setup.host)
        self.port_spin.setValue(setup.port)
        self.identifier_edit.setText(setup.identifier)
        self._save_settings()

    # ---- connection ----------------------------------------------------------------

    def _wire_session(self) -> None:
        self._session.status_changed.connect(self._on_status)
        self._session.connected.connect(self._on_connected)
        self._session.disconnected.connect(self._on_disconnected)
        self._session.log_message.connect(self._log)
        self._session.debug_message.connect(self._debug)
        self._session.run_progress.connect(self._on_run_progress)
        self._session.run_finished.connect(self._on_run_finished)
        self._session.run_failed.connect(self._on_run_failed)

    def connect_hardware(self) -> None:
        settings = self.current_settings()
        if not settings.container_image:
            QtWidgets.QMessageBox.warning(
                self, "BrainScaleS-2", "Choose the container image first."
            )
            return
        if not os.path.exists(settings.container_image):
            message = (
                f"The container image {settings.container_image} does not exist "
                "on this computer.\n\nEnter the path of the EBRAINS image in "
                "'Container image', or use Browse → 'Image file (.sif)…' or "
                "'Sandbox folder…'. To get the image, see the BrainScaleS-2 "
                "user guide."
            )
            self._log(message.replace("\n\n", " "))
            QtWidgets.QMessageBox.warning(self, "BrainScaleS-2", message)
            return
        if settings.container_app:
            apps = _container_apps(settings.container_image)
            if apps is not None and settings.container_app not in apps:
                self._log(
                    f"The image defines no app '{settings.container_app}' "
                    f"(apps: {', '.join(sorted(apps)) or 'none'}); "
                    "launching without --app."
                )
                self.app_edit.setText("")
                settings = self.current_settings()
        if not settings.setup_host:
            # hxcomm cannot resolve an empty host ("DNS lookup … failed").
            self._log("No quiggeldy host set; loading the setup list first.")
            self._connect_after_setups = True
            self.refresh_setups()
            return
        token = self.token_edit.text().strip() or os.environ.get("QUIGGELDY_TOKEN", "")
        if not token:
            # Allow-listed IPs connect without a token (user name only).
            self._log("Connecting without an EBRAINS token.")
        config = bp.LaunchConfig(
            container_image=settings.container_image,
            container_app=settings.container_app,
            host=settings.setup_host,
            port=settings.setup_port,
            user=bp.current_username(),
            cache_dir=_activation_cache_dir(),
        )
        self._save_settings()
        self._session.connect_helper(config, token, settings.setup_identifier)
        self._update_ui()

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)
        self._log(text)
        self._update_ui()

    def _on_connected(self, identifier: str) -> None:
        self.identifier_label.setText(identifier or "–")
        self._update_ui()

    def _on_disconnected(self, reason: str) -> None:
        self.identifier_label.setText("–")
        lines = reason.splitlines() or ["Disconnected"]
        self.status_label.setText(lines[0])
        for line in lines[1:]:  # helper stderr tail; the first line was logged
            self._log(line)
        self._update_ui()

    def _update_ui(self) -> None:
        state = self._session.state
        disconnected = state == STATE_DISCONNECTED
        self.btn_connect.setEnabled(disconnected)
        self.btn_disconnect.setEnabled(not disconnected and not self._session.busy)
        for widget in (self.image_edit, self.app_edit, self.btn_browse_image):
            widget.setEnabled(disconnected)
        self.btn_deliver.setEnabled(state == STATE_CONNECTED and not self._session.busy)
        # Stop also aborts a slow container start or connection attempt.
        self.btn_stop.setEnabled(self.run_in_progress or state in _STARTING_STATES)

    def _on_mode_changed(self, *_args: object) -> None:
        spikes = self.mode_combo.currentData() == bp.INPUT_SPIKES
        from_data = self.spike_source_combo.currentData() == bp.SPIKE_SOURCE_EXPERIMENT
        self.current_box.setVisible(not spikes)
        self.spike_box.setVisible(spikes)
        stim_widgets: tuple[QtWidgets.QWidget, ...] = (
            self.threshold_spin,
            self.event_ms_spin,
        )
        data_widgets: tuple[QtWidgets.QWidget, ...] = (
            self.experiment_combo,
            self.dataset_combo,
        )
        for widget in stim_widgets:
            widget.setEnabled(not from_data)
        for widget in data_widgets:
            widget.setEnabled(from_data)
        self._save_settings()

    # ---- spike data selection ------------------------------------------------------

    def _refresh_experiments(self) -> None:
        current = self.experiment_combo.currentText()
        self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()
        if self._project is not None:
            for name, experiment in self._project.experiments.items():
                if any(isinstance(d, SpikeData) for d in experiment.data):
                    self.experiment_combo.addItem(name)
        index = self.experiment_combo.findText(current)
        self.experiment_combo.setCurrentIndex(max(0, index))
        self.experiment_combo.blockSignals(False)
        self._refresh_datasets()

    def _refresh_datasets(self, *_args: object) -> None:
        self.dataset_combo.clear()
        experiment = self._selected_experiment()
        if experiment is None:
            return
        for data in experiment.data:
            if isinstance(data, SpikeData):
                self.dataset_combo.addItem(data.name)

    def _selected_experiment(self):
        if self._project is None:
            return None
        return self._project.experiments.get(self.experiment_combo.currentText())

    # ---- delivery ------------------------------------------------------------------

    def _plan(self, settings: bp.Bss2Settings) -> tuple[bp.Plan, str]:
        """Build the delivery plan and the name used for the readback."""
        if (
            settings.input_mode == bp.INPUT_SPIKES
            and settings.spike_source == bp.SPIKE_SOURCE_EXPERIMENT
        ):
            experiment = self._selected_experiment()
            dataset = next(
                (
                    d
                    for d in (experiment.data if experiment else [])
                    if isinstance(d, SpikeData)
                    and d.name == self.dataset_combo.currentText()
                ),
                None,
            )
            if experiment is None or dataset is None:
                raise ValueError("Choose a data group and spike dataset first.")
            plan = bp.plan_spikes_from_data(dataset, experiment.duration, settings)
            return plan, f"{experiment.name} {dataset.name}"
        stim = self._current_stim_provider() if self._current_stim_provider else None
        if stim is None:
            raise ValueError("There is no current input array to deliver.")
        if settings.input_mode == bp.INPUT_SPIKES:
            return bp.plan_spikes_from_stim(stim.values, stim.dt, settings), stim.name
        return bp.plan_constant_current(stim.values, stim.dt, settings), stim.name

    def deliver(self) -> None:
        if self._run is not None or self._session.state != STATE_CONNECTED:
            return
        settings = self.current_settings()
        try:
            plan, name = self._plan(settings)
            arguments = bp.build_run_arguments(plan, settings)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "BrainScaleS-2 delivery", str(exc))
            return
        summary = bp.plan_summary(plan, settings)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Deliver to BrainScaleS-2",
            f"Deliver '{name}' to {self._session.identifier or 'BrainScaleS-2'}?"
            f"\n\n{summary}",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        request_id = uuid4().hex
        if not self._session.run(request_id, arguments):
            QtWidgets.QMessageBox.warning(
                self,
                "BrainScaleS-2 delivery",
                "The hardware is busy or no longer connected.",
            )
            return
        self._run = (request_id, name, settings.speedup)
        self._log(f"Delivering '{name}': {summary.splitlines()[1]}")
        self._update_ui()

    def stop(self) -> None:
        if self.run_in_progress or self._session.state in _STARTING_STATES:
            self._session.stop()
        self._update_ui()

    def _on_run_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_host_experiment_added(self, name: str) -> None:
        """A routine's readback: show it like this tab's own deliveries."""
        self._refresh_experiments()
        self.experiment_added.emit(name)

    def _on_run_finished(self, request_id: str, result: object) -> None:
        if self._run is None or request_id != self._run[0]:
            self._update_ui()  # a routine's run ended: Stop no longer applies
            return
        _, name, speedup = self._run
        self._run = None
        self._update_ui()
        try:
            assert isinstance(result, dict)
            if self._project is None:
                raise ValueError("No project is open.")
            experiment, readback = bp.add_readback_experiment(
                self._project, result, speedup, name
            )
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            logger.exception("Could not add the BrainScaleS-2 readback")
            QtWidgets.QMessageBox.warning(
                self,
                "BrainScaleS-2 readback",
                f"The run finished, but its data could not be added: {exc}",
            )
            return
        spikes = readback.spikes.spike_indices.size
        self._log(f"Added '{experiment.name}' ({spikes} spike(s)).")
        self._refresh_experiments()
        self.experiment_added.emit(experiment.name)

    def _on_run_failed(self, request_id: str, message: str) -> None:
        if self._run is None or request_id != self._run[0]:
            self._update_ui()
            return
        self._run = None
        self._update_ui()
        self._log(f"Run failed: {message}")
        QtWidgets.QMessageBox.warning(self, "BrainScaleS-2 run failed", message)

    def _log(self, text: str) -> None:
        if text:
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            for line in text.splitlines() or [""]:
                self.log_view.appendPlainText(f"{stamp}  {line}")

    def _debug(self, text: str) -> None:
        if self.chk_protocol_trace.isChecked():
            self._log(text)

    def clear_log(self) -> None:
        self.log_view.clear()

    def debug_report(self) -> str:
        """The log with context (versions, host, settings); never the token."""
        token_given = bool(
            self.token_edit.text().strip() or os.environ.get("QUIGGELDY_TOKEN")
        )
        header = bp.debug_report_header(
            self.current_settings(),
            token_given=token_given,
            neuroseq_version=NEUROSEQ_VERSION,
            singularity_version=_singularity_version(),
            connection_state=self._session.state,
            identifier=self._session.identifier,
        )
        return "\n".join([*header, self.log_view.toPlainText()])

    def copy_log(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.debug_report())
        self._log("Debug log copied to the clipboard.")


def _activation_cache_dir() -> Path | None:
    """Per-user folder for the cached spack activation script (or None)."""
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.GenericCacheLocation
    )
    directory = (Path(base) if base else Path.home() / ".cache") / "neuroseq" / "bss2"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(f"Cannot create {directory}; spack activates on every start")
        return None
    return directory


def _container_apps(image: str) -> set[str] | None:
    """Apps defined in ``image`` (``singularity inspect --list-apps``).

    ``None`` when they cannot be determined, in which case the configured app
    is used as is.
    """
    program = shutil.which("singularity") or shutil.which("apptainer")
    if program is None:
        return None
    try:
        completed = subprocess.run(
            [program, "inspect", "--list-apps", image],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith(("INFO", "WARNING"))
    }


def _singularity_version() -> str:
    for program in ("singularity", "apptainer"):
        path = shutil.which(program)
        if path is None:
            continue
        try:
            completed = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{path} (version unknown: {exc})"
        return f"{path}: {completed.stdout.strip() or completed.stderr.strip()}"
    return "not found on PATH"
