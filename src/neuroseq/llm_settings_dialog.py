from __future__ import annotations

from qtpy import QtCore
from qtpy import QtWidgets

from neuroseq.gui_settings import GuiSettings
from neuroseq.llm import available_backends
from neuroseq.llm.anthropic_backend import AnthropicBackend
from neuroseq.llm.backend import LLMBackend
from neuroseq.llm.backend import LLMBackendError
from neuroseq.llm.ollama import OllamaBackend
from neuroseq.llm.openai_backend import OpenAIBackend
from neuroseq.llm.secrets import delete_api_key
from neuroseq.llm.secrets import set_api_key
from neuroseq.stdp_synaptic_tuning_dialog import _SelfRaisingDialog


class LLMSettingsDialog(_SelfRaisingDialog):
    """Non-modal dialog, so no Qt parent (see _SelfRaisingDialog's
    docstring): a parent sets WM_TRANSIENT_FOR, and on X11/KWin a transient
    dialog can end up sharing its owner's "raise group" so clicking it
    never brings it above another sibling popup."""

    settings_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setModal(False)
        self.setWindowTitle("LLM assistant settings")
        self._loading = False
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.provider_combo = QtWidgets.QComboBox(self)
        for provider_name, label in available_backends():
            self.provider_combo.addItem(label, provider_name)
        form.addRow("Provider:", self.provider_combo)

        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.setEditable(True)
        self.refresh_button = QtWidgets.QPushButton("Refresh", self)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.refresh_button)
        form.addRow("Model:", model_row)

        self.endpoint_edit = QtWidgets.QLineEdit(self)
        form.addRow("Endpoint:", self.endpoint_edit)

        self.api_key_edit = QtWidgets.QLineEdit(self)
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        form.addRow("API key:", self.api_key_edit)

        key_button_row = QtWidgets.QHBoxLayout()
        self.save_key_button = QtWidgets.QPushButton("Save key in OS keychain", self)
        self.clear_key_button = QtWidgets.QPushButton("Clear stored key", self)
        key_button_row.addWidget(self.save_key_button)
        key_button_row.addWidget(self.clear_key_button)
        form.addRow("", key_button_row)

        self.status_label = QtWidgets.QLabel("", self)
        form.addRow("Status:", self.status_label)
        layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
        self.test_button = QtWidgets.QPushButton("Test connection", self)
        self.close_button = QtWidgets.QPushButton("Close", self)
        button_row.addWidget(self.test_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.model_combo.currentTextChanged.connect(self._model_changed)
        self.endpoint_edit.editingFinished.connect(self._endpoint_changed)
        self.refresh_button.clicked.connect(self._refresh_models)
        self.save_key_button.clicked.connect(self._save_key)
        self.clear_key_button.clicked.connect(self._clear_key)
        self.test_button.clicked.connect(self._test_connection)
        self.close_button.clicked.connect(self.hide)

    def _load_settings(self) -> None:
        self._loading = True
        settings = GuiSettings()
        provider_index = self.provider_combo.findData(settings.llm_provider)
        if provider_index >= 0:
            self.provider_combo.setCurrentIndex(provider_index)
        self.model_combo.clear()
        self.model_combo.addItem(settings.llm_model)
        self.model_combo.setCurrentText(settings.llm_model)
        self.endpoint_edit.setText(self._current_endpoint_from_settings(settings))
        self.api_key_edit.clear()
        self._update_provider_visibility()
        self.status_label.setText("Idle")
        self._loading = False

    def _current_provider(self) -> str:
        return str(self.provider_combo.currentData())

    def _current_endpoint_from_settings(self, settings: GuiSettings) -> str:
        if settings.llm_provider == "openai":
            return settings.llm_openai_base_url
        return settings.llm_ollama_base_url

    def _update_provider_visibility(self) -> None:
        provider = self._current_provider()
        show_endpoint = provider in {"ollama", "openai"}
        show_key = provider in {"openai", "anthropic"}
        self.endpoint_edit.setVisible(show_endpoint)
        form = self.layout().itemAt(0).layout()
        assert isinstance(form, QtWidgets.QFormLayout)
        endpoint_label = form.labelForField(self.endpoint_edit)
        if endpoint_label is not None:
            endpoint_label.setVisible(show_endpoint)
        self.api_key_edit.setVisible(show_key)
        api_key_label = form.labelForField(self.api_key_edit)
        if api_key_label is not None:
            api_key_label.setVisible(show_key)
        self.save_key_button.setVisible(show_key)
        self.clear_key_button.setVisible(show_key)

    def _save_settings(self) -> None:
        if self._loading:
            return
        settings = GuiSettings()
        settings.llm_provider = self._current_provider()
        model = self.model_combo.currentText().strip()
        settings.llm_model = model or "qwen2.5:7b"
        endpoint = self.endpoint_edit.text().strip().rstrip("/")
        if settings.llm_provider == "openai":
            settings.llm_openai_base_url = endpoint
        else:
            settings.llm_ollama_base_url = endpoint or "http://localhost:11434"
        self.settings_changed.emit()

    def _provider_changed(self) -> None:
        self._update_provider_visibility()
        if self._loading:
            return
        settings = GuiSettings()
        provider = self._current_provider()
        if provider == "openai":
            self.endpoint_edit.setText(settings.llm_openai_base_url)
        else:
            self.endpoint_edit.setText(settings.llm_ollama_base_url)
        self._save_settings()

    def _model_changed(self) -> None:
        self._save_settings()

    def _endpoint_changed(self) -> None:
        self._save_settings()

    def _make_backend(self, use_typed_key: bool = False) -> LLMBackend:
        provider = self._current_provider()
        endpoint = self.endpoint_edit.text().strip().rstrip("/") or None
        api_key = self.api_key_edit.text().strip() or None
        if provider == "ollama":
            return OllamaBackend(base_url=endpoint or "http://localhost:11434")
        if provider == "openai":
            return OpenAIBackend(
                api_key=api_key if use_typed_key else None,
                base_url=endpoint,
            )
        return AnthropicBackend(api_key=api_key if use_typed_key else None)

    def _refresh_models(self) -> None:
        try:
            backend = self._make_backend(use_typed_key=True)
            models = backend.list_models()
        except LLMBackendError as e:
            self.status_label.setText(str(e))
            return
        current_model = self.model_combo.currentText()
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model)
        if current_model:
            if self.model_combo.findText(current_model) < 0:
                self.model_combo.addItem(current_model)
            self.model_combo.setCurrentText(current_model)
        self.status_label.setText("Models refreshed")

    def _save_key(self) -> None:
        provider = self._current_provider()
        key = self.api_key_edit.text().strip()
        if not key:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing API key",
                "Enter an API key before saving it.",
            )
            return
        if not set_api_key(provider, key):
            QtWidgets.QMessageBox.warning(
                self,
                "Keyring unavailable",
                "Could not store the key in the OS keychain. Install the "
                "'keyring' package in the active environment, or set the "
                "provider environment variable instead.",
            )
            return
        self.status_label.setText("API key saved")
        self.settings_changed.emit()

    def _clear_key(self) -> None:
        provider = self._current_provider()
        if (
            QtWidgets.QMessageBox.question(
                self,
                "Clear stored key",
                f"Delete the stored API key for {provider}?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        delete_api_key(provider)
        self.api_key_edit.clear()
        self.status_label.setText("Stored API key cleared")
        self.settings_changed.emit()

    def _test_connection(self) -> None:
        try:
            backend = self._make_backend(use_typed_key=True)
            ok, reason = backend.is_available()
        except LLMBackendError as e:
            self.status_label.setText(str(e))
            return
        self.status_label.setText("Connected" if ok else reason)
