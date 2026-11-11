from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.llm.claude_code_backend import ClaudeCodeBackend
from neuroseq.llm.codex_backend import CodexBackend
from neuroseq.llm.llm_settings_dialog import LLMSettingsDialog
from tests.wait_timeout import WAIT_TIMEOUT_MS


class _FakeSettings:
    def __init__(self) -> None:
        self.llm_provider = "ollama"
        self.llm_model = "qwen2.5:7b"
        self.llm_ollama_base_url = "http://localhost:11434"
        self.llm_openai_base_url = ""
        self.llm_codex_executable = ""
        self.llm_claude_executable = ""
        self.llm_private_project_data: dict[str, bool] = {}

    def llm_project_data_private(self, provider: str) -> bool:
        return bool(self.llm_private_project_data.get(provider, provider != "ollama"))

    def set_llm_project_data_private(self, provider: str, private: bool) -> None:
        self.llm_private_project_data = {
            **self.llm_private_project_data,
            provider: private,
        }


def _make_dialog(qtbot: QtBot, monkeypatch) -> LLMSettingsDialog:
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings",
        _FakeSettings,
    )
    dialog = LLMSettingsDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=WAIT_TIMEOUT_MS)
    return dialog


def test_dialog_has_no_qt_parent(qtbot: QtBot, monkeypatch) -> None:
    """No Qt parent - see SelfRaisingDialog's docstring for why a shared
    WM_TRANSIENT_FOR chain with a sibling popup breaks click-to-front."""
    dialog = _make_dialog(qtbot, monkeypatch)
    assert dialog.parent() is None


def test_dialog_lists_providers(qtbot: QtBot, monkeypatch) -> None:
    dialog = _make_dialog(qtbot, monkeypatch)
    items = [
        dialog.provider_combo.itemText(i) for i in range(dialog.provider_combo.count())
    ]
    assert "Ollama (local)" in items
    assert "OpenAI" in items
    assert "Anthropic" in items
    assert "Codex (local sign-in)" in items
    assert "Claude Code (local sign-in)" in items


def test_claude_code_settings_and_model_switching(qtbot: QtBot, monkeypatch) -> None:
    settings = _FakeSettings()
    dialog = _make_dialog(qtbot, monkeypatch)
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings", lambda: settings
    )
    dialog.provider_combo.setCurrentText("Claude Code (local sign-in)")
    assert dialog.claude_executable_edit.isVisible()
    assert dialog.claude_help_label.isVisible()
    assert not dialog.codex_executable_edit.isVisible()
    assert not dialog.endpoint_edit.isVisible()
    assert not dialog.api_key_edit.isVisible()
    assert dialog.model_combo.currentText() == "default"
    dialog.claude_executable_edit.setText("C:/Tools/claude.exe")
    dialog._save_settings()
    assert settings.llm_provider == "claude_code"
    assert settings.llm_claude_executable == "C:/Tools/claude.exe"
    backend = dialog._make_backend()
    assert isinstance(backend, ClaudeCodeBackend)
    assert backend.executable == "C:/Tools/claude.exe"
    dialog.model_combo.setCurrentText("opus")
    dialog.provider_combo.setCurrentText("Codex (local sign-in)")
    assert dialog.model_combo.currentText() == "default"
    dialog.provider_combo.setCurrentText("Claude Code (local sign-in)")
    assert dialog.model_combo.currentText() == "opus"


def test_codex_settings_and_model_switching(qtbot: QtBot, monkeypatch) -> None:
    settings = _FakeSettings()
    dialog = _make_dialog(qtbot, monkeypatch)
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings", lambda: settings
    )
    dialog.provider_combo.setCurrentText("Codex (local sign-in)")
    assert dialog.codex_executable_edit.isVisible()
    assert not dialog.endpoint_edit.isVisible()
    assert not dialog.api_key_edit.isVisible()
    assert dialog.model_combo.currentText() == "default"
    dialog.codex_executable_edit.setText("C:/Tools/codex.exe")
    dialog._save_settings()
    assert settings.llm_codex_executable == "C:/Tools/codex.exe"
    backend = dialog._make_backend()
    assert isinstance(backend, CodexBackend)
    assert backend.executable == "C:/Tools/codex.exe"
    assert settings.llm_provider == "codex"
    dialog.model_combo.setCurrentText("my-codex-model")
    dialog.provider_combo.setCurrentText("Ollama (local)")
    assert dialog.model_combo.currentText() == "qwen2.5:7b"
    dialog.provider_combo.setCurrentText("Codex (local sign-in)")
    assert dialog.model_combo.currentText() == "my-codex-model"


def test_switching_provider_shows_correct_fields(qtbot: QtBot, monkeypatch) -> None:
    dialog = _make_dialog(qtbot, monkeypatch)

    dialog.provider_combo.setCurrentText("Ollama (local)")
    assert dialog.endpoint_edit.isVisible()
    assert not dialog.api_key_edit.isVisible()

    dialog.provider_combo.setCurrentText("OpenAI")
    assert dialog.endpoint_edit.isVisible()
    assert dialog.api_key_edit.isVisible()

    dialog.provider_combo.setCurrentText("Anthropic")
    assert not dialog.endpoint_edit.isVisible()
    assert dialog.api_key_edit.isVisible()


def test_save_key_writes_to_secrets_module(qtbot: QtBot, monkeypatch) -> None:
    recorded: dict[tuple[str, str], bool] = {}
    dialog = _make_dialog(qtbot, monkeypatch)
    dialog.provider_combo.setCurrentText("OpenAI")
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.set_api_key",
        lambda provider, key: recorded.setdefault((provider, key), True),
    )

    dialog.api_key_edit.setText("sk-test")
    dialog.save_key_button.click()

    assert recorded == {("openai", "sk-test"): True}


def test_save_key_warns_when_keyring_missing(
    qtbot: QtBot,
    monkeypatch,
    qmessagebox_calls: list[dict[str, str]],
) -> None:
    dialog = _make_dialog(qtbot, monkeypatch)
    dialog.provider_combo.setCurrentText("OpenAI")
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.set_api_key",
        lambda provider, key: False,
    )

    dialog.api_key_edit.setText("sk-test")
    dialog.save_key_button.click()

    assert qmessagebox_calls[-1]["name"] == "warning"


def test_settings_changed_signal_fires_on_provider_switch(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    dialog = _make_dialog(qtbot, monkeypatch)
    with qtbot.waitSignal(dialog.settings_changed, timeout=WAIT_TIMEOUT_MS):
        dialog.provider_combo.setCurrentText("OpenAI")


def test_test_connection_button_calls_is_available(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    class _FakeBackend:
        def is_available(self):
            return False, "no server"

    dialog = _make_dialog(qtbot, monkeypatch)
    monkeypatch.setattr(
        dialog, "_make_backend", lambda use_typed_key=False: _FakeBackend()
    )

    dialog.test_button.click()

    assert "no server" in dialog.status_label.text()


def test_private_data_checkbox_defaults_per_provider_and_persists(
    qtbot: QtBot, monkeypatch
) -> None:
    settings = _FakeSettings()
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings", lambda: settings
    )
    dialog = LLMSettingsDialog()
    qtbot.addWidget(dialog)
    changed = []
    dialog.settings_changed.connect(lambda: changed.append(True))

    # Local Ollama shares data by default, hosted providers do not.
    assert not dialog.private_data_check.isChecked()
    dialog.provider_combo.setCurrentText("Anthropic")
    assert dialog.private_data_check.isChecked()

    dialog.private_data_check.setChecked(False)
    assert settings.llm_private_project_data == {"anthropic": False}
    assert changed

    dialog.provider_combo.setCurrentText("Codex (local sign-in)")
    assert dialog.private_data_check.isChecked()
    dialog.provider_combo.setCurrentText("Anthropic")
    assert not dialog.private_data_check.isChecked()


def test_codex_privacy_warning_shows_only_for_codex(qtbot: QtBot, monkeypatch) -> None:
    settings = _FakeSettings()
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings", lambda: settings
    )
    dialog = LLMSettingsDialog()
    qtbot.addWidget(dialog)

    assert dialog.codex_privacy_label.isHidden()
    dialog.provider_combo.setCurrentText("Codex (local sign-in)")
    assert not dialog.codex_privacy_label.isHidden()
    assert "OpenAI" in dialog.codex_privacy_label.text()
    dialog.provider_combo.setCurrentText("Anthropic")
    assert dialog.codex_privacy_label.isHidden()
