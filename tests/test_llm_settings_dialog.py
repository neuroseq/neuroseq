from __future__ import annotations

from pytestqt.qtbot import QtBot

from neuroseq.llm.llm_settings_dialog import LLMSettingsDialog


class _FakeSettings:
    def __init__(self) -> None:
        self.llm_provider = "ollama"
        self.llm_model = "qwen2.5:7b"
        self.llm_ollama_base_url = "http://localhost:11434"
        self.llm_openai_base_url = ""


def _make_dialog(qtbot: QtBot, monkeypatch) -> LLMSettingsDialog:
    monkeypatch.setattr(
        "neuroseq.llm.llm_settings_dialog.GuiSettings",
        _FakeSettings,
    )
    dialog = LLMSettingsDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.isVisible())
    return dialog


def test_dialog_has_no_qt_parent(qtbot: QtBot, monkeypatch) -> None:
    """No Qt parent - see _SelfRaisingDialog's docstring for why a shared
    WM_TRANSIENT_FOR chain with a sibling popup breaks click-to-front."""
    dialog = _make_dialog(qtbot, monkeypatch)
    assert dialog.parent() is None


def test_dialog_lists_three_providers(qtbot: QtBot, monkeypatch) -> None:
    dialog = _make_dialog(qtbot, monkeypatch)
    items = [
        dialog.provider_combo.itemText(i) for i in range(dialog.provider_combo.count())
    ]
    assert "Ollama (local)" in items
    assert "OpenAI" in items
    assert "Anthropic" in items


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
    with qtbot.waitSignal(dialog.settings_changed):
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
