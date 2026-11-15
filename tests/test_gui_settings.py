from __future__ import annotations

import json
from pathlib import Path

from neuroseq.gui_settings import GuiSettings


def test_codex_settings_round_trip(tmp_config_dir) -> None:
    settings = GuiSettings()
    settings.llm_provider = "codex"
    settings.llm_model = "default"
    settings.llm_codex_executable = "C:/Tools/codex.exe"
    loaded = GuiSettings()
    assert loaded.llm_provider == "codex"
    assert loaded.llm_model == "default"
    assert loaded.llm_codex_executable == "C:/Tools/codex.exe"


def test_claude_code_settings_round_trip(tmp_config_dir) -> None:
    settings = GuiSettings()
    settings.llm_provider = "claude_code"
    settings.llm_model = "default"
    settings.llm_claude_executable = "C:/Tools/claude.exe"
    loaded = GuiSettings()
    assert loaded.llm_provider == "claude_code"
    assert loaded.llm_model == "default"
    assert loaded.llm_claude_executable == "C:/Tools/claude.exe"


def test_gui_save_config_settings(tmp_config_dir) -> None:
    settings = GuiSettings()

    # change all settings
    settings.brian2_backend = "cython"
    settings.max_ram_synapse_plot_matrix = 512
    settings.max_ram_trace_plot_matrix = 384
    settings.stim_padding_secs = 6

    # check if settings file exists and contains the correct values
    config_dir = Path(tmp_config_dir)
    config_file = config_dir / "config.json"
    assert config_file.exists()

    # check if settings were saved correctly
    with open(config_file) as f:
        config = json.load(f)
    settings.theme = "dark"

    assert config["brian2_backend"] == settings.brian2_backend
    assert config["max_ram_synapse_plot_matrix"] == settings.max_ram_synapse_plot_matrix
    assert config["max_ram_trace_plot_matrix"] == settings.max_ram_trace_plot_matrix
    assert config["stim_padding_secs"] == settings.stim_padding_secs
    with open(config_file) as f:
        config = json.load(f)
    assert config["theme"] == "dark"


def test_gui_load_config_settings(tmp_config_dir) -> None:
    settings = GuiSettings()

    # change all settings
    settings.brian2_backend = "cython"
    settings.max_ram_synapse_plot_matrix = 512
    settings.max_ram_trace_plot_matrix = 384
    settings.stim_padding_secs = 6

    # set settings to original to check if loading works
    settings.__dict__["brian2_backend"] = "numpy"
    settings.__dict__["max_ram_synapse_plot_matrix"] = 256
    settings.__dict__["max_ram_trace_plot_matrix"] = 256
    settings.__dict__["stim_padding_secs"] = 0

    # load settings from file
    settings.load_config_settings()

    # check if settings were loaded correctly
    assert settings.brian2_backend == "cython"
    assert settings.max_ram_synapse_plot_matrix == 512
    assert settings.max_ram_trace_plot_matrix == 384
    assert settings.stim_padding_secs == 6

    # try loading invalid values
    settings.brian2_backend = "invalid backend"
    settings.max_ram_synapse_plot_matrix = -5
    settings.stim_padding_secs = -3

    # try loading invalid values
    # brian2 backend - should reset to default 'numpy'
    settings.load_config_settings()
    assert settings.brian2_backend == "numpy"
    # max_ram_synapse_plot_matrix - should be set to 0
    assert settings.max_ram_synapse_plot_matrix == 0
    # stim_padding_secs - should be set to 0
    assert settings.stim_padding_secs == 0

    # try setting RAM to string
    settings.brian2_backend = "numpy"  # valid again
    settings.max_ram_synapse_plot_matrix = "this is not even a number!"  # type: ignore[assignment]
    settings.stim_padding_secs = 0  # valid again
    # should log a ValueError and default to 256
    settings.load_config_settings()
    assert settings.max_ram_synapse_plot_matrix == 256

    # try setting stim_padding_secs to string
    settings.max_ram_synapse_plot_matrix = 256  # valid again
    settings.stim_padding_secs = "this is not even a number!"  # type: ignore[assignment]
    # should log a ValueError and default to 0s
    settings.load_config_settings()
    assert settings.stim_padding_secs == 0


def _resolved(path: str) -> str:
    """Match GuiSettings.add_recent_project's own Path(...).resolve() call,
    so these assertions hold on platforms where /tmp is itself a symlink
    (e.g. macOS, where it resolves to /private/tmp)."""
    return str(Path(path).resolve())


def test_add_recent_project_dedupes_and_orders_most_recent_first(
    tmp_config_dir,
) -> None:
    settings = GuiSettings()
    settings.add_recent_project("/tmp/a.nseq")
    settings.add_recent_project("/tmp/b.nseq")
    assert settings.recent_projects == [
        _resolved("/tmp/b.nseq"),
        _resolved("/tmp/a.nseq"),
    ]

    # re-adding an existing path moves it to the front instead of duplicating
    settings.add_recent_project("/tmp/a.nseq")
    assert settings.recent_projects == [
        _resolved("/tmp/a.nseq"),
        _resolved("/tmp/b.nseq"),
    ]


def test_add_recent_project_caps_list_length(tmp_config_dir) -> None:
    settings = GuiSettings()
    for i in range(10):
        settings.add_recent_project(f"/tmp/project_{i}.nseq")
    assert len(settings.recent_projects) == 8
    assert settings.recent_projects[0] == _resolved("/tmp/project_9.nseq")


def test_recent_projects_persists_and_reloads(tmp_config_dir) -> None:
    settings = GuiSettings()
    settings.add_recent_project("/tmp/a.nseq")
    settings.add_recent_project("/tmp/b.nseq")

    settings.__dict__["recent_projects"] = []
    settings.load_config_settings()
    assert settings.recent_projects == [
        _resolved("/tmp/b.nseq"),
        _resolved("/tmp/a.nseq"),
    ]


def test_recent_projects_falls_back_to_empty_list_for_invalid_config(
    tmp_config_dir,
) -> None:
    settings = GuiSettings()
    settings.add_recent_project("/tmp/a.nseq")

    config_dir = Path(tmp_config_dir)
    config_file = config_dir / "config.json"
    with open(config_file) as f:
        config = json.load(f)
    config["recent_projects"] = "not-a-list"
    with open(config_file, "w") as f:
        json.dump(config, f)

    settings.load_config_settings()
    assert settings.recent_projects == []


def test_gui_theme_setting_roundtrip_and_validation(tmp_config_dir) -> None:
    settings = GuiSettings()
    assert settings.theme == "retro"

    settings.theme = "dark"
    settings.__dict__["theme"] = "light"
    settings.load_config_settings()
    assert settings.theme == "dark"

    # invalid theme should fall back to the default retro theme
    settings.theme = "rainbow"
    settings.load_config_settings()
    assert settings.theme == "retro"


def test_llm_private_project_data_defaults_and_round_trip(tmp_config_dir) -> None:
    settings = GuiSettings()
    assert not settings.llm_project_data_private("ollama")
    assert settings.llm_project_data_private("openai")
    assert settings.llm_project_data_private("claude_code")

    settings.set_llm_project_data_private("ollama", True)
    settings.set_llm_project_data_private("openai", False)

    reloaded = GuiSettings()
    assert reloaded.llm_project_data_private("ollama")
    assert not reloaded.llm_project_data_private("openai")
    assert reloaded.llm_project_data_private("anthropic")


def test_llm_private_project_data_ignores_invalid_config(tmp_config_dir) -> None:
    config_file = Path(tmp_config_dir) / "config.json"
    config_file.write_text(
        json.dumps({"llm_private_project_data": {"openai": "no", "codex": False}}),
        encoding="utf-8",
    )
    settings = GuiSettings()
    assert settings.llm_private_project_data == {"codex": False}
    assert settings.llm_project_data_private("openai")


def test_long_lived_settings_do_not_overwrite_other_instances(tmp_config_dir) -> None:
    long_lived = GuiSettings()
    dialog = GuiSettings()
    dialog.llm_provider = "codex"
    dialog.set_llm_project_data_private("ollama", True)
    long_lived.set_llm_project_data_private("openai", False)
    long_lived.add_recent_project(str(tmp_config_dir / "x.nseq"))
    loaded = GuiSettings()
    assert loaded.llm_provider == "codex"
    assert loaded.llm_project_data_private("ollama") is True
    assert loaded.llm_project_data_private("openai") is False
    assert loaded.recent_projects == [_resolved(str(tmp_config_dir / "x.nseq"))]
