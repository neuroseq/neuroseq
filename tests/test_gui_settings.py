from __future__ import annotations

import json
from pathlib import Path

from neuroseq.gui_settings import GuiSettings


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


def test_add_recent_project_dedupes_and_orders_most_recent_first(
    tmp_config_dir,
) -> None:
    settings = GuiSettings()
    settings.add_recent_project("/tmp/a.nseq")
    settings.add_recent_project("/tmp/b.nseq")
    assert settings.recent_projects == ["/tmp/b.nseq", "/tmp/a.nseq"]

    # re-adding an existing path moves it to the front instead of duplicating
    settings.add_recent_project("/tmp/a.nseq")
    assert settings.recent_projects == ["/tmp/a.nseq", "/tmp/b.nseq"]


def test_add_recent_project_caps_list_length(tmp_config_dir) -> None:
    settings = GuiSettings()
    for i in range(10):
        settings.add_recent_project(f"/tmp/project_{i}.nseq")
    assert len(settings.recent_projects) == 8
    assert settings.recent_projects[0] == "/tmp/project_9.nseq"


def test_recent_projects_persists_and_reloads(tmp_config_dir) -> None:
    settings = GuiSettings()
    settings.add_recent_project("/tmp/a.nseq")
    settings.add_recent_project("/tmp/b.nseq")

    settings.__dict__["recent_projects"] = []
    settings.load_config_settings()
    assert settings.recent_projects == ["/tmp/b.nseq", "/tmp/a.nseq"]


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
