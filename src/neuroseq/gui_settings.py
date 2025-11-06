from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

import brian2 as b2
from platformdirs import user_config_dir


@dataclass
class GuiSettings:
    max_ram_synapse_plot_matrix: int = 256
    brian2_backend: str = "numpy"
    stim_padding_secs: int = 5

    def __init__(self) -> None:
        self.logger: logging.Logger
        self.__dict__["logger"] = logging.getLogger(__name__)
        self.logger.info("loading GUI settings")
        self.load_config_settings()
        b2.prefs.codegen.target = self.brian2_backend

    def __setattr__(self, prop, val):
        super().__setattr__(prop, val)
        # save changes to the config file
        self.save_config_settings()

    def save_config_settings(self) -> None:
        try:
            config_dir = pathlib.Path(user_config_dir("neuroseq", "ssciwr"))
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "brian2_backend": self.brian2_backend,
                        "max_ram_synapse_plot_matrix": self.max_ram_synapse_plot_matrix,
                        "stim_padding_secs": self.stim_padding_secs,
                    },
                    f,
                )
            self.logger.info(f"Saved config settings to {config_file}")
        except Exception as e:
            self.logger.warning(f"Could not save config file: {e}")
            self.logger.exception(e)

    def load_config_settings(self) -> None:
        # check if config file exists. If not, do nothing and run on default values
        config_dir = pathlib.Path(user_config_dir("neuroseq", "ssciwr"))
        config_file = config_dir / "config.json"
        if not config_file.exists():
            self.logger.info(
                f"No config file found at {config_file}, using default settings"
            )
            return
        # config file exists, reading...
        try:
            # open file
            config_dir = pathlib.Path(user_config_dir("neuroseq", "ssciwr"))
            config_file = config_dir / "config.json"
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # retrieving values
            brian2_backend = config.get("brian2_backend", self.brian2_backend)
            max_ram_synapse_plot_matrix = config.get(
                "max_ram_synapse_plot_matrix", self.max_ram_synapse_plot_matrix
            )
            stim_padding_secs = config.get("stim_padding_secs", self.stim_padding_secs)

            # setting values
            if brian2_backend in ["cython", "numpy"]:
                self.__dict__["brian2_backend"] = brian2_backend
                self.logger.info(
                    f"Loaded Brian2 backend from config file {config_file}: {brian2_backend}"
                )
            else:
                self.logger.warning(
                    f"Invalid Brian2 backend in config file {config_file}: {brian2_backend}. Restoring default of numpy"
                )
                self.__dict__["brian2_backend"] = "numpy"
            try:
                self.__dict__["max_ram_synapse_plot_matrix"] = max(
                    0, int(max_ram_synapse_plot_matrix)
                )
                self.logger.info(
                    f"Loaded max RAM for synapse plot matrix from config file {config_file}: {max_ram_synapse_plot_matrix}"
                )
            except ValueError:
                self.logger.warning(
                    f"Invalid max RAM synapse plot matrix in config file {config_file}: {max_ram_synapse_plot_matrix}. Restoring default of 256MB"
                )
                self.__dict__["max_ram_synapse_plot_matrix"] = 256
            try:
                self.__dict__["stim_padding_secs"] = max(0, int(stim_padding_secs))
                self.logger.info(
                    f"Loaded stimulus padding from config file {config_file}: {stim_padding_secs}"
                )
            except ValueError:
                self.logger.warning(
                    f"Invalid stimulus padding in config file {config_file}: {stim_padding_secs}. Restoring default of 5s"
                )
                self.__dict__["stim_padding_secs"] = 5
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}")
            self.logger.exception(e)
