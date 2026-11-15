from __future__ import annotations

import json
import logging
import pathlib
import shutil
from dataclasses import dataclass
from dataclasses import field

import brian2 as b2
from platformdirs import user_config_dir

from neuroseq import theme as theme_module

_FIJI_AUTODETECT_PATHS = (
    # New (2025+) Fiji layout: ~/Fiji/fiji shell-script launcher.
    "/opt/Fiji/fiji",
    "/opt/fiji/fiji",
    "/usr/local/Fiji/fiji",
    # Legacy Fiji.app layout with ImageJ-linux64 binary.
    "/opt/Fiji.app/ImageJ-linux64",
    "/opt/fiji/ImageJ-linux64",
    "/usr/local/Fiji.app/ImageJ-linux64",
    "/usr/bin/fiji",
    # macOS bundle.
    "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
)

_FIJI_HOME_CANDIDATES = (
    pathlib.Path("Fiji") / "fiji",
    pathlib.Path("Fiji.app") / "ImageJ-linux64",
)

MAX_RECENT_PROJECTS = 8


def _autodetect_fiji() -> str:
    home = pathlib.Path.home()
    for rel in _FIJI_HOME_CANDIDATES:
        home_candidate = home / rel
        if home_candidate.exists():
            return str(home_candidate)
    for path_str in _FIJI_AUTODETECT_PATHS:
        if pathlib.Path(path_str).exists():
            return path_str
    found = shutil.which("fiji") or shutil.which("ImageJ")
    return found or ""


def _without_secrets(values: dict) -> dict:
    """Drop anything token-like: credentials are never persisted."""
    return {
        key: value
        for key, value in values.items()
        if isinstance(key, str) and "token" not in key.lower()
    }


def _read_config_file() -> dict:
    config_file = pathlib.Path(user_config_dir("neuroseq", "ssciwr")) / "config.json"
    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


@dataclass
class GuiSettings:
    max_ram_synapse_plot_matrix: int = 256
    max_ram_trace_plot_matrix: int = 256
    brian2_backend: str = "numpy"
    stim_padding_secs: int = 0
    theme: str = theme_module.DEFAULT
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_ollama_base_url: str = "http://localhost:11434"
    llm_openai_base_url: str = ""
    llm_codex_executable: str = ""
    llm_claude_executable: str = ""
    #: Per provider: True keeps experimental data from the LLM assistant.
    #: Missing providers use ``llm_project_data_private``'s default.
    llm_private_project_data: dict = field(default_factory=dict)
    fiji_executable_path: str = ""
    scanimage_python_path: str = ""
    scanimage_recording_directory: str = ""
    syscon_export_directory: str = ""
    #: Folder of the file picked last in any project's data import dialogs;
    #: the start folder for a project that has none of its own (#646).
    import_directory: str = ""
    #: "Grab FOV for segmentation" applies Fiji's Auto brightness before Cellpose.
    fov_auto_brightness: bool = True
    #: BrainScaleS-2 tab settings (``bss2_protocol.Bss2Settings.to_dict()``).
    #: Never contains the EBRAINS token.
    bss2_settings: dict = field(default_factory=dict)
    recent_projects: list[str] = field(default_factory=list)

    def __init__(self) -> None:
        self.logger: logging.Logger
        self.__dict__["logger"] = logging.getLogger(__name__)
        self.logger.info("loading GUI settings")
        self.__dict__["fiji_executable_path"] = _autodetect_fiji()
        self.__dict__["recent_projects"] = []
        self.__dict__["bss2_settings"] = {}
        self.__dict__["llm_private_project_data"] = {}
        self.load_config_settings()
        b2.prefs.codegen.target = self.brian2_backend

    def add_recent_project(self, path: str) -> None:
        """Record ``path`` as the most recently opened/saved project file,
        for the startup screen's Recent Projects list."""
        normalized = str(pathlib.Path(path).resolve())
        updated = [normalized] + [p for p in self.recent_projects if p != normalized]
        self.recent_projects = updated[:MAX_RECENT_PROJECTS]

    def llm_project_data_private(self, provider: str) -> bool:
        """Whether the assistant hides experimental data from ``provider``.
        Defaults to hidden for every provider except local Ollama."""
        return bool(self.llm_private_project_data.get(provider, provider != "ollama"))

    def set_llm_project_data_private(self, provider: str, private: bool) -> None:
        stored = _read_config_file().get("llm_private_project_data")
        current = stored if isinstance(stored, dict) else {}
        self.llm_private_project_data = {
            **self.llm_private_project_data,
            **{k: v for k, v in current.items() if isinstance(v, bool)},
            provider: bool(private),
        }

    def __setattr__(self, prop, val):
        super().__setattr__(prop, val)
        # Write only this key: other GuiSettings instances (e.g. a settings
        # dialog) may have changed other keys since this one was loaded.
        self.save_config_settings(only=prop)

    def _config_values(self) -> dict:
        return {
            "brian2_backend": self.brian2_backend,
            "max_ram_synapse_plot_matrix": self.max_ram_synapse_plot_matrix,
            "max_ram_trace_plot_matrix": self.max_ram_trace_plot_matrix,
            "stim_padding_secs": self.stim_padding_secs,
            "theme": self.theme,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_ollama_base_url": self.llm_ollama_base_url,
            "llm_openai_base_url": self.llm_openai_base_url,
            "llm_codex_executable": self.llm_codex_executable,
            "llm_claude_executable": self.llm_claude_executable,
            "llm_private_project_data": self.llm_private_project_data,
            "fiji_executable_path": self.fiji_executable_path,
            "scanimage_python_path": self.scanimage_python_path,
            "scanimage_recording_directory": self.scanimage_recording_directory,
            "syscon_export_directory": self.syscon_export_directory,
            "import_directory": self.import_directory,
            "fov_auto_brightness": self.fov_auto_brightness,
            "bss2_settings": _without_secrets(self.bss2_settings),
            "recent_projects": self.recent_projects,
        }

    def save_config_settings(self, only: str | None = None) -> None:
        """Write the settings; with ``only``, merge just that key into the file."""
        values = self._config_values()
        if only is not None:
            if only not in values:
                return
            values = {**_read_config_file(), only: values[only]}
        try:
            config_dir = pathlib.Path(user_config_dir("neuroseq", "ssciwr"))
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(values, f)
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
            max_ram_trace_plot_matrix = config.get(
                "max_ram_trace_plot_matrix", self.max_ram_trace_plot_matrix
            )
            stim_padding_secs = config.get("stim_padding_secs", self.stim_padding_secs)
            theme = config.get("theme", self.theme)
            llm_provider = config.get("llm_provider", self.llm_provider)
            llm_model = config.get("llm_model", self.llm_model)
            llm_ollama_base_url = config.get(
                "llm_ollama_base_url", self.llm_ollama_base_url
            )
            llm_openai_base_url = config.get(
                "llm_openai_base_url", self.llm_openai_base_url
            )
            fiji_executable_path = config.get("fiji_executable_path", None)
            scanimage_python_path = config.get("scanimage_python_path", "")
            scanimage_recording_directory = config.get(
                "scanimage_recording_directory", ""
            )
            syscon_export_directory = config.get("syscon_export_directory", "")
            import_directory = config.get("import_directory", "")
            fov_auto_brightness = config.get("fov_auto_brightness", True)
            recent_projects = config.get("recent_projects", [])
            bss2_settings = config.get("bss2_settings", {})

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
                self.__dict__["max_ram_trace_plot_matrix"] = max(
                    0, int(max_ram_trace_plot_matrix)
                )
                self.logger.info(
                    f"Loaded max RAM for trace plot matrix from config file {config_file}: {max_ram_trace_plot_matrix}"
                )
            except ValueError:
                self.logger.warning(
                    f"Invalid max RAM trace plot matrix in config file {config_file}: {max_ram_trace_plot_matrix}. Restoring default of 256MB"
                )
                self.__dict__["max_ram_trace_plot_matrix"] = 256
            try:
                self.__dict__["stim_padding_secs"] = max(0, int(stim_padding_secs))
                self.logger.info(
                    f"Loaded input padding from config file {config_file}: {stim_padding_secs}"
                )
            except ValueError:
                self.logger.warning(
                    f"Invalid input padding in config file {config_file}: {stim_padding_secs}. Restoring default of 0 s"
                )
                self.__dict__["stim_padding_secs"] = 0
            if theme in theme_module.THEMES:
                self.__dict__["theme"] = theme
                self.logger.info(
                    f"Loaded theme from config file {config_file}: {theme}"
                )
            else:
                self.logger.warning(
                    f"Invalid theme in config file {config_file}: {theme}. "
                    f"Restoring default of {theme_module.DEFAULT}"
                )
                self.__dict__["theme"] = theme_module.DEFAULT
            valid_providers = {
                "ollama",
                "openai",
                "anthropic",
                "codex",
                "claude_code",
            }
            if llm_provider in valid_providers:
                self.__dict__["llm_provider"] = llm_provider
            else:
                self.logger.warning(
                    "Invalid LLM provider in config file %s: %s. Restoring "
                    "default of ollama",
                    config_file,
                    llm_provider,
                )
                self.__dict__["llm_provider"] = "ollama"
            llm_model = str(llm_model).strip()
            self.__dict__["llm_model"] = llm_model or "qwen2.5:7b"
            self.__dict__["llm_codex_executable"] = str(
                config.get("llm_codex_executable", "")
            ).strip()
            self.__dict__["llm_claude_executable"] = str(
                config.get("llm_claude_executable", "")
            ).strip()
            self.__dict__["llm_ollama_base_url"] = (
                str(llm_ollama_base_url).rstrip("/") or "http://localhost:11434"
            )
            self.__dict__["llm_openai_base_url"] = str(llm_openai_base_url).rstrip("/")
            private_data = config.get("llm_private_project_data", {})
            self.__dict__["llm_private_project_data"] = (
                {str(k): v for k, v in private_data.items() if isinstance(v, bool)}
                if isinstance(private_data, dict)
                else {}
            )
            if fiji_executable_path is None or not str(fiji_executable_path).strip():
                self.__dict__["fiji_executable_path"] = _autodetect_fiji()
            else:
                self.__dict__["fiji_executable_path"] = str(fiji_executable_path)
            self.__dict__["scanimage_python_path"] = str(scanimage_python_path or "")
            self.__dict__["scanimage_recording_directory"] = str(
                scanimage_recording_directory or ""
            )
            self.__dict__["syscon_export_directory"] = str(
                syscon_export_directory or ""
            )
            self.__dict__["import_directory"] = str(import_directory or "")
            self.__dict__["fov_auto_brightness"] = fov_auto_brightness is True
            self.__dict__["bss2_settings"] = (
                _without_secrets(bss2_settings)
                if isinstance(bss2_settings, dict)
                else {}
            )
            if isinstance(recent_projects, list) and all(
                isinstance(p, str) for p in recent_projects
            ):
                self.__dict__["recent_projects"] = recent_projects[:MAX_RECENT_PROJECTS]
            else:
                self.logger.warning(
                    f"Invalid recent_projects in config file {config_file}: "
                    f"{recent_projects!r}. Restoring empty list"
                )
                self.__dict__["recent_projects"] = []
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}")
            self.logger.exception(e)
