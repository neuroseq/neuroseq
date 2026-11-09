from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from qtpy import QtCore

from neuroseq.data import SpikeData
from neuroseq.data import SynapseData
from neuroseq.data import TraceData


@dataclass(frozen=True)
class AnalysisPlugin:
    id: str
    label: str
    icon: str
    render: Callable[
        [object, TraceData | SpikeData | SynapseData, slice | None, slice | None], None
    ]
    open_dialog: (
        Callable[
            [
                object,
                object,
                TraceData | SpikeData | SynapseData,
                slice | None,
                slice | None,
            ],
            None,
        ]
        | None
    ) = None

    def __str__(self) -> str:
        return self.label


#: Environment variable overriding the per-user plugin folder (used by tests).
USER_PLUGIN_DIR_ENV = "NEUROSEQ_USER_PLUGIN_DIR"


def _plugins_directory() -> Path:
    """Built-in plugins shipped inside the package."""
    return Path(__file__).resolve().parent / "resources" / "analysis_plugins"


def user_plugins_directory(create: bool = False) -> Path:
    """Per-user plugin folder (outside the package), e.g. for chat-authored plugins.

    ``NEUROSEQ_USER_PLUGIN_DIR`` overrides the location. Otherwise the folder
    lives under the platform's user data location (``.../neuroseq/analysis_plugins``).
    """
    override = os.environ.get(USER_PLUGIN_DIR_ENV)
    if override:
        directory = Path(override).expanduser()
    else:
        base = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.GenericDataLocation
        )
        root = Path(base) / "neuroseq" if base else Path.home() / ".neuroseq"
        directory = root / "analysis_plugins"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def builtin_plugin_names() -> set[str]:
    """Module stems in the built-in plugin folder, including helper modules."""
    plugins_dir = _plugins_directory()
    if not plugins_dir.is_dir():
        return set()
    return {path.stem for path in plugins_dir.glob("*.py")}


def _load_plugin_module(
    plugin_file: Path, namespace: str = "neuroseq.analysis_plugin"
) -> ModuleType | None:
    logger = logging.getLogger(__name__)
    spec = importlib.util.spec_from_file_location(
        f"{namespace}.{plugin_file.stem}",
        plugin_file,
    )
    if spec is None or spec.loader is None:
        logger.warning(
            f"Could not create module spec for analysis plugin: {plugin_file}"
        )
        return None
    module = importlib.util.module_from_spec(spec)
    # Match normal import semantics. Decorators such as ``@dataclass`` resolve
    # annotations through ``sys.modules`` while the module is executing.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        logger.exception(f"Failed to load analysis plugin {plugin_file}: {exc}")
        return None
    return module


def _plugin_files() -> list[tuple[Path, str]]:
    """Built-in plugin files followed by non-shadowing user plugin files."""
    logger = logging.getLogger(__name__)
    files: list[tuple[Path, str]] = []
    plugins_dir = _plugins_directory()
    if plugins_dir.is_dir():
        files.extend(
            (path, "neuroseq.analysis_plugin")
            for path in sorted(plugins_dir.glob("*.py"))
            if not path.stem.startswith("_")
        )
    builtin_names = builtin_plugin_names()
    user_dir = user_plugins_directory()
    if user_dir.is_dir() and user_dir.resolve() != plugins_dir.resolve():
        for path in sorted(user_dir.glob("*.py")):
            if path.stem.startswith("_"):
                continue
            if path.stem in builtin_names:
                logger.warning(
                    f"Skipping user analysis plugin {path}: name collides with a "
                    f"built-in plugin"
                )
                continue
            files.append((path, "neuroseq.user_analysis_plugin"))
    return files


_PluginFilesKey = tuple[tuple[str, str, int | None, int | None], ...]

#: Last discovery result, keyed on the plugin files and their mtimes/sizes.
_discovery_cache: tuple[_PluginFilesKey, list[AnalysisPlugin]] | None = None


def _plugin_files_key(files: list[tuple[Path, str]]) -> _PluginFilesKey:
    key: list[tuple[str, str, int | None, int | None]] = []
    for path, namespace in files:
        try:
            stat = path.stat()
        except OSError:
            key.append((str(path), namespace, None, None))
        else:
            key.append((str(path), namespace, stat.st_mtime_ns, stat.st_size))
    return tuple(key)


def discover_analysis_plugins() -> list[AnalysisPlugin]:
    """Load every analysis plugin.

    Plugin files are only re-executed when the set of files or one of their
    modification times/sizes changed since the last call (the trace context
    menu calls this every time it opens).
    """
    global _discovery_cache
    files = _plugin_files()
    key = _plugin_files_key(files)
    if _discovery_cache is not None and _discovery_cache[0] == key:
        return list(_discovery_cache[1])
    plugins = _discover_analysis_plugins(files)
    _discovery_cache = (key, plugins)
    return list(plugins)


def _discover_analysis_plugins(files: list[tuple[Path, str]]) -> list[AnalysisPlugin]:
    logger = logging.getLogger(__name__)
    plugins: list[AnalysisPlugin] = []
    builtin_ids: set[str] = set()

    for plugin_file, namespace in files:
        is_user = namespace != "neuroseq.analysis_plugin"
        module = _load_plugin_module(plugin_file, namespace)
        if module is None:
            continue

        render = getattr(module, "render_analysis", None)
        if not callable(render):
            logger.warning(
                f"Skipping analysis plugin {plugin_file}: missing callable render_analysis"
            )
            continue

        plugin_id = str(getattr(module, "PLUGIN_ID", plugin_file.stem))
        if is_user and plugin_id in builtin_ids:
            logger.warning(
                f"Skipping user analysis plugin {plugin_file}: PLUGIN_ID "
                f"{plugin_id!r} collides with a built-in plugin"
            )
            continue
        if not is_user:
            builtin_ids.add(plugin_id)
        plugin_label = getattr(module, "PLUGIN_NAME", plugin_file.stem)
        plugin_icon = getattr(module, "PLUGIN_ICON", "mdi6.puzzle")
        open_dialog = getattr(module, "open_analysis_dialog", None)
        if open_dialog is not None and not callable(open_dialog):
            logger.warning(
                f"Ignoring non-callable open_analysis_dialog in analysis plugin "
                f"{plugin_file}"
            )
            open_dialog = None

        plugins.append(
            AnalysisPlugin(
                id=plugin_id,
                label=str(plugin_label),
                icon=str(plugin_icon),
                render=render,
                open_dialog=open_dialog,
            )
        )

    return plugins
