from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

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


def _plugins_directory() -> Path:
    return Path(__file__).resolve().parent / "resources" / "analysis_plugins"


def _load_plugin_module(plugin_file: Path) -> ModuleType | None:
    logger = logging.getLogger(__name__)
    spec = importlib.util.spec_from_file_location(
        f"neuroseq.analysis_plugin.{plugin_file.stem}",
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


def discover_analysis_plugins() -> list[AnalysisPlugin]:
    logger = logging.getLogger(__name__)
    plugins: list[AnalysisPlugin] = []
    plugins_dir = _plugins_directory()
    if not plugins_dir.is_dir():
        return plugins

    for plugin_file in sorted(plugins_dir.glob("*.py")):
        if plugin_file.stem.startswith("_"):
            continue
        module = _load_plugin_module(plugin_file)
        if module is None:
            continue

        render = getattr(module, "render_analysis", None)
        if not callable(render):
            logger.warning(
                f"Skipping analysis plugin {plugin_file}: missing callable render_analysis"
            )
            continue

        plugin_id = getattr(module, "PLUGIN_ID", plugin_file.stem)
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
                id=str(plugin_id),
                label=str(plugin_label),
                icon=str(plugin_icon),
                render=render,
                open_dialog=open_dialog,
            )
        )

    return plugins
