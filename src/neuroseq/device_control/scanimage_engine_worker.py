"""Standalone Python 3.10+ MATLAB Engine helper; JSON lines over stdio.

Executed by file path, without importing NeuroSeq. Install the engine matching
the rig's MATLAB release, numpy and tifffile in this interpreter. stdout is
reserved for protocol replies. No arbitrary MATLAB evaluation is exposed.
"""

from __future__ import annotations

import functools
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


@functools.cache
def _load_importer():
    """Load the Qt-free indexer without importing the NeuroSeq package (once)."""
    path = Path(__file__).resolve().parents[1] / "importers" / "importer_scanimage.py"
    spec = importlib.util.spec_from_file_location("importer_scanimage", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ScanImage indexer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EngineAdapter:
    def __init__(self) -> None:
        self.engine: Any = None
        self.manifest_path: Path | None = None

    def dispatch(self, operation: str, arguments: dict) -> dict:
        if operation == "connect":
            import matlab.engine

            name = arguments["session"]
            if not name or name not in matlab.engine.find_matlab():
                raise ValueError(
                    "Named MATLAB session not found. Run matlab.engine.shareEngine('NeuroSeqScanImage') in the ScanImage MATLAB window."
                )
            self.engine = matlab.engine.connect_matlab(name)
            _load_importer()
            self.engine.addpath(
                str(Path(__file__).resolve().parents[1] / "resources" / "scanimage"),
                nargout=0,
            )
            return self._call("probe", {})
        if operation == "index":
            return _load_importer().index_recording(arguments["paths"])
        if operation not in {"status", "preview", "record", "abort", "disconnect"}:
            raise ValueError("Unknown ScanImage operation")
        if operation == "record":
            return self._record(arguments)
        result = self._call(operation, arguments)
        if self.manifest_path is not None and operation == "status":
            # Lifecycle state is available in the project; the sidecar is the
            # immutable request/snapshot and remains valid after interruption.
            result["manifest_path"] = str(self.manifest_path)
        return result

    def _record(self, arguments: dict) -> dict:
        """Start a recording, with its metadata next to the TIFF (issue #628).

        The metadata is written before the hardware command, so it survives a GUI
        crash even if the project has not been saved yet, and removed again if
        ScanImage refuses to start.
        """
        manifest = arguments.pop("manifest", None)
        self.manifest_path = None
        path = None
        if manifest is not None:
            directory = Path(arguments["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            path = (
                directory / f"{arguments['stem']}_{int(arguments['counter']):05d}.json"
            )
            with path.open("x", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
        try:
            result = self._call("record", arguments)
        except Exception:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        self.manifest_path = path
        return result

    def _call(self, operation: str, arguments: dict) -> dict:
        if self.engine is None:
            raise ValueError("Not connected to MATLAB")
        return json.loads(
            self.engine.neuroseq_scanimage(
                operation,
                json.dumps(arguments),
                nargout=1,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        )


def main() -> None:
    adapter = EngineAdapter()
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                result = adapter.dispatch(
                    request["operation"], request.get("arguments", {})
                )
            response = {"id": request["id"], "result": result}
        except Exception as exc:
            response = {"id": request.get("id"), "error": str(exc)}
        print(json.dumps(response), flush=True)
        if request.get("operation") == "disconnect":
            break
    # Do not call engine.quit(): the microscope's MATLAB session belongs to the user.


if __name__ == "__main__":
    main()
