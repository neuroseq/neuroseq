"""Standalone Python 3.10+ MATLAB Engine helper; JSON lines over stdio.

Executed by file path, without importing NeuroSeq. Install the engine matching
the rig's MATLAB release, numpy and tifffile in this interpreter. stdout is
reserved for protocol replies. No arbitrary MATLAB evaluation is exposed.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any


class EngineAdapter:
    def __init__(self) -> None:
        self.engine: Any = None

    def dispatch(self, operation: str, arguments: dict) -> dict:
        if operation == "connect":
            import matlab.engine

            name = arguments["session"]
            if not name or name not in matlab.engine.find_matlab():
                raise ValueError(
                    "Named MATLAB session not found. Run matlab.engine.shareEngine('NeuroSeqScanImage') in the ScanImage MATLAB window."
                )
            self.engine = matlab.engine.connect_matlab(name)
            self.engine.addpath(
                str(Path(__file__).parent / "resources" / "scanimage"), nargout=0
            )
            return self._call("probe", {})
        if operation == "index":
            from importer_scanimage import index_recording

            return index_recording(arguments["paths"])
        if operation not in {"status", "preview", "record", "abort", "disconnect"}:
            raise ValueError("Unknown ScanImage operation")
        return self._call(operation, arguments)

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
