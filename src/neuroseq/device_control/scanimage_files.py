"""Names of NeuroSeq's ScanImage recordings (issues #628, #651). Qt-free.

Each recording is one TIFF directly in the recording folder, ``<date>_<number>.tif``
(ScanImage appends the number to the stem), with ``<date>_<number>.json`` beside it.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path


def recording_stem(day: datetime.date | None = None) -> str:
    """File-name stem of today's recordings, e.g. ``2026-10-22``."""
    return (day or datetime.date.today()).isoformat()


def next_recording_number(directory: Path, stem: str) -> int:
    """One more than the highest recording number of ``stem`` in ``directory``."""
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)(?:_\d+)?\.(?:tif|tiff|json)$")
    numbers = [
        int(match[1])
        for path in (directory.iterdir() if directory.is_dir() else ())
        if (match := pattern.match(path.name))
    ]
    return max(numbers, default=0) + 1


def recording_file_name(stem: str, number: int) -> str:
    """ScanImage's name for recording ``number`` of ``stem``."""
    return f"{stem}_{number:05d}.tif"
