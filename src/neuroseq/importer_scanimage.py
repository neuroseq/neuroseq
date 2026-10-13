"""Index completed single-plane ScanImage TIFF recordings without loading movies.

This module also runs in the MATLAB helper's Python 3.10 environment. Keep it
independent of NeuroSeq/Qt and preserve source pixels and timestamps.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


def index_recording(paths: list[str]) -> dict:
    frames: list[dict] = []
    channels: list[int] = []
    fps = 0.0
    first_time: float | None = None
    for filename in paths:
        with tifffile.TiffFile(filename) as tif:
            metadata = tif.scanimage_metadata or {}
            settings = metadata.get("FrameData", {})
            saved = settings.get("SI.hChannels.channelSave", [])
            saved = np.atleast_1d(saved).astype(int).tolist()
            if not saved:
                raise ValueError("Missing ScanImage saved-channel metadata")
            if channels and saved != channels:
                raise ValueError("Saved channels changed within the recording")
            channels = saved
            if int(settings.get("SI.hStackManager.numSlices", 1)) != 1:
                raise ValueError("Only single-plane ScanImage recordings are supported")
            if settings.get("SI.hRoiManager.mroiEnable", False):
                raise ValueError("Multi-ROI recordings are not supported yet")
            fps = float(settings.get("SI.hRoiManager.scanFrameRate", 0))
            for i, page in enumerate(tif.pages):
                if len(page.shape) != 2:
                    raise ValueError("Expected a two-dimensional ScanImage TIFF page")
                if any(
                    offset + count > tif.filehandle.size
                    for offset, count in zip(
                        page.dataoffsets, page.databytecounts, strict=True
                    )
                ):
                    raise ValueError("ScanImage TIFF pixel data is incomplete")
                description = getattr(page, "description", "") or ""
                match = re.search(
                    r"(?:^|\n)frameTimestamps_sec\s*=\s*([-+\deE.]+)", description
                )
                if match is None:
                    raise ValueError("Missing ScanImage frame timestamp")
                timestamp = float(match[1])
                if not np.isfinite(timestamp):
                    raise ValueError("Invalid ScanImage frame timestamp")
                if first_time is None:
                    first_time = timestamp
                frames.append(
                    {
                        "path": str(Path(filename)),
                        "page": i,
                        "channel": channels[i % len(channels)],
                        "time_s": timestamp - first_time,
                        "scanimage_time_s": timestamp,
                        "description": description,
                    }
                )
            if len(tif.pages) % len(channels):
                raise ValueError("Incomplete channel group in ScanImage recording")
    if not frames or fps <= 0 or not np.isfinite(fps):
        raise ValueError("Recording has no frames or a valid frame rate")
    for channel in channels:
        times = [f["time_s"] for f in frames if f["channel"] == channel]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Frame timestamps are not strictly increasing")
    return {
        "paths": paths,
        "channels": channels,
        "frames": frames,
        "fps": fps,
        "duration_s": max(f["time_s"] for f in frames) + 1 / fps,
    }
