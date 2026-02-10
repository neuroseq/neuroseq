from __future__ import annotations

import cProfile
import logging
import pathlib
import time
from collections.abc import Iterator
from contextlib import contextmanager

_profiler: cProfile.Profile | None = None
_profile_output: pathlib.Path | None = None


def configure_profiling(enabled: bool, output: pathlib.Path | None) -> None:
    global _profiler
    global _profile_output
    if not enabled:
        _profiler = None
        _profile_output = None
        return
    _profiler = cProfile.Profile()
    _profile_output = output or pathlib.Path("neuroseq.prof")


def start_profiling() -> None:
    if _profiler is None:
        return
    _profiler.enable()


def stop_profiling() -> None:
    if _profiler is None:
        return
    _profiler.disable()
    if _profile_output is None:
        return
    _profile_output.parent.mkdir(parents=True, exist_ok=True)
    _profiler.dump_stats(str(_profile_output))


@contextmanager
def profile_section(name: str) -> Iterator[None]:
    if _profiler is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.getLogger("neuroseq.profile").info("section %s %.3fs", name, elapsed)
