from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files as _resource_files

_KV_RE = re.compile(r"^(\w+)=(.*)$")
_TUPLE_RE = re.compile(r"^Tuple(\d+)$")


@dataclass(frozen=True)
class PowerLut:
    """SysCon .lut: ordered (percent, mW) breakpoints for a single laser channel."""

    description: str
    tuples: tuple[tuple[float, float], ...]

    @property
    def max_mw(self) -> float:
        return max(mw for _, mw in self.tuples) if self.tuples else 0.0

    def to_serializable(self) -> dict:
        return {
            "description": self.description,
            "tuples": [list(pair) for pair in self.tuples],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> PowerLut:
        return cls(
            description=str(data.get("description", "")),
            tuples=tuple((float(p), float(mw)) for p, mw in data.get("tuples", [])),
        )


def parse_power_lut_text(text: str) -> PowerLut:
    """Parse a SysCon `.lut` file."""

    description = ""
    tuples_by_idx: dict[int, tuple[float, float]] = {}
    unit = "mW"
    in_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[_]":
            in_block = True
            continue
        if line == "[/_]":
            in_block = False
            continue
        if not in_block:
            continue
        match = _KV_RE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key == "Description":
            description = value
        elif key == "Unit":
            unit = value
        elif key.startswith("Tuple"):
            idx_match = _TUPLE_RE.match(key)
            if idx_match is None:
                continue
            idx = int(idx_match.group(1))
            if ";" not in value:
                raise ValueError(f"Malformed LUT tuple at {key}: {value!r}")
            percent_str, mw_str = value.split(";", 1)
            tuples_by_idx[idx] = (float(percent_str), float(mw_str))

    if unit != "mW":
        raise ValueError(f"Unsupported LUT unit {unit!r}; only 'mW' is supported")
    if not tuples_by_idx:
        raise ValueError("LUT contains no tuples")

    ordered = tuple(tuples_by_idx[idx] for idx in sorted(tuples_by_idx))
    for prev, curr in zip(ordered, ordered[1:], strict=False):
        if curr[0] < prev[0]:
            raise ValueError(
                f"LUT tuples must be sorted by percent (got {curr[0]} after {prev[0]})"
            )
    return PowerLut(description=description, tuples=ordered)


def power_lut_mw_to_percent(lut: PowerLut, target_mw: float) -> tuple[float, bool]:
    """Invert ``lut`` at ``target_mw`` as ``(percent, saturated)``."""

    if not lut.tuples:
        raise ValueError("LUT is empty")
    if target_mw <= 0:
        return 0.0, False

    prev_pct, prev_mw = lut.tuples[0]
    if prev_mw >= target_mw:
        return float(prev_pct), False

    for pct, mw in lut.tuples[1:]:
        if mw >= target_mw and mw > prev_mw:
            t = (target_mw - prev_mw) / (mw - prev_mw)
            return float(prev_pct + t * (pct - prev_pct)), False
        prev_pct, prev_mw = pct, mw

    max_mw = lut.max_mw
    saturating_pct = next(
        (pct for pct, mw in lut.tuples if mw >= max_mw),
        float(lut.tuples[-1][0]),
    )
    return float(saturating_pct), True


def load_bundled_default_luts() -> dict[int, PowerLut]:
    """Parse all bundled `default_<wavelength>nm.lut` files."""

    bundled: dict[int, PowerLut] = {}
    lut_dir = _resource_files("neuroseq.resources.syscon_luts")
    for entry in lut_dir.iterdir():
        name = entry.name
        if not name.startswith("default_") or not name.endswith("nm.lut"):
            continue
        try:
            wavelength_nm = int(name[len("default_") : -len("nm.lut")])
        except ValueError:
            continue
        bundled[wavelength_nm] = parse_power_lut_text(entry.read_text())
    return bundled


DEFAULT_POWER_LUTS: dict[int, PowerLut] = load_bundled_default_luts()


def resolve_power_lut(
    project_luts: dict[int, PowerLut],
    wavelength_nm: int,
) -> tuple[PowerLut | None, bool]:
    """Resolve the LUT to use at export time."""

    if wavelength_nm in project_luts:
        return project_luts[wavelength_nm], True
    return DEFAULT_POWER_LUTS.get(wavelength_nm), False
