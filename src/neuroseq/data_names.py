"""Qt-free naming rules for project data groups and data sets.

Project data is listed as ``"<group> :: <data set>"``; the simulation's data
uses the reserved group name ``"Simulation"``. Every place that checks or makes
up such a name (renaming in the data manager, imports, the trace comparison's
concatenated group, calcium videos) uses these helpers so the rules agree.
"""

from __future__ import annotations

from collections.abc import Iterable

# Separator between a data group (experiment) and a data set in project data
# names ("Repeat 1 :: dF/F"); neither part may contain it.
NAME_SEPARATOR = " :: "
# The separator's core: a name containing it would split ambiguously.
FORBIDDEN_NAME_PART = NAME_SEPARATOR.strip()
# Group name of the simulation's data; no imported data group may use it.
SIMULATION_GROUP = "Simulation"


def split_data_name(name: str) -> tuple[str, str]:
    """``("group", "data set")`` of a project data name (``""`` if no data set)."""
    group, _, data = name.partition(NAME_SEPARATOR)
    return group, data


def data_name_problem(name: str) -> str:
    """Why ``name`` cannot be a data group or data set name ('' when it can)."""
    name = name.strip()
    if not name:
        return "The name cannot be empty."
    if FORBIDDEN_NAME_PART in name:
        return f"The name cannot contain '{FORBIDDEN_NAME_PART}'."
    return ""


def group_name_problem(name: str, existing: Iterable[str]) -> str:
    """Why ``name`` cannot be a new data group's name ('' when it can):
    empty, containing ``'::'``, ``"Simulation"`` or an existing group."""
    problem = data_name_problem(name)
    if problem:
        return problem
    name = name.strip()
    if name == SIMULATION_GROUP or name in set(existing):
        return f"A data group named '{name}' already exists."
    return ""


def unique_name(base: str, existing: Iterable[str]) -> str:
    """``base``, or ``"base (2)"``, ``"base (3)"`` … when already taken."""
    taken = set(existing)
    if base not in taken:
        return base
    number = 2
    while f"{base} ({number})" in taken:
        number += 1
    return f"{base} ({number})"


def unique_group_name(base: str, existing: Iterable[str]) -> str:
    """A valid, unused data group name made from ``base`` (e.g. a file name).

    ``'::'`` is replaced, an empty base becomes ``"Data"``, and the reserved
    ``"Simulation"`` is treated as taken, so the result always passes
    :func:`group_name_problem`.
    """
    name = base.replace(FORBIDDEN_NAME_PART, "_").strip() or "Data"
    return unique_name(name, {*existing, SIMULATION_GROUP})
