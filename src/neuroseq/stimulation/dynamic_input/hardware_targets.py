"""Helpers shared by the routine dialogs that can run on hardware targets
(rheobase, STDP window): following the stimulation hosts' signals, and keeping
each target's parameter values (stim units for Brian2, fractions on hardware).
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

from qtpy import QtWidgets

if TYPE_CHECKING:
    from neuroseq.stimulation.stim_editor import StimEditor

#: (signal, slot) pairs to connect for one host.
Slots = list[tuple[Any, Callable]]


class HostConnections:
    """The (signal, slot) pairs a dialog connected, per host, so it disconnects
    exactly those, even if the editor's hosts were replaced meanwhile."""

    def __init__(self) -> None:
        self._connected: dict[int, Slots] = {}

    def connect(self, host: object, slots: Slots) -> None:
        """Connect ``slots`` of ``host`` unless that host is connected already."""
        if id(host) in self._connected:
            return
        for signal, slot in slots:
            signal.connect(slot)
        self._connected[id(host)] = slots

    def disconnect_all(self) -> None:
        for slots in self._connected.values():
            for signal, slot in slots:
                signal.disconnect(slot)
        self._connected.clear()


class PerTargetValues:
    """Remembers some spin boxes' values per stimulation target.

    Switching target stores the old target's values and shows the new target's:
    what it had before, else its defaults (``defaults(target)``; ``None`` keeps the
    values shown).
    """

    def __init__(
        self,
        spins: Sequence[QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox],
        defaults: Callable[[str], Sequence[float] | None],
        target: str,
    ) -> None:
        self._spins = tuple(spins)
        self._defaults = defaults
        self._target = target
        self._values: dict[str, tuple[float, ...]] = {}

    def switch(self, target: str) -> None:
        if target == self._target:
            return
        self._values[self._target] = tuple(spin.value() for spin in self._spins)
        self._target = target
        values = self._values.get(target)
        if values is None:
            defaults = self._defaults(target)
            values = tuple(defaults) if defaults is not None else None
        if values is None:
            return
        for spin, value in zip(self._spins, values, strict=True):
            if isinstance(spin, QtWidgets.QSpinBox):
                spin.setValue(int(value))
            else:
                spin.setValue(float(value))


def hardware_start_sample(editor: StimEditor, choice: str) -> int:
    """Where a hardware routine places its first block (no simulation to catch up
    with): at 0 after clearing the array (``choice == "clear"``), else right
    after its last stimulus."""
    if choice == "clear":
        editor.zero_current_stim_array()
        editor.reset_stim_array()
        return 0
    return editor.current_stim_array_last_stimulus_sample()


def delivery_settings_note(host: Any) -> str:
    """The laser and power a SysCon2 routine delivers with, for its log at Start,
    so a wrong laser row is seen before the first block."""
    settings = host.export_settings()
    if settings is None:
        return "No export settings: set them under Hardware settings."
    return (
        f"Delivering on the {settings.laser_wavelength_nm} nm laser at "
        f"{settings.target_power_mw:g} mW (amplitude 1); change them under "
        "Hardware settings."
    )
