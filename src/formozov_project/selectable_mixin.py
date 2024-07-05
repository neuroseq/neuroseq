from __future__ import annotations

import abc

from qtpy import QtCore


class QtMixinMeta(type(QtCore.QObject), abc.ABCMeta):  # type: ignore
    pass


class SelectableMixin(metaclass=QtMixinMeta):
    """Adds a set_selected_slice method which by default does nothing."""

    @abc.abstractmethod
    def set_selected_slice(self, selected_slice: slice) -> None:
        pass
