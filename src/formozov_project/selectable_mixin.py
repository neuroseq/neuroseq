from __future__ import annotations

import abc

from formozov_project.utils import QtABCMeta


class SelectableMixin(metaclass=QtABCMeta):
    """Adds a set_selected_slice method which by default does nothing."""

    @abc.abstractmethod
    def set_selected_slice(self, selected_slice: slice) -> None:
        # fallback for pyside6 which apparently does not fully support abstract methods in a metaclass
        raise NotImplementedError("Subclasses must implement set_selected_slice")
