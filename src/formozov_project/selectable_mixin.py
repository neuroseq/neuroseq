from __future__ import annotations


class SelectableMixin:
    """Adds a set_selected_slice method which by default does nothing."""

    def set_selected_slice(self, selected_slice: slice) -> None:
        pass
