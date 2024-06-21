from __future__ import annotations

from re import escape

import brian2 as b2
import formozov_project.b2_model_widget as B2W
import numpy as np
import pytest
from formozov_project.b2_model import B2Model
from pytestqt.qtbot import QtBot


def test_quantity_conversion(qtbot: QtBot) -> None:
    wdgt = B2W.ModelDialog(B2Model(), None)

    qtbot.addWidget(wdgt)

    wdgt.show()

    value = "3.0 * volt"

    quantity = wdgt._convert_string_to_quantity(value, int)

    assert type(quantity) == b2.Quantity
    assert quantity == 3 * b2.volt

    wrong = "42 x"

    different_wrong = "42/volt"

    with pytest.raises(
        AttributeError,
        match="module 'brian2.units' has no attribute 'x'",
    ):
        wdgt._convert_string_to_quantity(wrong, int)

    with pytest.raises(
        ValueError,
        match=escape(
            "Quantities must follow the pattern 'value * unit ' or 'value unit'"
        ),
    ):
        wdgt._convert_string_to_quantity(different_wrong, int)

    array_value = "[3,2,1] * metre"

    quantity = wdgt._convert_string_to_quantity(array_value, np.ndarray)

    assert isinstance(quantity, b2.Quantity)
    assert np.all(quantity == np.ndarray([3, 2, 1]) * b2.metre)


def test_optional_conversion(qtbot: QtBot) -> None:
    wdgt = B2W.ModelDialog(B2Model(), None)

    qtbot.addWidget(wdgt)

    wdgt.show()

    value = "[3,2,1] * metre"

    quantity = wdgt._convert_value_optional(value, np.ndarray)
    assert isinstance(quantity, b2.Quantity)

    expr = "x * y : mVolt"

    quantity = wdgt._convert_value_optional(expr, np.ndarray)
    assert isinstance(quantity, str)
